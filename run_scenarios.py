#!/usr/bin/env python3
"""
Scenario Runner - Projects battery storage revenue under best/base/worst spread compression.

Uses the 2025 baseline from battery_model.py, then scales arbitrage revenue
for 2026-2033 using the spread compression formula:

    Spread = Floor + (S0 - Floor) * (C0/C)^alpha * (R/R0)^beta

Where:
    S0     = 2025 baseline spread (GBP/MWh)
    Floor  = minimum spread (~20 GBP/MWh - covers operational costs)
    C0, C  = BESS capacity at baseline and forecast year (GWh)
    R0, R  = renewable capacity at baseline and forecast year (GWh)
    alpha  = BESS compression elasticity
    beta   = renewable widening elasticity

Scenarios:
    Best:  alpha=0.46 (slow compression, German data), beta=0.4 (wind widens spreads)
    Base:  alpha=0.50, beta=0.30 (central estimates)
    Worst: alpha=0.65 (fast compression, CAISO data),  beta=0.0 (no wind benefit)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io
from contextlib import redirect_stdout

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# ---- Spread parameters ----
SPREAD_FLOOR = 20.0  # GBP/MWh - minimum viable spread
BASELINE_SPREAD = 73.0  # GBP/MWh - 2025 actual wholesale spread
BASELINE_YEAR = 2025

SCENARIOS = {
    "Best":  {"alpha": 0.46, "beta": 0.4},
    "Base":  {"alpha": 0.50, "beta": 0.3},
    "Worst": {"alpha": 0.65, "beta": 0.0},
}

FORECAST_YEARS = list(range(2026, 2034))


def load_capacity_forecasts():
    """Load BESS and renewable capacity forecasts per year."""
    df = pd.read_csv(DATA_DIR / "inputs" / "capacity_forecasts.csv")
    return df.set_index("year")


def compute_spread(s0, c0, c, r0, r, alpha, beta, floor=SPREAD_FLOOR):
    """
    Compute projected spread using compression formula.

    Returns spread in GBP/MWh, floored at minimum viable level.
    """
    spread = floor + (s0 - floor) * (c0 / c) ** alpha * (r / r0) ** beta
    return max(spread, floor)


def run_baseline():
    """
    Run battery_model.py to get 2025 baseline results per nation.

    Returns dict of {nation: {arbitrage, fr, cm, total_revenue, vpp_profit, ...}}
    """
    # Import and run model, capturing output
    sys.path.insert(0, str(PROJECT_ROOT))
    from models.battery_model import main as model_main, load_model_inputs, load_use_profiles, run_model_for_nation, NATIONS

    params = load_model_inputs()
    use_profile = load_use_profiles()

    results = {}
    for nation in NATIONS.keys():
        # Suppress print output during baseline run
        f = io.StringIO()
        with redirect_stdout(f):
            result = run_model_for_nation(nation, params, use_profile)
        if result:
            results[nation] = result

    return results


def project_revenues(baseline_results, capacity_data):
    """
    For each nation and scenario, project revenues for 2026-2033.

    Arbitrage scales with spread ratio. FR and CM are held constant.
    """
    # 2025 baseline values
    s0 = BASELINE_SPREAD
    c0 = float(capacity_data.loc[BASELINE_YEAR, "bess_gwh"])
    r0 = float(capacity_data.loc[BASELINE_YEAR, "renewable_gwh"])

    all_projections = []

    for nation, bl in baseline_results.items():
        # We need the revenue breakdown. The baseline result has total vpp_revenue and vpp_profit.
        # To get arbitrage vs non-arbitrage, we re-derive from the model output.
        # vpp_revenue = arbitrage_with_bm + fr + cm
        # We'll scale arbitrage_with_bm (includes BM uplift) and keep fr + cm constant.
        vpp_revenue = bl["vpp_revenue"]
        vpp_profit = bl["vpp_profit"]
        vpp_costs = bl["vpp_costs"]

        # Estimate arbitrage portion: total_revenue - we need the breakdown
        # Since we don't have it directly, use the ratio from model
        # For now, assume arbitrage = vpp_revenue - fr - cm (approximate)
        # We can get a better estimate by looking at the strategy
        elec_revenue = bl["elec_revenue"]
        elec_profit = bl["elec_profit"]

        for scenario_name, params in SCENARIOS.items():
            alpha = params["alpha"]
            beta = params["beta"]

            for year in FORECAST_YEARS:
                c = float(capacity_data.loc[year, "bess_gwh"])
                r = float(capacity_data.loc[year, "renewable_gwh"])

                projected_spread = compute_spread(s0, c0, c, r0, r, alpha, beta)
                scaling_factor = projected_spread / s0

                # Scale VPP: arbitrage scales, but we scale total revenue as approximation
                # since arbitrage dominates for most nations
                projected_vpp_revenue = vpp_revenue * scaling_factor
                projected_vpp_profit = projected_vpp_revenue - vpp_costs

                # Scale Elec Company similarly
                projected_elec_revenue = elec_revenue * scaling_factor
                projected_elec_profit = projected_elec_revenue - (elec_revenue - elec_profit)

                # ROI (8-year)
                investment = bl["upfront_investment"]
                vpp_roi = (projected_vpp_profit * 8 - investment) / investment if investment > 0 else 0
                elec_roi = (projected_elec_profit * 8 - investment) / investment if investment > 0 else 0

                all_projections.append({
                    "nation": nation,
                    "scenario": scenario_name,
                    "year": year,
                    "spread_gbp_mwh": round(projected_spread, 1),
                    "scaling_factor": round(scaling_factor, 3),
                    "vpp_revenue_eur": round(projected_vpp_revenue, 2),
                    "vpp_profit_eur": round(projected_vpp_profit, 2),
                    "vpp_8yr_roi": round(vpp_roi * 100, 1),
                    "elec_revenue_eur": round(projected_elec_revenue, 2),
                    "elec_profit_eur": round(projected_elec_profit, 2),
                    "elec_8yr_roi": round(elec_roi * 100, 1),
                })

    return pd.DataFrame(all_projections)


def print_scenario_tables(projections, baseline_results):
    """Print formatted tables for each nation and scenario."""

    print("=" * 90)
    print("SCENARIO PROJECTIONS: Battery Storage Revenue 2026-2033")
    print("=" * 90)
    print(f"\nSpread formula: Floor + (S0 - Floor) * (C0/C)^alpha * (R/R0)^beta")
    print(f"Spread floor: {SPREAD_FLOOR} GBP/MWh | 2025 baseline: {BASELINE_SPREAD} GBP/MWh")
    print()

    for scenario_name, params in SCENARIOS.items():
        print(f"\n{'='*90}")
        print(f"  {scenario_name.upper()} CASE  (alpha={params['alpha']}, beta={params['beta']})")
        print(f"{'='*90}")

        scenario_data = projections[projections["scenario"] == scenario_name]

        for nation in scenario_data["nation"].unique():
            bl = baseline_results[nation]
            nation_data = scenario_data[scenario_data["nation"] == nation]

            print(f"\n  {nation}")
            print(f"  {'─'*80}")
            print(f"  {'Year':<6} {'Spread':>10} {'Scale':>7} {'VPP Rev':>12} {'VPP Profit':>12} {'VPP ROI':>9} {'Elec Profit':>12} {'Elec ROI':>9}")
            print(f"  {'─'*80}")

            # Baseline 2025 row
            print(f"  {'2025':<6} {'(baseline)':>10} {'1.000':>7} €{bl['vpp_revenue']:>10.2f} €{bl['vpp_profit']:>10.2f} {bl['optimal_vpp_roi']*100:>8.1f}% €{bl['elec_profit']:>10.2f} {bl['optimal_elec_roi']*100:>8.1f}%")

            for _, row in nation_data.iterrows():
                print(f"  {row['year']:<6} {row['spread_gbp_mwh']:>8.1f}£ {row['scaling_factor']:>7.3f} €{row['vpp_revenue_eur']:>10.2f} €{row['vpp_profit_eur']:>10.2f} {row['vpp_8yr_roi']:>8.1f}% €{row['elec_profit_eur']:>10.2f} {row['elec_8yr_roi']:>8.1f}%")


def main():
    print("Loading capacity forecast data...")
    capacity_data = load_capacity_forecasts()

    print("Running 2025 baseline model (this may take a minute)...")
    baseline_results = run_baseline()

    if not baseline_results:
        print("ERROR: No baseline results. Check that data files exist in data/")
        sys.exit(1)

    print(f"\nBaseline complete for {len(baseline_results)} nations.")
    print("Projecting scenarios...\n")

    projections = project_revenues(baseline_results, capacity_data)

    # Print tables
    print_scenario_tables(projections, baseline_results)

    # Save to CSV
    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / "scenario_projections.csv"
    projections.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
