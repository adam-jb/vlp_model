"""
VLP Battery — Agile Consumer Model
====================================
Battery + installation total: £2,800

The consumer stays on Octopus Agile; the company operates the battery as a
Virtual Lead Party (VLP) earning grid services revenue.  Consumer benefit =
Agile arbitrage savings.  Company revenue = wholesale arb + FR + CM + BM.

No supply cost (company doesn't supply electricity), no network charges,
no grid levy (company is VLP, not electricity supplier).

Three spread-compression scenarios × three consumption levels
(4000 / 5000 / 6000 kWh/yr), four loan periods.

Joint MILP splits discharge between consumer (saves Agile rate) and grid
(earns wholesale rate) hour-by-hour.

Outputs saved to this folder (vlp_agile/):
  pl_{Best,Base,Worst}_{consumption}.csv   — P&L
  cashflow_{Best,Base,Worst}_{consumption}.csv — cash flow statements
  loan_schedule.csv                         — amortisation schedules
  report.html                               — interactive HTML report
"""

import sys
import textwrap
import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json

try:
    import pulp
except ImportError:
    print("ERROR: PuLP is required.  pip install pulp")
    sys.exit(1)

# -- paths -------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_DIR      = SCRIPT_DIR

# -- battery hardware (15.36 kWh / 6 kW system) ------------------------------
BAT_KWH      = 15.36
BAT_KW       = 6.0
EFF_RT       = 0.90
ETA          = EFF_RT ** 0.5          # one-way ~ 0.9487

BAT_COST     = 2_800.00
BAT_LIFE     = 10
DEPRECIATION = BAT_COST / BAT_LIFE

LOAN_RATE    = 0.07
CORP_TAX     = 0.25

# -- loan scenarios -----------------------------------------------------------
LOAN_YEARS   = [3, 5, 7, 10]
MODEL_YEARS  = 12

# -- consumption levels -------------------------------------------------------
CONSUMPTION_LEVELS = [4000, 5000, 6000]   # kWh/yr

# -- battery degradation ------------------------------------------------------
DEGRADATION_PA = 0.02   # 2% capacity loss per year

def degradation_factor(year):
    """Capacity retention factor at start of a given year (year 1 = 1.0)."""
    return (1 - DEGRADATION_PA) ** (year - 1)

# -- EUR/GBP ------------------------------------------------------------------
EUR_TO_GBP   = 0.85

# -- load model_input.csv — UK column ----------------------------------------
def load_model_inputs():
    df = pd.read_csv(PROJECT_ROOT / "data" / "inputs" / "model_input.csv")
    params = {}
    for _, row in df.iterrows():
        key = row["input"]
        val = row["UK"]
        if pd.isna(val) or val == "":
            params[key] = 0.0
            continue
        s = str(val).strip().replace("%", "")
        try:
            params[key] = float(s) / 100 if "%" in str(val) else float(s)
        except ValueError:
            params[key] = 0.0
    return params

PARAMS = load_model_inputs()

# -- grid services rates from CSV --------------------------------------------
FR_PER_KW    = PARAMS.get("Frequency response per kw per year", 34)
CM_PER_KW    = PARAMS.get("Capacity market /kW/year", 40)
CM_DERATING  = 0.2094   # de-rating: 2-hr battery class
BM_WIN_RATE  = PARAMS.get("Balancing mechanism (BM) avg win rate", 0.10)
BM_UPLIFT    = PARAMS.get("Avg £/kwh uplift for BM", 0.05)

# -- VLP cost lines from CSV -------------------------------------------------
CAC          = PARAMS.get("customer acquisition cost (CAC)", 100)
STAFF_COST   = PARAMS.get("annual staff cost per customer", 50)
CUST_MGMT    = PARAMS.get("Cost of managing customers per customer per annum if VLP", 10)
GRID_LEVY    = 0   # VLP is NOT an electricity supplier — no levy

# -- load hourly consumption profile -----------------------------------------
def load_use_profile():
    df = pd.read_csv(PROJECT_ROOT / "data" / "inputs" / "use_profiles.csv")
    uk = df[df["Nation"] == "UK"].sort_values("hour")
    if len(uk) < 24:
        print("  WARNING: UK use profile not found or incomplete, using flat profile")
        return np.ones(24) / 24
    return uk["consumption_kwh"].values.astype(float)

HOURLY_SHAPE = load_use_profile()              # raw Wh values (sum ~ 9875 Wh)
SHAPE_DAILY_WH = HOURLY_SHAPE.sum()            # daily total in Wh
SHAPE_ANNUAL_KWH = SHAPE_DAILY_WH * 365 / 1000 # ~ 3604.4 kWh

# -- spread compression -------------------------------------------------------
SPREAD_FLOOR    = 20.0   # GBP/MWh
BASELINE_SPREAD = 73.0   # GBP/MWh 2025
BASELINE_YEAR   = 2025

SPREAD_SCENARIOS = {
    "Best":  {"alpha": 0.46, "beta": 0.4},
    "Base":  {"alpha": 0.50, "beta": 0.3},
    "Worst": {"alpha": 0.65, "beta": 0.0},
}

def load_capacity_forecasts():
    df = pd.read_csv(PROJECT_ROOT / "data" / "inputs" / "capacity_forecasts.csv")
    return df.set_index("year")

CAP_DATA = load_capacity_forecasts()

def compute_spread(s0, c0, c, r0, r, alpha, beta, floor=SPREAD_FLOOR):
    spread = floor + (s0 - floor) * (c0 / c) ** alpha * (r / r0) ** beta
    return max(spread, floor)

def get_scaling_factors(spread_name):
    p = SPREAD_SCENARIOS[spread_name]
    c0 = float(CAP_DATA.loc[BASELINE_YEAR, "bess_gwh"])
    r0 = float(CAP_DATA.loc[BASELINE_YEAR, "renewable_gwh"])
    factors = {}
    for yr_offset in range(1, MODEL_YEARS + 1):
        cal_year = BASELINE_YEAR + yr_offset - 1
        if cal_year in CAP_DATA.index:
            c = float(CAP_DATA.loc[cal_year, "bess_gwh"])
            r = float(CAP_DATA.loc[cal_year, "renewable_gwh"])
            sp = compute_spread(BASELINE_SPREAD, c0, c, r0, r, p["alpha"], p["beta"])
            factors[yr_offset] = sp / BASELINE_SPREAD
        else:
            last_yr = CAP_DATA.index.max()
            c = float(CAP_DATA.loc[last_yr, "bess_gwh"])
            r = float(CAP_DATA.loc[last_yr, "renewable_gwh"])
            sp = compute_spread(BASELINE_SPREAD, c0, c, r0, r, p["alpha"], p["beta"])
            factors[yr_offset] = sp / BASELINE_SPREAD
    return factors

# -- chart colours -------------------------------------------------------------
LOAN_COLORS   = {3: "#E04B3A", 5: "#F5A623", 7: "#4A90D9", 10: "#27AE60"}
CONS_COLORS   = {4000: "#4A90D9", 5000: "#F5A623", 6000: "#E04B3A"}
SPREAD_COLORS = {"Best": "#27AE60", "Base": "#F5A623", "Worst": "#E04B3A"}


# =============================================================================
# AGILE + WHOLESALE DATA LOADING
# =============================================================================

AGILE_CACHE_FILE = PROJECT_ROOT / "data" / "agile" / "agile_rates_2025_cache.csv"

def load_agile_cache():
    """Load cached Agile rates and return hourly averages."""
    df = pd.read_csv(AGILE_CACHE_FILE, parse_dates=["date"])
    # Average half-hourly to hourly
    hourly = df.groupby(["date", "hour"])["rate_p_kwh"].mean().reset_index()
    hourly["date"] = pd.to_datetime(hourly["date"]).dt.date
    return hourly

def load_wholesale_prices():
    """Load UK wholesale day-ahead prices for 2025 (matching Agile cache year)."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "prices" / "United Kingdom.csv")
    df["dt"]   = pd.to_datetime(df["Datetime (UTC)"])
    df         = df[df["dt"].dt.year == 2025].copy()
    df["date"] = df["dt"].dt.date
    df["hour"] = df["dt"].dt.hour
    return df


# =============================================================================
# JOINT MILP: CONSUMER (AGILE) + GRID (WHOLESALE) DISCHARGE SPLIT
# =============================================================================

def solve_daily_milp(agile_prices_24h, wholesale_prices_24h, consumer_demand_24h,
                     battery_kwh, battery_kw, efficiency, fr_rate_hourly):
    """
    Solve daily MILP: jointly optimise consumer savings (Agile) + grid arb
    (wholesale) + FR.

    Decision variables per hour:
      c[h]          : charge power (kW)
      d_consumer[h] : discharge to consumer — saves them Agile rate
      d_grid[h]     : discharge to grid — earns wholesale rate
      fr[h]         : binary FR commitment

    Returns dict with daily profit breakdown.
    """
    eta = np.sqrt(efficiency)
    hours = range(24)
    E_start = battery_kwh * 0.5

    prob = pulp.LpProblem("Daily_Battery_Opt", pulp.LpMaximize)

    c          = {h: pulp.LpVariable(f"c_{h}", 0, battery_kw)          for h in hours}
    d_consumer = {h: pulp.LpVariable(f"dc_{h}", 0, battery_kw)         for h in hours}
    d_grid     = {h: pulp.LpVariable(f"dg_{h}", 0, battery_kw)         for h in hours}
    fr         = {h: pulp.LpVariable(f"fr_{h}", cat="Binary")          for h in hours}
    E          = {h: pulp.LpVariable(f"E_{h}", 0, battery_kwh)         for h in range(25)}

    consumer_value = []
    grid_revenue   = []
    charge_cost    = []
    fr_income      = []

    for h in hours:
        agile_gbp     = agile_prices_24h[h] / 100          # p/kWh -> GBP/kWh
        wholesale_gbp = (wholesale_prices_24h[h] / 1000) * EUR_TO_GBP  # EUR/MWh -> GBP/kWh

        consumer_value.append(d_consumer[h] * agile_gbp)
        grid_revenue.append(d_grid[h] * wholesale_gbp)
        charge_cost.append(c[h] * wholesale_gbp)
        fr_income.append(fr[h] * fr_rate_hourly * battery_kw)

    prob += (pulp.lpSum(consumer_value) + pulp.lpSum(grid_revenue)
             + pulp.lpSum(fr_income) - pulp.lpSum(charge_cost)), "Total_Value"

    # Constraints
    prob += E[0]  == E_start, "Initial_SoC"
    prob += E[24] == E_start, "Final_SoC"

    for h in hours:
        prob += E[h+1] == E[h] + c[h] * eta - (d_consumer[h] + d_grid[h]) / eta, f"EB_{h}"
        prob += d_consumer[h] <= consumer_demand_24h[h], f"Demand_{h}"
        prob += d_consumer[h] + d_grid[h] <= battery_kw, f"DischLim_{h}"
        prob += c[h] <= battery_kw * (1 - fr[h]), f"FRblkC_{h}"
        prob += d_consumer[h] + d_grid[h] <= battery_kw * (1 - fr[h]), f"FRblkD_{h}"

    M = battery_kwh
    for h in hours:
        prob += E[h] >= E_start - M * (1 - fr[h]), f"FRsocL_{h}"
        prob += E[h] <= E_start + M * (1 - fr[h]), f"FRsocU_{h}"

    # One cycle per day
    prob += pulp.lpSum([d_consumer[h] + d_grid[h] for h in hours]) <= battery_kwh, "DayDisch"
    prob += pulp.lpSum([c[h] for h in hours]) <= battery_kwh, "DayCharge"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        return {"status": "Failed", "consumer_savings": 0, "wholesale_arbitrage": 0,
                "fr_income": 0, "total": 0, "fr_hours": 0,
                "consumer_kwh": 0, "grid_kwh": 0}

    # Extract results
    total_consumer_value = sum(
        (d_consumer[h].varValue or 0) * (agile_prices_24h[h] / 100) for h in hours)
    total_grid_revenue = sum(
        (d_grid[h].varValue or 0) * ((wholesale_prices_24h[h] / 1000) * EUR_TO_GBP) for h in hours)
    total_charge_cost = sum(
        (c[h].varValue or 0) * ((wholesale_prices_24h[h] / 1000) * EUR_TO_GBP) for h in hours)
    total_fr = sum(
        (fr[h].varValue or 0) * fr_rate_hourly * battery_kw for h in hours)

    total_discharge = sum((d_consumer[h].varValue or 0) + (d_grid[h].varValue or 0) for h in hours)
    total_consumer_discharge = sum((d_consumer[h].varValue or 0) for h in hours)
    total_grid_discharge = sum((d_grid[h].varValue or 0) for h in hours)

    if total_discharge > 0:
        consumer_share = total_consumer_discharge / total_discharge
        grid_share     = total_grid_discharge / total_discharge
    else:
        consumer_share = 0.5
        grid_share     = 0.5

    consumer_charge_cost = total_charge_cost * consumer_share
    grid_charge_cost     = total_charge_cost * grid_share

    net_consumer_savings  = total_consumer_value - consumer_charge_cost
    net_wholesale_profit  = total_grid_revenue - grid_charge_cost

    return {
        "status": "Optimal",
        "consumer_savings": net_consumer_savings,
        "wholesale_arbitrage": net_wholesale_profit,
        "fr_income": total_fr,
        "total": net_consumer_savings + net_wholesale_profit + total_fr,
        "fr_hours": sum(int(fr[h].varValue or 0) for h in hours),
        "consumer_kwh": total_consumer_discharge,
        "grid_kwh": total_grid_discharge,
    }


# =============================================================================
# ANNUAL OPTIMISATION
# =============================================================================

def optimize_annual(agile_hourly, wholesale_df, annual_kwh):
    """
    Run daily MILP across the year. Returns annual totals for consumer savings,
    wholesale arb, FR, CM, BM.
    """
    efficiency   = EFF_RT
    fr_rate_hourly = FR_PER_KW / (365 * 24)

    # Normalise consumption profile
    profile_total = HOURLY_SHAPE.sum()
    profile_norm  = HOURLY_SHAPE / profile_total       # fractional shape
    daily_kwh     = annual_kwh / 365

    # Build per-hour consumer demand (kWh per hour)
    consumer_demand_24h = [daily_kwh * profile_norm[h] / 1000 * 1000
                           for h in range(24)]
    # Simplify: profile_norm[h] is Wh / total Wh, daily_kwh is kWh
    consumer_demand_24h = [(daily_kwh * (HOURLY_SHAPE[h] / SHAPE_DAILY_WH))
                           for h in range(24)]

    agile_dates     = set(agile_hourly["date"].unique())
    wholesale_dates = set(wholesale_df["date"].unique())
    common_dates    = agile_dates & wholesale_dates

    total_consumer = 0.0
    total_wholesale = 0.0
    total_fr = 0.0
    days = 0

    for date in sorted(common_dates):
        ag = agile_hourly[agile_hourly["date"] == date].sort_values("hour")
        ws = wholesale_df[wholesale_df["date"] == date]
        # Average duplicates per hour for wholesale
        ws = ws.groupby("hour")["Price (EUR/MWhe)"].mean()

        if len(ag) < 24 or len(ws) < 24:
            continue

        agile_prices     = ag["rate_p_kwh"].values.tolist()
        wholesale_prices = [ws.loc[h] for h in range(24)]

        result = solve_daily_milp(
            agile_prices, wholesale_prices, consumer_demand_24h,
            BAT_KWH, BAT_KW, efficiency, fr_rate_hourly)

        total_consumer  += result["consumer_savings"]
        total_wholesale += result["wholesale_arbitrage"]
        total_fr        += result["fr_income"]
        days += 1

    # CM and BM
    cm_income = BAT_KW * CM_PER_KW * CM_DERATING
    bm_income = total_wholesale * BM_WIN_RATE * BM_UPLIFT

    return {
        "days_processed":      days,
        "consumer_savings":    total_consumer,
        "wholesale_arbitrage": total_wholesale,
        "bm_income":           bm_income,
        "fr_income":           total_fr,
        "cm_income":           cm_income,
        "total_vpp":           total_wholesale + bm_income + total_fr + cm_income,
        "total_value":         total_consumer + total_wholesale + bm_income + total_fr + cm_income,
    }


# =============================================================================
# FINANCIAL CALCULATIONS
# =============================================================================

def build_pl(vpp_rev_base, consumer_savings_base, loan_years, scaling_factors):
    """
    Build VLP company P&L.

    VLP Revenue = wholesale arb + FR + CM + BM  (all subject to spread compression + degradation)
    Opex = staff + customer management (VLP)
    No COGS, no network charges, no levy.
    """
    opex = STAFF_COST + CUST_MGMT
    loan_bal = float(BAT_COST)
    ann_prin = BAT_COST / loan_years
    rows = []
    for yr in range(1, MODEL_YEARS + 1):
        sf = scaling_factors.get(yr, 1.0)
        df = degradation_factor(yr)
        vpp_rev_yr = vpp_rev_base * sf * df
        ebitda     = vpp_rev_yr - opex
        ebit       = ebitda - DEPRECIATION
        interest   = loan_bal * LOAN_RATE
        ebt        = ebit - interest
        tax        = max(0.0, ebt * CORP_TAX)
        net_inc    = ebt - tax
        principal  = min(ann_prin, loan_bal) if loan_bal > 0.001 else 0.0
        loan_bal   = max(0.0, loan_bal - principal)
        rows.append({
            "Year":              yr,
            "Loan Yrs":          loan_years,
            "VLP Revenue (£)":   round(vpp_rev_yr, 2),
            "  Spread Scale":    round(sf, 3),
            "  Degrade Scale":   round(df, 3),
            "Staff Cost (£)":    round(STAFF_COST, 2),
            "Cust Mgmt (£)":    round(CUST_MGMT, 2),
            "Total Opex (£)":   round(opex, 2),
            "EBITDA (£)":        round(ebitda, 2),
            "Depreciation (£)":  round(DEPRECIATION, 2),
            "EBIT (£)":          round(ebit, 2),
            "Interest (£)":      round(interest, 2),
            "EBT (£)":           round(ebt, 2),
            "Tax 25% (£)":       round(tax, 2),
            "Net Income (£)":    round(net_inc, 2),
        })
    return pd.DataFrame(rows)


def build_cashflow(vpp_rev_base, loan_years, scaling_factors):
    """Build cashflow: EBITDA - Tax = OpCF, then interest + principal + CAC."""
    opex = STAFF_COST + CUST_MGMT
    loan_bal = float(BAT_COST)
    ann_prin = BAT_COST / loan_years
    cum_cash = 0.0
    rows = []
    for yr in range(1, MODEL_YEARS + 1):
        sf = scaling_factors.get(yr, 1.0)
        df = degradation_factor(yr)
        vpp_rev_yr = vpp_rev_base * sf * df
        ebitda     = vpp_rev_yr - opex
        ebit       = ebitda - DEPRECIATION
        interest   = loan_bal * LOAN_RATE
        ebt        = ebit - interest
        tax        = max(0.0, ebt * CORP_TAX)
        ocf        = ebitda - tax
        principal  = min(ann_prin, loan_bal) if loan_bal > 0.001 else 0.0
        loan_bal   = max(0.0, loan_bal - principal)
        cac_out    = CAC if yr == 1 else 0.0
        free_cash  = ocf - interest - principal - cac_out
        cum_cash  += free_cash
        rows.append({
            "Year":                 yr,
            "Loan Yrs":             loan_years,
            "VLP Revenue (£)":      round(vpp_rev_yr, 2),
            "EBITDA (£)":           round(ebitda, 2),
            "Tax (£)":              round(tax, 2),
            "Operating CF (£)":     round(ocf, 2),
            "Interest (£)":         round(interest, 2),
            "Principal Repaid (£)": round(principal, 2),
            "CAC (£)":              round(cac_out, 2),
            "Free Cash (£)":        round(free_cash, 2),
            "Cumulative Cash (£)":  round(cum_cash, 2),
            "CF Positive":          free_cash >= 0,
        })
    return pd.DataFrame(rows)


def build_loan_schedule(loan_years):
    bal      = float(BAT_COST)
    ann_prin = BAT_COST / loan_years
    rows = []
    for yr in range(1, loan_years + 1):
        open_bal  = bal
        interest  = round(open_bal * LOAN_RATE, 2)
        principal = round(min(ann_prin, open_bal), 2)
        total_pmt = round(interest + principal, 2)
        bal       = round(max(0.0, bal - principal), 6)
        rows.append({
            "Year":                  yr,
            "Loan Yrs":              loan_years,
            "Opening Balance (£)":   round(open_bal, 2),
            "Interest Payment (£)":  interest,
            "Principal Payment (£)": principal,
            "Total Payment (£)":     total_pmt,
            "Closing Balance (£)":   round(bal, 2),
        })
    df = pd.DataFrame(rows)
    totals = pd.DataFrame([{
        "Year":                  "TOTAL",
        "Loan Yrs":              loan_years,
        "Opening Balance (£)":   "",
        "Interest Payment (£)":  round(df["Interest Payment (£)"].sum(), 2),
        "Principal Payment (£)": round(df["Principal Payment (£)"].sum(), 2),
        "Total Payment (£)":     round(df["Total Payment (£)"].sum(), 2),
        "Closing Balance (£)":   "",
    }])
    return pd.concat([df, totals], ignore_index=True)


def build_consumer_savings_projection(savings_yr1, scaling_factors):
    """Project consumer Agile savings over MODEL_YEARS with spread compression + degradation."""
    rows = []
    cum = 0.0
    for yr in range(1, MODEL_YEARS + 1):
        sf = scaling_factors.get(yr, 1.0)
        df = degradation_factor(yr)
        annual = savings_yr1 * sf * df
        monthly = annual / 12
        cum += annual
        rows.append({
            "Year": yr,
            "Spread Scale": round(sf, 3),
            "Degrade Scale": round(df, 3),
            "Annual Savings (£)": round(annual, 2),
            "Monthly Savings (£)": round(monthly, 2),
            "Cumulative Savings (£)": round(cum, 2),
        })
    return pd.DataFrame(rows)


def find_combined_payback(consumer_savings_yr1, vpp_rev_yr1, scaling_factors):
    """Year when cumulative (consumer savings + VLP income) exceeds battery cost."""
    cum = 0.0
    for yr in range(1, MODEL_YEARS + 1):
        sf = scaling_factors.get(yr, 1.0)
        df = degradation_factor(yr)
        combined = (consumer_savings_yr1 + vpp_rev_yr1) * sf * df
        cum += combined
        if cum >= BAT_COST:
            return yr
    return None  # not reached within model horizon


# =============================================================================
# CHARTS
# =============================================================================

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_consumption_comparison(all_results, spread_name, loan_yr=5):
    """Compare consumption levels for a spread scenario."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(
        f"VLP Agile Model  |  {spread_name} spread  |  {loan_yr}-yr loan\n"
        f"VLP Company Free Cashflow by Consumption Level",
        fontsize=11, fontweight="bold", y=0.99)

    years = np.arange(1, MODEL_YEARS + 1)
    for ckwh in CONSUMPTION_LEVELS:
        key = (spread_name, ckwh, loan_yr)
        cf = all_results["cf"][key]
        color = CONS_COLORS[ckwh]
        label = f"{ckwh:,} kWh/yr"
        ax1.plot(years, cf["Free Cash (£)"].values, marker="o", markersize=4,
                 color=color, label=label, linewidth=2)
        ax2.plot(years, cf["Cumulative Cash (£)"].values, marker="o", markersize=4,
                 color=color, label=label, linewidth=2)

    for ax in [ax1, ax2]:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    ax1.set_ylabel("Annual Free Cash (£)")
    ax1.set_title("Annual Free Cashflow")
    ax2.set_ylabel("Cumulative Cash (£)")
    ax2.set_xlabel("Year")
    ax2.set_xticks(years)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig_to_base64(fig)


def plot_spread_comparison(all_results, consumption, loan_yr=5):
    """Compare spread scenarios for a consumption level."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(
        f"VLP Agile Model  |  {consumption:,} kWh/yr  |  {loan_yr}-yr loan\n"
        f"Spread Scenarios Compared",
        fontsize=11, fontweight="bold", y=0.99)

    years = np.arange(1, MODEL_YEARS + 1)
    for sp_name in SPREAD_SCENARIOS:
        key = (sp_name, consumption, loan_yr)
        cf = all_results["cf"][key]
        color = SPREAD_COLORS[sp_name]
        ax1.plot(years, cf["Free Cash (£)"].values, marker="o", markersize=4,
                 color=color, label=sp_name, linewidth=2)
        ax2.plot(years, cf["Cumulative Cash (£)"].values, marker="o", markersize=4,
                 color=color, label=sp_name, linewidth=2)

    for ax in [ax1, ax2]:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    ax1.set_ylabel("Annual Free Cash (£)")
    ax1.set_title("Annual Free Cashflow by Spread Scenario")
    ax2.set_ylabel("Cumulative Cash (£)")
    ax2.set_xlabel("Year")
    ax2.set_xticks(years)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig_to_base64(fig)


def plot_loan_comparison(all_results, spread_name, consumption):
    """Per spread+consumption: compare loan periods."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(
        f"VLP Agile Model  |  {spread_name} spread  |  {consumption:,} kWh/yr\n"
        f"Loan Period Comparison  |  Battery: £{BAT_COST:,.0f}",
        fontsize=11, fontweight="bold", y=0.99)

    years   = np.arange(1, MODEL_YEARS + 1)
    n_loans = len(LOAN_YEARS)
    bar_w   = 0.18
    offsets = np.linspace(-(n_loans-1)/2, (n_loans-1)/2, n_loans) * bar_w

    for i, ly in enumerate(LOAN_YEARS):
        key   = (spread_name, consumption, ly)
        cf    = all_results["cf"][key]
        color = LOAN_COLORS[ly]
        fcf   = cf["Free Cash (£)"].values
        cum   = cf["Cumulative Cash (£)"].values
        ax1.bar(years + offsets[i], fcf, width=bar_w, color=color,
                label=f"{ly}-yr loan", alpha=0.85)
        ax2.plot(years, cum, marker="o", markersize=4, color=color,
                 label=f"{ly}-yr loan", linewidth=2)

    for ax in [ax1, ax2]:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
        ax.legend(fontsize=9, ncol=4)
        ax.grid(axis="y", alpha=0.3)
    ax1.set_ylabel("Annual Free Cash (£)")
    ax1.set_title("Annual Free Cashflow")
    ax2.set_ylabel("Cumulative Cash (£)")
    ax2.set_xlabel("Year")
    ax2.set_xticks(years)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig_to_base64(fig)


def plot_consumer_savings(savings_projections):
    """Chart: consumer savings over time for each consumption level."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(
        "Consumer Agile Savings Projection\n"
        "With spread compression + battery degradation (Base scenario)",
        fontsize=11, fontweight="bold", y=0.99)

    years = np.arange(1, MODEL_YEARS + 1)
    for ckwh, proj in savings_projections.items():
        color = CONS_COLORS[ckwh]
        label = f"{ckwh:,} kWh/yr"
        ax1.plot(years, proj["Annual Savings (£)"].values, marker="o", markersize=4,
                 color=color, label=label, linewidth=2)
        ax2.plot(years, proj["Cumulative Savings (£)"].values, marker="o", markersize=4,
                 color=color, label=label, linewidth=2)

    ax2.axhline(BAT_COST, color="red", linewidth=1.5, linestyle=":", label=f"Battery cost £{BAT_COST:,.0f}")
    for ax in [ax1, ax2]:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    ax1.set_ylabel("Annual Savings (£)")
    ax1.set_title("Annual Consumer Savings from Agile Arbitrage")
    ax2.set_ylabel("Cumulative Savings (£)")
    ax2.set_xlabel("Year")
    ax2.set_xticks(years)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig_to_base64(fig)


# =============================================================================
# HTML REPORT
# =============================================================================

def _build_report_js(milp_results, all_scaling):
    """Build JavaScript for interactive recalculation."""
    all_sf = {}
    for sp_name in SPREAD_SCENARIOS:
        factors = get_scaling_factors(sp_name)
        all_sf[sp_name] = {str(k): round(v, 6) for k, v in factors.items()}

    milp_data = {}
    for ckwh, res in milp_results.items():
        milp_data[str(ckwh)] = {
            "consumer_savings":    round(res["consumer_savings"], 2),
            "wholesale_arbitrage": round(res["wholesale_arbitrage"], 2),
            "bm_income":           round(res["bm_income"], 2),
            "fr_income":           round(res["fr_income"], 2),
            "cm_income":           round(res["cm_income"], 2),
            "total_vpp":           round(res["total_vpp"], 2),
            "total_value":         round(res["total_value"], 2),
        }

    data = {
        "BAT_COST": BAT_COST, "DEPRECIATION": DEPRECIATION,
        "LOAN_RATE": LOAN_RATE, "CORP_TAX": CORP_TAX,
        "MODEL_YEARS": MODEL_YEARS, "DEGRADATION_PA": DEGRADATION_PA,
        "LOAN_YEARS": LOAN_YEARS, "CONSUMPTION_LEVELS": CONSUMPTION_LEVELS,
        "STAFF_COST": STAFF_COST, "CUST_MGMT": CUST_MGMT,
        "DEF_CAC": CAC, "DEF_STAFF": STAFF_COST, "DEF_MGMT": CUST_MGMT,
        "milp": milp_data,
        "scaling": all_sf,
        "spread_names": list(SPREAD_SCENARIOS.keys()),
    }

    js = "const D=" + json.dumps(data) + ";\n"
    js += r"""
function fmt(v){return v.toLocaleString('en-GB',{minimumFractionDigits:2,maximumFractionDigits:2});}
function fmt0(v){return Math.round(v).toLocaleString('en-GB');}
function degradeFactor(yr){return Math.pow(1-D.DEGRADATION_PA,yr-1);}
function getInputs(){
    return {
        cac:parseFloat(document.getElementById('inp-cac').value)||0,
        staff:parseFloat(document.getElementById('inp-staff').value)||0,
        mgmt:parseFloat(document.getElementById('inp-mgmt').value)||0
    };
}
function buildCF(vppRev,loanYrs,sf,inp){
    var opex=inp.staff+inp.mgmt;
    var bal=D.BAT_COST,annP=D.BAT_COST/loanYrs,cum=0,rows=[];
    for(var yr=1;yr<=D.MODEL_YEARS;yr++){
        var s=sf[String(yr)]||1, df=degradeFactor(yr);
        var rev=vppRev*s*df;
        var ebitda=rev-opex;
        var ebit=ebitda-D.DEPRECIATION, interest=bal*D.LOAN_RATE;
        var ebt=ebit-interest, tax=Math.max(0,ebt*D.CORP_TAX), ocf=ebitda-tax;
        var prin=bal>0.001?Math.min(annP,bal):0; bal=Math.max(0,bal-prin);
        var cacOut=yr===1?inp.cac:0, fc=ocf-interest-prin-cacOut; cum+=fc;
        rows.push({yr:yr,rev:rev,s:s,df:df,opex:opex,
            ebitda:ebitda,ebit:ebit,interest:interest,ebt:ebt,tax:tax,
            netInc:ebt-tax,ocf:ocf,prin:prin,cacOut:cacOut,fc:fc,cum:cum});
    }
    return rows;
}
function renderSummary(){
    var inp=getInputs();
    var h='<table class="data-table summary-table"><thead><tr>';
    ['Spread','Consumption','VPP Rev £','Opex £','EBIT £',
     'Yr1 FCF £',D.MODEL_YEARS+'yr Cum £','Break-even'].forEach(function(c){h+='<th>'+c+'</th>';});
    h+='</tr></thead><tbody>';
    D.spread_names.forEach(function(sp){
        var sf=D.scaling[sp];
        D.CONSUMPTION_LEVELS.forEach(function(ckwh){
            var m=D.milp[String(ckwh)];
            var cf=buildCF(m.total_vpp,5,sf,inp);
            var yr1=cf[0].fc, cumF=cf[cf.length-1].cum;
            var beYr='Never';
            for(var i=0;i<cf.length;i++){if(cf[i].cum>=0){beYr=i+1;break;}}
            var cls=beYr==='Never'?' class="negative"':'';
            h+='<tr><td>'+sp+'</td>';
            h+='<td>'+ckwh.toLocaleString()+'</td>';
            h+='<td>'+fmt0(m.total_vpp)+'</td>';
            h+='<td>'+fmt0(inp.staff+inp.mgmt)+'</td>';
            h+='<td>'+fmt0(m.total_vpp-inp.staff-inp.mgmt-D.DEPRECIATION)+'</td>';
            h+='<td>'+fmt0(yr1)+'</td><td>'+fmt0(cumF)+'</td>';
            h+='<td'+cls+'>'+beYr+'</td></tr>';
        });
    });
    h+='</tbody></table>';
    document.getElementById('summary-dashboard').innerHTML=h;
}
function renderPLTable(vppRev,loanYrs,sf,inp){
    var opex=inp.staff+inp.mgmt;
    var bal=D.BAT_COST,annP=D.BAT_COST/loanYrs;
    var h='<table class="data-table"><thead><tr>';
    ['Year','VLP Revenue (£)','Spread','Degrade','Staff (£)','Mgmt (£)','Opex (£)',
     'EBITDA (£)','Depr (£)','EBIT (£)','Interest (£)','EBT (£)','Tax 25% (£)','Net Inc (£)'].forEach(function(c){h+='<th>'+c+'</th>';});
    h+='</tr></thead><tbody>';
    for(var yr=1;yr<=D.MODEL_YEARS;yr++){
        var s=sf[String(yr)]||1, df=degradeFactor(yr);
        var rev=vppRev*s*df;
        var ebitda=rev-opex, ebit=ebitda-D.DEPRECIATION;
        var interest=bal*D.LOAN_RATE, ebt=ebit-interest;
        var tax=Math.max(0,ebt*D.CORP_TAX), ni=ebt-tax;
        var prin=bal>0.001?Math.min(annP,bal):0; bal=Math.max(0,bal-prin);
        h+='<tr><td>'+yr+'</td><td>'+fmt(rev)+'</td><td>'+s.toFixed(3)+'</td><td>'+df.toFixed(3)+'</td>';
        h+='<td>'+fmt(inp.staff)+'</td><td>'+fmt(inp.mgmt)+'</td><td>'+fmt(opex)+'</td>';
        h+='<td>'+fmt(ebitda)+'</td><td>'+fmt(D.DEPRECIATION)+'</td><td>'+fmt(ebit)+'</td>';
        h+='<td>'+fmt(interest)+'</td><td>'+fmt(ebt)+'</td><td>'+fmt(tax)+'</td>';
        h+='<td>'+fmt(ni)+'</td></tr>';
    }
    h+='</tbody></table>'; return h;
}
function renderCFTable(vppRev,loanYrs,sf,inp){
    var cf=buildCF(vppRev,loanYrs,sf,inp);
    var h='<table class="data-table"><thead><tr>';
    ['Year','VLP Rev (£)','EBITDA (£)','Tax (£)','Op CF (£)','Interest (£)','Principal (£)',
     'CAC (£)','Free Cash (£)','Cumulative (£)'].forEach(function(c){h+='<th>'+c+'</th>';});
    h+='</tr></thead><tbody>';
    cf.forEach(function(r){
        var cls=r.fc<0?' class="negative"':' class="positive"';
        h+='<tr><td>'+r.yr+'</td><td>'+fmt(r.rev)+'</td><td>'+fmt(r.ebitda)+'</td><td>'+fmt(r.tax)+'</td>';
        h+='<td>'+fmt(r.ocf)+'</td><td>'+fmt(r.interest)+'</td><td>'+fmt(r.prin)+'</td>';
        h+='<td>'+fmt(r.cacOut)+'</td><td'+cls+'>'+fmt(r.fc)+'</td><td>'+fmt(r.cum)+'</td></tr>';
    });
    h+='</tbody></table>'; return h;
}
function renderDetails(){
    var inp=getInputs();
    D.spread_names.forEach(function(sp){
        var sf=D.scaling[sp], h='';
        D.CONSUMPTION_LEVELS.forEach(function(ckwh){
            var m=D.milp[String(ckwh)];
            h+='<details><summary>P&L — '+sp+' spread, '+ckwh.toLocaleString()+' kWh/yr, 5-yr loan</summary>';
            h+=renderPLTable(m.total_vpp,5,sf,inp)+'</details>';
            h+='<details><summary>Cashflow — '+sp+' spread, '+ckwh.toLocaleString()+' kWh/yr, 5-yr loan</summary>';
            h+=renderCFTable(m.total_vpp,5,sf,inp)+'</details>';
        });
        document.getElementById('detail-tables-'+sp).innerHTML=h;
    });
}
function renderCombinedPayback(){
    var inp=getInputs();
    var h='<table class="data-table"><thead><tr><th>Spread</th><th>Consumption</th>';
    h+='<th>Consumer Savings £/yr</th><th>VLP Income £/yr</th><th>Combined £/yr</th>';
    h+='<th>Simple Payback (yrs)</th></tr></thead><tbody>';
    D.spread_names.forEach(function(sp){
        var sf=D.scaling[sp];
        D.CONSUMPTION_LEVELS.forEach(function(ckwh){
            var m=D.milp[String(ckwh)];
            var combined=m.consumer_savings+m.total_vpp;
            // Find payback year with degradation + compression
            var cum=0,pbYr='Never';
            for(var yr=1;yr<=D.MODEL_YEARS;yr++){
                var s=sf[String(yr)]||1,df=degradeFactor(yr);
                cum+=combined*s*df;
                if(cum>=D.BAT_COST){pbYr=yr;break;}
            }
            var cls=pbYr==='Never'?' class="negative"':'';
            h+='<tr><td>'+sp+'</td><td>'+ckwh.toLocaleString()+'</td>';
            h+='<td>'+fmt0(m.consumer_savings)+'</td><td>'+fmt0(m.total_vpp)+'</td>';
            h+='<td>'+fmt0(combined)+'</td>';
            h+='<td'+cls+'>'+pbYr+'</td></tr>';
        });
    });
    h+='</tbody></table>';
    document.getElementById('combined-payback').innerHTML=h;
}
function recalcAll(){renderSummary();renderDetails();renderCombinedPayback();}
document.addEventListener('DOMContentLoaded',recalcAll);
"""
    return js


def build_html_report(all_results, milp_results, savings_projections, charts, out_path):
    """Generate interactive HTML report."""

    # Battery revenue breakdown table
    rev_html = """<table class='data-table'><thead><tr>
    <th>Consumption</th><th>Consumer Savings £</th><th>Wholesale Arb £</th>
    <th>FR £</th><th>CM £</th><th>BM £</th><th>Total VPP £</th><th>Total Value £</th>
    </tr></thead><tbody>\n"""
    for ckwh in CONSUMPTION_LEVELS:
        r = milp_results[ckwh]
        rev_html += f"""<tr>
            <td>{ckwh:,} kWh/yr</td>
            <td>{r['consumer_savings']:,.0f}</td>
            <td>{r['wholesale_arbitrage']:,.0f}</td>
            <td>{r['fr_income']:,.0f}</td>
            <td>{r['cm_income']:,.0f}</td>
            <td>{r['bm_income']:,.0f}</td>
            <td><strong>{r['total_vpp']:,.0f}</strong></td>
            <td><strong>{r['total_value']:,.0f}</strong></td>
        </tr>\n"""
    rev_html += "</tbody></table>\n"

    # Consumer savings table
    savings_html = ""
    for ckwh in CONSUMPTION_LEVELS:
        r = milp_results[ckwh]
        annual = r["consumer_savings"]
        monthly = annual / 12
        savings_html += f"""<div style="display:inline-block; margin:10px 20px; text-align:center;">
            <h4>{ckwh:,} kWh/yr</h4>
            <div style="font-size:28px; font-weight:bold; color:#27ae60;">£{annual:,.0f}/yr</div>
            <div style="font-size:16px; color:#666;">£{monthly:,.1f}/mo</div>
        </div>\n"""

    # Savings projection table (Base spread)
    proj_html = ""
    for ckwh in CONSUMPTION_LEVELS:
        proj = savings_projections[ckwh]
        proj_html += f"<details><summary>{ckwh:,} kWh/yr savings projection</summary>"
        proj_html += '<table class="data-table"><thead><tr>'
        for c in ["Year", "Spread Scale", "Degrade Scale", "Annual Savings (£)",
                   "Monthly Savings (£)", "Cumulative Savings (£)"]:
            proj_html += f"<th>{c}</th>"
        proj_html += "</tr></thead><tbody>"
        for _, row in proj.iterrows():
            proj_html += "<tr>"
            for c in proj.columns:
                val = row[c]
                if isinstance(val, float):
                    proj_html += f"<td>{val:,.2f}</td>"
                else:
                    proj_html += f"<td>{val}</td>"
            proj_html += "</tr>"
        proj_html += "</tbody></table></details>\n"

    # Spread compression table
    spread_table = ""
    for sp_name, sp_params in SPREAD_SCENARIOS.items():
        factors = get_scaling_factors(sp_name)
        yr1 = factors[1]
        yr6 = factors[6]
        yr12 = factors[MODEL_YEARS]
        spread_table += f'<tr><td>{sp_name}</td><td>{sp_params["alpha"]}</td><td>{sp_params["beta"]}</td><td>{yr1:.3f}</td><td>{yr6:.3f}</td><td>{yr12:.3f}</td></tr>\n'

    # Detail sections by spread
    detail_sections = ""
    for sp_name in SPREAD_SCENARIOS:
        detail_sections += f"""
        <div class="section">
        <h2>{sp_name} Spread — Detailed P&L and Cashflow</h2>
        """
        # Charts
        chart_key = f"cons_{sp_name}"
        if chart_key in charts:
            detail_sections += f"""
            <details>
            <summary>Chart: Consumption comparison — {sp_name} spread</summary>
            <img src="data:image/png;base64,{charts[chart_key]}" style="max-width:100%">
            </details>"""

        for ckwh in CONSUMPTION_LEVELS:
            chart_key = f"loan_{sp_name}_{ckwh}"
            if chart_key in charts:
                detail_sections += f"""
                <details>
                <summary>Chart: Loan comparison — {sp_name} spread, {ckwh:,} kWh/yr</summary>
                <img src="data:image/png;base64,{charts[chart_key]}" style="max-width:100%">
                </details>"""

        detail_sections += f'<div id="detail-tables-{sp_name}"></div>\n'
        detail_sections += "</div>\n"

    # Build JS
    all_scaling = {}
    for sp_name in SPREAD_SCENARIOS:
        all_scaling[sp_name] = get_scaling_factors(sp_name)
    js_code = _build_report_js(milp_results, all_scaling)

    # Consumer savings chart
    savings_chart_html = ""
    if "consumer_savings" in charts:
        savings_chart_html = f'<img src="data:image/png;base64,{charts["consumer_savings"]}" style="max-width:100%">'

    # Spread comparison charts
    spread_chart_html = ""
    for ckwh in CONSUMPTION_LEVELS:
        chart_key = f"spread_{ckwh}"
        if chart_key in charts:
            spread_chart_html += f"""
            <details>
            <summary>Chart: Spread comparison — {ckwh:,} kWh/yr</summary>
            <img src="data:image/png;base64,{charts[chart_key]}" style="max-width:100%">
            </details>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VLP Battery — Agile Consumer Model</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           max-width: 1200px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #333; }}
    h1 {{ color: #2c3e50; border-bottom: 3px solid #27ae60; padding-bottom: 10px; }}
    h2 {{ color: #2c3e50; margin-top: 30px; }}
    h3 {{ color: #34495e; margin-top: 20px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
    .section {{ background: white; padding: 20px; margin: 15px 0; border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    .data-table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 10px 0; }}
    .data-table th {{ background: #2c3e50; color: white; padding: 8px 12px; text-align: right; }}
    .data-table th:first-child {{ text-align: left; }}
    .data-table td {{ padding: 6px 12px; border-bottom: 1px solid #eee; text-align: right; }}
    .data-table td:first-child {{ text-align: left; }}
    .data-table tr:hover {{ background: #f5f5f5; }}
    .summary-table th {{ font-size: 12px; }}
    .positive {{ color: #27ae60; font-weight: bold; }}
    .negative {{ color: #e74c3c; font-weight: bold; }}
    details {{ margin: 8px 0; }}
    summary {{ cursor: pointer; padding: 10px; background: #ecf0f1; border-radius: 5px;
               font-weight: 600; }}
    summary:hover {{ background: #d5dbdb; }}
    .highlight {{ background: #d4edda; padding: 15px; border-left: 4px solid #27ae60;
                  border-radius: 4px; margin: 15px 0; }}
    .assumption {{ background: #fef3f3; padding: 15px; border-left: 4px solid #e74c3c;
                   border-radius: 4px; margin: 10px 0; font-size: 13px; }}
    img {{ border-radius: 5px; margin: 10px 0; }}
    .controls {{ position: sticky; top: 0; z-index: 100; background: #2c3e50;
                 color: white; padding: 12px 20px; border-radius: 8px;
                 margin-bottom: 15px; display: flex; gap: 24px; align-items: center;
                 box-shadow: 0 4px 6px rgba(0,0,0,0.2); flex-wrap: wrap; }}
    .controls label {{ font-size: 13px; font-weight: 600; display: flex;
                       align-items: center; gap: 6px; }}
    .controls input {{ width: 70px; padding: 4px 8px; border: 1px solid #7f8c8d;
                       border-radius: 4px; font-size: 13px; text-align: right; }}
    .controls .btn {{ background: #27ae60; border: none; color: white; padding: 6px 16px;
                      border-radius: 4px; cursor: pointer; font-weight: 600; font-size: 13px; }}
    .controls .btn:hover {{ background: #219a52; }}
    .savings-cards {{ display: flex; justify-content: center; flex-wrap: wrap; }}
</style>
</head>
<body>

<h1>VLP Battery — Agile Consumer Model</h1>
<p>The consumer stays on <strong>Octopus Agile</strong> and benefits from battery arbitrage on Agile rates.
   The company operates the battery as a <strong>Virtual Lead Party (VLP)</strong>, earning grid services revenue
   (wholesale arbitrage + FR + CM + BM). No supply cost, no network charges, no grid levy.</p>

<div class="controls">
    <span style="font-weight:700; font-size:14px;">Adjust Inputs</span>
    <label>CAC &pound; <input type="number" id="inp-cac" value="{CAC}" step="10"></label>
    <label>Staff &pound;/yr <input type="number" id="inp-staff" value="{STAFF_COST}" step="5"></label>
    <label>VLP Mgmt &pound;/yr <input type="number" id="inp-mgmt" value="{CUST_MGMT}" step="5"></label>
    <button class="btn" onclick="recalcAll()">Recalculate</button>
    <span style="font-size:11px; opacity:0.7;">Charts are static &mdash; tables &amp; dashboard update live</span>
</div>

<div class="highlight">
    <strong>Business model:</strong> Consumer keeps their Agile tariff and gets battery savings from arbitrage
    (charge at low Agile rates, discharge during expensive hours). The VLP company earns wholesale arbitrage,
    frequency response, capacity market, and balancing mechanism revenue on the remaining battery capacity.
    No electricity supply licence needed &mdash; no levies, no network charges for the company.
</div>

<div class="section">
<h2>Battery Revenue Breakdown (Year 1)</h2>
<p>The MILP jointly optimises consumer savings and VPP income hour-by-hour.
   Higher consumption = more consumer demand hours = more savings from Agile arbitrage,
   but less grid discharge available for VPP wholesale income.</p>
{rev_html}
</div>

<div class="section">
<h2>Consumer Savings Analysis</h2>
<p>Annual savings from battery Agile arbitrage — the consumer avoids expensive Agile hours
   by using battery-stored cheap-hour electricity instead.</p>
<div class="savings-cards">
{savings_html}
</div>
{savings_chart_html}
{proj_html}
</div>

<div class="section">
<h2>Key Parameters</h2>

<h3>Battery Hardware</h3>
<table class="data-table">
<thead><tr><th>Parameter</th><th>Value</th><th>Notes</th></tr></thead>
<tbody>
<tr><td>Capacity</td><td>{BAT_KWH} kWh</td><td>Usable LFP cells</td></tr>
<tr><td>Inverter power</td><td>{BAT_KW:.0f} kW</td><td>Max charge/discharge rate</td></tr>
<tr><td>Round-trip efficiency</td><td>{EFF_RT*100:.0f}%</td><td>One-way: {ETA*100:.2f}%</td></tr>
<tr><td>Total cost</td><td>&pound;{BAT_COST:,.0f}</td><td>Hardware + installation</td></tr>
<tr><td>Economic life</td><td>{BAT_LIFE} years</td><td>Straight-line depreciation: &pound;{DEPRECIATION:,.0f}/yr</td></tr>
<tr><td>Annual degradation</td><td>{DEGRADATION_PA*100:.0f}%/yr</td><td>Yr5: {degradation_factor(5)*100:.1f}%, Yr10: {degradation_factor(10)*100:.1f}%</td></tr>
</tbody>
</table>

<h3>Financing</h3>
<table class="data-table">
<thead><tr><th>Parameter</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Loan interest rate</td><td>{LOAN_RATE*100:.0f}% p.a.</td></tr>
<tr><td>Corporation tax</td><td>{CORP_TAX*100:.0f}%</td></tr>
<tr><td>Loan periods modelled</td><td>{', '.join(str(y)+'-yr' for y in LOAN_YEARS)}</td></tr>
<tr><td>Model horizon</td><td>{MODEL_YEARS} years</td></tr>
</tbody>
</table>

<h3>Grid Services Revenue</h3>
<table class="data-table">
<thead><tr><th>Service</th><th>Rate</th><th>Annual (full battery)</th></tr></thead>
<tbody>
<tr><td>Frequency Response (FR)</td><td>&pound;{FR_PER_KW:.0f}/kW/yr</td><td>&pound;{BAT_KW*FR_PER_KW:,.0f} max</td></tr>
<tr><td>Capacity Market (CM)</td><td>&pound;{CM_PER_KW:.0f}/kW/yr</td><td>&pound;{BAT_KW*CM_PER_KW*CM_DERATING:,.0f} (de-rated {CM_DERATING*100:.1f}%)</td></tr>
<tr><td>Balancing Mechanism (BM)</td><td>{BM_WIN_RATE*100:.0f}% win x {BM_UPLIFT*100:.0f}% uplift</td><td>Varies</td></tr>
</tbody>
</table>

<h3>VLP Company Cost Stack (per customer/yr)</h3>
<table class="data-table">
<thead><tr><th>Cost line</th><th>Amount</th><th>Notes</th></tr></thead>
<tbody>
<tr><td>Staff cost</td><td>&pound;{STAFF_COST:,.0f}</td><td>Annual per customer</td></tr>
<tr><td>Customer management (VLP)</td><td>&pound;{CUST_MGMT:,.0f}</td><td>From model_input.csv</td></tr>
<tr><td>Grid levy</td><td>&pound;{GRID_LEVY:,.0f}</td><td>VLP is not electricity supplier &mdash; no levy</td></tr>
<tr><td>Network charges</td><td>&pound;0</td><td>Consumer pays via Agile tariff</td></tr>
<tr><td>Supply cost (COGS)</td><td>&pound;0</td><td>Company does not supply electricity</td></tr>
<tr><td>CAC (year 1 only)</td><td>&pound;{CAC:,.0f}</td><td>Customer acquisition cost</td></tr>
<tr><td><strong>Total annual opex</strong></td><td><strong>&pound;{STAFF_COST + CUST_MGMT:,.0f}</strong></td><td></td></tr>
</tbody>
</table>

<h3>Data Sources</h3>
<table class="data-table">
<thead><tr><th>Data</th><th>Source</th><th>Period</th></tr></thead>
<tbody>
<tr><td>Agile retail rates</td><td>agile_rates_2025_cache.csv</td><td>2025 (half-hourly, averaged to hourly)</td></tr>
<tr><td>Wholesale prices</td><td>United Kingdom.csv</td><td>Jul 2024 &ndash; Jun 2025</td></tr>
<tr><td>Use profile</td><td>use_profiles.csv (UK)</td><td>24-hour shape, {SHAPE_DAILY_WH:.0f} Wh/day</td></tr>
<tr><td>Capacity forecasts</td><td>capacity_forecasts.csv</td><td>2024&ndash;2035</td></tr>
</tbody>
</table>

<h3>Spread Compression Scenarios</h3>
<table class="data-table">
<thead><tr><th>Scenario</th><th>alpha</th><th>beta</th><th>Yr 1</th><th>Yr 6</th><th>Yr 12</th></tr></thead>
<tbody>
{spread_table}
</tbody>
</table>
</div>

<div class="section">
<h2>VLP Company Economics Dashboard (5-yr loan)</h2>
<div id="summary-dashboard"><p>Loading...</p></div>
</div>

{spread_chart_html}

{detail_sections}

<div class="section">
<h2>Sensitivity: Combined Payback (if ALL VLP income given to consumer)</h2>
<p>If the VLP company passed all its revenue to the consumer as additional savings,
   how quickly would the combined income cover the battery cost?</p>
<div id="combined-payback"><p>Loading...</p></div>
</div>

<div class="section">
<h2>Assumptions &amp; Limitations</h2>

<div class="assumption">
<strong>1. Consumer stays on Octopus Agile.</strong>
Consumer savings are based on the Agile tariff spread (charge low, discharge high).
If the consumer switches tariff, savings would differ. Agile rates may change structurally
in future years.
</div>

<div class="assumption">
<strong>2. Perfect foresight on prices.</strong>
The MILP optimises each day's schedule using actual prices. A real dispatch algorithm would
achieve 10-30% lower returns.
</div>

<div class="assumption">
<strong>3. Battery degradation at {DEGRADATION_PA*100:.0f}%/yr.</strong>
All revenue (consumer savings + VPP income) is scaled by degradation factor each year.
</div>

<div class="assumption">
<strong>4. Grid services assumed contractually secured.</strong>
FR and CM revenues require competitive tender wins. Actual DC rates have fallen recently.
</div>

<div class="assumption">
<strong>5. BM estimated, not modelled.</strong>
BM revenue = {BM_WIN_RATE*100:.0f}% x {BM_UPLIFT*100:.0f}% uplift on wholesale arbitrage.
</div>

<div class="assumption">
<strong>6. VLP has no supply obligations.</strong>
The company is a Virtual Lead Party, not an electricity supplier. No Ofgem supply licence
needed, no grid levy, no network charges, no BSUoS/DUoS/TNUoS. Consumer bears all supply
costs via their Agile tariff.
</div>

<div class="assumption">
<strong>7. Same spread compression as electricity company model.</strong>
Agile rate spreads compress at the same rate as wholesale spreads. In practice, Agile
spreads may compress differently as retail tariff methodology evolves.
</div>

<div class="assumption">
<strong>8. One cycle per day limit.</strong>
The MILP constrains total daily charge and discharge to one battery capacity. This is
conservative but protects battery life.
</div>

<div class="assumption">
<strong>9. CM stress events assumed rare.</strong>
Battery is usually available for other activities alongside CM obligations.
</div>

<div class="assumption">
<strong>10. Fixed loan repayment and {LOAN_RATE*100:.0f}% interest.</strong>
Equal annual principal instalments, not annuity-style.
</div>

<div class="assumption">
<strong>11. Corp tax simplified at {CORP_TAX*100:.0f}%.</strong>
No loss carry-forward, group relief, or capital allowances.
</div>

<div class="assumption">
<strong>12. UK use profile applied.</strong>
Hourly consumption shape from use_profiles.csv (UK), scaled proportionally.
</div>

<div class="assumption">
<strong>13. EUR/GBP fixed at {EUR_TO_GBP}.</strong>
Wholesale prices in EUR/MWh converted at fixed rate.
</div>
</div>

<div class="section" style="text-align: center; color: #999; font-size: 12px;">
Generated by VLP Battery — Agile Consumer Model  |  Agile rates 2025  |  UK wholesale Jul 2024 - Jun 2025
</div>

<script>
{js_code}
</script>
</body>
</html>"""

    out_path.write_text(html)
    print(f"  HTML report saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("=" * 90)
    print(f"  VLP Battery — Agile Consumer Model  (Battery £{BAT_COST:,.0f})")
    print(f"  Consumer on Agile | VLP grid services | Joint MILP optimisation")
    print("=" * 90)

    # -- Step 1: Load data --
    print("\n  Loading Agile cache...")
    agile_hourly = load_agile_cache()
    print(f"    {len(agile_hourly)} hourly records loaded")

    print("  Loading wholesale prices...")
    wholesale_df = load_wholesale_prices()
    print(f"    {len(wholesale_df)} records loaded")

    # -- Step 2: Run MILP for each consumption level --
    milp_results = {}
    for ckwh in CONSUMPTION_LEVELS:
        print(f"\n  Running MILP for {ckwh:,} kWh/yr...")
        result = optimize_annual(agile_hourly, wholesale_df, ckwh)
        milp_results[ckwh] = result
        print(f"    Days: {result['days_processed']}")
        print(f"    Consumer savings: £{result['consumer_savings']:,.0f}")
        print(f"    Wholesale arb:    £{result['wholesale_arbitrage']:,.0f}")
        print(f"    FR income:        £{result['fr_income']:,.0f}")
        print(f"    CM income:        £{result['cm_income']:,.0f}")
        print(f"    BM income:        £{result['bm_income']:,.0f}")
        print(f"    Total VPP:        £{result['total_vpp']:,.0f}")
        print(f"    Total value:      £{result['total_value']:,.0f}")

    # -- Step 3: Compute scaling factors --
    all_scaling = {}
    for sp_name in SPREAD_SCENARIOS:
        all_scaling[sp_name] = get_scaling_factors(sp_name)
        yr1 = all_scaling[sp_name][1]
        yr12 = all_scaling[sp_name][MODEL_YEARS]
        print(f"\n  Spread {sp_name}: yr1={yr1:.3f}  yr{MODEL_YEARS}={yr12:.3f}")

    # -- Step 4: Build all P&L and cashflow combinations --
    all_results = {"pl": {}, "cf": {}, "loan": {}}

    for sp_name in SPREAD_SCENARIOS:
        sf = all_scaling[sp_name]
        for ckwh in CONSUMPTION_LEVELS:
            r = milp_results[ckwh]
            vpp_rev = r["total_vpp"]
            consumer_sav = r["consumer_savings"]

            for ly in LOAN_YEARS:
                key = (sp_name, ckwh, ly)
                all_results["pl"][key]  = build_pl(vpp_rev, consumer_sav, ly, sf)
                all_results["cf"][key]  = build_cashflow(vpp_rev, ly, sf)

    # Loan schedules (independent of spread/consumption)
    for ly in LOAN_YEARS:
        all_results["loan"][ly] = build_loan_schedule(ly)

    # Print summary
    for sp_name in SPREAD_SCENARIOS:
        print(f"\n  -- {sp_name} spread, 5-yr loan --")
        for ckwh in CONSUMPTION_LEVELS:
            key = (sp_name, ckwh, 5)
            cf = all_results["cf"][key]
            cum = cf["Cumulative Cash (£)"].iloc[-1]
            print(f"    {ckwh:,} kWh: 12yr cum £{cum:,.0f}")

    # -- Step 5: Consumer savings projections --
    base_sf = all_scaling["Base"]
    savings_projections = {}
    for ckwh in CONSUMPTION_LEVELS:
        sav = milp_results[ckwh]["consumer_savings"]
        savings_projections[ckwh] = build_consumer_savings_projection(sav, base_sf)

    # -- Step 6: Save CSVs --
    print("\n  Saving CSVs...")
    for sp_name in SPREAD_SCENARIOS:
        for ckwh in CONSUMPTION_LEVELS:
            pl_frames = []
            cf_frames = []
            for ly in LOAN_YEARS:
                key = (sp_name, ckwh, ly)
                pl = all_results["pl"][key].copy()
                cf = all_results["cf"][key].copy()
                pl["Consumption kWh"] = ckwh
                cf["Consumption kWh"] = ckwh
                pl_frames.append(pl)
                cf_frames.append(cf)
            pd.concat(pl_frames, ignore_index=True).to_csv(
                OUT_DIR / f"pl_{sp_name}_{ckwh}.csv", index=False)
            pd.concat(cf_frames, ignore_index=True).to_csv(
                OUT_DIR / f"cashflow_{sp_name}_{ckwh}.csv", index=False)

    loan_frames = [all_results["loan"][ly] for ly in LOAN_YEARS]
    pd.concat(loan_frames, ignore_index=True).to_csv(
        OUT_DIR / "loan_schedule.csv", index=False)
    print("    CSVs saved.")

    # -- Step 7: Generate charts --
    print("  Generating charts...")
    charts = {}

    # Consumption comparison per spread
    for sp_name in SPREAD_SCENARIOS:
        charts[f"cons_{sp_name}"] = plot_consumption_comparison(
            all_results, sp_name, loan_yr=5)

    # Spread comparison per consumption
    for ckwh in CONSUMPTION_LEVELS:
        charts[f"spread_{ckwh}"] = plot_spread_comparison(
            all_results, ckwh, loan_yr=5)

    # Loan comparison per spread x consumption
    for sp_name in SPREAD_SCENARIOS:
        for ckwh in CONSUMPTION_LEVELS:
            charts[f"loan_{sp_name}_{ckwh}"] = plot_loan_comparison(
                all_results, sp_name, ckwh)

    # Consumer savings chart
    charts["consumer_savings"] = plot_consumer_savings(savings_projections)
    print("    Charts generated.")

    # -- Step 8: HTML report --
    print("  Generating HTML report...")
    build_html_report(all_results, milp_results, savings_projections, charts,
                      OUT_DIR / "report.html")

    print()
    print(f"  All files written to: {OUT_DIR}")
    print("=" * 90)


if __name__ == "__main__":
    main()
