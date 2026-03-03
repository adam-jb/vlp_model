"""
VLP Battery — Aggressive Customer Offering  (25–35% savings vs competitor)
===========================================================================
Battery + installation total: £2,800

Three tariff scenarios × three spread-compression scenarios × three
consumption levels (4000 / 5000 / 6000 kWh/yr), four loan periods.

Consumer-first allocation: battery serves customer demand before arbitrage.
MILP-based leftover capacity optimisation for joint arbitrage + FR.
Spread compression applied year-over-year using capacity forecast data.

Outputs saved to this folder (qlp_aggressive/):
  pl_{A,B,C}_{Best,Base,Worst}.csv   — P&L
  cashflow_{A,B,C}_{Best,Base,Worst}.csv — cash flow statements
  loan_schedule_{A,B,C}.csv          — amortisation schedules (spread-independent)
  chart_{A,B,C}.png                  — FCF + cumulative cash per tariff (base spread)
  chart_summary.png                  — scenario comparison
  report.html                        — interactive HTML report
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

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_DIR      = SCRIPT_DIR

# ── battery hardware (override vs CSV defaults — 15.36 kWh / 6 kW system) ────
BAT_KWH      = 15.36
BAT_KW       = 6.0
EFF_RT       = 0.90
ETA          = EFF_RT ** 0.5          # one-way ≈ 0.9487

GRID_DRAW    = BAT_KWH / ETA
DISCHARGE    = BAT_KWH * ETA
T_CHARGE     = GRID_DRAW / BAT_KW
T_DISCHARGE  = DISCHARGE / BAT_KW
N_FC         = int(T_CHARGE)
FRAC_C       = T_CHARGE  - N_FC
N_FD         = int(T_DISCHARGE)
FRAC_D       = T_DISCHARGE - N_FD

BAT_COST     = 2_800.00
BAT_LIFE     = 10
DEPRECIATION = BAT_COST / BAT_LIFE

LOAN_RATE    = 0.07
CORP_TAX     = 0.25

# ── loan scenarios ────────────────────────────────────────────────────────────
LOAN_YEARS   = [3, 5, 7, 10]
MODEL_YEARS  = 12

# ── consumption levels to model ──────────────────────────────────────────────
CONSUMPTION_LEVELS = [4000, 5000, 6000]   # kWh/yr

# ── battery degradation ──────────────────────────────────────────────────────
DEGRADATION_PA = 0.02   # 2% capacity loss per year

def degradation_factor(year):
    """Capacity retention factor at start of a given year (year 1 = 1.0)."""
    return (1 - DEGRADATION_PA) ** (year - 1)

# ── load model_input.csv — UK column ─────────────────────────────────────────
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

# ── grid services rates from CSV ─────────────────────────────────────────────
FR_PER_KW    = PARAMS.get("Frequency response per kw per year", 34)
CM_PER_KW    = PARAMS.get("Capacity market /kW/year", 40)
CM_DERATING  = 0.2094   # de-rating: 2-hr battery class
BM_WIN_RATE  = PARAMS.get("Balancing mechanism (BM) avg win rate", 0.10)
BM_UPLIFT    = PARAMS.get("Avg £/kwh uplift for BM", 0.05)

# ── new cost lines from CSV ──────────────────────────────────────────────────
CAC          = PARAMS.get("customer acquisition cost (CAC)", 100)
STAFF_COST   = PARAMS.get("annual staff cost per customer", 50)
BAT_REPLACE  = PARAMS.get("battery replacement cost", 200)
CUST_MGMT    = PARAMS.get("Cost of managing customers per customer per annum if electricity company", 10)

# ── load hourly consumption profile ──────────────────────────────────────────
def load_use_profile():
    """Load UK hourly consumption shape from use_profiles.csv.
    Returns array of 24 hourly values (Wh) that sum to one day's total."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "inputs" / "use_profiles.csv")
    uk = df[df["Nation"] == "UK"].sort_values("hour")
    if len(uk) < 24:
        print("  WARNING: UK use profile not found or incomplete, using flat profile")
        return np.ones(24) / 24
    return uk["consumption_kwh"].values.astype(float)

HOURLY_SHAPE = load_use_profile()              # raw Wh values (sum ≈ 9875 Wh)
SHAPE_DAILY_WH = HOURLY_SHAPE.sum()            # daily total in Wh
SHAPE_ANNUAL_KWH = SHAPE_DAILY_WH * 365 / 1000 # ≈ 3604.4 kWh

# ── spread compression ──────────────────────────────────────────────────────
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
    """Return dict {year_offset: scaling_factor} for years 1..MODEL_YEARS.
    Year 1 = baseline (2025), year 2 = 2026, etc."""
    p = SPREAD_SCENARIOS[spread_name]
    c0 = float(CAP_DATA.loc[BASELINE_YEAR, "bess_gwh"])
    r0 = float(CAP_DATA.loc[BASELINE_YEAR, "renewable_gwh"])
    factors = {}
    for yr_offset in range(1, MODEL_YEARS + 1):
        cal_year = BASELINE_YEAR + yr_offset - 1   # yr1 = 2025
        if cal_year in CAP_DATA.index:
            c = float(CAP_DATA.loc[cal_year, "bess_gwh"])
            r = float(CAP_DATA.loc[cal_year, "renewable_gwh"])
            sp = compute_spread(BASELINE_SPREAD, c0, c, r0, r, p["alpha"], p["beta"])
            factors[yr_offset] = sp / BASELINE_SPREAD
        else:
            # Beyond forecast horizon — use last available
            last_yr = CAP_DATA.index.max()
            c = float(CAP_DATA.loc[last_yr, "bess_gwh"])
            r = float(CAP_DATA.loc[last_yr, "renewable_gwh"])
            sp = compute_spread(BASELINE_SPREAD, c0, c, r0, r, p["alpha"], p["beta"])
            factors[yr_offset] = sp / BASELINE_SPREAD
    return factors

EUR_TO_GBP   = 0.85

# ── electricity cost stack  (Ofgem Q1 2025) ──────────────────────────────────
# These use a reference annual consumption for the cost stack calculation
REF_KWH           = 3_604.4
DAYS              = 365
WHOLESALE_EUR_MWH = 99.84
WHOLESALE_P_KWH   = WHOLESALE_EUR_MWH * EUR_TO_GBP / 1000

SUPPLY_COST   = REF_KWH * WHOLESALE_P_KWH

TNUOS         = REF_KWH * 0.0087 + DAYS * 0.0359
DUOS          = REF_KWH * 0.0363 + DAYS * 0.0731
BSUOS         = REF_KWH * 0.0075
SMART_METER   = REF_KWH * 0.0054 + DAYS * 0.0356
SUPPLIER_OPEX = REF_KWH * 0.0137 + DAYS * 0.1396 - 10
BAD_DEBT      = REF_KWH * 0.0041
GRID_LEVY     = 95.0

TOTAL_OPEX = (GRID_LEVY + CUST_MGMT + TNUOS + DUOS + BSUOS
              + SMART_METER + SUPPLIER_OPEX + BAD_DEBT + STAFF_COST)

# ── competitor ────────────────────────────────────────────────────────────────
COMP_PKW   = 25
COMP_PD    = 50

# ── aggressive tariff scenarios ───────────────────────────────────────────────
TARIFF_SCENARIOS = {
    "A": dict(label="Scenario A — Entry aggressive",
              pkwh=20, pmo=8,  saving_target="~25%"),
    "B": dict(label="Scenario B — Mid aggressive",
              pkwh=19, pmo=6,  saving_target="~30%"),
    "C": dict(label="Scenario C — Maximum aggressive",
              pkwh=18, pmo=5,  saving_target="~35%"),
}

# ── chart colours ─────────────────────────────────────────────────────────────
LOAN_COLORS  = {3: "#E04B3A", 5: "#F5A623", 7: "#4A90D9", 10: "#27AE60"}
SC_COLORS    = {"A": "#4A90D9", "B": "#F5A623", "C": "#E04B3A"}
CONS_COLORS  = {4000: "#4A90D9", 5000: "#F5A623", 6000: "#E04B3A"}
SPREAD_COLORS = {"Best": "#27AE60", "Base": "#F5A623", "Worst": "#E04B3A"}

# ═════════════════════════════════════════════════════════════════════════════
# CONSUMER-FIRST ALLOCATION + MILP OPTIMISATION
# ═════════════════════════════════════════════════════════════════════════════
# The battery sits behind the customer's meter.  When it discharges:
#   - Consumer demand in that hour is served first (offset import, valued
#     at wholesale price — VLP avoids buying at spot)
#   - Excess discharge beyond consumer demand exports to grid (also at
#     wholesale for a licensed supplier)
# Because VLP is a licensed supplier, export price ≈ wholesale price, so
# the hourly consumer demand constraint does NOT reduce the total value of
# each kWh discharged — the MILP runs on the FULL battery.
#
# The only real constraint: during charging hours the consumer still draws
# from the grid, so VLP buys consumer demand + charge power.  But in the
# MILP the charge cost is already priced at the wholesale price of that hour.
#
# Net effect: consumer demand has minimal impact on battery revenue.
# Higher consumption DOES increase supply cost (COGS) on the other side
# of the P&L.
# ═════════════════════════════════════════════════════════════════════════════

def solve_daily_milp(prices_24h, battery_kwh, battery_kw, efficiency,
                     fr_rate_per_kw_year, consumer_demand_kw=None):
    """
    Solve optimal hourly allocation for one day using MILP on FULL battery.

    consumer_demand_kw: optional array of 24 hourly consumer demand values (kW).
        Used for reporting breakdown (consumer-served vs exported) but does NOT
        constrain the optimisation because VLP exports at wholesale.
    """
    eta = np.sqrt(efficiency)
    fr_hourly_rate = fr_rate_per_kw_year * battery_kw / 8760

    E_min = 0.1 * battery_kwh
    E_max = 0.9 * battery_kwh
    E_rest = E_min    # start/end at min SoC — allows full charge-discharge cycle
    E_fr   = 0.5 * battery_kwh   # FR commitment needs 50% SoC

    prob = pulp.LpProblem("Daily_Battery_Opt", pulp.LpMaximize)
    hours = range(24)

    c  = [pulp.LpVariable(f"c_{h}", lowBound=0, upBound=battery_kw) for h in hours]
    d  = [pulp.LpVariable(f"d_{h}", lowBound=0, upBound=battery_kw) for h in hours]
    fr = [pulp.LpVariable(f"fr_{h}", cat='Binary') for h in hours]
    E  = [pulp.LpVariable(f"E_{h}", lowBound=E_min, upBound=E_max) for h in range(25)]

    arb_profit = []
    fr_income  = []
    for h in hours:
        p_kwh = prices_24h[h] / 1000
        # c[h] and d[h] are grid-side (AC) power — efficiency is already
        # handled in the energy balance, so NO additional eta here.
        arb_profit.append(d[h] * p_kwh - c[h] * p_kwh)
        fr_income.append(fr[h] * fr_hourly_rate)

    prob += pulp.lpSum(arb_profit) + pulp.lpSum(fr_income)

    prob += E[0]  == E_rest
    prob += E[24] == E_rest

    for h in hours:
        prob += E[h+1] == E[h] + c[h] * eta - d[h] / eta
        prob += c[h] <= battery_kw * (1 - fr[h])
        prob += d[h] <= battery_kw * (1 - fr[h])

    # FR hours require SoC at 50% (to respond symmetrically)
    M = battery_kwh
    for h in hours:
        prob += E[h] >= E_fr - M * (1 - fr[h])
        prob += E[h] <= E_fr + M * (1 - fr[h])

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        return {"arbitrage_profit": 0, "fr_income": 0, "total_profit": 0,
                "consumer_served_kwh": 0, "exported_kwh": 0}

    total_arb = 0
    total_fr  = 0
    consumer_served = 0
    exported = 0
    for h in hours:
        p_kwh = prices_24h[h] / 1000
        dv = d[h].varValue or 0
        cv = c[h].varValue or 0
        fv = int(fr[h].varValue or 0)
        # d[h] and c[h] are grid-side power — no eta adjustment
        total_arb += dv * p_kwh - cv * p_kwh
        total_fr  += fv * fr_hourly_rate
        # Breakdown: dv is already delivered kW (grid-side), 1 hour = kWh
        if consumer_demand_kw is not None:
            served = min(dv, consumer_demand_kw[h])
            consumer_served += served
            exported += max(0, dv - consumer_demand_kw[h])

    return {"arbitrage_profit": total_arb, "fr_income": total_fr,
            "total_profit": total_arb + total_fr,
            "consumer_served_kwh": consumer_served, "exported_kwh": exported}


def compute_battery_revenue(annual_kwh):
    """
    Run MILP on the FULL battery (15.36 kWh / 6 kW) — consumer-first is an
    hourly power constraint, NOT a capacity reservation.

    The consumer's hourly demand profile is used to report the split between
    consumer-served discharge and grid-exported discharge, but because VLP is
    a licensed supplier (export price ≈ wholesale), the total battery value
    is the same regardless of consumption level.

    What DOES change with consumption level:
      - Supply cost (COGS) — more kWh purchased at wholesale
      - Opex (network charges scale with volume)
      - Customer bill revenue (more kWh × tariff rate)

    Returns dict with annual arbitrage, FR, CM, BM, and breakdown info.
    """
    # Scale hourly profile to target consumption (kW per hour)
    scale = (annual_kwh / SHAPE_ANNUAL_KWH) if SHAPE_ANNUAL_KWH > 0 else 1.0
    consumer_demand_kw = HOURLY_SHAPE * scale / 1000  # Wh → kW (1-hr periods)

    # Load UK prices
    price_file = PROJECT_ROOT / "data" / "prices" / "United Kingdom.csv"
    df = pd.read_csv(price_file)
    df["dt"]    = pd.to_datetime(df["Datetime (UTC)"])
    df          = df[(df["dt"] >= "2024-07-01") & (df["dt"] < "2025-07-01")].copy()
    df["date"]  = df["dt"].dt.date
    df["hour"]  = df["dt"].dt.hour

    total_arb = 0
    total_fr  = 0
    total_consumer_served = 0
    total_exported = 0
    ndays = 0

    for _, grp in df.groupby("date"):
        if len(grp) < 24:
            continue
        grp_sorted = grp.sort_values("hour")
        prices_24 = grp_sorted["Price (EUR/MWhe)"].values  # EUR/MWh

        result = solve_daily_milp(prices_24, BAT_KWH, BAT_KW, EFF_RT,
                                  FR_PER_KW, consumer_demand_kw)
        # MILP arb profit is in EUR; convert to GBP
        total_arb += result["arbitrage_profit"] * EUR_TO_GBP
        total_fr  += result["fr_income"]  # FR rate already in GBP
        total_consumer_served += result.get("consumer_served_kwh", 0)
        total_exported += result.get("exported_kwh", 0)
        ndays += 1

    # CM and BM on full battery kW
    cm_income = BAT_KW * CM_PER_KW * CM_DERATING
    bm_income = total_arb * BM_WIN_RATE * BM_UPLIFT
    grid_svcs = total_fr + cm_income + bm_income

    # Avoided network charges: kWh served from battery don't incur grid
    # network charges (TNUoS 0.87p + DUoS 3.63p + BSUoS 0.75p = 5.25p/kWh)
    AVOIDED_NETWORK_P_KWH = 0.0087 + 0.0363 + 0.0075   # £/kWh
    avoided_network = total_consumer_served * AVOIDED_NETWORK_P_KWH

    return {
        "annual_kwh": annual_kwh,
        "daily_consumption": annual_kwh / 365,
        "arb_net": total_arb,
        "fr_income": total_fr,
        "cm_income": cm_income,
        "bm_income": bm_income,
        "grid_svcs": grid_svcs,
        "avoided_network": avoided_network,
        "total_battery_rev": total_arb + grid_svcs + avoided_network,
        "consumer_served_kwh": total_consumer_served,
        "exported_kwh": total_exported,
        "days_used": ndays,
    }


# ═════════════════════════════════════════════════════════════════════════════
# CORE CALCULATIONS
# ═════════════════════════════════════════════════════════════════════════════

def comp_bill(annual_kwh):
    return annual_kwh * COMP_PKW / 100 + DAYS * COMP_PD / 100

def supply_cost_for(annual_kwh):
    """Scale supply cost to actual consumption level."""
    return annual_kwh * WHOLESALE_P_KWH

def total_opex_for(annual_kwh):
    """Scale volume-dependent opex to actual consumption; fixed costs stay."""
    tnuos  = annual_kwh * 0.0087 + DAYS * 0.0359
    duos   = annual_kwh * 0.0363 + DAYS * 0.0731
    bsuos  = annual_kwh * 0.0075
    smart  = annual_kwh * 0.0054 + DAYS * 0.0356
    sup_op = annual_kwh * 0.0137 + DAYS * 0.1396 - 10
    bad_d  = annual_kwh * 0.0041
    return (GRID_LEVY + CUST_MGMT + tnuos + duos + bsuos
            + smart + sup_op + bad_d + STAFF_COST)

def unit_economics(pkwh, pmo, annual_kwh, battery_info_entry):
    """Per-customer unit economics for a given tariff, consumption, and battery info."""
    elec_rev  = annual_kwh * pkwh / 100
    svc_fee   = pmo * 12
    cust_bill = elec_rev + svc_fee
    battery_rev = battery_info_entry["total_battery_rev"]
    avoided_nw  = battery_info_entry["avoided_network"]
    total_rev = cust_bill + battery_rev
    sc        = supply_cost_for(annual_kwh)
    # Opex: full network charges on ALL consumption, then subtract avoided
    # charges for kWh served from battery (those kWh didn't use the grid)
    opex_gross = total_opex_for(annual_kwh)
    opex       = opex_gross - avoided_nw
    gross     = total_rev - sc
    ebitda    = gross - opex
    ebit      = ebitda - DEPRECIATION
    cb        = comp_bill(annual_kwh)
    saving    = cb - cust_bill
    saving_pct = saving / cb * 100 if cb > 0 else 0
    return dict(
        elec_rev=elec_rev, svc_fee=svc_fee, cust_bill=cust_bill,
        total_rev=total_rev, supply_cost=sc,
        opex_gross=opex_gross, avoided_network=avoided_nw, opex=opex,
        gross=gross, ebitda=ebitda, ebit=ebit,
        saving=saving, saving_pct=saving_pct,
        comp_bill=cb, battery_rev=battery_rev,
    )


def build_pl(ue, loan_years, scaling_factors):
    """Build P&L — revenue from arbitrage/FR/BM scales year-over-year,
    with battery degradation applied."""
    loan_bal = float(BAT_COST)
    ann_prin = BAT_COST / loan_years
    rows = []
    for yr in range(1, MODEL_YEARS + 1):
        sf = scaling_factors.get(yr, 1.0)
        df = degradation_factor(yr)
        # Battery-derived revenue scales by spread compression AND degradation
        battery_rev_yr = ue["battery_rev"] * sf * df
        rev_yr = ue["cust_bill"] + battery_rev_yr
        gross_yr = rev_yr - ue["supply_cost"]
        ebitda_yr = gross_yr - ue["opex"]
        ebit_yr  = ebitda_yr - DEPRECIATION

        interest = loan_bal * LOAN_RATE
        ebt      = ebit_yr - interest
        tax      = max(0.0, ebt * CORP_TAX)
        net_inc  = ebt - tax
        principal = min(ann_prin, loan_bal) if loan_bal > 0.001 else 0.0
        loan_bal  = max(0.0, loan_bal - principal)
        rows.append({
            "Year":             yr,
            "Loan Yrs":         loan_years,
            "Revenue (£)":      round(rev_yr, 2),
            "  Customer Bill":  round(ue["cust_bill"], 2),
            "  Battery Rev":    round(battery_rev_yr, 2),
            "  Spread Scale":   round(sf, 3),
            "  Degrade Scale":  round(df, 3),
            "COGS (£)":         round(ue["supply_cost"], 2),
            "Gross Profit (£)": round(gross_yr, 2),
            "Opex (£)":         round(ue["opex"], 2),
            "  Staff Cost":     round(STAFF_COST, 2),
            "EBITDA (£)":       round(ebitda_yr, 2),
            "Depreciation (£)": round(DEPRECIATION, 2),
            "EBIT (£)":         round(ebit_yr, 2),
            "Interest (£)":     round(interest, 2),
            "EBT (£)":          round(ebt, 2),
            "Tax 25% (£)":      round(tax, 2),
            "Net Income (£)":   round(net_inc, 2),
        })
    return pd.DataFrame(rows)


def build_cashflow(ue, loan_years, scaling_factors):
    """Build cashflow — no depreciation (non-cash, balance sheet only).
    Includes CAC (yr 1) and battery replacement (yr = BAT_LIFE).
    Battery degradation applied year-over-year."""
    loan_bal = float(BAT_COST)
    ann_prin = BAT_COST / loan_years
    cum_cash = 0.0
    rows = []
    for yr in range(1, MODEL_YEARS + 1):
        sf = scaling_factors.get(yr, 1.0)
        df = degradation_factor(yr)
        battery_rev_yr = ue["battery_rev"] * sf * df
        rev_yr   = ue["cust_bill"] + battery_rev_yr
        gross_yr = rev_yr - ue["supply_cost"]
        ebitda_yr = gross_yr - ue["opex"]

        # Tax uses P&L profit (which includes depreciation) — but cashflow
        # itself excludes depreciation.  Use EBIT for tax calc only.
        ebit_yr   = ebitda_yr - DEPRECIATION
        interest  = loan_bal * LOAN_RATE
        ebt       = ebit_yr - interest
        tax       = max(0.0, ebt * CORP_TAX)

        # Operating cashflow = EBITDA - tax (no depreciation add-back needed
        # because we start from EBITDA, not net income)
        ocf       = ebitda_yr - tax
        principal = min(ann_prin, loan_bal) if loan_bal > 0.001 else 0.0
        loan_bal  = max(0.0, loan_bal - principal)

        # Extra cash items
        cac_outflow = CAC if yr == 1 else 0.0

        free_cash = ocf - interest - principal - cac_outflow
        cum_cash += free_cash
        rows.append({
            "Year":                     yr,
            "Loan Yrs":                 loan_years,
            "EBITDA (£)":               round(ebitda_yr, 2),
            "Tax (£)":                  round(tax, 2),
            "Operating CF (£)":         round(ocf, 2),
            "Interest (£)":             round(interest, 2),
            "Principal Repaid (£)":     round(principal, 2),
            "CAC (£)":                  round(cac_outflow, 2),
            "Free Cash (£)":            round(free_cash, 2),
            "Cumulative Cash (£)":      round(cum_cash, 2),
            "CF Positive":              free_cash >= 0,
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


# ═════════════════════════════════════════════════════════════════════════════
# BREAK-EVEN TARIFF CALCULATOR
# ═════════════════════════════════════════════════════════════════════════════

def _simulate_cum_cash(cust_bill_annual, battery_rev_base, supply_cost, opex,
                       loan_years, scaling_factors):
    """Simulate cumulative cash at end of loan period for a given annual customer bill."""
    loan_bal = float(BAT_COST)
    ann_prin = BAT_COST / loan_years
    cum = 0.0
    for yr in range(1, loan_years + 1):
        sf = scaling_factors.get(yr, 1.0)
        df = degradation_factor(yr)
        bat_rev = battery_rev_base * sf * df
        rev = cust_bill_annual + bat_rev
        gross = rev - supply_cost
        ebitda = gross - opex
        ebit = ebitda - DEPRECIATION
        interest = loan_bal * LOAN_RATE
        ebt = ebit - interest
        tax = max(0.0, ebt * CORP_TAX)
        ocf = ebitda - tax
        principal = min(ann_prin, loan_bal) if loan_bal > 0.001 else 0.0
        loan_bal = max(0.0, loan_bal - principal)
        cac_out = CAC if yr == 1 else 0.0
        cum += ocf - interest - principal - cac_out
    return cum


def find_breakeven_tariffs(battery_rev_base, supply_cost, opex,
                           loan_years, scaling_factors):
    """Find break-even tariff expressed multiple ways.

    Returns dict with:
      - min_bill: minimum annual customer bill for cumulative cash = 0 at end of loan
      - pkwh_at_sc: for several standing charges, the required p/kWh at each consumption level
    """
    # Binary search for total annual customer bill that gives cum cash = 0
    lo, hi = 0.0, 10000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _simulate_cum_cash(mid, battery_rev_base, supply_cost, opex,
                              loan_years, scaling_factors) < 0:
            lo = mid
        else:
            hi = mid
    min_bill = (lo + hi) / 2

    # Express as p/kWh for a few standing charge levels, at each consumption
    standing_charges_monthly = [0, 5, 8, 10, 15]
    pkwh_table = {}
    for ckwh in CONSUMPTION_LEVELS:
        pkwh_table[ckwh] = {}
        for sc_mo in standing_charges_monthly:
            sc_annual = sc_mo * 12
            remaining = max(0, min_bill - sc_annual)
            pkwh = remaining / ckwh * 100  # convert £ to pence
            pkwh_table[ckwh][sc_mo] = round(pkwh, 2)

    return {"min_bill": round(min_bill, 2), "pkwh_table": pkwh_table}


# ═════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═════════════════════════════════════════════════════════════════════════════

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_consumption_comparison(all_results, tariff_key, spread_name, loan_yr=5):
    """
    Chart comparing all consumption levels for a given tariff + spread combo.
    Top: annual FCF  |  Bottom: cumulative cash
    Returns base64 PNG.
    """
    sc = TARIFF_SCENARIOS[tariff_key]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    fig.suptitle(
        f"{sc['label']}  |  {spread_name} spread  |  {loan_yr}-yr loan\n"
        f"Tariff: {sc['pkwh']}p/kWh + £{sc['pmo']}/mo",
        fontsize=11, fontweight="bold", y=0.99,
    )

    years = np.arange(1, MODEL_YEARS + 1)

    for ckwh in CONSUMPTION_LEVELS:
        key = (tariff_key, spread_name, ckwh, loan_yr)
        cf = all_results["cf"][key]
        color = CONS_COLORS[ckwh]
        fcf = cf["Free Cash (£)"].values
        cum = cf["Cumulative Cash (£)"].values
        label = f"{ckwh:,} kWh/yr"
        ax1.plot(years, fcf, marker="o", markersize=4, color=color, label=label, linewidth=2)
        ax2.plot(years, cum, marker="o", markersize=4, color=color, label=label, linewidth=2)

    for ax in [ax1, ax2]:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    ax1.set_ylabel("Annual Free Cash (£)")
    ax1.set_title("Annual Free Cashflow by Consumption Level")
    ax2.set_ylabel("Cumulative Cash (£)")
    ax2.set_xlabel("Year")
    ax2.set_xticks(years)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig_to_base64(fig)


def plot_spread_comparison(all_results, tariff_key, consumption, loan_yr=5):
    """
    Chart comparing all spread scenarios for a given tariff + consumption.
    Returns base64 PNG.
    """
    sc = TARIFF_SCENARIOS[tariff_key]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    fig.suptitle(
        f"{sc['label']}  |  {consumption:,} kWh/yr  |  {loan_yr}-yr loan\n"
        f"Spread scenarios compared",
        fontsize=11, fontweight="bold", y=0.99,
    )

    years = np.arange(1, MODEL_YEARS + 1)

    for sp_name in SPREAD_SCENARIOS:
        key = (tariff_key, sp_name, consumption, loan_yr)
        cf = all_results["cf"][key]
        color = SPREAD_COLORS[sp_name]
        fcf = cf["Free Cash (£)"].values
        cum = cf["Cumulative Cash (£)"].values
        ax1.plot(years, fcf, marker="o", markersize=4, color=color, label=sp_name, linewidth=2)
        ax2.plot(years, cum, marker="o", markersize=4, color=color, label=sp_name, linewidth=2)

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


def plot_scenario_chart(sc_key, sc, all_results, spread_name, consumption, out_path):
    """Per-tariff chart with all loan periods, for a given spread + consumption."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    fig.suptitle(
        f"VLP Battery — {sc['label']}  ({consumption:,} kWh/yr, {spread_name} spread)\n"
        f"Tariff: {sc['pkwh']}p/kWh + £{sc['pmo']}/mo  |  Battery: £{BAT_COST:,.0f}",
        fontsize=11, fontweight="bold", y=0.99,
    )

    years   = np.arange(1, MODEL_YEARS + 1)
    n_loans = len(LOAN_YEARS)
    bar_w   = 0.18
    offsets = np.linspace(-(n_loans-1)/2, (n_loans-1)/2, n_loans) * bar_w

    for i, ly in enumerate(LOAN_YEARS):
        key   = (sc_key, spread_name, consumption, ly)
        cf    = all_results["cf"][key]
        color = LOAN_COLORS[ly]
        fcf   = cf["Free Cash (£)"].values
        cum   = cf["Cumulative Cash (£)"].values

        ax1.bar(years + offsets[i], fcf,
                width=bar_w, color=color, label=f"{ly}-yr loan", alpha=0.85)
        ax2.plot(years, cum, marker="o", markersize=4,
                 color=color, label=f"{ly}-yr loan", linewidth=2)

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
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_chart(all_results, spread_name, consumption, out_path):
    """Compare all 3 tariff scenarios at 5-yr loan."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(
        f"VLP Battery — Scenario Comparison  ({consumption:,} kWh/yr, {spread_name} spread, 5-yr loan)",
        fontsize=11, fontweight="bold",
    )

    years   = np.arange(1, MODEL_YEARS + 1)
    bar_w   = 0.25
    offsets = [-0.25, 0.0, 0.25]

    for i, sc_key in enumerate(["A", "B", "C"]):
        sc    = TARIFF_SCENARIOS[sc_key]
        key   = (sc_key, spread_name, consumption, 5)
        cf    = all_results["cf"][key]
        color = SC_COLORS[sc_key]
        tariff = f"{sc['pkwh']}p+£{sc['pmo']}/mo ({sc['saving_target']})"
        fcf   = cf["Free Cash (£)"].values
        cum   = cf["Cumulative Cash (£)"].values

        ax1.bar(years + offsets[i], fcf,
                width=bar_w, color=color, label=tariff, alpha=0.85)
        ax2.plot(years, cum, marker="o", markersize=4,
                 color=color, label=tariff, linewidth=2)

    for ax in [ax1, ax2]:
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    ax1.set_ylabel("Annual Free Cash (£)")
    ax2.set_ylabel("Cumulative Cash (£)")
    ax2.set_xlabel("Year")
    ax2.set_xticks(years)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ═════════════════════════════════════════════════════════════════════════════

def _build_report_js(battery_info):
    """Build JavaScript for interactive report recalculation."""
    all_sf = {}
    for sp_name in SPREAD_SCENARIOS:
        factors = get_scaling_factors(sp_name)
        all_sf[sp_name] = {str(k): round(v, 6) for k, v in factors.items()}

    bi_data = {}
    for ckwh, bi in battery_info.items():
        bi_data[str(ckwh)] = {
            "total_battery_rev": round(bi["total_battery_rev"], 2),
            "avoided_network": round(bi["avoided_network"], 2),
        }

    tariff_data = {}
    for tk, sc in TARIFF_SCENARIOS.items():
        tariff_data[tk] = {
            "pkwh": sc["pkwh"], "pmo": sc["pmo"],
            "label": sc["label"], "saving_target": sc["saving_target"],
        }

    data = {
        "BAT_COST": BAT_COST, "DEPRECIATION": DEPRECIATION,
        "LOAN_RATE": LOAN_RATE, "CORP_TAX": CORP_TAX,
        "MODEL_YEARS": MODEL_YEARS, "DEGRADATION_PA": DEGRADATION_PA,
        "LOAN_YEARS": LOAN_YEARS, "CONSUMPTION_LEVELS": CONSUMPTION_LEVELS,
        "WHOLESALE_P_KWH": round(WHOLESALE_P_KWH, 6),
        "GRID_LEVY": GRID_LEVY,
        "COMP_PKW": COMP_PKW, "COMP_PD": COMP_PD, "DAYS": DAYS,
        "DEF_CAC": CAC, "DEF_STAFF": STAFF_COST, "DEF_MGMT": CUST_MGMT,
        "battery_info": bi_data,
        "scaling": all_sf,
        "tariffs": tariff_data,
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
function calcOpex(kwh,staff,mgmt){
    return D.GRID_LEVY+mgmt+kwh*0.0087+365*0.0359+kwh*0.0363+365*0.0731
        +kwh*0.0075+kwh*0.0054+365*0.0356+kwh*0.0137+365*0.1396-10+kwh*0.0041+staff;
}
function calcSupply(kwh){return kwh*D.WHOLESALE_P_KWH;}
function compBill(kwh){return kwh*D.COMP_PKW/100+D.DAYS*D.COMP_PD/100;}
function unitEcon(pkwh,pmo,kwh,inp){
    var bi=D.battery_info[String(kwh)];
    var custBill=kwh*pkwh/100+pmo*12;
    var batteryRev=bi.total_battery_rev, avoidedNw=bi.avoided_network;
    var totalRev=custBill+batteryRev;
    var sc=calcSupply(kwh);
    var opex=calcOpex(kwh,inp.staff,inp.mgmt)-avoidedNw;
    var gross=totalRev-sc, ebitda=gross-opex, ebit=ebitda-D.DEPRECIATION;
    return {custBill:custBill,batteryRev:batteryRev,totalRev:totalRev,
            supplyCost:sc,opex:opex,gross:gross,ebitda:ebitda,ebit:ebit};
}
function buildCF(ue,loanYrs,sf,cac){
    var bal=D.BAT_COST,annP=D.BAT_COST/loanYrs,cum=0,rows=[];
    for(var yr=1;yr<=D.MODEL_YEARS;yr++){
        var s=sf[String(yr)]||1, df=degradeFactor(yr);
        var batRev=ue.batteryRev*s*df, rev=ue.custBill+batRev;
        var gross=rev-ue.supplyCost, ebitda=gross-ue.opex;
        var ebit=ebitda-D.DEPRECIATION, interest=bal*D.LOAN_RATE;
        var ebt=ebit-interest, tax=Math.max(0,ebt*D.CORP_TAX), ocf=ebitda-tax;
        var prin=bal>0.001?Math.min(annP,bal):0; bal=Math.max(0,bal-prin);
        var cacOut=yr===1?cac:0, fc=ocf-interest-prin-cacOut; cum+=fc;
        rows.push({yr:yr,rev:rev,batRev:batRev,s:s,df:df,gross:gross,
            ebitda:ebitda,ebit:ebit,interest:interest,ebt:ebt,tax:tax,
            netInc:ebt-tax,ocf:ocf,prin:prin,cacOut:cacOut,fc:fc,cum:cum});
    }
    return rows;
}
function renderSummary(){
    var inp=getInputs();
    var h='<table class="data-table summary-table"><thead><tr>';
    ['Tariff','Spread','Consumption','Cust Bill £','Battery Rev £',
     'Total Rev £','Supply Cost £','Opex £','EBIT £',
     'Yr1 FCF £',D.MODEL_YEARS+'yr Cum £','Break-even'].forEach(function(c){h+='<th>'+c+'</th>';});
    h+='</tr></thead><tbody>';
    Object.keys(D.tariffs).forEach(function(tk){
        var t=D.tariffs[tk];
        D.spread_names.forEach(function(sp){
            var sf=D.scaling[sp];
            D.CONSUMPTION_LEVELS.forEach(function(ckwh){
                var ue=unitEcon(t.pkwh,t.pmo,ckwh,inp);
                var cf=buildCF(ue,5,sf,inp.cac);
                var yr1=cf[0].fc, cumF=cf[cf.length-1].cum;
                var beYr='Never';
                for(var i=0;i<cf.length;i++){if(cf[i].cum>=0){beYr=i+1;break;}}
                var cls=beYr==='Never'?' class="negative"':'';
                h+='<tr><td>'+t.pkwh+'p+\u00a3'+t.pmo+'/mo</td><td>'+sp+'</td>';
                h+='<td>'+ckwh.toLocaleString()+'</td>';
                h+='<td>'+fmt0(ue.custBill)+'</td><td>'+fmt0(ue.batteryRev)+'</td>';
                h+='<td>'+fmt0(ue.totalRev)+'</td><td>'+fmt0(ue.supplyCost)+'</td>';
                h+='<td>'+fmt0(ue.opex)+'</td><td>'+fmt0(ue.ebit)+'</td>';
                h+='<td>'+fmt0(yr1)+'</td><td>'+fmt0(cumF)+'</td>';
                h+='<td'+cls+'>'+beYr+'</td></tr>';
            });
        });
    });
    h+='</tbody></table>';
    document.getElementById('summary-dashboard').innerHTML=h;
}
function renderPLTable(ue,loanYrs,sf){
    var bal=D.BAT_COST,annP=D.BAT_COST/loanYrs;
    var h='<table class="data-table"><thead><tr>';
    ['Year','Revenue (£)','  Cust Bill','  Battery Rev','  Spread','  Degrade',
     'COGS (£)','Gross (£)','Opex (£)','EBITDA (£)','Depr (£)','EBIT (£)',
     'Interest (£)','EBT (£)','Tax 25% (£)','Net Inc (£)'].forEach(function(c){h+='<th>'+c+'</th>';});
    h+='</tr></thead><tbody>';
    for(var yr=1;yr<=D.MODEL_YEARS;yr++){
        var s=sf[String(yr)]||1, df=degradeFactor(yr);
        var batRev=ue.batteryRev*s*df, rev=ue.custBill+batRev;
        var gross=rev-ue.supplyCost, ebitda=gross-ue.opex;
        var ebit=ebitda-D.DEPRECIATION, interest=bal*D.LOAN_RATE;
        var ebt=ebit-interest, tax=Math.max(0,ebt*D.CORP_TAX), ni=ebt-tax;
        var prin=bal>0.001?Math.min(annP,bal):0; bal=Math.max(0,bal-prin);
        h+='<tr><td>'+yr+'</td><td>'+fmt(rev)+'</td><td>'+fmt(ue.custBill)+'</td>';
        h+='<td>'+fmt(batRev)+'</td><td>'+s.toFixed(3)+'</td><td>'+df.toFixed(3)+'</td>';
        h+='<td>'+fmt(ue.supplyCost)+'</td><td>'+fmt(gross)+'</td><td>'+fmt(ue.opex)+'</td>';
        h+='<td>'+fmt(ebitda)+'</td><td>'+fmt(D.DEPRECIATION)+'</td><td>'+fmt(ebit)+'</td>';
        h+='<td>'+fmt(interest)+'</td><td>'+fmt(ebt)+'</td><td>'+fmt(tax)+'</td>';
        h+='<td>'+fmt(ni)+'</td></tr>';
    }
    h+='</tbody></table>'; return h;
}
function renderCFTable(ue,loanYrs,sf,cac){
    var cf=buildCF(ue,loanYrs,sf,cac);
    var h='<table class="data-table"><thead><tr>';
    ['Year','EBITDA (£)','Tax (£)','Op CF (£)','Interest (£)','Principal (£)',
     'CAC (£)','Free Cash (£)','Cumulative (£)'].forEach(function(c){h+='<th>'+c+'</th>';});
    h+='</tr></thead><tbody>';
    cf.forEach(function(r){
        var cls=r.fc<0?' class="negative"':' class="positive"';
        h+='<tr><td>'+r.yr+'</td><td>'+fmt(r.ebitda)+'</td><td>'+fmt(r.tax)+'</td>';
        h+='<td>'+fmt(r.ocf)+'</td><td>'+fmt(r.interest)+'</td><td>'+fmt(r.prin)+'</td>';
        h+='<td>'+fmt(r.cacOut)+'</td><td'+cls+'>'+fmt(r.fc)+'</td><td>'+fmt(r.cum)+'</td></tr>';
    });
    h+='</tbody></table>'; return h;
}
function renderDetails(){
    var inp=getInputs(), sf=D.scaling['Base'];
    Object.keys(D.tariffs).forEach(function(tk){
        var t=D.tariffs[tk], h='';
        D.CONSUMPTION_LEVELS.forEach(function(ckwh){
            var ue=unitEcon(t.pkwh,t.pmo,ckwh,inp);
            h+='<details><summary>\ud83d\udccb P&L \u2014 Base spread, '+ckwh.toLocaleString()+' kWh/yr, 5-yr loan</summary>';
            h+=renderPLTable(ue,5,sf)+'</details>';
            h+='<details><summary>\ud83d\udccb Cashflow \u2014 Base spread, '+ckwh.toLocaleString()+' kWh/yr, 5-yr loan</summary>';
            h+=renderCFTable(ue,5,sf,inp.cac)+'</details>';
        });
        document.getElementById('detail-tables-'+tk).innerHTML=h;
    });
}
function findBreakevenBill(batteryRev,supplyCost,opex,loanYrs,sf,cac){
    var lo=0,hi=10000;
    for(var i=0;i<200;i++){
        var mid=(lo+hi)/2, bal=D.BAT_COST, annP=D.BAT_COST/loanYrs, cum=0;
        for(var yr=1;yr<=loanYrs;yr++){
            var s=sf[String(yr)]||1, df=degradeFactor(yr);
            var batRev=batteryRev*s*df, rev=mid+batRev;
            var gross=rev-supplyCost, ebitda=gross-opex;
            var ebit=ebitda-D.DEPRECIATION, interest=bal*D.LOAN_RATE;
            var ebt=ebit-interest, tax=Math.max(0,ebt*D.CORP_TAX), ocf=ebitda-tax;
            var prin=bal>0.001?Math.min(annP,bal):0; bal=Math.max(0,bal-prin);
            var cacOut=yr===1?cac:0; cum+=ocf-interest-prin-cacOut;
        }
        if(cum<0) lo=mid; else hi=mid;
    }
    return (lo+hi)/2;
}
function renderBreakeven(){
    var inp=getInputs(), sf=D.scaling['Base'], scs=[0,5,8,10,15];
    var h='<div class="section"><h2>Break-Even Tariff Analysis</h2>';
    h+='<p>Minimum tariff for <strong>cumulative FCF = 0</strong> at end of loan. ';
    h+='Includes degradation, spread compression (Base), loan repayments, tax, CAC, and all opex.</p>';
    [5,7,10].forEach(function(loanYr){
        h+='<h3>'+loanYr+'-Year Loan Break-Even</h3>';
        D.CONSUMPTION_LEVELS.forEach(function(ckwh){
            var bi=D.battery_info[String(ckwh)];
            var opex=calcOpex(ckwh,inp.staff,inp.mgmt)-bi.avoided_network;
            var sc=calcSupply(ckwh);
            var minBill=findBreakevenBill(bi.total_battery_rev,sc,opex,loanYr,sf,inp.cac);
            var comp=compBill(ckwh);
            h+='<details open><summary>'+ckwh.toLocaleString()+' kWh/yr \u2014 minimum annual bill: \u00a3'+fmt0(minBill)+'</summary>';
            h+='<table class="data-table"><thead><tr><th>Standing charge (\u00a3/mo)</th>';
            h+='<th>Required p/kWh</th><th>Annual bill \u00a3</th><th>vs competitor saving</th></tr></thead><tbody>';
            scs.forEach(function(scMo){
                var remaining=Math.max(0,minBill-scMo*12);
                var pkwh=remaining/ckwh*100;
                var bill=ckwh*pkwh/100+scMo*12;
                var savPct=comp>0?(comp-bill)/comp*100:0;
                var cls=savPct>0?'positive':'negative';
                h+='<tr><td>\u00a3'+scMo+'/mo</td><td>'+pkwh.toFixed(1)+'p</td>';
                h+='<td>\u00a3'+fmt0(bill)+'</td>';
                h+='<td class="'+cls+'">'+(savPct>=0?'+':'')+savPct.toFixed(1)+'%</td></tr>';
            });
            h+='</tbody></table></details>';
        });
    });
    h+='<div class="highlight"><strong>Reading the table:</strong> Pick a standing charge, ';
    h+='read across for the minimum p/kWh to break even. "vs competitor" = customer saving (positive = saves).</div></div>';
    document.getElementById('breakeven-section').innerHTML=h;
}
function recalcAll(){renderSummary();renderDetails();renderBreakeven();}
document.addEventListener('DOMContentLoaded',recalcAll);
"""
    return js


def build_html_report(all_results, battery_info, breakeven_data, out_path):
    """Generate interactive HTML report with all scenario combinations."""

    # Helper: dataframe to HTML table
    def df_to_html(df, highlight_col=None):
        cols_to_drop = [c for c in ["Loan Yrs", "CF Positive"] if c in df.columns]
        df_disp = df.drop(columns=cols_to_drop, errors="ignore")
        html = '<table class="data-table">\n<thead><tr>'
        for c in df_disp.columns:
            html += f"<th>{c}</th>"
        html += "</tr></thead>\n<tbody>\n"
        for _, row in df_disp.iterrows():
            html += "<tr>"
            for c in df_disp.columns:
                val = row[c]
                cls = ""
                if highlight_col and c == highlight_col:
                    try:
                        cls = ' class="negative"' if float(val) < 0 else ' class="positive"'
                    except (ValueError, TypeError):
                        pass
                if isinstance(val, float):
                    html += f"<td{cls}>{val:,.2f}</td>"
                else:
                    html += f"<td{cls}>{val}</td>"
            html += "</tr>\n"
        html += "</tbody></table>\n"
        return html

    # Generate charts as base64
    charts = {}
    for tk in TARIFF_SCENARIOS:
        for sp in SPREAD_SCENARIOS:
            charts[f"cons_{tk}_{sp}"] = plot_consumption_comparison(
                all_results, tk, sp, loan_yr=5)
        for ckwh in CONSUMPTION_LEVELS:
            charts[f"spread_{tk}_{ckwh}"] = plot_spread_comparison(
                all_results, tk, ckwh, loan_yr=5)

    # Build JS code for interactive recalculation
    js_code = _build_report_js(battery_info)

    # Build detailed sections
    detail_sections = ""
    for tk in TARIFF_SCENARIOS:
        sc = TARIFF_SCENARIOS[tk]
        detail_sections += f"""
        <div class="section">
        <h2>{sc['label']}</h2>
        <p>Tariff: {sc['pkwh']}p/kWh + £{sc['pmo']}/mo  |  Target saving: {sc['saving_target']}</p>
        """

        # Charts: consumption comparison for each spread
        for sp in SPREAD_SCENARIOS:
            chart_key = f"cons_{tk}_{sp}"
            detail_sections += f"""
            <details>
            <summary>📊 Consumption comparison — {sp} spread</summary>
            <img src="data:image/png;base64,{charts[chart_key]}" style="max-width:100%">
            </details>
            """

        # Charts: spread comparison for each consumption
        for ckwh in CONSUMPTION_LEVELS:
            chart_key = f"spread_{tk}_{ckwh}"
            detail_sections += f"""
            <details>
            <summary>📊 Spread comparison — {ckwh:,} kWh/yr</summary>
            <img src="data:image/png;base64,{charts[chart_key]}" style="max-width:100%">
            </details>
            """

        # P&L/CF tables rendered by JavaScript
        detail_sections += f'<div id="detail-tables-{tk}"></div>\n'
        detail_sections += "</div>\n"

    # Battery revenue breakdown
    bi0 = battery_info[CONSUMPTION_LEVELS[0]]
    alloc_html = f"""<p>Trading revenue (arb + FR + CM + BM) is the same for all consumption
    levels — the battery trades on the full {BAT_KWH} kWh / {BAT_KW:.0f} kW. But
    <strong>avoided network charges</strong> vary: kWh served to the consumer from the
    battery don't incur TNUoS/DUoS/BSUoS (5.25p/kWh), so higher consumption = more
    consumer-served discharge = more avoided charges.</p>
    <table class='data-table'><thead><tr>
    <th>Consumption</th><th>Arb £</th><th>FR £</th><th>CM £</th><th>BM £</th>
    <th>Consumer Served kWh</th><th>Avoided Network £</th><th>Total Battery Rev £</th>
    </tr></thead><tbody>\n"""
    for ckwh in CONSUMPTION_LEVELS:
        bi = battery_info[ckwh]
        alloc_html += f"""<tr>
            <td>{ckwh:,} kWh/yr</td>
            <td>{bi['arb_net']:,.0f}</td>
            <td>{bi['fr_income']:,.0f}</td>
            <td>{bi['cm_income']:,.0f}</td>
            <td>{bi['bm_income']:,.0f}</td>
            <td>{bi['consumer_served_kwh']:,.0f}</td>
            <td>{bi['avoided_network']:,.0f}</td>
            <td><strong>{bi['total_battery_rev']:,.0f}</strong></td>
        </tr>\n"""
    alloc_html += "</tbody></table>\n"

    # Build tariff rows
    tariff_rows = ""
    for sk, sc in TARIFF_SCENARIOS.items():
        b4 = 4000 * sc['pkwh'] / 100 + sc['pmo'] * 12
        b5 = 5000 * sc['pkwh'] / 100 + sc['pmo'] * 12
        b6 = 6000 * sc['pkwh'] / 100 + sc['pmo'] * 12
        tariff_rows += f'<tr><td>{sc["label"]}</td><td>{sc["pkwh"]}p/kWh</td><td>£{sc["pmo"]}/mo</td><td>£{b4:,.0f}</td><td>£{b5:,.0f}</td><td>£{b6:,.0f}</td><td>{sc["saving_target"]}</td></tr>\n'

    comp4 = 4000 * COMP_PKW / 100 + DAYS * COMP_PD / 100
    comp5 = 5000 * COMP_PKW / 100 + DAYS * COMP_PD / 100
    comp6 = 6000 * COMP_PKW / 100 + DAYS * COMP_PD / 100

    # Build spread compression table
    spread_table = ""
    for sp_name, sp_params in SPREAD_SCENARIOS.items():
        factors = get_scaling_factors(sp_name)
        yr1 = factors[1]
        yr6 = factors[6]
        yr12 = factors[MODEL_YEARS]
        spread_table += f'<tr><td>{sp_name}</td><td>{sp_params["alpha"]}</td><td>{sp_params["beta"]}</td><td>{yr1:.3f}</td><td>{yr6:.3f}</td><td>{yr12:.3f}</td></tr>\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VLP Battery — Aggressive Offering Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           max-width: 1200px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #333; }}
    h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
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
    .highlight {{ background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107;
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
    .controls .btn {{ background: #3498db; border: none; color: white; padding: 6px 16px;
                      border-radius: 4px; cursor: pointer; font-weight: 600; font-size: 13px; }}
    .controls .btn:hover {{ background: #2980b9; }}
</style>
</head>
<body>

<h1>VLP Battery — Electricity Company Model</h1>
<p>Per-customer unit economics for a behind-the-meter battery operated by a licensed electricity supplier.
   Battery trades on wholesale + ancillary markets. Consumer demand served from grid, offset by battery discharge.</p>

<div class="controls">
    <span style="font-weight:700; font-size:14px;">Adjust Inputs</span>
    <label>CAC &pound; <input type="number" id="inp-cac" value="{CAC}" step="10"></label>
    <label>Staff &pound;/yr <input type="number" id="inp-staff" value="{STAFF_COST}" step="5"></label>
    <label>Cust Mgmt &pound;/yr <input type="number" id="inp-mgmt" value="{CUST_MGMT}" step="5"></label>
    <button class="btn" onclick="recalcAll()">Recalculate</button>
    <span style="font-size:11px; opacity:0.7;">Charts are static &mdash; tables &amp; break-even update live</span>
</div>

<div class="highlight">
    Battery revenue includes arbitrage (MILP-optimised), frequency response, capacity market, balancing mechanism,
    and <strong>avoided network charges</strong> on kWh served from battery. Higher consumption = more avoided charges.
    Consumption levels modelled: {', '.join(f'{c:,} kWh' for c in CONSUMPTION_LEVELS)}.
</div>

<div class="section">
<h2>Battery Revenue Breakdown</h2>
{alloc_html}
</div>

<div class="section">
<h2>Key Parameters</h2>

<h3>Battery Hardware</h3>
<table class="data-table">
<thead><tr><th>Parameter</th><th>Value</th><th>Notes</th></tr></thead>
<tbody>
<tr><td>Capacity</td><td>{BAT_KWH} kWh</td><td>Usable LFP cells</td></tr>
<tr><td>Inverter power</td><td>{BAT_KW:.0f} kW</td><td>Max charge/discharge rate</td></tr>
<tr><td>Duration</td><td>{BAT_KWH/BAT_KW:.2f} hrs</td><td>Capacity / power</td></tr>
<tr><td>Round-trip efficiency</td><td>{EFF_RT*100:.0f}%</td><td>One-way: {ETA*100:.2f}%</td></tr>
<tr><td>SoC operating range</td><td>10% – 90%</td><td>Usable: {BAT_KWH*0.8:.2f} kWh ({BAT_KWH*0.8/BAT_KWH*100:.0f}% of total)</td></tr>
<tr><td>Daily SoC start/end</td><td>10% ({BAT_KWH*0.1:.2f} kWh)</td><td>Allows full charge-discharge cycle per day</td></tr>
<tr><td>Grid draw per full charge</td><td>{GRID_DRAW:.2f} kWh</td><td>= capacity / one-way efficiency</td></tr>
<tr><td>Energy delivered per discharge</td><td>{DISCHARGE:.2f} kWh</td><td>= capacity x one-way efficiency</td></tr>
<tr><td>Charge time (full)</td><td>{T_CHARGE:.2f} hrs</td><td>Inverter-constrained</td></tr>
<tr><td>Discharge time (full)</td><td>{T_DISCHARGE:.2f} hrs</td><td>Inverter-constrained</td></tr>
</tbody>
</table>

<h3>Battery Cost &amp; Financing</h3>
<table class="data-table">
<thead><tr><th>Parameter</th><th>Value</th><th>Notes</th></tr></thead>
<tbody>
<tr><td>Total cost (hardware + install)</td><td>£{BAT_COST:,.0f}</td><td>Overrides CSV default of £{PARAMS.get("battery cost inc install", 2600):,.0f}</td></tr>
<tr><td>Economic life</td><td>{BAT_LIFE} years</td><td>Straight-line depreciation</td></tr>
<tr><td>Annual depreciation</td><td>£{DEPRECIATION:,.0f}/yr</td><td>P&amp;L only — excluded from cashflow</td></tr>
<tr><td>Loan interest rate</td><td>{LOAN_RATE*100:.0f}% p.a.</td><td>On declining balance, equal principal repayments</td></tr>
<tr><td>Corporation tax</td><td>{CORP_TAX*100:.0f}%</td><td>Applied to EBT each year, no loss carry-forward</td></tr>
<tr><td>Loan periods modelled</td><td>{', '.join(str(y)+'-yr' for y in LOAN_YEARS)}</td><td></td></tr>
<tr><td>Model horizon</td><td>{MODEL_YEARS} years</td><td>10-yr loan gets 2 post-payoff years</td></tr>
<tr><td>Annual degradation</td><td>{DEGRADATION_PA*100:.0f}%/yr</td><td>Compound; yr5 retention {degradation_factor(5)*100:.1f}%, yr10 {degradation_factor(10)*100:.1f}%, yr12 {degradation_factor(12)*100:.1f}%</td></tr>
</tbody>
</table>

<h3>Grid Services Revenue</h3>
<table class="data-table">
<thead><tr><th>Service</th><th>Rate</th><th>Annual (full battery)</th><th>Notes</th></tr></thead>
<tbody>
<tr><td>Frequency Response (FR)</td><td>£{FR_PER_KW:.0f}/kW/yr</td><td>£{BAT_KW*FR_PER_KW:,.0f} max</td><td>MILP jointly optimises FR vs arb hour-by-hour; actual FR &lt; max</td></tr>
<tr><td>Capacity Market (CM)</td><td>£{CM_PER_KW:.0f}/kW/yr</td><td>£{BAT_KW*CM_PER_KW*CM_DERATING:,.0f}</td><td>De-rated {CM_DERATING*100:.1f}% (2-hr battery class, {BAT_KWH/BAT_KW:.1f}h duration)</td></tr>
<tr><td>Balancing Mechanism (BM)</td><td>{BM_WIN_RATE*100:.0f}% win x {BM_UPLIFT*100:.0f}% uplift on arb</td><td>Varies</td><td>Estimated as % of arbitrage, not explicitly dispatched</td></tr>
<tr><td>Avoided network charges</td><td>5.25p/kWh served from battery</td><td>Varies by consumption</td><td>TNUoS 0.87p + DUoS 3.63p + BSUoS 0.75p</td></tr>
</tbody>
</table>

<h3>Wholesale &amp; FX</h3>
<table class="data-table">
<thead><tr><th>Parameter</th><th>Value</th><th>Notes</th></tr></thead>
<tbody>
<tr><td>Price data source</td><td>United Kingdom.csv</td><td>Day-ahead hourly prices (EUR/MWh)</td></tr>
<tr><td>Price period</td><td>Jul 2024 – Jun 2025</td><td>Latest full year, 365 days</td></tr>
<tr><td>Average wholesale price</td><td>{WHOLESALE_EUR_MWH} EUR/MWh = {WHOLESALE_P_KWH*100:.2f}p/kWh</td><td></td></tr>
<tr><td>EUR/GBP rate</td><td>{EUR_TO_GBP}</td><td>Fixed throughout model</td></tr>
<tr><td>Use profile source</td><td>use_profiles.csv (UK)</td><td>24-hour shape, {SHAPE_DAILY_WH:.0f} Wh/day = {SHAPE_ANNUAL_KWH:,.0f} kWh/yr base</td></tr>
<tr><td>Consumption levels</td><td>{', '.join(f'{c:,}' for c in CONSUMPTION_LEVELS)} kWh/yr</td><td>Profile scaled proportionally to each level</td></tr>
</tbody>
</table>

<h3>Cost Stack (Ofgem Q1 2025 methodology, per customer/yr)</h3>
<table class="data-table">
<thead><tr><th>Cost line</th><th>Rate</th><th>At 4,000 kWh</th><th>At 5,000 kWh</th><th>At 6,000 kWh</th></tr></thead>
<tbody>
<tr><td><strong>Wholesale electricity (COGS)</strong></td><td>{WHOLESALE_P_KWH*100:.2f}p/kWh</td><td>£{4000*WHOLESALE_P_KWH:,.0f}</td><td>£{5000*WHOLESALE_P_KWH:,.0f}</td><td>£{6000*WHOLESALE_P_KWH:,.0f}</td></tr>
<tr><td>TNUoS — transmission</td><td>0.87p/kWh + 3.59p/day</td><td>£{4000*0.0087+365*0.0359:,.0f}</td><td>£{5000*0.0087+365*0.0359:,.0f}</td><td>£{6000*0.0087+365*0.0359:,.0f}</td></tr>
<tr><td>DUoS — distribution</td><td>3.63p/kWh + 7.31p/day</td><td>£{4000*0.0363+365*0.0731:,.0f}</td><td>£{5000*0.0363+365*0.0731:,.0f}</td><td>£{6000*0.0363+365*0.0731:,.0f}</td></tr>
<tr><td>BSUoS — balancing</td><td>0.75p/kWh</td><td>£{4000*0.0075:,.0f}</td><td>£{5000*0.0075:,.0f}</td><td>£{6000*0.0075:,.0f}</td></tr>
<tr><td>Smart metering</td><td>0.54p/kWh + 3.56p/day</td><td>£{4000*0.0054+365*0.0356:,.0f}</td><td>£{5000*0.0054+365*0.0356:,.0f}</td><td>£{6000*0.0054+365*0.0356:,.0f}</td></tr>
<tr><td>Supplier opex</td><td>1.37p/kWh + 13.96p/day - £10</td><td>£{4000*0.0137+365*0.1396-10:,.0f}</td><td>£{5000*0.0137+365*0.1396-10:,.0f}</td><td>£{6000*0.0137+365*0.1396-10:,.0f}</td></tr>
<tr><td>Bad debt provision</td><td>0.41p/kWh</td><td>£{4000*0.0041:,.0f}</td><td>£{5000*0.0041:,.0f}</td><td>£{6000*0.0041:,.0f}</td></tr>
<tr><td>Grid levy (policy)</td><td>Fixed</td><td>£{GRID_LEVY:,.0f}</td><td>£{GRID_LEVY:,.0f}</td><td>£{GRID_LEVY:,.0f}</td></tr>
<tr><td>Customer management</td><td>Fixed</td><td>£{CUST_MGMT:,.0f}</td><td>£{CUST_MGMT:,.0f}</td><td>£{CUST_MGMT:,.0f}</td></tr>
<tr><td>Staff cost</td><td>Fixed</td><td>£{STAFF_COST:,.0f}</td><td>£{STAFF_COST:,.0f}</td><td>£{STAFF_COST:,.0f}</td></tr>
<tr><td>CAC (year 1 cashflow only)</td><td>One-off</td><td>£{CAC:,.0f}</td><td>£{CAC:,.0f}</td><td>£{CAC:,.0f}</td></tr>
</tbody>
</table>
<p style="font-size:12px; color:#666;">Avoided network charges on battery-served kWh (TNUoS + DUoS + BSUoS = 5.25p/kWh) are subtracted from opex.</p>

<h3>Tariff Scenarios</h3>
<table class="data-table">
<thead><tr><th>Scenario</th><th>Unit rate</th><th>Monthly fee</th><th>Bill at 4,000</th><th>Bill at 5,000</th><th>Bill at 6,000</th><th>Target saving</th></tr></thead>
<tbody>
{tariff_rows}
<tr style="background:#f0f0f0"><td><em>Competitor reference</em></td><td>{COMP_PKW}p/kWh</td><td>{COMP_PD}p/day</td><td>£{comp4:,.0f}</td><td>£{comp5:,.0f}</td><td>£{comp6:,.0f}</td><td>--</td></tr>
</tbody>
</table>

<h3>Spread Compression Scenarios</h3>
<table class="data-table">
<thead><tr><th>Scenario</th><th>alpha (BESS compression)</th><th>beta (renewable widening)</th><th>Yr 1 scale</th><th>Yr 6 scale</th><th>Yr 12 scale</th></tr></thead>
<tbody>
{spread_table}
</tbody>
</table>
<p style="font-size:12px; color:#666;">
    Formula: Spread = Floor + (S0 - Floor) x (C0/C)^alpha x (R/R0)^beta<br>
    Floor: {SPREAD_FLOOR} GBP/MWh | 2025 baseline spread: {BASELINE_SPREAD} GBP/MWh |
    Capacity data: data/inputs/capacity_forecasts.csv<br>
    Battery revenue (arb + FR + BM) in years 2+ is scaled by each year's factor. CM and avoided network are unscaled.
</p>
</div>

<div class="section">
<h2>Assumptions &amp; Limitations</h2>

<div class="assumption">
<strong>1. Perfect foresight on prices.</strong>
The MILP optimises each day's charge/discharge schedule using the full day's actual prices.
A real dispatch algorithm would achieve a lower spread, potentially 10-30% below this estimate.
</div>

<div class="assumption">
<strong>2. Battery degradation modelled at {DEGRADATION_PA*100:.0f}%/yr.</strong>
Capacity declines {DEGRADATION_PA*100:.0f}% per year (compound). By year 10, retention is
{degradation_factor(11)*100:.1f}%. All battery revenue (arbitrage, FR, CM, BM, avoided network charges)
is scaled by the degradation factor each year, in addition to spread compression.
</div>

<div class="assumption">
<strong>3. Static customer tariff.</strong>
Customer revenue is fixed at the modelled p/kWh + monthly fee for the entire period. In practice tariffs
are reviewed annually against changing wholesale prices, network charges, and competitor rates.
</div>

<div class="assumption">
<strong>4. Grid services assumed contractually secured.</strong>
FR and CM revenues are modelled as certain annual income. In practice both require competitive tender wins.
CM payments are subject to interconnector and demand-side response competition. FR rate (£{FR_PER_KW}/kW/yr)
assumes Dynamic Containment or equivalent; actual DC rates have fallen to £1-5/MW/h in recent clearing.
</div>

<div class="assumption">
<strong>5. BM estimated, not modelled.</strong>
Balancing Mechanism revenue is a simple {BM_WIN_RATE*100:.0f}% x {BM_UPLIFT*100:.0f}% uplift on arbitrage net,
not an explicit gate-closure dispatch model. Actual BM participation is volume-based and highly variable.
</div>

<div class="assumption">
<strong>6. Fixed equal-principal loan repayment.</strong>
The model uses equal annual principal instalments (not annuity/amortising). Real facilities typically use
equal total payments, giving higher early payments and lower later ones — slightly different cashflow profiles.
</div>

<div class="assumption">
<strong>7. Interest rate held constant at {LOAN_RATE*100:.0f}%.</strong>
Reasonable for a fixed-rate facility but does not capture floating-rate risk or refinancing at maturity.
</div>

<div class="assumption">
<strong>8. Corporation tax simplified.</strong>
Tax at {CORP_TAX*100:.0f}% on standalone EBT each year. Group relief, prior-year losses, R&amp;D credits,
Full Expensing capital allowances, and ring-fencing are all excluded. Effective tax rate likely lower
in early years due to capital allowances.
</div>

<div class="assumption">
<strong>9. No working capital.</strong>
The cashflow model omits receivables, payables, and cash reserves. A real business would require operating
capital, increasing the effective funding need above the battery cost alone.
</div>

<div class="assumption">
<strong>10. Single customer, single battery.</strong>
All results are per-customer. Portfolio effects — load diversification, operational leverage, shared grid
service contracts — are not captured and would improve unit economics at scale.
</div>

<div class="assumption">
<strong>11. EUR/GBP fixed at {EUR_TO_GBP}.</strong>
GBP appreciation would reduce both supply cost and arbitrage revenue; depreciation would increase both.
Net effect uncertain depending on pass-through arrangements.
</div>

<div class="assumption">
<strong>12. UK use profile applied.</strong>
Hourly consumption shape from use_profiles.csv (UK). Scaled proportionally to each consumption level.
Different customer types (heat pumps, EVs) would have different peak patterns affecting avoided network charges.
</div>

<div class="assumption">
<strong>13. Spread compression is market-level only.</strong>
The compression formula models how aggregate BESS build-out reduces wholesale spreads. It does not capture:
individual battery positioning, intraday price volatility changes, or potential new revenue streams (e.g.
local flexibility markets, EV smart charging coordination).
</div>

<div class="assumption">
<strong>14. MILP allows multiple intraday cycles.</strong>
The optimiser can charge-discharge more than once per day if profitable (avg ~1.7 cycles/day observed).
This is physically possible but increases wear; real dispatch may be more conservative to preserve battery life.
</div>

<div class="assumption">
<strong>15. CM stress events assumed rare.</strong>
Capacity Market revenue assumes the battery is usually available for other activities. If CM delivery
obligations conflict with arbitrage/FR, actual revenue would be lower.
</div>
</div>

<div class="section">
<h2>Summary Dashboard (5-yr loan)</h2>
<div id="summary-dashboard"><p>Loading...</p></div>
</div>

{detail_sections}

<div id="breakeven-section"></div>

<div class="section" style="text-align: center; color: #999; font-size: 12px;">
Generated by VLP Battery Electricity Company Model  |  UK wholesale prices Jul 2024 - Jun 2025
</div>

<script>
{js_code}
</script>
</body>
</html>"""

    out_path.write_text(html)
    print(f"  HTML report saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 90)
    print(f"  VLP Battery — Aggressive Offering  (Battery £{BAT_COST:,.0f})")
    print(f"  Consumer-first allocation | MILP optimisation | Spread compression")
    print("=" * 90)

    # ── Step 1: compute battery revenue via MILP ──
    # Battery revenue is the SAME for all consumption levels (VLP exports at
    # wholesale). We run once with the middle consumption for the demand breakdown,
    # then derive consumer-served splits for the other levels cheaply.
    print(f"\n  Running MILP on full battery ({BAT_KWH} kWh / {BAT_KW} kW) ...")
    mid_kwh = CONSUMPTION_LEVELS[len(CONSUMPTION_LEVELS) // 2]
    bi_base = compute_battery_revenue(mid_kwh)
    print(f"    Arb: £{bi_base['arb_net']:,.0f}  FR: £{bi_base['fr_income']:,.0f}  "
          f"CM: £{bi_base['cm_income']:,.0f}  BM: £{bi_base['bm_income']:,.0f}")
    print(f"    Total battery revenue: £{bi_base['total_battery_rev']:,.0f}/yr  "
          f"({bi_base['days_used']} days)")

    # For consumer-served breakdown at other consumption levels, scale
    # proportionally from the middle run (avoids re-running MILP).
    AVOIDED_NETWORK_P_KWH = 0.0087 + 0.0363 + 0.0075  # £/kWh
    battery_info = {}
    for ckwh in CONSUMPTION_LEVELS:
        bi = dict(bi_base)  # copy — trading revenue is identical
        # Scale consumer-served/exported proportionally to demand
        ratio = ckwh / mid_kwh
        bi["consumer_served_kwh"] = bi_base["consumer_served_kwh"] * ratio
        bi["exported_kwh"] = max(0, bi_base["exported_kwh"]
                                 + bi_base["consumer_served_kwh"] * (1 - ratio))
        # Recalculate avoided network charges for this consumption level
        bi["avoided_network"] = bi["consumer_served_kwh"] * AVOIDED_NETWORK_P_KWH
        bi["total_battery_rev"] = bi["arb_net"] + bi["grid_svcs"] + bi["avoided_network"]
        bi["annual_kwh"] = ckwh
        bi["daily_consumption"] = ckwh / 365
        battery_info[ckwh] = bi
        print(f"    {ckwh:,} kWh/yr: served {bi['consumer_served_kwh']:.0f} kWh "
              f"(avoided network £{bi['avoided_network']:.0f}), "
              f"exported {bi['exported_kwh']:.0f} kWh, "
              f"total battery rev £{bi['total_battery_rev']:.0f}")

    # ── Step 2: compute scaling factors for each spread scenario ──
    all_scaling = {}
    for sp_name in SPREAD_SCENARIOS:
        all_scaling[sp_name] = get_scaling_factors(sp_name)
        yr1 = all_scaling[sp_name][1]
        yr12 = all_scaling[sp_name][MODEL_YEARS]
        print(f"\n  Spread {sp_name}: yr1 scale={yr1:.3f}  yr{MODEL_YEARS} scale={yr12:.3f}")

    # ── Step 3: run all combinations ──
    all_results = {"ue": {}, "pl": {}, "cf": {}, "loan": {}}

    for tk, sc in TARIFF_SCENARIOS.items():
        print(f"\n  ── Tariff {tk}: {sc['pkwh']}p/kWh + £{sc['pmo']}/mo ──")

        for sp_name in SPREAD_SCENARIOS:
            sf = all_scaling[sp_name]

            for ckwh in CONSUMPTION_LEVELS:
                bi = battery_info[ckwh]
                ue = unit_economics(sc["pkwh"], sc["pmo"], ckwh, bi)

                for ly in LOAN_YEARS:
                    key = (tk, sp_name, ckwh, ly)
                    all_results["ue"][key]   = ue
                    all_results["pl"][key]   = build_pl(ue, ly, sf)
                    all_results["cf"][key]   = build_cashflow(ue, ly, sf)
                    all_results["loan"][key] = build_loan_schedule(ly)

        # Print summary for base spread, 5-yr loan
        for ckwh in CONSUMPTION_LEVELS:
            key = (tk, "Base", ckwh, 5)
            ue = all_results["ue"][key]
            cf = all_results["cf"][key]
            cum = cf["Cumulative Cash (£)"].iloc[-1]
            print(f"    {ckwh:,} kWh: EBIT £{ue['ebit']:,.0f}  "
                  f"| 12yr cum (Base,5yr) £{cum:,.0f}")

    # ── Step 4: save CSVs ──
    print("\n  Saving CSVs...")
    for tk in TARIFF_SCENARIOS:
        for sp_name in SPREAD_SCENARIOS:
            pl_frames = []
            cf_frames = []
            for ckwh in CONSUMPTION_LEVELS:
                for ly in LOAN_YEARS:
                    key = (tk, sp_name, ckwh, ly)
                    pl = all_results["pl"][key].copy()
                    cf = all_results["cf"][key].copy()
                    pl["Consumption kWh"] = ckwh
                    cf["Consumption kWh"] = ckwh
                    pl_frames.append(pl)
                    cf_frames.append(cf)

            pd.concat(pl_frames, ignore_index=True).to_csv(
                OUT_DIR / f"pl_{tk}_{sp_name}.csv", index=False)
            pd.concat(cf_frames, ignore_index=True).to_csv(
                OUT_DIR / f"cashflow_{tk}_{sp_name}.csv", index=False)

        # Loan schedules are spread-independent
        loan_frames = [all_results["loan"][(tk, "Base", CONSUMPTION_LEVELS[0], ly)]
                       for ly in LOAN_YEARS]
        pd.concat(loan_frames, ignore_index=True).to_csv(
            OUT_DIR / f"loan_schedule_{tk}.csv", index=False)

    print("    CSVs saved.")

    # ── Step 5: charts (base spread, 5000 kWh as default) ──
    print("  Generating charts...")
    default_cons = 5000
    for tk, sc in TARIFF_SCENARIOS.items():
        plot_scenario_chart(tk, sc, all_results, "Base", default_cons,
                            OUT_DIR / f"chart_{tk}.png")
        print(f"    chart_{tk}.png")

    plot_summary_chart(all_results, "Base", default_cons,
                       OUT_DIR / "chart_summary.png")
    print("    chart_summary.png")

    # ── Step 6: break-even tariff analysis ──
    print("  Computing break-even tariffs...")
    base_sf = all_scaling["Base"]
    breakeven_data = {}
    for loan_yr in [5, 7, 10]:
        for ckwh in CONSUMPTION_LEVELS:
            bi = battery_info[ckwh]
            ue = unit_economics(20, 8, ckwh, bi)  # tariff doesn't matter, we solve for bill
            be = find_breakeven_tariffs(
                ue["battery_rev"], ue["supply_cost"], ue["opex"],
                loan_yr, base_sf)
            breakeven_data[(loan_yr, ckwh)] = be
            print(f"    {loan_yr}-yr loan, {ckwh:,} kWh: min bill £{be['min_bill']:,.0f}/yr")

    # ── Step 7: HTML report ──
    print("  Generating HTML report...")
    build_html_report(all_results, battery_info, breakeven_data, OUT_DIR / "report.html")

    print()
    print(f"  All files written to: {OUT_DIR}")
    print("=" * 90)


if __name__ == "__main__":
    main()
