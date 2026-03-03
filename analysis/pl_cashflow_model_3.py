"""
Three-Statement Financial Model — Version 3
============================================
Changes from v2:
  - Battery: 15.36 kWh / 6 kW inverter (was 10kWh/5kW)
  - Charging time modelled explicitly: 2.70h charge / 2.43h discharge
    Arbitrage uses actual UK hourly wholesale prices (Jul 2024–Jun 2025),
    selecting the cheapest 2.70 hours to charge and most expensive 2.43 hours
    to discharge — properly constrained by inverter power limit.
  - Full cost stack based on Ofgem Q1 2025 methodology:
      Wholesale supply, TNUoS, DUoS, BSUoS, smart metering,
      supplier opex, bad debt, policy levy, customer management.
  - Battery cost recalculated: 15.36kWh × £220/kWh + £400 install = £3,779
  - Grid services scaled to 6kW: FR £204/yr, CM £50.26/yr (de-rated 20.94%)

Outputs (new files, no overwrites):
  results/pl_cashflow_full_report_3.txt
  results/pl_cashflow_base_case_3.csv
  results/pl_cashflow_portfolio_3.csv
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# ─────────────────────────────────────────────────────────
# BATTERY HARDWARE
# ─────────────────────────────────────────────────────────
BAT_KWH         = 15.36          # kWh capacity
BAT_KW          = 6.0            # kW inverter power
EFF_RT          = 0.90           # round-trip efficiency
ETA             = EFF_RT ** 0.5  # one-way efficiency ≈ 0.9487

# Charging / discharging time at full inverter power
GRID_DRAW_KWH   = BAT_KWH / ETA          # kWh drawn from grid for a full charge ≈ 16.19
DISCHARGE_KWH   = BAT_KWH * ETA          # kWh delivered to grid at discharge ≈ 14.57
T_CHARGE        = GRID_DRAW_KWH / BAT_KW  # hours to fully charge ≈ 2.699
T_DISCHARGE     = DISCHARGE_KWH / BAT_KW  # hours to fully discharge ≈ 2.429
N_FC            = int(T_CHARGE)           # full charge hours = 2
FRAC_C          = T_CHARGE - N_FC         # partial 3rd hour ≈ 0.699
N_FD            = int(T_DISCHARGE)        # full discharge hours = 2
FRAC_D          = T_DISCHARGE - N_FD      # partial 3rd hour ≈ 0.429

# Cost (from model_input.csv formula: £220/kWh + £400 install)
BAT_COST_GBP    = BAT_KWH * 220 + 400    # £3,779.20
BAT_REMOVAL_GBP = 400
BAT_LIFE_YRS    = 10
DEPRECIATION    = BAT_COST_GBP / BAT_LIFE_YRS   # £377.92/yr straight-line
LOAN_RATE       = 0.07
CORP_TAX        = 0.25

# ─────────────────────────────────────────────────────────
# GRID SERVICES  (6 kW battery, de-rating for 2-hr class)
# ─────────────────────────────────────────────────────────
FR_PER_KW       = 34             # £/kW/yr frequency response
CM_PER_KW       = 40             # £/kW/yr capacity market
CM_DERATING     = 0.2094         # 2-hour battery (15.36/6 = 2.56h → <3h → 2hr class)
BM_WIN_RATE     = 0.10
BM_UPLIFT       = 0.05

FR_GBP          = BAT_KW * FR_PER_KW                   # £204.00
CM_GBP          = BAT_KW * CM_PER_KW * CM_DERATING     # £50.26

# ─────────────────────────────────────────────────────────
# ARBITRAGE — computed from hourly wholesale prices
# ─────────────────────────────────────────────────────────
EUR_TO_GBP      = 0.85

def compute_arbitrage():
    """
    For each day Jul 2024–Jun 2025, find cheapest T_CHARGE hours to buy and
    most expensive T_DISCHARGE hours to sell, respecting 6kW inverter limit.
    Returns (buy_cost_annual, sell_rev_annual, net_annual).
    """
    price_file = PROJECT_ROOT / "data" / "prices" / "United Kingdom.csv"
    df = pd.read_csv(price_file)
    df["dt"]   = pd.to_datetime(df["Datetime (UTC)"])
    df         = df[(df["dt"] >= "2024-07-01") & (df["dt"] < "2025-07-01")].copy()
    df["date"] = df["dt"].dt.date
    df["p_gbp_kwh"] = df["Price (EUR/MWhe)"] * EUR_TO_GBP / 1000

    buy_total = sell_total = 0.0
    for _, grp in df.groupby("date"):
        p = np.sort(grp["p_gbp_kwh"].values)
        if len(p) < 24:
            continue
        p_desc = p[::-1]
        # Buy: cheapest N_FC full hours + FRAC_C of next
        buy_cost  = BAT_KW * (p[:N_FC].sum()      + FRAC_C * p[N_FC])
        # Sell: dearest N_FD full hours + FRAC_D of next
        sell_rev  = BAT_KW * (p_desc[:N_FD].sum() + FRAC_D * p_desc[N_FD])
        buy_total  += buy_cost
        sell_total += sell_rev

    return buy_total, sell_total, sell_total - buy_total

ARB_BUY, ARB_SELL, ARB_NET = compute_arbitrage()
BM_GBP    = ARB_NET * BM_WIN_RATE * BM_UPLIFT
GRID_SVCS = FR_GBP + CM_GBP + BM_GBP       # £255.92

# ─────────────────────────────────────────────────────────
# ELECTRICITY SUPPLY COST (wholesale average, customer only)
# ─────────────────────────────────────────────────────────
ANNUAL_KWH      = 3604.4          # kWh/yr from UK use profile
WHOLESALE_EUR_MWH = 99.84         # UK day-ahead avg Jul 2024–Jun 2025
WHOLESALE_P_KWH   = WHOLESALE_EUR_MWH * EUR_TO_GBP / 1000   # 8.486p/kWh
SUPPLY_COST     = ANNUAL_KWH * WHOLESALE_P_KWH              # £305.88

# ─────────────────────────────────────────────────────────
# FULL COST STACK  (Ofgem Q1 2025 methodology, per customer/yr)
# ─────────────────────────────────────────────────────────
DAYS = 365
# Network
TNUOS   = ANNUAL_KWH * 0.0087 + DAYS * 0.0359   # £44.46  transmission
DUOS    = ANNUAL_KWH * 0.0363 + DAYS * 0.0731   # £157.52 distribution
BSUOS   = ANNUAL_KWH * 0.0075                   # £27.03  balancing
# Obligations (policy levies already covered by existing grid_levy line below)
SMART_METER  = ANNUAL_KWH * 0.0054 + DAYS * 0.0356  # £32.46
# Supplier opex: Ofgem allows £100.33; subtract the £10 customer-mgmt already in model
SUPPLIER_OPEX = ANNUAL_KWH * 0.0137 + DAYS * 0.1396 - 10   # £90.33
BAD_DEBT      = ANNUAL_KWH * 0.0041                         # £14.78

GRID_LEVY     = 95.0    # policy levies — already in model_input.csv
CUST_MGMT     = 10.0    # CRM — already in model_input.csv

# Total cost (all lines)
TOTAL_COST = (SUPPLY_COST + GRID_LEVY + CUST_MGMT
              + TNUOS + DUOS + BSUOS + SMART_METER + SUPPLIER_OPEX + BAD_DEBT)

# ─────────────────────────────────────────────────────────
# COMPETITOR & PRICING OPTIONS
# ─────────────────────────────────────────────────────────
COMP_PKW  = 25      # p/kWh
COMP_PD   = 50      # p/day
COMP_BILL = ANNUAL_KWH * COMP_PKW / 100 + DAYS * COMP_PD / 100   # £1,083.60

# (p_per_kwh, £_per_month, label)
PRICING_OPTIONS = [
    (19,  8, "19p/kWh + £8/mo"),
    (20,  8, "20p/kWh + £8/mo"),
    (21,  8, "21p/kWh + £8/mo"),
    (22, 10, "22p/kWh + £10/mo  ★ base case"),
    (23, 10, "23p/kWh + £10/mo"),
    (24, 10, "24p/kWh + £10/mo"),
]
BASE_P, BASE_M = 22, 10


# ─────────────────────────────────────────────────────────
# UNIT ECONOMICS
# ─────────────────────────────────────────────────────────

def unit_economics(pkwh, pmo):
    elec_rev  = ANNUAL_KWH * pkwh / 100
    svc_fee   = pmo * 12
    cust_bill = elec_rev + svc_fee
    total_rev = elec_rev + svc_fee + ARB_NET + GRID_SVCS
    gross     = total_rev - SUPPLY_COST
    ebitda    = gross - (GRID_LEVY + CUST_MGMT + TNUOS + DUOS + BSUOS
                         + SMART_METER + SUPPLIER_OPEX + BAD_DEBT)
    ebit      = ebitda - DEPRECIATION
    saving    = COMP_BILL - cust_bill
    saving_pct = saving / COMP_BILL * 100
    return dict(elec_rev=elec_rev, svc_fee=svc_fee, cust_bill=cust_bill,
                total_rev=total_rev, gross=gross, ebitda=ebitda, ebit=ebit,
                saving=saving, saving_pct=saving_pct)


# ─────────────────────────────────────────────────────────
# THREE-STATEMENT MODEL — single customer
# ─────────────────────────────────────────────────────────

def three_statements(pkwh, pmo, years=10):
    ue       = unit_economics(pkwh, pmo)
    ebit     = ue["ebit"]
    ebitda   = ue["ebitda"]
    loan_bal = float(BAT_COST_GBP)
    cum_cash = 0.0
    cum_earn = 0.0
    rows     = []

    for yr in range(1, years + 1):
        # Income statement
        interest  = loan_bal * LOAN_RATE
        ebt       = ebit - interest
        tax       = max(0.0, ebt * CORP_TAX)
        net_inc   = ebt - tax

        # Cash flow (indirect): OCF = NI + D&A; interest already in NI
        ocf       = net_inc + DEPRECIATION
        principal = min(ocf, loan_bal)
        free_cash = ocf - principal
        cum_cash += free_cash
        loan_bal -= principal

        # Balance sheet
        bat_nbv   = max(0.0, BAT_COST_GBP - DEPRECIATION * yr)
        cum_earn += net_inc
        total_assets = cum_cash + bat_nbv
        bs_check  = total_assets - loan_bal - cum_earn   # should be ~0

        rows.append({
            # IS
            "Year":          yr,
            "Revenue":       ue["total_rev"],
            "COGS":          SUPPLY_COST,
            "Gross Profit":  ue["gross"],
            "EBITDA":        ebitda,
            "Depreciation":  DEPRECIATION,
            "EBIT":          ebit,
            "Interest":      interest,
            "EBT":           ebt,
            "Tax (25%)":     tax,
            "Net Income":    net_inc,
            # CFS
            "Operating CF":  ocf,
            "Principal":     principal,
            "Financing CF":  -principal,
            "Free Cash":     free_cash,
            "Cum Cash":      cum_cash,
            # BS
            "Battery NBV":   bat_nbv,
            "Total Assets":  total_assets,
            "Loan Balance":  loan_bal,
            "Retained Earn": cum_earn,
            "BS Check":      bs_check,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
# PORTFOLIO — three-statement summary
# ─────────────────────────────────────────────────────────

def portfolio(pkwh, pmo, schedule, years=12):
    """schedule = [(year, new_customers), ...]"""
    ue         = unit_economics(pkwh, pmo)
    ebit_u     = ue["ebit"]
    ebitda_u   = ue["ebitda"]
    adds       = dict(schedule)
    total_c    = 0
    cohort_bal = {}   # {start_yr: remaining_balance}
    cum_cash   = 0.0
    cum_earn   = 0.0
    rows       = []

    for yr in range(1, years + 1):
        new_c     = adds.get(yr, 0)
        total_c  += new_c
        if new_c > 0:
            cohort_bal[yr] = new_c * BAT_COST_GBP

        t_ebitda  = total_c * ebitda_u
        t_ebit    = total_c * ebit_u
        t_dep     = total_c * DEPRECIATION
        t_int     = sum(b * LOAN_RATE for b in cohort_bal.values())
        t_ebt     = t_ebit - t_int
        t_tax     = max(0.0, t_ebt * CORP_TAX)
        t_ni      = t_ebt - t_tax
        t_ocf     = t_ni + t_dep

        # Repay oldest cohorts first from OCF
        remaining     = t_ocf
        t_principal   = 0.0
        for cy in sorted(cohort_bal):
            if remaining <= 0:
                break
            pay = min(remaining, cohort_bal[cy])
            cohort_bal[cy] -= pay
            t_principal    += pay
            remaining      -= pay
        cohort_bal = {k: v for k, v in cohort_bal.items() if v > 0.01}

        free_cash  = t_ocf - t_principal
        cum_cash  += free_cash
        cum_earn  += t_ni
        loan_total = sum(cohort_bal.values())

        rows.append({
            "Year":       yr,
            "New Custs":  new_c,
            "Total Custs":total_c,
            "EBITDA":     t_ebitda,
            "EBIT":       t_ebit,
            "Interest":   t_int,
            "EBT":        t_ebt,
            "Tax":        t_tax,
            "Net Income": t_ni,
            "Op CF":      t_ocf,
            "Principal":  t_principal,
            "Free Cash":  free_cash,
            "Cum Cash":   cum_cash,
            "Loan Bal":   loan_total,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
# PRINT FUNCTIONS
# ─────────────────────────────────────────────────────────
W = 92

def hdr(s):
    print(); print("=" * W); print(s); print("=" * W)

def ln(label, val, indent=2, neg=False):
    sign = "-" if neg else " "
    print(f"{'  '*indent}{label:<52}{sign}£{abs(val):>8,.2f}")

def sep(indent=2):
    print("  " * indent + "─" * 60)


def print_assumptions():
    hdr("VLP BATTERY — THREE-STATEMENT MODEL v3  (UK, Jul 2024–Jun 2025 wholesale)")
    print()
    print("BATTERY SPECS")
    print(f"  Capacity   : {BAT_KWH} kWh  |  Inverter: {BAT_KW:.0f} kW  |  Duration: {BAT_KWH/BAT_KW:.2f} hrs")
    print(f"  Cost       : {BAT_KWH} × £220/kWh + £400 install = £{BAT_COST_GBP:,.2f}")
    print(f"  Loan       : {LOAN_RATE*100:.0f}% p.a.  |  Life: {BAT_LIFE_YRS} yrs  |  Dep: £{DEPRECIATION:.2f}/yr")
    print()
    print("CHARGING TIME MODEL  (inverter-constrained, hourly price data)")
    print(f"  Grid draw to charge : {GRID_DRAW_KWH:.3f} kWh  →  {T_CHARGE:.3f} hrs at {BAT_KW:.0f} kW")
    print(f"  Energy discharged   : {DISCHARGE_KWH:.3f} kWh  →  {T_DISCHARGE:.3f} hrs at {BAT_KW:.0f} kW")
    print(f"  Strategy            : buy cheapest {T_CHARGE:.2f} hrs/day, sell dearest {T_DISCHARGE:.2f} hrs/day")
    print(f"  (uses 2 full + {FRAC_C:.3f} partial hour charging; 2 full + {FRAC_D:.3f} partial discharging)")
    print()
    print("BATTERY MARKET REVENUES  (direct wholesale, Jul 2024–Jun 2025)")
    print(f"  Charge cost (buy)  : £{ARB_BUY:,.2f}/yr  at daily cheapest hours")
    print(f"  Sell revenue       : £{ARB_SELL:,.2f}/yr  at daily dearest hours")
    print(f"  Net arbitrage      : £{ARB_NET:,.2f}/yr")
    print()
    print("GRID SERVICES  (6 kW battery)")
    print(f"  Frequency response : {BAT_KW:.0f}kW × £{FR_PER_KW}/kW            = £{FR_GBP:,.2f}/yr")
    print(f"  Capacity market    : {BAT_KW:.0f}kW × £{CM_PER_KW}/kW × {CM_DERATING*100:.2f}% de-rate = £{CM_GBP:,.2f}/yr")
    print(f"  Balancing mech     : arb £{ARB_NET:.2f} × {BM_WIN_RATE*100:.0f}% × {BM_UPLIFT*100:.0f}%     = £{BM_GBP:,.2f}/yr")
    print(f"  Total              :                               £{GRID_SVCS:,.2f}/yr")
    print()
    print("FULL COST STACK  (Ofgem Q1 2025 methodology)")
    print(f"  {'Cost line':<40} {'£/yr':>8}")
    print(f"  {'─'*50}")
    items = [
        ("Wholesale electricity supply",  SUPPLY_COST),
        ("Policy levy (grid_levy)",        GRID_LEVY),
        ("Customer management",            CUST_MGMT),
        ("TNUoS (transmission network)",   TNUOS),
        ("DUoS  (distribution network)",   DUOS),
        ("BSUoS (balancing services)",     BSUOS),
        ("Smart metering obligation",      SMART_METER),
        ("Supplier opex (billing/compl.)", SUPPLIER_OPEX),
        ("Bad debt provision (~0.4%)",     BAD_DEBT),
    ]
    for name, val in items:
        print(f"  {name:<40} £{val:>8.2f}")
    print(f"  {'─'*50}")
    print(f"  {'TOTAL COSTS':<40} £{TOTAL_COST:>8.2f}")
    print()
    print(f"  Competitor: {COMP_PKW}p/kWh + {COMP_PD}p/day = £{COMP_BILL:,.2f}/yr  ({ANNUAL_KWH:.0f} kWh)")


def print_sensitivity():
    print()
    print("─" * W)
    print("PRICING SENSITIVITY")
    print("─" * W)
    print(f"  {'Tariff':<26} {'Cust Bill':>10} {'Saving £':>9} {'Saving%':>8} "
          f"{'EBITDA':>9} {'EBIT':>8} {'Yr1 NI':>8} {'Yr1 OCF':>9}")
    print("─" * W)
    for pkwh, pmo, label in PRICING_OPTIONS:
        ue = unit_economics(pkwh, pmo)
        yr1_int = BAT_COST_GBP * LOAN_RATE
        yr1_ebt = ue["ebit"] - yr1_int
        yr1_tax = max(0, yr1_ebt * CORP_TAX)
        yr1_ni  = yr1_ebt - yr1_tax
        yr1_ocf = yr1_ni + DEPRECIATION
        profitable = "✓" if yr1_ni > 0 else "~"  # ~ = cashflow +ve but NI negative
        print(f"  {label:<26} £{ue['cust_bill']:>8,.2f} "
              f" £{ue['saving']:>7,.2f}  {ue['saving_pct']:>6.1f}%"
              f" £{ue['ebitda']:>7,.2f} £{ue['ebit']:>6,.2f}"
              f" £{yr1_ni:>6,.0f}{profitable} £{yr1_ocf:>7,.0f}")
    print(f"\n  ✓ = profitable from year 1   ~ = cashflow positive but NI negative yr1")


def print_unit_economics(pkwh, pmo):
    ue = unit_economics(pkwh, pmo)
    hdr(f"ANNUAL UNIT ECONOMICS — {pkwh}p/kWh + £{pmo}/month  (base case)")
    print(f"  Customer bill: £{ue['cust_bill']:,.2f}/yr  vs competitor £{COMP_BILL:,.2f}/yr"
          f"  →  saves £{ue['saving']:,.2f}  ({ue['saving_pct']:.1f}% cheaper)")
    print()
    print("  ── INCOME STATEMENT ──")
    print("  REVENUE")
    ln(f"Electricity sales ({ANNUAL_KWH:.0f} kWh × {pkwh}p/kWh)", ue["elec_rev"])
    ln(f"Monthly service fee (£{pmo}/mo × 12)",                    ue["svc_fee"])
    ln(f"Battery arbitrage — net (direct wholesale market)",        ARB_NET)
    ln(f"  Sell revenue  ({BAT_KWH}kWh × eff × {T_DISCHARGE:.2f}h daily peak)",  ARB_SELL,  indent=3)
    ln(f"  Charge cost   ({BAT_KWH}kWh / eff × {T_CHARGE:.2f}h daily trough)",   ARB_BUY,   indent=3, neg=True)
    ln(f"Frequency response ({BAT_KW:.0f}kW × £{FR_PER_KW}/kW)",                 FR_GBP)
    ln(f"Capacity market    ({BAT_KW:.0f}kW × £{CM_PER_KW} × {CM_DERATING*100:.2f}% de-rate)", CM_GBP)
    ln(f"Balancing mechanism",                                       BM_GBP)
    sep()
    ln("Total Revenue",                                             ue["total_rev"])
    print()
    print("  COST OF REVENUE")
    ln(f"Wholesale electricity ({ANNUAL_KWH:.0f}kWh × {WHOLESALE_P_KWH*100:.2f}p/kWh)", SUPPLY_COST, neg=True)
    sep()
    ln("Gross Profit",                                              ue["gross"])
    print()
    print("  OPERATING EXPENSES")
    ln("Policy levy (grid_levy per customer)",                      GRID_LEVY,      neg=True)
    ln("Customer management (CRM)",                                 CUST_MGMT,      neg=True)
    ln("TNUoS — transmission network",                              TNUOS,          neg=True)
    ln("DUoS  — distribution network",                              DUOS,           neg=True)
    ln("BSUoS — balancing services",                                BSUOS,          neg=True)
    ln("Smart metering obligation",                                  SMART_METER,    neg=True)
    ln("Supplier opex (billing, compliance, hedging)",               SUPPLIER_OPEX,  neg=True)
    ln("Bad debt provision",                                         BAD_DEBT,       neg=True)
    sep()
    ln("EBITDA",                                                     ue["ebitda"])
    ln(f"Depreciation (£{BAT_COST_GBP:,.0f} / {BAT_LIFE_YRS}yr)",  DEPRECIATION,   neg=True)
    sep()
    ln("EBIT",                                                       ue["ebit"])
    yr1_int = BAT_COST_GBP * LOAN_RATE
    yr1_ebt = ue["ebit"] - yr1_int
    yr1_tax = max(0, yr1_ebt * CORP_TAX)
    ln(f"Interest — year 1  (7% × £{BAT_COST_GBP:,.0f})",          yr1_int,        neg=True)
    sep()
    ln("EBT — year 1",                                              yr1_ebt)
    ln(f"Tax 25%",                                                   yr1_tax,        neg=True)
    sep()
    ln("Net Income — year 1",                                        yr1_ebt - yr1_tax)


def print_three_statements(df, pkwh, pmo):
    hdr(f"THREE-STATEMENT FORECAST — {pkwh}p/kWh + £{pmo}/mo  "
        f"(battery 100% loan-funded, max repayment each year)")

    print("\n  ── INCOME STATEMENT (£, single customer) ──")
    is_cols = ["Year", "Revenue", "COGS", "Gross Profit", "EBITDA",
               "Depreciation", "EBIT", "Interest", "EBT", "Tax (25%)", "Net Income"]
    print(df[is_cols].to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    print("\n  ── CASH FLOW STATEMENT (£) ──")
    cf_cols = ["Year", "Operating CF", "Principal", "Financing CF", "Free Cash", "Cum Cash"]
    print(df[cf_cols].to_string(index=False, float_format=lambda x: f"{x:,.0f}"))
    print("  Note: interest paid included in Operating CF (indirect method)")

    print("\n  ── BALANCE SHEET (£) ──")
    bs_cols = ["Year", "Cum Cash", "Battery NBV", "Total Assets",
               "Loan Balance", "Retained Earn", "BS Check"]
    print(df[bs_cols].to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    # Payoff stats
    paid_rows = df[df["Loan Balance"] < 1]
    payoff_yr  = paid_rows["Year"].min() if not paid_rows.empty else ">" + str(len(df))
    post_cf    = df[df["Loan Balance"] < 1]["Free Cash"].mean() if not paid_rows.empty else 0
    print()
    print(f"  Loan paid off            : Year {payoff_yr}")
    print(f"  Post-payoff free cashflow: £{post_cf:,.0f}/yr/customer")
    print(f"  Total interest paid      : £{df['Interest'].sum():,.0f}")
    print(f"  Total tax paid           : £{df['Tax (25%)'].sum():,.0f}")
    print(f"  10-yr cumulative cash    : £{df['Cum Cash'].iloc[-1]:,.0f}")


def print_portfolio(df, pkwh, pmo):
    hdr(f"PORTFOLIO FORECAST — {pkwh}p/kWh + £{pmo}/mo  "
        f"(illustrative 50→1,050 customers, years 1–5)")
    cols = ["Year", "New Custs", "Total Custs", "EBITDA", "EBIT", "Interest",
            "EBT", "Tax", "Net Income", "Op CF", "Free Cash", "Cum Cash", "Loan Bal"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    final_row   = df.iloc[-1]
    steady_rows = df[df["Loan Bal"] < 1]
    print()
    print(f"  At 1,050 customers (steady state):")
    print(f"    Annual EBITDA     : £{1050 * unit_economics(pkwh,pmo)['ebitda']:>10,.0f}")
    print(f"    Annual Net Income : £{1050 * unit_economics(pkwh,pmo)['ebit'] * (1-CORP_TAX):>10,.0f}  (approx, post-payoff)")
    print(f"    Annual Free CF    : £{1050 * (unit_economics(pkwh,pmo)['net_inc_post'] if 'net_inc_post' in unit_economics(pkwh,pmo) else unit_economics(pkwh,pmo)['ebitda']):>10,.0f}  (EBITDA, post-payoff)")
    if not steady_rows.empty:
        print(f"    Post-payoff OCF   : £{steady_rows['Free Cash'].iloc[0]:>10,.0f}/yr")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    print_assumptions()
    print_sensitivity()
    print_unit_economics(BASE_P, BASE_M)

    df_base = three_statements(BASE_P, BASE_M, years=10)
    print_three_statements(df_base, BASE_P, BASE_M)

    df_port = portfolio(
        BASE_P, BASE_M,
        schedule=[(1, 50), (2, 100), (3, 200), (4, 300), (5, 400)],
        years=12,
    )
    print_portfolio(df_port, BASE_P, BASE_M)

    # Save files
    base_csv = PROJECT_ROOT / "results" / "pl_cashflow_base_case_3.csv"
    port_csv = PROJECT_ROOT / "results" / "pl_cashflow_portfolio_3.csv"
    df_base.to_csv(base_csv, index=False)
    df_port.to_csv(port_csv, index=False)
    print(f"\n  Saved: {base_csv}")
    print(f"  Saved: {port_csv}")


if __name__ == "__main__":
    out_txt = PROJECT_ROOT / "results" / "pl_cashflow_full_report_3.txt"
    tee     = open(out_txt, "w")
    orig    = sys.stdout

    class Tee:
        def write(self, m): orig.write(m); tee.write(m)
        def flush(self):    orig.flush();  tee.flush()

    sys.stdout = Tee()
    main()
    sys.stdout = orig
    tee.close()
    print(f"\nFull report saved: {out_txt}")
