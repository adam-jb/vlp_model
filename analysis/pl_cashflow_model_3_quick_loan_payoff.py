"""
Quick Loan Payoff Comparison — Base Case
=========================================
Derived from pl_cashflow_model_3.py (15.36 kWh / 6 kW battery, Ofgem Q1 2025
cost stack, UK Jul 2024–Jun 2025 wholesale prices).

This model replaces the "pay as much as you can each year" approach with three
fixed equal-annual-principal loan schedules:
  - 3-year payoff  →  largest annual payment, longest cashflow-negative period
  - 4-year payoff
  - 5-year payoff  →  smallest annual payment, shortest cashflow-negative period

Tariff: 22p/kWh + £10/month  (base case, same as v3)

Free cashflow can go negative in early years when the fixed principal
repayment exceeds OCF.  The model highlights those periods explicitly.

Outputs (new files):
  results/pl_cashflow_qlp_base_case.csv       — year-by-year table (all 3 scenarios)
  results/pl_cashflow_qlp_full_report.txt     — full text report
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# ─────────────────────────────────────────────────────────
# BATTERY HARDWARE  (identical to model_3)
# ─────────────────────────────────────────────────────────
BAT_KWH         = 15.36
BAT_KW          = 6.0
EFF_RT          = 0.90
ETA             = EFF_RT ** 0.5

GRID_DRAW_KWH   = BAT_KWH / ETA
DISCHARGE_KWH   = BAT_KWH * ETA
T_CHARGE        = GRID_DRAW_KWH / BAT_KW
T_DISCHARGE     = DISCHARGE_KWH / BAT_KW
N_FC            = int(T_CHARGE)
FRAC_C          = T_CHARGE - N_FC
N_FD            = int(T_DISCHARGE)
FRAC_D          = T_DISCHARGE - N_FD

BAT_COST_GBP    = BAT_KWH * 220 + 400      # £3,779.20
BAT_REMOVAL_GBP = 400
BAT_LIFE_YRS    = 10
DEPRECIATION    = BAT_COST_GBP / BAT_LIFE_YRS
LOAN_RATE       = 0.07
CORP_TAX        = 0.25

# ─────────────────────────────────────────────────────────
# GRID SERVICES  (identical to model_3)
# ─────────────────────────────────────────────────────────
FR_PER_KW       = 34
CM_PER_KW       = 40
CM_DERATING     = 0.2094
BM_WIN_RATE     = 0.10
BM_UPLIFT       = 0.05

FR_GBP          = BAT_KW * FR_PER_KW
CM_GBP          = BAT_KW * CM_PER_KW * CM_DERATING

# ─────────────────────────────────────────────────────────
# ARBITRAGE  (identical to model_3)
# ─────────────────────────────────────────────────────────
EUR_TO_GBP      = 0.85

def compute_arbitrage():
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
        p_desc   = p[::-1]
        buy_cost = BAT_KW * (p[:N_FC].sum()      + FRAC_C * p[N_FC])
        sell_rev = BAT_KW * (p_desc[:N_FD].sum() + FRAC_D * p_desc[N_FD])
        buy_total  += buy_cost
        sell_total += sell_rev

    return buy_total, sell_total, sell_total - buy_total

ARB_BUY, ARB_SELL, ARB_NET = compute_arbitrage()
BM_GBP    = ARB_NET * BM_WIN_RATE * BM_UPLIFT
GRID_SVCS = FR_GBP + CM_GBP + BM_GBP

# ─────────────────────────────────────────────────────────
# COST STACK  (Ofgem Q1 2025, identical to model_3)
# ─────────────────────────────────────────────────────────
ANNUAL_KWH        = 3604.4
WHOLESALE_EUR_MWH = 99.84
WHOLESALE_P_KWH   = WHOLESALE_EUR_MWH * EUR_TO_GBP / 1000
SUPPLY_COST       = ANNUAL_KWH * WHOLESALE_P_KWH

DAYS = 365
TNUOS        = ANNUAL_KWH * 0.0087 + DAYS * 0.0359
DUOS         = ANNUAL_KWH * 0.0363 + DAYS * 0.0731
BSUOS        = ANNUAL_KWH * 0.0075
SMART_METER  = ANNUAL_KWH * 0.0054 + DAYS * 0.0356
SUPPLIER_OPEX = ANNUAL_KWH * 0.0137 + DAYS * 0.1396 - 10
BAD_DEBT      = ANNUAL_KWH * 0.0041
GRID_LEVY     = 95.0
CUST_MGMT     = 10.0

TOTAL_COST = (SUPPLY_COST + GRID_LEVY + CUST_MGMT
              + TNUOS + DUOS + BSUOS + SMART_METER + SUPPLIER_OPEX + BAD_DEBT)

# ─────────────────────────────────────────────────────────
# BASE CASE TARIFF
# ─────────────────────────────────────────────────────────
COMP_PKW   = 25
COMP_PD    = 50
COMP_BILL  = ANNUAL_KWH * COMP_PKW / 100 + DAYS * COMP_PD / 100
BASE_P, BASE_M = 22, 10

LOAN_SCENARIOS = [3, 4, 5]   # payoff years

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
# FIXED-TERM LOAN MODEL
# ─────────────────────────────────────────────────────────

def fixed_term_loan(pkwh, pmo, loan_years, model_years=10):
    """
    Models fixed equal-annual-principal repayment over loan_years.

    Principal payment per year = BAT_COST_GBP / loan_years  (fixed schedule).
    After loan is fully repaid, no further principal payments.

    Free Cash = OCF - Principal  (can be negative in early years).
    """
    ue         = unit_economics(pkwh, pmo)
    ebit       = ue["ebit"]
    annual_principal = BAT_COST_GBP / loan_years  # fixed equal instalments
    loan_bal   = float(BAT_COST_GBP)
    cum_cash   = 0.0
    cum_earn   = 0.0
    rows       = []

    for yr in range(1, model_years + 1):
        # Income statement
        interest  = loan_bal * LOAN_RATE
        ebt       = ebit - interest
        tax       = max(0.0, ebt * CORP_TAX)
        net_inc   = ebt - tax

        # OCF (indirect method)
        ocf = net_inc + DEPRECIATION

        # Principal: fixed schedule while loan remains
        if loan_bal > 0.01:
            principal = min(annual_principal, loan_bal)  # don't overpay in final yr
        else:
            principal = 0.0

        free_cash  = ocf - principal   # can be negative
        cum_cash  += free_cash
        loan_bal   = max(0.0, loan_bal - principal)

        # Balance sheet
        bat_nbv  = max(0.0, BAT_COST_GBP - DEPRECIATION * yr)
        cum_earn += net_inc
        total_assets = cum_cash + bat_nbv
        bs_check = total_assets - loan_bal - cum_earn

        rows.append({
            "Year":          yr,
            "Loan Yrs":      loan_years,
            # IS
            "Revenue":       ue["total_rev"],
            "COGS":          SUPPLY_COST,
            "Gross Profit":  ue["gross"],
            "EBITDA":        ue["ebitda"],
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
            "CF Negative":   free_cash < 0,
            # BS
            "Battery NBV":   bat_nbv,
            "Total Assets":  total_assets,
            "Loan Balance":  loan_bal,
            "Retained Earn": cum_earn,
            "BS Check":      bs_check,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────────────────────
W = 100

def hdr(s):
    print(); print("=" * W); print(s); print("=" * W)

def sub(s):
    print(); print("  " + s); print("  " + "─" * (W - 4))

def ln(label, val, indent=2, neg=False):
    sign = "-" if neg else " "
    print(f"{'  '*indent}{label:<52}{sign}£{abs(val):>9,.2f}")

def sep(indent=2):
    print("  " * indent + "─" * 62)


# ─────────────────────────────────────────────────────────
# PRINT FUNCTIONS
# ─────────────────────────────────────────────────────────

def print_assumptions(ue):
    hdr("VLP BATTERY — QUICK LOAN PAYOFF COMPARISON  (Base Case, UK Jul 2024–Jun 2025)")
    print()
    print("BATTERY SPECS")
    print(f"  Capacity   : {BAT_KWH} kWh  |  Inverter: {BAT_KW:.0f} kW  |  Duration: {BAT_KWH/BAT_KW:.2f} hrs")
    print(f"  Cost       : {BAT_KWH} × £220/kWh + £400 install = £{BAT_COST_GBP:,.2f}")
    print(f"  Loan rate  : {LOAN_RATE*100:.0f}% p.a.  |  Battery life: {BAT_LIFE_YRS} yrs  |  Dep: £{DEPRECIATION:.2f}/yr")
    print()
    print("LOAN SCENARIOS COMPARED")
    for ly in LOAN_SCENARIOS:
        ann_prin = BAT_COST_GBP / ly
        print(f"  {ly}-year payoff : £{ann_prin:,.2f}/yr principal  "
              f"(total interest ≈ £{_est_total_interest(ly):,.2f})")
    print()
    print("ARBITRAGE  (UK wholesale, Jul 2024–Jun 2025)")
    print(f"  Charge cost : £{ARB_BUY:,.2f}/yr   Sell revenue : £{ARB_SELL:,.2f}/yr")
    print(f"  Net arbitrage : £{ARB_NET:,.2f}/yr")
    print()
    print("UNIT ECONOMICS  (22p/kWh + £10/month)")
    print(f"  Annual revenue   : £{ue['total_rev']:,.2f}")
    print(f"  Annual EBITDA    : £{ue['ebitda']:,.2f}")
    print(f"  Annual EBIT      : £{ue['ebit']:,.2f}")
    print(f"  Customer bill    : £{ue['cust_bill']:,.2f}/yr  vs competitor £{COMP_BILL:,.2f}/yr")
    print(f"  Saving           : £{ue['saving']:,.2f}  ({ue['saving_pct']:.1f}% cheaper)")


def _est_total_interest(loan_years):
    """Quick estimate of total interest for a fixed-principal schedule."""
    bal   = BAT_COST_GBP
    ann_p = BAT_COST_GBP / loan_years
    total = 0.0
    for _ in range(loan_years):
        total += bal * LOAN_RATE
        bal   -= ann_p
    return total


def print_scenario(df, loan_years):
    sub(f"{loan_years}-YEAR LOAN PAYOFF  —  annual principal £{BAT_COST_GBP/loan_years:,.2f}")

    print()
    # IS columns
    is_cols = ["Year", "Revenue", "EBITDA", "EBIT", "Interest", "EBT", "Tax (25%)", "Net Income"]
    print("  Income Statement (£/yr, single customer):")
    print(df[is_cols].to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    print()
    cf_cols = ["Year", "Operating CF", "Principal", "Free Cash", "Cum Cash", "Loan Balance"]
    print("  Cash Flow & Balance Sheet (£):")
    # Build display df with marker for negative free cash
    disp = df[cf_cols + ["CF Negative"]].copy()
    disp["Free Cash"] = disp.apply(
        lambda r: f"  {'*** ' if r['CF Negative'] else '    '}{r['Free Cash']:,.0f}", axis=1
    )
    disp = disp.drop(columns=["CF Negative"])
    print(disp.to_string(index=False))
    print("  *** = negative free cashflow year (fixed principal > OCF)")

    # Summary stats
    neg_yrs  = df[df["CF Negative"]]["Year"].tolist()
    paid_off = df[df["Loan Balance"] < 1]["Year"].min() if (df["Loan Balance"] < 1).any() else f">{len(df)}"
    post_ocf = df[df["Loan Balance"] < 1]["Free Cash"].mean() if (df["Loan Balance"] < 1).any() else 0
    print()
    print(f"  Loan fully repaid    : Year {paid_off}")
    print(f"  Cashflow-negative yrs: {neg_yrs if neg_yrs else 'none'}")
    print(f"  Total interest paid  : £{df['Interest'].sum():,.0f}")
    print(f"  Total tax paid       : £{df['Tax (25%)'].sum():,.0f}")
    print(f"  10-yr cumulative cash: £{df['Cum Cash'].iloc[-1]:,.0f}")
    print(f"  Post-payoff free CF  : £{post_ocf:,.0f}/yr")


def print_side_by_side(dfs):
    """Print free cash and cumulative cash for all scenarios side by side."""
    hdr("SIDE-BY-SIDE COMPARISON  —  Free Cashflow & Cumulative Cash (£, single customer)")
    print()

    # Build merged table
    base_df = dfs[LOAN_SCENARIOS[0]][["Year"]].copy()
    for ly in LOAN_SCENARIOS:
        df = dfs[ly]
        base_df[f"OCF {ly}yr"]      = df["Operating CF"].values
        base_df[f"Principal {ly}yr"] = df["Principal"].values
        base_df[f"FreeCF {ly}yr"]   = df["Free Cash"].values
        base_df[f"CumCash {ly}yr"]  = df["Cum Cash"].values

    print(base_df.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    print()
    print("  CASHFLOW-NEGATIVE YEARS (fixed principal exceeds OCF):")
    for ly in LOAN_SCENARIOS:
        df    = dfs[ly]
        neg   = df[df["CF Negative"]]["Year"].tolist()
        msg   = str(neg) if neg else "none — OCF covers repayment throughout"
        print(f"    {ly}-year payoff: {msg}")

    print()
    print("  10-YEAR CUMULATIVE FREE CASH:")
    for ly in LOAN_SCENARIOS:
        df = dfs[ly]
        print(f"    {ly}-year payoff: £{df['Cum Cash'].iloc[-1]:,.0f}")

    print()
    print("  TOTAL INTEREST COST:")
    for ly in LOAN_SCENARIOS:
        df = dfs[ly]
        print(f"    {ly}-year payoff: £{df['Interest'].sum():,.0f}")

    print()
    print("  BREAK-EVEN (cum cash turns positive):")
    for ly in LOAN_SCENARIOS:
        df = dfs[ly]
        pos = df[df["Cum Cash"] > 0]["Year"].min()
        if pd.isna(pos):
            pos = f">{len(df)}"
        print(f"    {ly}-year payoff: Year {pos}")


def print_recommendation(dfs, ue):
    hdr("SUMMARY & RECOMMENDATION")
    print()
    print("  Key trade-off: shorter payoff = higher near-term cash drain but")
    print("  less total interest and faster route to free cash generation.")
    print()
    print(f"  {'Scenario':<16} {'Ann. Principal':>14} {'Total Interest':>14} "
          f"{'CF-neg yrs':>12} {'10yr Cum Cash':>14}")
    print("  " + "─" * 72)
    for ly in LOAN_SCENARIOS:
        df      = dfs[ly]
        ann_p   = BAT_COST_GBP / ly
        tot_int = df["Interest"].sum()
        neg_yrs = df[df["CF Negative"]]["Year"].tolist()
        cum10   = df["Cum Cash"].iloc[-1]
        neg_str = str(neg_yrs) if neg_yrs else "none"
        print(f"  {ly}-year payoff   £{ann_p:>12,.2f}  £{tot_int:>12,.2f}  "
              f"  {neg_str:<10}  £{cum10:>12,.0f}")

    print()
    print("  Note: all scenarios use same EBIT — pricing and battery economics unchanged.")
    print(f"  EBIT per customer per year: £{ue['ebit']:,.2f}")
    print(f"  Battery cost funded: £{BAT_COST_GBP:,.2f}")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    ue = unit_economics(BASE_P, BASE_M)

    print_assumptions(ue)

    dfs = {}
    for ly in LOAN_SCENARIOS:
        dfs[ly] = fixed_term_loan(BASE_P, BASE_M, loan_years=ly, model_years=10)

    # Detailed scenario sections
    for ly in LOAN_SCENARIOS:
        print_scenario(dfs[ly], ly)

    # Side-by-side summary
    print_side_by_side(dfs)

    # Recommendation
    print_recommendation(dfs, ue)

    # Save CSV — all scenarios stacked
    combined = pd.concat([dfs[ly] for ly in LOAN_SCENARIOS], ignore_index=True)
    out_csv  = PROJECT_ROOT / "results" / "pl_cashflow_qlp_base_case.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")


if __name__ == "__main__":
    out_txt = PROJECT_ROOT / "results" / "pl_cashflow_qlp_full_report.txt"
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
