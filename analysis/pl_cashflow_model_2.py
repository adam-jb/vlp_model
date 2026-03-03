"""
Three-Statement Financial Model for VLP Battery Business (UK) — Version 2
=========================================================================
Key differences from version 1:
  - Battery revenue from DIRECT WHOLESALE MARKET (buy daily min, sell daily max)
    using United_Kingdom_daily_spread.csv — NOT Agile Octopus retail rates
  - Electricity supply cost = actual UK wholesale average (8.49p/kWh)
  - Capacity Market properly de-rated for 2-hour (10kWh/5kW) battery: 20.94%
  - Full three-statement model: Income Statement, Cash Flow, Balance Sheet
  - UK corporation tax (25%) applied to positive EBT

Data sources:
  - results/United_Kingdom_daily_spread.csv  → wholesale arbitrage
  - data/prices/United Kingdom.csv           → average wholesale supply cost
  - data/inputs/model_input.csv              → battery params, grid services
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# ─────────────────────────────────────────────────────────
# ASSUMPTIONS — all derived from model data / model_input.csv
# ─────────────────────────────────────────────────────────

# Battery
BATTERY_COST_GBP    = 2_600    # £ inc install
BATTERY_REMOVAL_GBP = 400      # £ at end of life (year 10+)
BATTERY_LIFE_YEARS  = 10
BATTERY_KWH         = 10.0
BATTERY_KW          = 5.0
EFFICIENCY          = 0.90     # round-trip; one-way = sqrt(0.90)
LOAN_RATE           = 0.07     # 7% p.a.
CORP_TAX_RATE       = 0.25     # UK corporation tax

# Customer
ANNUAL_KWH          = 3_604.4  # kWh/year from UK use profile

# ── Direct market revenues (from wholesale price data) ──
# Arbitrage: buy at daily min, sell daily max, 10kWh battery, one-way eff = sqrt(0.9)
# Derived from United_Kingdom_daily_spread.csv (Jul 2024 – Jun 2025)
EUR_TO_GBP          = 0.85
ETA                 = EFFICIENCY ** 0.5          # one-way efficiency ~0.949
ARB_BUY_COST_GBP    = 191.59   # annual electricity cost to charge battery (£)
ARB_SELL_REV_GBP    = 440.88   # annual revenue from selling battery energy to grid (£)
ARB_NET_GBP         = ARB_SELL_REV_GBP - ARB_BUY_COST_GBP  # = £249.29 net arbitrage

# ── Grid services ──
FR_PER_KW_YEAR      = 34       # £/kW/year frequency response
CM_PER_KW_YEAR      = 40       # £/kW/year capacity market (pre de-rating)
CM_DERATING         = 0.2094   # de-rating for 2-hour (10kWh/5kW) battery — EMR DB
BM_WIN_RATE         = 0.10
BM_UPLIFT           = 0.05

FR_INCOME_GBP       = BATTERY_KW * FR_PER_KW_YEAR                          # £170.00
CM_INCOME_GBP       = BATTERY_KW * CM_PER_KW_YEAR * CM_DERATING            # £41.88
BM_INCOME_GBP       = ARB_NET_GBP * BM_WIN_RATE * BM_UPLIFT                # £1.25
GRID_SERVICES_GBP   = FR_INCOME_GBP + CM_INCOME_GBP + BM_INCOME_GBP       # £213.13

# ── Electricity supply cost ──
# Average UK day-ahead wholesale Jul 2024–Jun 2025
WHOLESALE_AVG_EUR_MWH = 99.84  # EUR/MWh
WHOLESALE_AVG_GBP_KWH = WHOLESALE_AVG_EUR_MWH * EUR_TO_GBP / 1000         # 8.49p/kWh
SUPPLY_COST_GBP       = ANNUAL_KWH * WHOLESALE_AVG_GBP_KWH                # £305.84

# ── Operating costs (per customer/year) ──
GRID_LEVY_GBP       = 95       # electricity company levy per customer (model_input.csv)
CUST_MGMT_GBP       = 10       # customer management cost

# ── Competitor tariff (for comparison) ──
COMPETITOR_PKW      = 25       # p/kWh
COMPETITOR_PD       = 50       # p/day
COMPETITOR_ANNUAL   = ANNUAL_KWH * COMPETITOR_PKW / 100 + 365 * COMPETITOR_PD / 100

# ── Depreciation ──
DEPRECIATION_GBP    = BATTERY_COST_GBP / BATTERY_LIFE_YEARS  # £260/year straight-line

# ─────────────────────────────────────────────────────────
# PRICING OPTIONS  (p/kWh,  £/month, label)
# ─────────────────────────────────────────────────────────
PRICING_OPTIONS = [
    (17, 8,  "17p/kWh + £8/mo"),
    (18, 8,  "18p/kWh + £8/mo"),
    (19, 8,  "19p/kWh + £8/mo  ★ base case"),   # matches battery_model.py defaults
    (20, 8,  "20p/kWh + £8/mo"),
    (20, 10, "20p/kWh + £10/mo"),
    (22, 10, "22p/kWh + £10/mo"),
]
BASE_PKWH, BASE_PMO = 19, 8


# ─────────────────────────────────────────────────────────
# UNIT ECONOMICS
# ─────────────────────────────────────────────────────────

def unit_economics(p_per_kwh, gbp_per_month):
    elec_rev    = ANNUAL_KWH * p_per_kwh / 100
    svc_fee     = gbp_per_month * 12
    cust_bill   = elec_rev + svc_fee
    total_rev   = elec_rev + svc_fee + ARB_NET_GBP + GRID_SERVICES_GBP
    gross       = total_rev - SUPPLY_COST_GBP
    ebitda      = gross - GRID_LEVY_GBP - CUST_MGMT_GBP
    ebit        = ebitda - DEPRECIATION_GBP
    saving      = COMPETITOR_ANNUAL - cust_bill
    saving_pct  = saving / COMPETITOR_ANNUAL * 100
    return dict(
        elec_rev=elec_rev, svc_fee=svc_fee, cust_bill=cust_bill,
        total_rev=total_rev, gross=gross, ebitda=ebitda, ebit=ebit,
        saving=saving, saving_pct=saving_pct
    )


# ─────────────────────────────────────────────────────────
# THREE-STATEMENT ANNUAL MODEL (single customer)
# ─────────────────────────────────────────────────────────

def three_statements(p_per_kwh, gbp_per_month, years=10):
    ue     = unit_economics(p_per_kwh, gbp_per_month)
    ebitda = ue["ebitda"]
    ebit   = ue["ebit"]

    rows = []
    loan_bal    = float(BATTERY_COST_GBP)  # drawn down at start
    cum_ret_earn = 0.0
    cum_cash     = 0.0
    bat_gross    = float(BATTERY_COST_GBP)

    for yr in range(1, years + 1):
        # ── Income Statement ──
        interest    = loan_bal * LOAN_RATE
        ebt         = ebit - interest
        tax         = max(0, ebt * CORP_TAX_RATE)
        net_income  = ebt - tax

        # ── Cash Flow Statement ──
        # OCF (indirect method): net income already includes interest & tax paid,
        # so add back only the non-cash D&A charge.
        ocf         = net_income + DEPRECIATION_GBP
        # All OCF available for principal repayment (interest already settled via IS)
        principal   = min(ocf, loan_bal)
        free_cash   = ocf - principal
        cum_cash   += free_cash
        loan_bal   -= principal

        # ── Balance Sheet ──
        accum_dep   = DEPRECIATION_GBP * yr
        bat_nbv     = max(0, bat_gross - accum_dep)
        cum_ret_earn += net_income
        total_assets = cum_cash + bat_nbv
        total_liab   = loan_bal
        equity       = cum_ret_earn            # paid-up capital = 0 (100% loan funded)
        bs_check     = total_assets - total_liab - equity   # should be ~0

        rows.append({
            "Year": yr,
            # IS
            "Revenue":            ue["total_rev"],
            "COGS (wholesale)":   SUPPLY_COST_GBP,
            "Gross Profit":       ue["gross"],
            "OpEx":               GRID_LEVY_GBP + CUST_MGMT_GBP,
            "EBITDA":             ebitda,
            "Depreciation":       DEPRECIATION_GBP,
            "EBIT":               ebit,
            "Interest":           interest,
            "EBT":                ebt,
            "Tax (25%)":          tax,
            "Net Income":         net_income,
            # CFS  (indirect method: interest paid is within Operating CF)
            "Operating CF":       ocf,
            "Investing CF":       0,
            "Principal Paid":     principal,
            "Financing CF":       -principal,
            "Free Cash":          free_cash,
            "Cum Cash":           cum_cash,
            # BS
            "Battery NBV":        bat_nbv,
            "Total Assets":       total_assets,
            "Loan Balance":       loan_bal,
            "Retained Earnings":  cum_ret_earn,
            "Total Equity":       equity,
            "BS Check":           bs_check,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
# PORTFOLIO FORECAST
# ─────────────────────────────────────────────────────────

def portfolio_three_statements(p_per_kwh, gbp_per_month, customer_schedule, years=12):
    """
    Portfolio model: each cohort gets its own loan.
    customer_schedule: [(year, new_customers), ...]
    """
    ue          = unit_economics(p_per_kwh, gbp_per_month)
    ebit_unit   = ue["ebit"]
    ebitda_unit = ue["ebitda"]
    adds        = dict(customer_schedule)

    total_custs  = 0
    # cohort_balances: {start_year: remaining_loan_balance}
    cohort_bal   = {}
    cum_cash     = 0.0
    cum_earnings = 0.0

    rows = []
    for yr in range(1, years + 1):
        new_custs   = adds.get(yr, 0)
        total_custs += new_custs
        if new_custs > 0:
            cohort_bal[yr] = new_custs * float(BATTERY_COST_GBP)

        total_ebitda    = total_custs * ebitda_unit
        total_ebit      = total_custs * ebit_unit
        total_dep       = total_custs * DEPRECIATION_GBP

        # Interest across all cohorts
        total_interest  = sum(b * LOAN_RATE for b in cohort_bal.values())
        total_ebt       = total_ebit - total_interest
        total_tax       = max(0, total_ebt * CORP_TAX_RATE)
        total_ni        = total_ebt - total_tax
        total_ocf       = total_ni + total_dep

        # OCF is after interest (already in net income); repay principal from OCF
        remaining       = total_ocf
        total_principal = 0
        for cy in sorted(cohort_bal.keys()):
            if remaining <= 0:
                break
            pay = min(remaining, cohort_bal[cy])
            cohort_bal[cy] -= pay
            total_principal += pay
            remaining       -= pay
        cohort_bal = {k: v for k, v in cohort_bal.items() if v > 0.01}

        total_debt_svc  = total_principal   # interest already in OCF via net income
        free_cash       = total_ocf - total_debt_svc
        cum_cash       += free_cash
        cum_earnings   += total_ni
        total_loan      = sum(cohort_bal.values())

        rows.append({
            "Year": yr,
            "New Custs":     new_custs,
            "Total Custs":   total_custs,
            "EBITDA":        total_ebitda,
            "EBIT":          total_ebit,
            "Interest":      total_interest,
            "EBT":           total_ebt,
            "Tax":           total_tax,
            "Net Income":    total_ni,
            "Op CF":         total_ocf,
            "Principal":     total_principal,
            "Free Cash":     free_cash,
            "Cum Cash":      cum_cash,
            "Loan Balance":  total_loan,
            "Cum Earnings":  cum_earnings,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────────────────────

W = 90
def hdr(title):
    print()
    print("=" * W)
    print(title)
    print("=" * W)

def row(label, val, indent=2, prefix="£"):
    sp = " " * indent
    print(f"{sp}{label:<50}{prefix}{val:>9,.2f}")

def divider(indent=2):
    print(" " * indent + "─" * 58)


def print_assumptions():
    hdr("VLP BATTERY BUSINESS — THREE-STATEMENT MODEL v2  (UK, wholesale market rates)")
    print()
    print("KEY ASSUMPTIONS")
    print(f"  Battery       : {BATTERY_KWH:.0f}kWh / {BATTERY_KW:.0f}kW  |  cost £{BATTERY_COST_GBP:,} inc install  "
          f"|  life {BATTERY_LIFE_YEARS}yr  |  loan {LOAN_RATE*100:.0f}% p.a.")
    print(f"  Consumption   : {ANNUAL_KWH:.0f} kWh/year (UK profile)")
    print(f"  Wholesale cost: £{WHOLESALE_AVG_EUR_MWH:.2f}/MWh ({WHOLESALE_AVG_GBP_KWH*100:.2f}p/kWh) UK day-ahead avg Jul24–Jun25")
    print(f"  Annual supply : {ANNUAL_KWH:.0f}kWh × {WHOLESALE_AVG_GBP_KWH*100:.2f}p = £{SUPPLY_COST_GBP:.2f}")
    print()
    print("  BATTERY DIRECT MARKET REVENUES (from United_Kingdom_daily_spread.csv)")
    print(f"    Arb buy cost  : £{ARB_BUY_COST_GBP:.2f}/yr  (charging at daily min price)")
    print(f"    Arb sell rev  : £{ARB_SELL_REV_GBP:.2f}/yr  (selling at daily max price)")
    print(f"    Net arbitrage : £{ARB_NET_GBP:.2f}/yr")
    print()
    print("  GRID SERVICES")
    print(f"    Frequency response : {BATTERY_KW:.0f}kW × £{FR_PER_KW_YEAR}/kW = £{FR_INCOME_GBP:.2f}/yr")
    print(f"    Capacity market    : {BATTERY_KW:.0f}kW × £{CM_PER_KW_YEAR}/kW × {CM_DERATING*100:.2f}% de-rating "
          f"(2hr battery) = £{CM_INCOME_GBP:.2f}/yr")
    print(f"    Balancing mech.    : arb × {BM_WIN_RATE*100:.0f}% win × {BM_UPLIFT*100:.0f}% uplift = £{BM_INCOME_GBP:.2f}/yr")
    print(f"    Total grid services: £{GRID_SERVICES_GBP:.2f}/yr")
    print()
    print(f"  Grid levy     : £{GRID_LEVY_GBP}/yr/cust  |  Customer mgmt: £{CUST_MGMT_GBP}/yr")
    print(f"  Corporation tax: {CORP_TAX_RATE*100:.0f}%  |  Depreciation: £{BATTERY_COST_GBP:,}/{BATTERY_LIFE_YEARS}yr = £{DEPRECIATION_GBP:.0f}/yr")
    print()
    print(f"  ⚠  NOTE: Wholesale supply cost (£{SUPPLY_COST_GBP:.0f}/yr) covers raw energy only.")
    print(f"     Network charges (DUoS/TNUoS ~£200-350/yr) and BSUoS may apply additionally.")
    print(f"     These are NOT in model_input.csv — treat EBITDA as pre-network-cost upper bound.")
    print(f"     Conservative EBITDA (subtract est. £275 network) shown in sensitivity table.")
    print()
    print(f"  Competitor: {COMPETITOR_PKW}p/kWh + {COMPETITOR_PD}p/day = £{COMPETITOR_ANNUAL:,.2f}/yr  ({ANNUAL_KWH:.0f} kWh)")


def print_sensitivity():
    print()
    print("─" * W)
    print(f"PRICING SENSITIVITY")
    print("─" * W)
    print(f"{'Tariff':<28} {'Cust Bill':>10} {'Saving':>8} {'Saving%':>8} "
          f"{'EBITDA':>9} {'EBIT':>8} {'Net Inc*':>9}")
    print(f"  (* approx yr1 net income, interest = £{BATTERY_COST_GBP*LOAN_RATE:.0f})")
    print("─" * W)
    for (pkwh, pmo, label) in PRICING_OPTIONS:
        ue = unit_economics(pkwh, pmo)
        approx_ni = (ue["ebit"] - BATTERY_COST_GBP * LOAN_RATE) * (1 - CORP_TAX_RATE)
        print(f"  {label:<26} £{ue['cust_bill']:>8,.2f} "
              f" £{ue['saving']:>6,.2f}  {ue['saving_pct']:>6.1f}% "
              f" £{ue['ebitda']:>7,.2f} £{ue['ebit']:>6,.2f} £{approx_ni:>7,.2f}")
    print()
    # Conservative version (subtract est. network charges)
    NETWORK_COST = 275
    print(f"  Conservative (subtract est. £{NETWORK_COST} network charges):")
    print("─" * W)
    for (pkwh, pmo, label) in PRICING_OPTIONS:
        ue = unit_economics(pkwh, pmo)
        ebitda_c = ue["ebitda"] - NETWORK_COST
        ebit_c   = ebitda_c - DEPRECIATION_GBP
        approx_ni = max(0, ebit_c - BATTERY_COST_GBP * LOAN_RATE) * (1 - CORP_TAX_RATE)
        print(f"  {label:<26} £{ue['cust_bill']:>8,.2f} "
              f" £{ue['saving']:>6,.2f}  {ue['saving_pct']:>6.1f}% "
              f" £{ebitda_c:>7,.2f} £{ebit_c:>6,.2f} £{approx_ni:>7,.2f}")


def print_unit_economics(p_per_kwh, gbp_per_month):
    ue = unit_economics(p_per_kwh, gbp_per_month)
    hdr(f"ANNUAL UNIT ECONOMICS — {p_per_kwh}p/kWh + £{gbp_per_month}/month (base case)")
    print(f"  Customer bill: £{ue['cust_bill']:,.2f}/yr  vs competitor £{COMPETITOR_ANNUAL:,.2f}/yr  "
          f"→  saves £{ue['saving']:,.2f}  ({ue['saving_pct']:.1f}% cheaper)")
    print()
    print("  INCOME STATEMENT")
    print("  REVENUE")
    row(f"Electricity sales ({ANNUAL_KWH:.0f} kWh × {p_per_kwh}p)",
        ue["elec_rev"])
    row(f"Monthly service fee (£{gbp_per_month}/mo × 12)",
        ue["svc_fee"])
    row(f"Battery arbitrage — wholesale market (net)",
        ARB_NET_GBP)
    row(f"  Sell revenue ({BATTERY_KWH:.0f}kWh × √eff × daily max)",
        ARB_SELL_REV_GBP, indent=4)
    row(f"  Charge cost  ({BATTERY_KWH:.0f}kWh × daily min)",
        -ARB_BUY_COST_GBP, indent=4)
    row(f"Frequency response ({BATTERY_KW:.0f}kW × £{FR_PER_KW_YEAR}/kW)",
        FR_INCOME_GBP)
    row(f"Capacity market ({BATTERY_KW:.0f}kW × £{CM_PER_KW_YEAR} × {CM_DERATING*100:.1f}% de-rate)",
        CM_INCOME_GBP)
    row(f"Balancing mechanism (est.)",
        BM_INCOME_GBP)
    divider()
    row("Total Revenue", ue["total_rev"])
    print()
    print("  COST OF REVENUE")
    row(f"Wholesale electricity ({ANNUAL_KWH:.0f}kWh × {WHOLESALE_AVG_GBP_KWH*100:.2f}p/kWh)",
        SUPPLY_COST_GBP)
    divider()
    row("Gross Profit", ue["gross"])
    print()
    print("  OPERATING EXPENSES")
    row("Grid levy (electricity co., per customer)", GRID_LEVY_GBP)
    row("Customer management", CUST_MGMT_GBP)
    divider()
    row("EBITDA", ebitda := ue["ebitda"])
    row(f"Depreciation (£{BATTERY_COST_GBP:,} / {BATTERY_LIFE_YEARS}yr)", -DEPRECIATION_GBP)
    divider()
    row("EBIT", ue["ebit"])
    row(f"Interest (yr1, loan £{BATTERY_COST_GBP:,} × {LOAN_RATE*100:.0f}%)",
        -(BATTERY_COST_GBP * LOAN_RATE))
    divider()
    yr1_ebt = ue["ebit"] - BATTERY_COST_GBP * LOAN_RATE
    row("EBT (year 1)", yr1_ebt)
    row(f"Tax ({CORP_TAX_RATE*100:.0f}%)", -(yr1_ebt * CORP_TAX_RATE))
    divider()
    row("Net Income (year 1)", yr1_ebt * (1 - CORP_TAX_RATE))


def print_three_statements(df, p_per_kwh, gbp_per_month):
    ue = unit_economics(p_per_kwh, gbp_per_month)
    hdr(f"THREE-STATEMENT FORECAST — {p_per_kwh}p/kWh + £{gbp_per_month}/mo  "
        f"(Battery 100% loan-funded, max repayment)")

    # ── Income Statement ──
    print("\n  ── INCOME STATEMENT (£) ──")
    cols_is = ["Year","Revenue","COGS (wholesale)","Gross Profit","EBITDA",
               "Depreciation","EBIT","Interest","EBT","Tax (25%)","Net Income"]
    is_df = df[cols_is].copy()
    print(is_df.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    # ── Cash Flow Statement ──
    print("\n  ── CASH FLOW STATEMENT (£) ──")
    cols_cf = ["Year","Operating CF","Investing CF","Financing CF",
               "Principal Paid","Free Cash","Cum Cash"]
    cf_df = df[cols_cf].copy()
    print(cf_df.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    # ── Balance Sheet ──
    print("\n  ── BALANCE SHEET (£) ──")
    cols_bs = ["Year","Cum Cash","Battery NBV","Total Assets",
               "Loan Balance","Retained Earnings","Total Equity","BS Check"]
    bs_df = df[cols_bs].copy()
    print(bs_df.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    payoff = df[df["Loan Balance"] < 1]["Year"].min()
    post_cf = df[df["Loan Balance"] < 1]["Free Cash"].mean()
    total_interest = df["Interest"].sum()
    total_tax = df["Tax (25%)"].sum()
    print()
    print(f"  Loan paid off:              Year {payoff}")
    print(f"  Post-payoff free cashflow:  £{post_cf:,.0f}/year")
    print(f"  Total interest paid:        £{total_interest:,.0f}")
    print(f"  Total tax paid:             £{total_tax:,.0f}")
    print(f"  10-year cumulative cash:    £{df['Cum Cash'].iloc[-1]:,.0f}")


def print_portfolio(df):
    hdr("PORTFOLIO FORECAST — 19p/kWh + £8/mo  (illustrative growth, 3-statement summary)")
    print()
    cols = ["Year","New Custs","Total Custs","EBITDA","EBIT","Interest",
            "EBT","Tax","Net Income","Op CF","Free Cash","Cum Cash","Loan Balance"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:,.0f}"))
    print()
    print(f"  At 1,050 customers post-growth:")
    row1050 = df[df["Total Custs"] == 1050].iloc[-1]
    print(f"    Annual EBITDA:    £{row1050['EBITDA']:,.0f}")
    print(f"    Annual Net Income:£{row1050['Net Income']:,.0f}")
    print(f"    Free Cashflow:    £{row1050['Free Cash']:,.0f}")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    print_assumptions()
    print_sensitivity()
    print_unit_economics(BASE_PKWH, BASE_PMO)

    df_base = three_statements(BASE_PKWH, BASE_PMO, years=10)
    print_three_statements(df_base, BASE_PKWH, BASE_PMO)

    df_port = portfolio_three_statements(
        BASE_PKWH, BASE_PMO,
        customer_schedule=[(1,50),(2,100),(3,200),(4,300),(5,400)],
        years=12
    )
    print_portfolio(df_port)

    # Save CSVs
    base_csv = PROJECT_ROOT / "results" / "pl_cashflow_base_case_2.csv"
    port_csv = PROJECT_ROOT / "results" / "pl_cashflow_portfolio_2.csv"
    df_base.to_csv(base_csv, index=False)
    df_port.to_csv(port_csv, index=False)
    print(f"\n  Saved: {base_csv}")
    print(f"  Saved: {port_csv}")


if __name__ == "__main__":
    out_txt = PROJECT_ROOT / "results" / "pl_cashflow_full_report_2.txt"
    tee_file = open(out_txt, "w")
    orig_stdout = sys.stdout

    class Tee:
        def write(self, msg):
            orig_stdout.write(msg)
            tee_file.write(msg)
        def flush(self):
            orig_stdout.flush()
            tee_file.flush()

    sys.stdout = Tee()
    main()
    sys.stdout = orig_stdout
    tee_file.close()
    print(f"\nFull report saved to: {out_txt}")
