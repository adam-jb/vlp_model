"""
P&L and Cashflow Forecast for VLP Battery Business Model (UK)
=============================================================
Based on 2025 Agile price data from results/agile_analysis_2025_5kw.csv

Business model:
- We act as retail electricity provider (energy company)
- We install 10kWh/5kW battery at customer premises (no upfront charge to customer)
- We fund the battery via a 7% loan, repay as fast as possible
- We charge customer: £X per kWh + £Y per month
- Target: significantly cheaper than competitor (25p/kWh + 50p/day)
- Battery generates arbitrage savings + grid service revenues

Usage:
    python analysis/pl_cashflow_model.py
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# BASE ASSUMPTIONS (from model results & inputs)
# ─────────────────────────────────────────────

# Battery hardware
BATTERY_COST_GBP = 2_600          # £ inc install (from model_input.csv)
BATTERY_REMOVAL_GBP = 400         # £ at end of life
BATTERY_LIFE_YEARS = 10           # assumed useful life
LOAN_RATE = 0.07                  # 7% annual
DEPRECIATION_GBP = BATTERY_COST_GBP / BATTERY_LIFE_YEARS  # £260/yr straight-line

# Annual consumption
ANNUAL_KWH = 3_604.4              # kWh/year (from UK use profile)

# Wholesale electricity costs (2025 Agile actuals, from results)
WHOLESALE_NO_BATTERY_GBP = 775.07  # £/year without battery
WHOLESALE_WITH_BATTERY_GBP = 442.76  # £/year after battery arbitrage (net cost to us)
ARBITRAGE_SAVING_GBP = 332.31     # £/year battery saves on wholesale

# Grid service revenues (from model_input.csv)
BATTERY_KW = 5.0
FREQ_RESPONSE_PER_KW = 34         # £/kW/year
CAPACITY_MARKET_PER_KW = 40       # £/kW/year
FREQ_RESPONSE_GBP = BATTERY_KW * FREQ_RESPONSE_PER_KW   # £170/year
CAPACITY_MARKET_GBP = BATTERY_KW * CAPACITY_MARKET_PER_KW  # £200/year
BM_REVENUE_GBP = 12               # £/year (small; 10% win rate × 5% uplift on small volume)

TOTAL_GRID_SERVICES_GBP = FREQ_RESPONSE_GBP + CAPACITY_MARKET_GBP + BM_REVENUE_GBP

# Operating costs (per customer per year)
GRID_LEVY_GBP = 95                # £/year (UK electricity company levy, model_input.csv)
CUSTOMER_MGMT_GBP = 10            # £/year

# Competitor tariff (for comparison)
COMPETITOR_PENCE_PER_KWH = 25     # p/kWh
COMPETITOR_PENCE_PER_DAY = 50     # p/day
COMPETITOR_ANNUAL_GBP = (ANNUAL_KWH * COMPETITOR_PENCE_PER_KWH / 100
                          + 365 * COMPETITOR_PENCE_PER_DAY / 100)

# ─────────────────────────────────────────────
# PRICING OPTIONS (our tariff to customer)
# ─────────────────────────────────────────────
# Format: (pence_per_kwh, pounds_per_month, label)
PRICING_OPTIONS = [
    (18, 8,  "18p/kWh + £8/mo"),
    (19, 8,  "19p/kWh + £8/mo"),
    (20, 8,  "20p/kWh + £8/mo"),
    (20, 10, "20p/kWh + £10/mo  ★ base case"),
    (21, 10, "21p/kWh + £10/mo"),
    (22, 10, "22p/kWh + £10/mo"),
    (22, 12, "22p/kWh + £12/mo"),
]

# ─────────────────────────────────────────────
# HELPER: loan repayment schedule (maximise principal)
# ─────────────────────────────────────────────

def loan_schedule(principal, rate, annual_free_cashflow):
    """
    Repay as much as possible each year from free cash flow.
    Returns list of dicts per year until paid off.
    """
    schedule = []
    balance = principal
    for yr in range(1, 30):
        if balance <= 0:
            break
        interest = balance * rate
        # Pay all free cash toward loan (interest first, then principal)
        if annual_free_cashflow >= interest:
            principal_paid = min(annual_free_cashflow - interest, balance)
        else:
            principal_paid = 0  # can't even cover interest
        balance -= principal_paid
        schedule.append({
            "year": yr,
            "opening_balance": balance + principal_paid,
            "interest": interest,
            "principal_paid": principal_paid,
            "closing_balance": max(balance, 0),
            "total_payment": interest + principal_paid,
        })
        if balance <= 0.01:
            break
    return schedule


# ─────────────────────────────────────────────
# SINGLE-CUSTOMER UNIT ECONOMICS
# ─────────────────────────────────────────────

def unit_economics(p_per_kwh, gbp_per_month):
    """Return dict of annual unit economics for one customer."""
    elec_revenue = ANNUAL_KWH * p_per_kwh / 100
    service_fee  = gbp_per_month * 12
    customer_bill = elec_revenue + service_fee

    total_revenue = (elec_revenue + service_fee
                     + TOTAL_GRID_SERVICES_GBP)

    electricity_cost = WHOLESALE_WITH_BATTERY_GBP  # net after arbitrage
    total_opex = GRID_LEVY_GBP + CUSTOMER_MGMT_GBP

    ebitda = total_revenue - electricity_cost - total_opex

    saving_vs_competitor = COMPETITOR_ANNUAL_GBP - customer_bill
    saving_pct = saving_vs_competitor / COMPETITOR_ANNUAL_GBP * 100

    return {
        "elec_revenue": elec_revenue,
        "service_fee": service_fee,
        "customer_bill": customer_bill,
        "saving_vs_competitor": saving_vs_competitor,
        "saving_pct": saving_pct,
        "grid_services": TOTAL_GRID_SERVICES_GBP,
        "total_revenue": total_revenue,
        "electricity_cost": electricity_cost,
        "total_opex": total_opex,
        "ebitda": ebitda,
    }


# ─────────────────────────────────────────────
# PRINT SENSITIVITY TABLE
# ─────────────────────────────────────────────

def print_sensitivity():
    print("\n" + "=" * 85)
    print("PRICING SENSITIVITY  |  Competitor: 25p/kWh + 50p/day"
          f" = £{COMPETITOR_ANNUAL_GBP:,.2f}/yr for {ANNUAL_KWH:.0f} kWh")
    print("=" * 85)
    print(f"{'Tariff':<26} {'Cust Bill':>10} {'Saving £':>9} {'Saving %':>9} "
          f"{'EBITDA':>9} {'Loan off':>9}")
    print("-" * 85)
    for (pkwh, pmo, label) in PRICING_OPTIONS:
        ue = unit_economics(pkwh, pmo)
        sched = loan_schedule(BATTERY_COST_GBP, LOAN_RATE, ue["ebitda"])
        payoff = sched[-1]["year"] if sched else "N/A"
        print(f"{label:<26} £{ue['customer_bill']:>8,.2f} "
              f" £{ue['saving_vs_competitor']:>7,.2f} "
              f"  {ue['saving_pct']:>6.1f}%"
              f"  £{ue['ebitda']:>7,.2f}"
              f"  {payoff:>5} yrs")
    print()


# ─────────────────────────────────────────────
# DETAILED SINGLE-CUSTOMER 10-YEAR P&L + CASHFLOW
# ─────────────────────────────────────────────

def detailed_forecast(p_per_kwh, gbp_per_month, label, years=10):
    ue = unit_economics(p_per_kwh, gbp_per_month)
    ebitda = ue["ebitda"]
    sched = loan_schedule(BATTERY_COST_GBP, LOAN_RATE, ebitda)
    payoff_year = sched[-1]["year"] if sched else 0

    print("\n" + "=" * 90)
    print(f"DETAILED FORECAST  |  {label}")
    print(f"Customer tariff: {p_per_kwh}p/kWh + £{gbp_per_month}/month")
    print(f"Customer annual bill: £{ue['customer_bill']:,.2f}  "
          f"vs competitor £{COMPETITOR_ANNUAL_GBP:,.2f}  "
          f"(saves £{ue['saving_vs_competitor']:,.2f} = {ue['saving_pct']:.1f}% cheaper)")
    print("=" * 90)

    # ── P&L section ──
    print("\n── ANNUAL UNIT ECONOMICS (per customer) ──")
    print(f"  {'REVENUE'}")
    print(f"    Electricity sales ({ANNUAL_KWH:.0f} kWh × {p_per_kwh}p):   £{ue['elec_revenue']:>8,.2f}")
    print(f"    Monthly service fee (£{gbp_per_month}/mo × 12):           £{ue['service_fee']:>8,.2f}")
    print(f"    Frequency response (5kW × £{FREQ_RESPONSE_PER_KW}/kW):    £{FREQ_RESPONSE_GBP:>8,.2f}")
    print(f"    Capacity market (5kW × £{CAPACITY_MARKET_PER_KW}/kW):       £{CAPACITY_MARKET_GBP:>8,.2f}")
    print(f"    Balancing mechanism (est.):                    £{BM_REVENUE_GBP:>8,.2f}")
    print(f"    {'─'*42}")
    print(f"    Total Revenue:                                 £{ue['total_revenue']:>8,.2f}")
    print()
    print(f"  {'COST OF ELECTRICITY'}")
    print(f"    Wholesale (Agile, no battery):                 £{WHOLESALE_NO_BATTERY_GBP:>8,.2f}")
    print(f"    Battery arbitrage saving:                     -£{ARBITRAGE_SAVING_GBP:>8,.2f}")
    print(f"    Net wholesale cost:                            £{ue['electricity_cost']:>8,.2f}")
    print()
    print(f"  {'OPERATING COSTS'}")
    print(f"    Grid levy (electricity co. levy/cust):         £{GRID_LEVY_GBP:>8,.2f}")
    print(f"    Customer management:                           £{CUSTOMER_MGMT_GBP:>8,.2f}")
    print(f"    Total OpEx:                                    £{ue['total_opex']:>8,.2f}")
    print()
    ebit = ebitda - DEPRECIATION_GBP
    print(f"  EBITDA (pre battery debt service):               £{ebitda:>8,.2f}")
    print(f"    Battery depreciation (£{BATTERY_COST_GBP:,}/10yr):            -£{DEPRECIATION_GBP:>7,.2f}")
    print(f"  EBIT:                                            £{ebit:>8,.2f}")
    print()

    # ── Loan schedule ──
    print(f"── BATTERY LOAN SCHEDULE (£{BATTERY_COST_GBP:,} @ {LOAN_RATE*100:.0f}%, max repayment) ──")
    print(f"  Battery is 100% loan-funded (zero equity outlay).")
    print(f"  All EBITDA applied to debt until paid off, then flows to equity.")
    print()
    print(f"  {'Year':>4}  {'Loan Bal':>9}  {'Interest':>9}  {'Principal':>10}  "
          f"{'Net P&L':>9}  {'Cashflow':>9}  {'Cum Cash':>10}")
    print(f"  {'────':>4}  {'────────':>9}  {'────────':>9}  {'─────────':>10}  "
          f"{'───────':>9}  {'───────':>9}  {'────────':>10}")

    cum_cash = 0  # zero equity invested — battery fully loan funded
    sched_dict = {s["year"]: s for s in sched}

    rows = []
    for yr in range(1, years + 1):
        if yr in sched_dict:
            s = sched_dict[yr]
            interest = s["interest"]
            principal = s["principal_paid"]
            debt_service = s["total_payment"]
            open_bal = s["opening_balance"]
        else:
            interest = 0
            principal = 0
            debt_service = 0
            open_bal = 0

        # P&L view: EBIT minus interest (depreciation already deducted from EBIT)
        net_pl = ebit - interest
        # Cashflow view: EBITDA minus total cash debt service (add-back depreciation vs principal)
        net_cashflow = ebitda - debt_service
        cum_cash += net_cashflow

        rows.append({
            "Year": yr,
            "Loan Balance": open_bal,
            "Interest": interest,
            "Principal Paid": principal,
            "Net P&L (after interest)": net_pl,
            "Net Cashflow": net_cashflow,
            "Cum Cashflow": cum_cash,
        })

        marker = " ← loan repaid" if yr == payoff_year else ""
        print(f"  {yr:>4}  £{open_bal:>7,.0f}   £{interest:>7,.0f}   £{principal:>8,.0f}  "
              f" £{net_pl:>7,.0f}   £{net_cashflow:>7,.0f}   £{cum_cash:>8,.0f}{marker}")

    print()
    print(f"  Battery loan paid off: Year {payoff_year}")
    print(f"  Post-payoff annual net profit: £{ebit:,.2f}  |  post-payoff cashflow: £{ebitda:,.2f}/year")
    print(f"  Total interest paid over loan life: £{sum(s['interest'] for s in sched):,.2f}")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# PORTFOLIO FORECAST (scaling customers)
# ─────────────────────────────────────────────

def portfolio_forecast(p_per_kwh, gbp_per_month, customer_schedule, label):
    """
    customer_schedule: list of (year, new_customers_added) tuples
    Each new customer gets their own battery loan.
    """
    ue = unit_economics(p_per_kwh, gbp_per_month)
    ebitda_per_cust = ue["ebitda"]

    print("\n" + "=" * 90)
    print(f"PORTFOLIO FORECAST  |  {label}  |  {p_per_kwh}p/kWh + £{gbp_per_month}/mo")
    print("=" * 90)

    # Build cumulative customer count per year
    adds_by_year = dict(customer_schedule)
    max_year = max(y for y, _ in customer_schedule) + 5

    print(f"\n  {'Year':>4}  {'New Custs':>9}  {'Total Custs':>11}  {'EBITDA':>10}  "
          f"{'Interest':>9}  {'Principal':>10}  {'Net Cash':>9}  {'Cum Cash':>10}")
    print(f"  {'────':>4}  {'────────':>9}  {'──────────':>11}  {'──────':>10}  "
          f"{'────────':>9}  {'─────────':>10}  {'───────':>9}  {'────────':>10}")

    # Track each cohort's loan balance
    cohort_balances = {}  # cohort_year -> remaining_balance
    total_customers = 0
    cum_cash = 0

    for yr in range(1, max_year + 1):
        new_custs = adds_by_year.get(yr, 0)
        total_customers += new_custs

        # New loan drawdown this year
        new_loan = new_custs * BATTERY_COST_GBP
        if new_custs > 0:
            cohort_balances[yr] = new_custs * BATTERY_COST_GBP

        # Total EBITDA from all customers this year
        total_ebitda = total_customers * ebitda_per_cust

        # Interest and principal across all cohorts
        total_interest = sum(bal * LOAN_RATE for bal in cohort_balances.values())
        free_cash_for_debt = total_ebitda  # use all free cash for debt

        # Allocate repayment to cohorts (oldest first)
        remaining_repay = max(0, free_cash_for_debt - total_interest)
        total_principal = 0
        for cohort_yr in sorted(cohort_balances.keys()):
            if remaining_repay <= 0:
                break
            bal = cohort_balances[cohort_yr]
            pay = min(remaining_repay, bal)
            cohort_balances[cohort_yr] -= pay
            total_principal += pay
            remaining_repay -= pay

        # Clean up paid-off cohorts
        cohort_balances = {k: v for k, v in cohort_balances.items() if v > 0.01}

        total_debt_service = total_interest + total_principal
        net_cashflow = total_ebitda - total_debt_service - new_loan
        # Actually loan is funded externally; we draw down on loan to fund batteries
        # So: we get loan drawdown in, pay battery cost (net 0), then service debt from EBITDA
        # Net cashflow to equity = EBITDA - debt service
        net_cashflow_equity = total_ebitda - total_debt_service

        cum_cash += net_cashflow_equity

        total_loan_bal = sum(cohort_balances.values())

        print(f"  {yr:>4}  {new_custs:>9,}  {total_customers:>11,}  "
              f"£{total_ebitda:>8,.0f}  £{total_interest:>7,.0f}  "
              f"£{total_principal:>8,.0f}  £{net_cashflow_equity:>7,.0f}  £{cum_cash:>8,.0f}")

    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 90)
    print("VLP BATTERY BUSINESS — P&L & CASHFLOW MODEL  (UK, 2025 Agile actuals)")
    print("=" * 90)
    print(f"\nKey assumptions:")
    print(f"  Battery: 10kWh / 5kW  |  Cost: £{BATTERY_COST_GBP:,} inc install  |  Loan: {LOAN_RATE*100:.0f}% p.a.")
    print(f"  Customer annual consumption: {ANNUAL_KWH:.0f} kWh")
    print(f"  Wholesale cost (no battery): £{WHOLESALE_NO_BATTERY_GBP:.2f}/yr  →  "
          f"with battery: £{WHOLESALE_WITH_BATTERY_GBP:.2f}/yr  "
          f"(arbitrage saves £{ARBITRAGE_SAVING_GBP:.2f}/yr)")
    print(f"  Grid services: freq response £{FREQ_RESPONSE_GBP} + capacity market £{CAPACITY_MARKET_GBP} "
          f"+ BM £{BM_REVENUE_GBP} = £{TOTAL_GRID_SERVICES_GBP}/yr")
    print(f"  Grid levy: £{GRID_LEVY_GBP}/yr  |  Customer mgmt: £{CUSTOMER_MGMT_GBP}/yr")
    print(f"\n  Competitor: {COMPETITOR_PENCE_PER_KWH}p/kWh + {COMPETITOR_PENCE_PER_DAY}p/day "
          f"= £{COMPETITOR_ANNUAL_GBP:,.2f}/yr  ({ANNUAL_KWH:.0f} kWh)")
    print(f"  Note: Agile 2025 prices used. Wholesale prices are variable-rate Octopus Agile.")

    # 1. Sensitivity table
    print_sensitivity()

    # 2. Detailed base case
    BASE_PKWH = 20
    BASE_PMO  = 10
    df_base = detailed_forecast(
        BASE_PKWH, BASE_PMO,
        f"{BASE_PKWH}p/kWh + £{BASE_PMO}/month (base case)",
        years=10
    )

    # 3. Portfolio forecast — illustrative growth
    portfolio_forecast(
        BASE_PKWH, BASE_PMO,
        customer_schedule=[
            (1,  50),
            (2, 100),
            (3, 200),
            (4, 300),
            (5, 400),
        ],
        label="Illustrative portfolio growth"
    )

    # 4. Save base case to CSV
    out_path = "results/pl_cashflow_base_case.csv"
    df_base.to_csv(out_path, index=False)
    print(f"Base case saved to {out_path}\n")


if __name__ == "__main__":
    import sys
    # If run directly, also save full output to a text file
    out_txt = "results/pl_cashflow_full_report.txt"
    tee = open(out_txt, "w")
    original_stdout = sys.stdout

    class Tee:
        def write(self, msg):
            original_stdout.write(msg)
            tee.write(msg)
        def flush(self):
            original_stdout.flush()
            tee.flush()

    sys.stdout = Tee()
    main()
    sys.stdout = original_stdout
    tee.close()
    print(f"Full report also saved to {out_txt}")
