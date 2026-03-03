"""
VLP Battery — Fixed-Term Loan Payoff Model  (Battery cost £2,800)
=================================================================
Battery + installation total: £2,800  (updated from v3's £3,779)

Compares three loan repayment schedules:
  - 3-year payoff:  fastest principal clearance, deepest early cash drain
  - 5-year payoff:  moderate schedule
  - 7-year payoff:  slowest, lowest annual payment, may be CF-positive yr 1

For each scenario produces:
  P&L (income statement)
  Cash flow statement
  Loan amortisation / payoff schedule
  Cashflow chart (matplotlib PNG)
  Full text report

All revenue and cost data:
  UK wholesale prices  Jul 2024–Jun 2025  (United Kingdom.csv)
  Battery: 15.36 kWh / 6 kW inverter
  Arbitrage: buys cheapest 2.70 hrs/day, sells dearest 2.43 hrs/day
  Grid services: frequency response + capacity market + balancing mechanism
  Cost stack: Ofgem Q1 2025 methodology

Outputs saved to this folder (qlp_2800/):
  pl_3yr.csv, pl_5yr.csv, pl_7yr.csv
  cashflow_3yr.csv, cashflow_5yr.csv, cashflow_7yr.csv
  loan_schedule_3yr.csv, loan_schedule_5yr.csv, loan_schedule_7yr.csv
  cashflow_chart.png
  report.txt
"""

import sys
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_DIR      = SCRIPT_DIR           # save everything alongside this script

# ── battery hardware ─────────────────────────────────────────────────────────
BAT_KWH         = 15.36
BAT_KW          = 6.0
EFF_RT          = 0.90
ETA             = EFF_RT ** 0.5     # one-way efficiency ≈ 0.9487

GRID_DRAW_KWH   = BAT_KWH / ETA          # kWh drawn from grid per full charge
DISCHARGE_KWH   = BAT_KWH * ETA          # kWh delivered per full discharge
T_CHARGE        = GRID_DRAW_KWH / BAT_KW  # hrs to fully charge  ≈ 2.699
T_DISCHARGE     = DISCHARGE_KWH / BAT_KW  # hrs to fully discharge ≈ 2.429
N_FC            = int(T_CHARGE)            # 2 full charge hours
FRAC_C          = T_CHARGE - N_FC          # 0.699 partial hour
N_FD            = int(T_DISCHARGE)         # 2 full discharge hours
FRAC_D          = T_DISCHARGE - N_FD       # 0.429 partial hour

# ── battery cost (UPDATED) ───────────────────────────────────────────────────
BAT_COST_GBP    = 2_800.00   # £ total (hardware + installation)
BAT_LIFE_YRS    = 10
DEPRECIATION    = BAT_COST_GBP / BAT_LIFE_YRS   # £280/yr straight-line

# ── financing ────────────────────────────────────────────────────────────────
LOAN_RATE   = 0.07    # 7% p.a. on outstanding balance
CORP_TAX    = 0.25    # 25% corporation tax

LOAN_SCENARIOS = [3, 5, 7]   # payoff years compared

# ── grid services ─────────────────────────────────────────────────────────────
FR_PER_KW   = 34      # £/kW/yr frequency response
CM_PER_KW   = 40      # £/kW/yr capacity market
CM_DERATING = 0.2094  # de-rating for 2-hr battery class (2.56 h duration)
BM_WIN_RATE = 0.10    # fraction of arbitrage cycles monetised via BM
BM_UPLIFT   = 0.05    # BM premium on top of arbitrage net

FR_GBP      = BAT_KW * FR_PER_KW                   # £204.00/yr
CM_GBP      = BAT_KW * CM_PER_KW * CM_DERATING     # £50.26/yr

# ── arbitrage (computed from actual UK day-ahead prices) ─────────────────────
EUR_TO_GBP  = 0.85

def compute_arbitrage():
    """
    Scan every day in Jul 2024–Jun 2025.
    Buy: cheapest T_CHARGE hours at BAT_KW.
    Sell: dearest T_DISCHARGE hours at BAT_KW.
    Returns (annual_buy_cost, annual_sell_revenue, annual_net).
    """
    price_file = PROJECT_ROOT / "data" / "prices" / "United Kingdom.csv"
    df = pd.read_csv(price_file)
    df["dt"]   = pd.to_datetime(df["Datetime (UTC)"])
    df         = df[(df["dt"] >= "2024-07-01") & (df["dt"] < "2025-07-01")].copy()
    df["date"] = df["dt"].dt.date
    df["p_gbp_kwh"] = df["Price (EUR/MWhe)"] * EUR_TO_GBP / 1000

    buy_total = sell_total = 0.0
    days_used = 0
    for _, grp in df.groupby("date"):
        p = np.sort(grp["p_gbp_kwh"].values)
        if len(p) < 24:
            continue
        p_desc    = p[::-1]
        buy_cost  = BAT_KW * (p[:N_FC].sum()      + FRAC_C * p[N_FC])
        sell_rev  = BAT_KW * (p_desc[:N_FD].sum() + FRAC_D * p_desc[N_FD])
        buy_total  += buy_cost
        sell_total += sell_rev
        days_used  += 1

    return buy_total, sell_total, sell_total - buy_total, days_used

ARB_BUY, ARB_SELL, ARB_NET, DAYS_USED = compute_arbitrage()
BM_GBP      = ARB_NET * BM_WIN_RATE * BM_UPLIFT
GRID_SVCS   = FR_GBP + CM_GBP + BM_GBP

# ── electricity supply cost stack  (Ofgem Q1 2025 methodology) ───────────────
ANNUAL_KWH        = 3_604.4    # kWh/yr per UK average household use profile
DAYS              = 365
WHOLESALE_EUR_MWH = 99.84      # day-ahead avg Jul 2024–Jun 2025
WHOLESALE_P_KWH   = WHOLESALE_EUR_MWH * EUR_TO_GBP / 1000   # £/kWh

SUPPLY_COST   = ANNUAL_KWH * WHOLESALE_P_KWH   # wholesale electricity purchase

TNUOS         = ANNUAL_KWH * 0.0087 + DAYS * 0.0359   # transmission network
DUOS          = ANNUAL_KWH * 0.0363 + DAYS * 0.0731   # distribution network
BSUOS         = ANNUAL_KWH * 0.0075                    # balancing services
SMART_METER   = ANNUAL_KWH * 0.0054 + DAYS * 0.0356   # smart metering obligation
SUPPLIER_OPEX = ANNUAL_KWH * 0.0137 + DAYS * 0.1396 - 10  # billing, compliance, hedging
BAD_DEBT      = ANNUAL_KWH * 0.0041                    # ~0.4% bad debt provision
GRID_LEVY     = 95.0    # policy levy (from model_input.csv)
CUST_MGMT     = 10.0    # CRM cost (from model_input.csv)

TOTAL_OPEX = (GRID_LEVY + CUST_MGMT + TNUOS + DUOS + BSUOS
              + SMART_METER + SUPPLIER_OPEX + BAD_DEBT)

# ── competitor pricing (used for savings calculation only) ────────────────────
COMP_PKW   = 25    # p/kWh
COMP_PD    = 50    # p/day standing charge
COMP_BILL  = ANNUAL_KWH * COMP_PKW / 100 + DAYS * COMP_PD / 100

# ── base-case tariff ──────────────────────────────────────────────────────────
BASE_P = 22   # p/kWh charged to customer
BASE_M = 10   # £/month service fee

# ═════════════════════════════════════════════════════════════════════════════
# CORE CALCULATIONS
# ═════════════════════════════════════════════════════════════════════════════

def unit_economics(pkwh=BASE_P, pmo=BASE_M):
    """Annual per-customer revenue and profit building blocks."""
    elec_rev   = ANNUAL_KWH * pkwh / 100
    svc_fee    = pmo * 12
    cust_bill  = elec_rev + svc_fee
    total_rev  = elec_rev + svc_fee + ARB_NET + GRID_SVCS
    gross      = total_rev - SUPPLY_COST
    ebitda     = gross - TOTAL_OPEX
    ebit       = ebitda - DEPRECIATION
    saving     = COMP_BILL - cust_bill
    saving_pct = saving / COMP_BILL * 100
    return dict(
        elec_rev=elec_rev, svc_fee=svc_fee, cust_bill=cust_bill,
        total_rev=total_rev, gross=gross, ebitda=ebitda, ebit=ebit,
        saving=saving, saving_pct=saving_pct,
    )


def build_pl(ue, loan_years, model_years=10):
    """
    P&L (Income Statement) — year-by-year.
    Fixed equal-principal loan: principal = BAT_COST / loan_years each year
    until balance clears.
    """
    ebit        = ue["ebit"]
    loan_bal    = float(BAT_COST_GBP)
    annual_prin = BAT_COST_GBP / loan_years
    rows = []
    for yr in range(1, model_years + 1):
        interest = round(loan_bal * LOAN_RATE, 6)
        ebt      = ebit - interest
        tax      = max(0.0, ebt * CORP_TAX)
        net_inc  = ebt - tax
        principal = min(annual_prin, loan_bal) if loan_bal > 0.001 else 0.0
        loan_bal  = max(0.0, loan_bal - principal)
        rows.append({
            "Year":           yr,
            "Revenue (£)":    round(ue["total_rev"], 2),
            "COGS (£)":       round(SUPPLY_COST, 2),
            "Gross Profit (£)": round(ue["gross"], 2),
            "EBITDA (£)":     round(ue["ebitda"], 2),
            "Depreciation (£)": round(DEPRECIATION, 2),
            "EBIT (£)":       round(ebit, 2),
            "Interest (£)":   round(interest, 2),
            "EBT (£)":        round(ebt, 2),
            "Tax 25% (£)":    round(tax, 2),
            "Net Income (£)": round(net_inc, 2),
        })
    return pd.DataFrame(rows)


def build_cashflow(ue, loan_years, model_years=10):
    """
    Cash Flow Statement.
    OCF = Net Income + Depreciation (indirect method; interest already in NI).
    Financing CF = principal repayment (negative).
    Free Cash = OCF + Financing CF (can be negative early on).
    """
    ebit        = ue["ebit"]
    loan_bal    = float(BAT_COST_GBP)
    annual_prin = BAT_COST_GBP / loan_years
    cum_cash    = 0.0
    rows = []
    for yr in range(1, model_years + 1):
        interest  = round(loan_bal * LOAN_RATE, 6)
        ebt       = ebit - interest
        tax       = max(0.0, ebt * CORP_TAX)
        net_inc   = ebt - tax
        ocf       = net_inc + DEPRECIATION
        principal = min(annual_prin, loan_bal) if loan_bal > 0.001 else 0.0
        loan_bal  = max(0.0, loan_bal - principal)
        free_cash = ocf - principal
        cum_cash += free_cash
        rows.append({
            "Year":              yr,
            "Net Income (£)":    round(net_inc, 2),
            "Add: Depreciation (£)": round(DEPRECIATION, 2),
            "Operating CF (£)":  round(ocf, 2),
            "Principal Repaid (£)": round(principal, 2),
            "Financing CF (£)":  round(-principal, 2),
            "Free Cash (£)":     round(free_cash, 2),
            "Cumulative Cash (£)": round(cum_cash, 2),
            "CF Positive":       free_cash >= 0,
        })
    return pd.DataFrame(rows)


def build_loan_schedule(loan_years):
    """
    Loan amortisation schedule: opening balance, interest, principal, closing.
    Fixed equal-principal repayment each year.
    """
    bal         = float(BAT_COST_GBP)
    annual_prin = BAT_COST_GBP / loan_years
    rows = []
    for yr in range(1, loan_years + 1):
        open_bal   = bal
        interest   = round(open_bal * LOAN_RATE, 2)
        principal  = round(min(annual_prin, open_bal), 2)
        total_pmt  = round(interest + principal, 2)
        bal        = round(max(0.0, bal - principal), 6)
        rows.append({
            "Year":                  yr,
            "Opening Balance (£)":   round(open_bal, 2),
            "Interest Payment (£)":  interest,
            "Principal Payment (£)": principal,
            "Total Payment (£)":     total_pmt,
            "Closing Balance (£)":   round(bal, 2),
        })
    # Summary row
    df = pd.DataFrame(rows)
    totals = pd.DataFrame([{
        "Year":                  "TOTAL",
        "Opening Balance (£)":   "",
        "Interest Payment (£)":  round(df["Interest Payment (£)"].sum(), 2),
        "Principal Payment (£)": round(df["Principal Payment (£)"].sum(), 2),
        "Total Payment (£)":     round(df["Total Payment (£)"].sum(), 2),
        "Closing Balance (£)":   "",
    }])
    return pd.concat([df, totals], ignore_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# CHART
# ═════════════════════════════════════════════════════════════════════════════

COLORS = {3: "#E04B3A", 5: "#F5A623", 7: "#4A90D9"}
LABELS = {3: "3-year payoff", 5: "5-year payoff", 7: "7-year payoff"}

def plot_cashflows(cf_dfs, out_path):
    """
    Two-panel chart:
      Top: Annual Free Cash Flow for each scenario
      Bottom: Cumulative Cash for each scenario
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        "VLP Battery — Free Cashflow by Loan Payoff Scenario\n"
        f"(Battery £{BAT_COST_GBP:,.0f}, 22p/kWh + £{BASE_M}/mo, UK prices Jul 2024–Jun 2025)",
        fontsize=12, fontweight="bold", y=0.98,
    )

    years = cf_dfs[3]["Year"].values

    for ly in LOAN_SCENARIOS:
        df    = cf_dfs[ly]
        color = COLORS[ly]
        label = LABELS[ly]
        fcf   = df["Free Cash (£)"].values
        cum   = df["Cumulative Cash (£)"].values

        ax1.bar(years + (LOAN_SCENARIOS.index(ly) - 1) * 0.25,
                fcf, width=0.25, color=color, label=label, alpha=0.85)
        ax2.plot(years, cum, marker="o", color=color, label=label, linewidth=2)

    # Zero line
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax1.set_ylabel("Annual Free Cash (£)", fontsize=10)
    ax1.set_title("Annual Free Cashflow  (*** negative = principal > OCF)", fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    ax2.set_ylabel("Cumulative Cash (£)", fontsize=10)
    ax2.set_title("Cumulative Free Cash Over 10 Years", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax2.set_xlabel("Year", fontsize=10)
    ax2.set_xticks(years)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ═════════════════════════════════════════════════════════════════════════════
W = 96

def hdr(s): print(); print("=" * W); print(s); print("=" * W)
def sub(s): print(); print(f"  {'─'*4}  {s}"); print(f"  {'─'*(W-2)}")
def ln(label, val, neg=False, indent=2):
    s = "-" if neg else " "
    print(f"{'  '*indent}{label:<54}{s}£{abs(val):>9,.2f}")
def sep(indent=2): print("  "*indent + "─"*62)


def fmt_df(df):
    """Pretty-print a DataFrame, formatting float columns with commas."""
    float_cols = df.select_dtypes(include="float").columns
    fmt = {c: lambda x, _c=c: f"{x:,.2f}" if isinstance(x, float) else str(x)
           for c in float_cols}
    return df.to_string(index=False,
                        formatters={c: (lambda x, f=f: f(x, None))
                                    for c, f in fmt.items()})


# ═════════════════════════════════════════════════════════════════════════════
# REPORT
# ═════════════════════════════════════════════════════════════════════════════

def write_report(ue, pl_dfs, cf_dfs, loan_dfs, out_path):
    lines = []
    W2 = 96
    def h(s):  lines.append(""); lines.append("=" * W2); lines.append(s); lines.append("=" * W2)
    def sh(s): lines.append(""); lines.append(f"  ── {s} ──")
    def p(s=""):  lines.append(s)
    def pr(label, val, unit=""):
        lines.append(f"  {label:<52} {val}{unit}")

    h("VLP BATTERY MODEL — FIXED-TERM LOAN PAYOFF ANALYSIS")
    p(f"  Battery cost (hardware + installation): £{BAT_COST_GBP:,.2f}")
    p(f"  Loan scenarios compared: {', '.join(str(x)+'-year' for x in LOAN_SCENARIOS)}")
    p(f"  Tariff: {BASE_P}p/kWh + £{BASE_M}/month service fee")
    p(f"  Price data: UK day-ahead wholesale, Jul 2024 – Jun 2025 ({DAYS_USED} days used)")

    h("1. MODEL PARAMETERS")

    sh("Battery Hardware")
    pr("Capacity", f"{BAT_KWH} kWh")
    pr("Inverter power", f"{BAT_KW:.0f} kW")
    pr("Duration (capacity / power)", f"{BAT_KWH/BAT_KW:.2f} hrs")
    pr("Round-trip efficiency", f"{EFF_RT*100:.0f}%")
    pr("One-way efficiency (√RTE)", f"{ETA*100:.2f}%")
    pr("Charge time (full cycle at BAT_KW)", f"{T_CHARGE:.3f} hrs  ({N_FC} full hrs + {FRAC_C:.3f} partial)")
    pr("Discharge time (full cycle at BAT_KW)", f"{T_DISCHARGE:.3f} hrs  ({N_FD} full hrs + {FRAC_D:.3f} partial)")
    pr("Grid draw per charge (kWh / ETA)", f"{GRID_DRAW_KWH:.3f} kWh")
    pr("Energy delivered per discharge (kWh × ETA)", f"{DISCHARGE_KWH:.3f} kWh")

    sh("Battery Cost & Financing")
    pr("Total cost (hardware + install)", f"£{BAT_COST_GBP:,.2f}")
    pr("Economic life", f"{BAT_LIFE_YRS} years")
    pr("Annual depreciation (straight-line)", f"£{DEPRECIATION:,.2f}/yr")
    pr("Loan interest rate", f"{LOAN_RATE*100:.0f}% p.a. on outstanding balance")
    pr("Corporation tax", f"{CORP_TAX*100:.0f}%")
    for ly in LOAN_SCENARIOS:
        ann_p = BAT_COST_GBP / ly
        tot_i = sum((BAT_COST_GBP - k*(BAT_COST_GBP/ly)) * LOAN_RATE for k in range(ly))
        pr(f"  {ly}-year payoff: annual principal", f"£{ann_p:,.2f}  (total interest ≈ £{tot_i:,.2f})")

    sh("Revenue Streams")
    pr("Electricity sold to customer", f"{ANNUAL_KWH:,.1f} kWh/yr × {BASE_P}p/kWh = £{ue['elec_rev']:,.2f}/yr")
    pr("Monthly service fee", f"£{BASE_M}/mo × 12 = £{ue['svc_fee']:,.2f}/yr")
    pr("Battery arbitrage — buy (cheapest hours)", f"-£{ARB_BUY:,.2f}/yr")
    pr("Battery arbitrage — sell (dearest hours)", f"+£{ARB_SELL:,.2f}/yr")
    pr("Battery arbitrage — net", f"£{ARB_NET:,.2f}/yr")
    pr("Frequency response", f"{BAT_KW:.0f}kW × £{FR_PER_KW}/kW = £{FR_GBP:,.2f}/yr")
    pr("Capacity market", f"{BAT_KW:.0f}kW × £{CM_PER_KW}/kW × {CM_DERATING*100:.2f}% de-rate = £{CM_GBP:,.2f}/yr")
    pr("Balancing mechanism", f"ARB × {BM_WIN_RATE*100:.0f}% × {BM_UPLIFT*100:.0f}% uplift = £{BM_GBP:,.2f}/yr")
    pr("TOTAL ANNUAL REVENUE", f"£{ue['total_rev']:,.2f}/yr")

    sh("Cost Stack  (Ofgem Q1 2025 methodology)")
    pr("Wholesale electricity supply", f"£{SUPPLY_COST:,.2f}/yr")
    pr("TNUoS — transmission network", f"£{TNUOS:,.2f}/yr")
    pr("DUoS  — distribution network", f"£{DUOS:,.2f}/yr")
    pr("BSUoS — balancing services", f"£{BSUOS:,.2f}/yr")
    pr("Smart metering obligation", f"£{SMART_METER:,.2f}/yr")
    pr("Supplier opex (billing, compliance)", f"£{SUPPLIER_OPEX:,.2f}/yr")
    pr("Bad debt provision (~0.4%)", f"£{BAD_DEBT:,.2f}/yr")
    pr("Policy levy (grid_levy)", f"£{GRID_LEVY:,.2f}/yr")
    pr("Customer management (CRM)", f"£{CUST_MGMT:,.2f}/yr")
    pr("TOTAL COSTS (excl. depreciation)", f"£{SUPPLY_COST+TOTAL_OPEX:,.2f}/yr")

    sh("Unit Economics Summary")
    pr("Gross Profit", f"£{ue['gross']:,.2f}/yr")
    pr("EBITDA", f"£{ue['ebitda']:,.2f}/yr")
    pr("EBIT (after £{:.0f}/yr depreciation)".format(DEPRECIATION), f"£{ue['ebit']:,.2f}/yr")
    pr("Customer bill", f"£{ue['cust_bill']:,.2f}/yr")
    pr("Competitor bill", f"£{COMP_BILL:,.2f}/yr")
    pr("Customer saving", f"£{ue['saving']:,.2f}/yr  ({ue['saving_pct']:.1f}% cheaper)")

    h("2. P&L, CASH FLOW & LOAN SCHEDULE — EACH SCENARIO")
    for ly in LOAN_SCENARIOS:
        p()
        p(f"  {'='*70}")
        p(f"  {ly}-YEAR LOAN PAYOFF  (annual principal £{BAT_COST_GBP/ly:,.2f})")
        p(f"  {'='*70}")
        p()
        p("  PROFIT & LOSS (Income Statement)")
        p(pl_dfs[ly].to_string(index=False))
        p()
        p("  CASH FLOW STATEMENT")
        cf_disp = cf_dfs[ly].drop(columns=["CF Positive"]).copy()
        p(cf_disp.to_string(index=False))
        p()
        p("  LOAN PAYOFF SCHEDULE")
        p(loan_dfs[ly].to_string(index=False))
        neg_yrs = cf_dfs[ly][~cf_dfs[ly]["CF Positive"]]["Year"].tolist()
        paid_yr = cf_dfs[ly][cf_dfs[ly]["Free Cash (£)"].cumsum() > 0]["Year"].min()
        p()
        p(f"  Cash-flow-negative years   : {neg_yrs if neg_yrs else 'none'}")
        p(f"  Break-even (cum cash > 0)  : Year {paid_yr if not pd.isna(paid_yr) else '>10'}")
        p(f"  10-yr cumulative cash      : £{cf_dfs[ly]['Cumulative Cash (£)'].iloc[-1]:,.2f}")
        p(f"  Total interest cost        : £{pl_dfs[ly]['Interest (£)'].sum():,.2f}")

    h("3. SIDE-BY-SIDE SUMMARY")
    p()
    p(f"  {'Metric':<40} {'3-year':>12} {'5-year':>12} {'7-year':>12}")
    p(f"  {'─'*76}")
    rows_cmp = [
        ("Annual principal (£)",
         *[f"{BAT_COST_GBP/ly:,.2f}" for ly in LOAN_SCENARIOS]),
        ("Total interest paid (£)",
         *[f"{pl_dfs[ly]['Interest (£)'].sum():,.2f}" for ly in LOAN_SCENARIOS]),
        ("Year 1 free cashflow (£)",
         *[f"{cf_dfs[ly]['Free Cash (£)'].iloc[0]:,.2f}" for ly in LOAN_SCENARIOS]),
        ("CF-negative years",
         *[str(cf_dfs[ly][~cf_dfs[ly]["CF Positive"]]["Year"].tolist() or "none")
           for ly in LOAN_SCENARIOS]),
        ("10-yr cumulative cash (£)",
         *[f"{cf_dfs[ly]['Cumulative Cash (£)'].iloc[-1]:,.2f}" for ly in LOAN_SCENARIOS]),
        ("Post-payoff annual free CF (£)",
         *[f"{cf_dfs[ly][cf_dfs[ly]['Year'] > ly]['Free Cash (£)'].mean():,.2f}"
           for ly in LOAN_SCENARIOS]),
    ]
    for row in rows_cmp:
        p(f"  {row[0]:<40} {row[1]:>12} {row[2]:>12} {row[3]:>12}")

    h("4. LIMITATIONS & ASSUMPTIONS")
    limitations = [
        ("1. Single cycle per day",
         "The model assumes one full charge–discharge cycle every day throughout "
         "the year. In practice, arbitrage opportunities vary by season and market "
         "conditions. Days with flat price curves yield little or no spread."),
        ("2. Perfect foresight on prices",
         "The algorithm selects the cheapest charging hours and most expensive "
         "discharging hours using the actual historical price series — an upper "
         "bound on achievable arbitrage. Real dispatch would require forecasting "
         "and would capture a smaller spread."),
        ("3. No battery degradation",
         "Capacity and efficiency are held constant for 10 years. Real lithium "
         "batteries degrade ~2–3% per year; by year 10 usable capacity and "
         "therefore arbitrage revenue could be 20–30% lower."),
        ("4. Static tariff and competitor pricing",
         "Customer revenue is modelled at a fixed 22p/kWh + £10/mo throughout. "
         "Energy prices, network charges, and competitor tariffs will change."),
        ("5. Grid services assumed available 100%",
         "Frequency response and capacity market revenues are modelled as "
         "guaranteed annual contracts. In practice, contracts are won/lost "
         "competitively and availability may be constrained by arbitrage cycles."),
        ("6. Balancing Mechanism simplification",
         "BM revenue is estimated as a fixed percentage uplift on arbitrage net. "
         "Actual BM participation is volume-based and highly variable."),
        ("7. Fixed loan interest rate",
         "A flat 7% p.a. on the declining balance is assumed. Rates may vary; "
         "a fixed-rate term loan would have equal total payments but different "
         "interest/principal splits."),
        ("8. Corporation tax simplified",
         "Tax is applied at 25% on each year's EBT independently. Group relief, "
         "loss carry-forward, R&D credits, and capital allowances are excluded."),
        ("9. Single customer / single battery",
         "All figures are per-customer, per-battery. Portfolio effects "
         "(diversification, operational leverage) are not modelled here."),
        ("10. No working capital or capex variation",
         "The model does not include cash locked in receivables, payables, or "
         "unexpected capex such as inverter replacement or grid connection costs."),
        ("11. EUR/GBP exchange rate held constant",
         f"All EUR wholesale prices converted at {EUR_TO_GBP} throughout. "
         "Currency movements affect both supply cost and arbitrage revenue."),
        ("12. UK use profile applied",
         "Annual demand of 3,604 kWh is based on the UK use profile from "
         "use_profiles.csv, as instructed when no country-specific profile exists."),
    ]
    for title, body in limitations:
        p()
        p(f"  {title}")
        for line in textwrap.wrap(body, width=88):
            p(f"    {line}")

    p()
    p("=" * W2)
    p("END OF REPORT")
    p("=" * W2)

    text = "\n".join(lines)
    out_path.write_text(text)
    print(f"  Report saved: {out_path}")
    return text


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ue = unit_economics()

    print()
    print("=" * W)
    print(f"  VLP Battery — Quick Loan Payoff Model  (Battery £{BAT_COST_GBP:,.0f})")
    print("=" * W)
    print(f"  Arbitrage computed from {DAYS_USED} days of UK price data")
    print(f"  Net arbitrage: £{ARB_NET:,.2f}/yr   |   EBITDA: £{ue['ebitda']:,.2f}/yr   "
          f"|   EBIT: £{ue['ebit']:,.2f}/yr")
    print()

    pl_dfs   = {}
    cf_dfs   = {}
    loan_dfs = {}

    for ly in LOAN_SCENARIOS:
        pl_dfs[ly]   = build_pl(ue, ly)
        cf_dfs[ly]   = build_cashflow(ue, ly)
        loan_dfs[ly] = build_loan_schedule(ly)

        print(f"  [{ly}-yr] Year-1 FCF: £{cf_dfs[ly]['Free Cash (£)'].iloc[0]:,.2f}  "
              f"| 10-yr cum: £{cf_dfs[ly]['Cumulative Cash (£)'].iloc[-1]:,.2f}  "
              f"| Total interest: £{pl_dfs[ly]['Interest (£)'].sum():,.2f}")

        pl_dfs[ly].to_csv(OUT_DIR / f"pl_{ly}yr.csv", index=False)
        cf_dfs[ly].to_csv(OUT_DIR / f"cashflow_{ly}yr.csv", index=False)
        loan_dfs[ly].to_csv(OUT_DIR / f"loan_schedule_{ly}yr.csv", index=False)
        print(f"  Saved: pl_{ly}yr.csv  cashflow_{ly}yr.csv  loan_schedule_{ly}yr.csv")

    # Chart
    chart_path = OUT_DIR / "cashflow_chart.png"
    plot_cashflows(cf_dfs, chart_path)

    # Report
    report_path = OUT_DIR / "report.txt"
    write_report(ue, pl_dfs, cf_dfs, loan_dfs, report_path)

    print()
    print("  All files written to:", OUT_DIR)
    print("=" * W)


if __name__ == "__main__":
    main()
