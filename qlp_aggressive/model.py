"""
VLP Battery — Aggressive Customer Offering  (25–35% savings vs competitor)
===========================================================================
Battery + installation total: £2,800  (same as qlp_2800)

Three tariff scenarios, all targeting the 25–35% customer saving band:

  Scenario A  20p/kWh + £8/mo   →  cust bill £817/yr  ≈ 25% saving  (entry aggressive)
  Scenario B  19p/kWh + £6/mo   →  cust bill £757/yr  ≈ 30% saving  (mid aggressive)
  Scenario C  18p/kWh + £5/mo   →  cust bill £709/yr  ≈ 35% saving  (maximum aggressive)

For each scenario, four loan repayment periods are compared:
  3-year, 5-year, 7-year, 10-year

Competitor reference: 25p/kWh + 50p/day standing charge = £1,083.60/yr

Revenue and cost data:
  UK wholesale prices Jul 2024–Jun 2025
  Battery: 15.36 kWh / 6 kW inverter, 90% round-trip efficiency
  Arbitrage: cheapest 2.70 hrs/day to charge, dearest 2.43 hrs/day to discharge
  Grid services: frequency response, capacity market, balancing mechanism
  Cost stack: Ofgem Q1 2025 methodology (TNUoS, DUoS, BSUoS, smart metering,
              supplier opex, bad debt, policy levy, customer management)

Outputs saved to this folder (qlp_aggressive/):
  pl_{A,B,C}.csv                — P&L for all loan periods (stacked)
  cashflow_{A,B,C}.csv          — cash flow statements (stacked)
  loan_schedule_{A,B,C}.csv     — amortisation schedules (stacked)
  chart_{A,B,C}.png             — FCF + cumulative cash chart per scenario
  chart_summary.png             — side-by-side comparison across scenarios (5yr loan)
  report.txt                    — full parameter documentation + limitations
"""

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_DIR      = SCRIPT_DIR

# ── battery hardware ──────────────────────────────────────────────────────────
BAT_KWH      = 15.36
BAT_KW       = 6.0
EFF_RT       = 0.90
ETA          = EFF_RT ** 0.5          # one-way ≈ 0.9487

GRID_DRAW    = BAT_KWH / ETA          # kWh drawn from grid per full charge  ≈16.19
DISCHARGE    = BAT_KWH * ETA          # kWh delivered per full discharge     ≈14.57
T_CHARGE     = GRID_DRAW / BAT_KW     # hrs to fully charge  ≈2.699
T_DISCHARGE  = DISCHARGE / BAT_KW     # hrs to fully discharge ≈2.429
N_FC         = int(T_CHARGE)          # 2 full charge hours
FRAC_C       = T_CHARGE  - N_FC       # 0.699 partial hour
N_FD         = int(T_DISCHARGE)       # 2 full discharge hours
FRAC_D       = T_DISCHARGE - N_FD     # 0.429 partial hour

BAT_COST     = 2_800.00               # £ total (hardware + installation, updated)
BAT_LIFE     = 10                     # years
DEPRECIATION = BAT_COST / BAT_LIFE    # £280/yr straight-line

LOAN_RATE    = 0.07                   # 7% p.a. on outstanding balance
CORP_TAX     = 0.25                   # 25% corporation tax

# ── loan scenarios ────────────────────────────────────────────────────────────
LOAN_YEARS   = [3, 5, 7, 10]
MODEL_YEARS  = 12    # 12 years so 10-yr loan has 2 post-payoff years visible

# ── grid services ─────────────────────────────────────────────────────────────
FR_PER_KW    = 34       # £/kW/yr frequency response
CM_PER_KW    = 40       # £/kW/yr capacity market
CM_DERATING  = 0.2094   # de-rating: 2-hr battery class (15.36/6 = 2.56h)
BM_WIN_RATE  = 0.10     # fraction of arb cycles monetised via BM
BM_UPLIFT    = 0.05     # BM premium on arbitrage net

FR_GBP       = BAT_KW * FR_PER_KW
CM_GBP       = BAT_KW * CM_PER_KW * CM_DERATING

# ── arbitrage ─────────────────────────────────────────────────────────────────
EUR_TO_GBP   = 0.85

def compute_arbitrage():
    price_file = PROJECT_ROOT / "data" / "prices" / "United Kingdom.csv"
    df = pd.read_csv(price_file)
    df["dt"]   = pd.to_datetime(df["Datetime (UTC)"])
    df         = df[(df["dt"] >= "2024-07-01") & (df["dt"] < "2025-07-01")].copy()
    df["date"] = df["dt"].dt.date
    df["p_gbp"] = df["Price (EUR/MWhe)"] * EUR_TO_GBP / 1000
    buy = sell = 0.0
    ndays = 0
    for _, grp in df.groupby("date"):
        p = np.sort(grp["p_gbp"].values)
        if len(p) < 24:
            continue
        pd_ = p[::-1]
        buy  += BAT_KW * (p[:N_FC].sum()  + FRAC_C * p[N_FC])
        sell += BAT_KW * (pd_[:N_FD].sum() + FRAC_D * pd_[N_FD])
        ndays += 1
    return buy, sell, sell - buy, ndays

ARB_BUY, ARB_SELL, ARB_NET, DAYS_USED = compute_arbitrage()
BM_GBP    = ARB_NET * BM_WIN_RATE * BM_UPLIFT
GRID_SVCS = FR_GBP + CM_GBP + BM_GBP

# ── electricity cost stack  (Ofgem Q1 2025) ──────────────────────────────────
ANNUAL_KWH        = 3_604.4
DAYS              = 365
WHOLESALE_EUR_MWH = 99.84
WHOLESALE_P_KWH   = WHOLESALE_EUR_MWH * EUR_TO_GBP / 1000

SUPPLY_COST   = ANNUAL_KWH * WHOLESALE_P_KWH

TNUOS         = ANNUAL_KWH * 0.0087 + DAYS * 0.0359
DUOS          = ANNUAL_KWH * 0.0363 + DAYS * 0.0731
BSUOS         = ANNUAL_KWH * 0.0075
SMART_METER   = ANNUAL_KWH * 0.0054 + DAYS * 0.0356
SUPPLIER_OPEX = ANNUAL_KWH * 0.0137 + DAYS * 0.1396 - 10
BAD_DEBT      = ANNUAL_KWH * 0.0041
GRID_LEVY     = 95.0
CUST_MGMT     = 10.0

TOTAL_OPEX = (GRID_LEVY + CUST_MGMT + TNUOS + DUOS + BSUOS
              + SMART_METER + SUPPLIER_OPEX + BAD_DEBT)

# ── competitor ────────────────────────────────────────────────────────────────
COMP_PKW   = 25
COMP_PD    = 50
COMP_BILL  = ANNUAL_KWH * COMP_PKW / 100 + DAYS * COMP_PD / 100   # £1,083.60

# ── aggressive tariff scenarios ───────────────────────────────────────────────
# (label, p/kWh, £/month, approx_saving_pct)
SCENARIOS = {
    "A": dict(label="Scenario A — Entry aggressive",
              pkwh=20, pmo=8,  saving_target="~25%"),
    "B": dict(label="Scenario B — Mid aggressive",
              pkwh=19, pmo=6,  saving_target="~30%"),
    "C": dict(label="Scenario C — Maximum aggressive",
              pkwh=18, pmo=5,  saving_target="~35%"),
}

# ── chart colours ─────────────────────────────────────────────────────────────
LOAN_COLORS = {3: "#E04B3A", 5: "#F5A623", 7: "#4A90D9", 10: "#27AE60"}
SC_COLORS   = {"A": "#4A90D9", "B": "#F5A623", "C": "#E04B3A"}

# ═════════════════════════════════════════════════════════════════════════════
# CORE CALCULATIONS
# ═════════════════════════════════════════════════════════════════════════════

def unit_economics(pkwh, pmo):
    elec_rev  = ANNUAL_KWH * pkwh / 100
    svc_fee   = pmo * 12
    cust_bill = elec_rev + svc_fee
    total_rev = cust_bill + ARB_NET + GRID_SVCS
    gross     = total_rev - SUPPLY_COST
    ebitda    = gross - TOTAL_OPEX
    ebit      = ebitda - DEPRECIATION
    saving    = COMP_BILL - cust_bill
    saving_pct = saving / COMP_BILL * 100
    return dict(
        elec_rev=elec_rev, svc_fee=svc_fee, cust_bill=cust_bill,
        total_rev=total_rev, gross=gross, ebitda=ebitda, ebit=ebit,
        saving=saving, saving_pct=saving_pct,
    )


def build_pl(ue, loan_years):
    ebit     = ue["ebit"]
    loan_bal = float(BAT_COST)
    ann_prin = BAT_COST / loan_years
    rows = []
    for yr in range(1, MODEL_YEARS + 1):
        interest  = loan_bal * LOAN_RATE
        ebt       = ebit - interest
        tax       = max(0.0, ebt * CORP_TAX)
        net_inc   = ebt - tax
        principal = min(ann_prin, loan_bal) if loan_bal > 0.001 else 0.0
        loan_bal  = max(0.0, loan_bal - principal)
        rows.append({
            "Year":             yr,
            "Loan Yrs":         loan_years,
            "Revenue (£)":      round(ue["total_rev"], 2),
            "COGS (£)":         round(SUPPLY_COST, 2),
            "Gross Profit (£)": round(ue["gross"], 2),
            "EBITDA (£)":       round(ue["ebitda"], 2),
            "Depreciation (£)": round(DEPRECIATION, 2),
            "EBIT (£)":         round(ebit, 2),
            "Interest (£)":     round(interest, 2),
            "EBT (£)":          round(ebt, 2),
            "Tax 25% (£)":      round(tax, 2),
            "Net Income (£)":   round(net_inc, 2),
        })
    return pd.DataFrame(rows)


def build_cashflow(ue, loan_years):
    ebit     = ue["ebit"]
    loan_bal = float(BAT_COST)
    ann_prin = BAT_COST / loan_years
    cum_cash = 0.0
    rows = []
    for yr in range(1, MODEL_YEARS + 1):
        interest  = loan_bal * LOAN_RATE
        ebt       = ebit - interest
        tax       = max(0.0, ebt * CORP_TAX)
        net_inc   = ebt - tax
        ocf       = net_inc + DEPRECIATION
        principal = min(ann_prin, loan_bal) if loan_bal > 0.001 else 0.0
        loan_bal  = max(0.0, loan_bal - principal)
        free_cash = ocf - principal
        cum_cash += free_cash
        rows.append({
            "Year":                  yr,
            "Loan Yrs":              loan_years,
            "Net Income (£)":        round(net_inc, 2),
            "Add: Depreciation (£)": round(DEPRECIATION, 2),
            "Operating CF (£)":      round(ocf, 2),
            "Principal Repaid (£)":  round(principal, 2),
            "Financing CF (£)":      round(-principal, 2),
            "Free Cash (£)":         round(free_cash, 2),
            "Cumulative Cash (£)":   round(cum_cash, 2),
            "CF Positive":           free_cash >= 0,
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
# CHARTS
# ═════════════════════════════════════════════════════════════════════════════

def plot_scenario_chart(sc_key, sc, cf_dfs, out_path):
    """
    Two-panel chart for one scenario:
      Top:    Annual Free Cash Flow — bars grouped by year, series = loan period
      Bottom: Cumulative Cash — lines, series = loan period
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    tariff_str = f"{sc['pkwh']}p/kWh + £{sc['pmo']}/mo"
    bill_str   = f"£{ANNUAL_KWH*sc['pkwh']/100 + sc['pmo']*12:,.0f}/yr"

    fig.suptitle(
        f"VLP Battery — {sc['label']}\n"
        f"Tariff: {tariff_str}  |  Customer bill: {bill_str}  |  "
        f"Saving: {sc['saving_target']} vs competitor\n"
        f"Battery: £{BAT_COST:,.0f}, UK prices Jul 2024–Jun 2025",
        fontsize=11, fontweight="bold", y=0.99,
    )

    years    = cf_dfs[3]["Year"].values
    n_loans  = len(LOAN_YEARS)
    bar_w    = 0.18
    offsets  = np.linspace(-(n_loans-1)/2, (n_loans-1)/2, n_loans) * bar_w

    for i, ly in enumerate(LOAN_YEARS):
        df    = cf_dfs[ly]
        color = LOAN_COLORS[ly]
        fcf   = df["Free Cash (£)"].values
        cum   = df["Cumulative Cash (£)"].values

        ax1.bar(years + offsets[i], fcf,
                width=bar_w, color=color, label=f"{ly}-yr loan", alpha=0.85)
        ax2.plot(years, cum, marker="o", markersize=4,
                 color=color, label=f"{ly}-yr loan", linewidth=2)

    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax1.set_ylabel("Annual Free Cash (£)", fontsize=10)
    ax1.set_title("Annual Free Cashflow  (negative bars = principal > OCF in that year)",
                  fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax1.legend(fontsize=9, ncol=4)
    ax1.grid(axis="y", alpha=0.3)

    ax2.set_ylabel("Cumulative Cash (£)", fontsize=10)
    ax2.set_title("Cumulative Free Cash", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax2.set_xlabel("Year", fontsize=10)
    ax2.set_xticks(years)
    ax2.legend(fontsize=9, ncol=4)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_chart(all_cf, out_path):
    """
    Summary chart: compare all 3 scenarios using the 5-year loan.
    Top: annual FCF  |  Bottom: cumulative cash
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(
        "VLP Battery — Scenario Comparison at 5-Year Loan Payoff\n"
        "Aggressive tariff options vs £1,083 competitor bill (UK, Jul 2024–Jun 2025)",
        fontsize=11, fontweight="bold",
    )

    years   = all_cf["A"][5]["Year"].values
    bar_w   = 0.25
    offsets = [-0.25, 0.0, 0.25]

    for i, sc_key in enumerate(["A", "B", "C"]):
        sc    = SCENARIOS[sc_key]
        df    = all_cf[sc_key][5]
        color = SC_COLORS[sc_key]
        tariff = f"{sc['pkwh']}p/kWh+£{sc['pmo']}/mo ({sc['saving_target']} saving)"
        fcf   = df["Free Cash (£)"].values
        cum   = df["Cumulative Cash (£)"].values

        ax1.bar(years + offsets[i], fcf,
                width=bar_w, color=color, label=tariff, alpha=0.85)
        ax2.plot(years, cum, marker="o", markersize=4,
                 color=color, label=tariff, linewidth=2)

    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax1.set_ylabel("Annual Free Cash (£)", fontsize=10)
    ax1.set_title("Annual Free Cashflow — 5-year loan (lower tariff = lower early FCF)",
                  fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    ax2.set_ylabel("Cumulative Cash (£)", fontsize=10)
    ax2.set_title("Cumulative Free Cash", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax2.set_xlabel("Year", fontsize=10)
    ax2.set_xticks(years)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# REPORT
# ═════════════════════════════════════════════════════════════════════════════

def write_report(all_ue, all_pl, all_cf, all_loan, out_path):
    W2 = 96
    lines = []

    def h(s):
        lines.extend(["", "=" * W2, s, "=" * W2])
    def sh(s):
        lines.extend(["", f"  ── {s} ──"])
    def p(s=""):
        lines.append(s)
    def pr(label, val):
        lines.append(f"  {label:<54} {val}")

    # ── title ──
    h("VLP BATTERY — AGGRESSIVE CUSTOMER OFFERING  (25–35% SAVINGS)")
    p(f"  Battery cost (hardware + installation): £{BAT_COST:,.2f}")
    p(f"  Loan scenarios compared: {', '.join(str(x)+'-year' for x in LOAN_YEARS)}")
    p(f"  Model horizon: {MODEL_YEARS} years")
    p(f"  Price data: UK day-ahead wholesale Jul 2024–Jun 2025 ({DAYS_USED} days)")
    p()
    p("  SCENARIO OVERVIEW")
    p(f"  {'Scenario':<10} {'Tariff':<22} {'Cust Bill':>10} {'Saving £':>9} {'Saving%':>8} "
      f"{'EBITDA':>9} {'EBIT':>8}")
    p("  " + "─" * 78)
    for sk, sc in SCENARIOS.items():
        ue = all_ue[sk]
        p(f"  {sc['label'][:8]:<10} {sc['pkwh']}p/kWh + £{sc['pmo']:>2}/mo"
          f"       £{ue['cust_bill']:>8,.2f}  £{ue['saving']:>7,.2f}  "
          f"{ue['saving_pct']:>6.1f}%  £{ue['ebitda']:>7,.2f}  £{ue['ebit']:>6,.2f}")
    p()
    p(f"  Competitor reference: {COMP_PKW}p/kWh + {COMP_PD}p/day = £{COMP_BILL:,.2f}/yr")

    # ── parameters ──
    h("1. MODEL PARAMETERS")

    sh("Battery Hardware")
    pr("Capacity", f"{BAT_KWH} kWh")
    pr("Inverter power", f"{BAT_KW:.0f} kW")
    pr("Duration (capacity / power)", f"{BAT_KWH/BAT_KW:.2f} hrs")
    pr("Round-trip efficiency", f"{EFF_RT*100:.0f}%")
    pr("One-way efficiency (√RTE)", f"{ETA*100:.2f}%")
    pr("Charge time (full cycle, inverter-constrained)",
       f"{T_CHARGE:.3f} hrs  ({N_FC} full + {FRAC_C:.3f} partial)")
    pr("Discharge time (full cycle, inverter-constrained)",
       f"{T_DISCHARGE:.3f} hrs  ({N_FD} full + {FRAC_D:.3f} partial)")
    pr("Grid draw per charge", f"{GRID_DRAW:.3f} kWh  (= BAT_KWH / ETA)")
    pr("Energy delivered per discharge", f"{DISCHARGE:.3f} kWh  (= BAT_KWH × ETA)")

    sh("Battery Cost & Financing")
    pr("Total cost (hardware + installation)", f"£{BAT_COST:,.2f}")
    pr("Economic life", f"{BAT_LIFE} years")
    pr("Annual depreciation (straight-line)", f"£{DEPRECIATION:,.2f}/yr")
    pr("Loan interest rate", f"{LOAN_RATE*100:.0f}% p.a. on outstanding balance")
    pr("Corporation tax", f"{CORP_TAX*100:.0f}%")
    p()
    p(f"  {'Loan term':<14} {'Ann. principal':>15} {'Total interest':>15} "
      f"{'Total repaid':>14}")
    p("  " + "─" * 60)
    for ly in LOAN_YEARS:
        ann_p = BAT_COST / ly
        tot_i = sum((BAT_COST - k * ann_p) * LOAN_RATE for k in range(ly))
        p(f"  {ly}-year loan    £{ann_p:>13,.2f}  £{tot_i:>13,.2f}  "
          f"  £{BAT_COST + tot_i:>12,.2f}")

    sh("Arbitrage Strategy  (UK day-ahead prices, Jul 2024–Jun 2025)")
    pr("Data source", "United Kingdom.csv (Datetime UTC, Price EUR/MWhe)")
    pr("Period", "2024-07-01 to 2025-06-30  (latest full year)")
    pr("Days with complete 24hr data", str(DAYS_USED))
    pr("EUR/GBP conversion rate", str(EUR_TO_GBP))
    pr("Average UK wholesale (EUR/MWh)", f"{WHOLESALE_EUR_MWH}")
    pr("Average UK wholesale (p/kWh)", f"{WHOLESALE_P_KWH*100:.4f}")
    pr("Strategy", f"Buy cheapest {T_CHARGE:.2f} hrs/day, sell dearest {T_DISCHARGE:.2f} hrs/day")
    pr("Annual charge cost", f"£{ARB_BUY:,.2f}/yr")
    pr("Annual sell revenue", f"£{ARB_SELL:,.2f}/yr")
    pr("Net arbitrage revenue", f"£{ARB_NET:,.2f}/yr")

    sh("Grid Services Revenue")
    pr("Frequency response",
       f"{BAT_KW:.0f}kW × £{FR_PER_KW}/kW = £{FR_GBP:,.2f}/yr")
    pr("Capacity market (de-rated 2-hr class)",
       f"{BAT_KW:.0f}kW × £{CM_PER_KW}/kW × {CM_DERATING*100:.2f}% = £{CM_GBP:,.2f}/yr")
    pr("Balancing mechanism",
       f"£{ARB_NET:,.2f} × {BM_WIN_RATE*100:.0f}% win × {BM_UPLIFT*100:.0f}% uplift = £{BM_GBP:,.2f}/yr")
    pr("Total grid services", f"£{GRID_SVCS:,.2f}/yr")

    sh("Supply Cost Stack  (Ofgem Q1 2025 methodology, per customer/yr)")
    items = [
        ("Wholesale electricity supply",    SUPPLY_COST),
        ("TNUoS — transmission network",    TNUOS),
        ("DUoS  — distribution network",    DUOS),
        ("BSUoS — balancing services",      BSUOS),
        ("Smart metering obligation",       SMART_METER),
        ("Supplier opex (billing/compl.)",  SUPPLIER_OPEX),
        ("Bad debt provision (~0.4%)",      BAD_DEBT),
        ("Policy levy (grid_levy)",         GRID_LEVY),
        ("Customer management (CRM)",       CUST_MGMT),
    ]
    p()
    p(f"  {'Cost line':<42} {'£/yr':>10}")
    p("  " + "─" * 54)
    for name, val in items:
        p(f"  {name:<42} £{val:>9.2f}")
    p("  " + "─" * 54)
    p(f"  {'TOTAL COSTS (excl. depreciation)':<42} £{SUPPLY_COST+TOTAL_OPEX:>9.2f}")

    # ── scenario financials ──
    h("2. DETAILED FINANCIALS BY SCENARIO")

    for sk in ["A", "B", "C"]:
        sc = SCENARIOS[sk]
        ue = all_ue[sk]
        p()
        p(f"  {'━'*90}")
        p(f"  {sc['label'].upper()}")
        p(f"  Tariff: {sc['pkwh']}p/kWh + £{sc['pmo']}/month")
        p(f"  Customer bill: £{ue['cust_bill']:,.2f}/yr  |  "
          f"Saving vs competitor: £{ue['saving']:,.2f} ({ue['saving_pct']:.1f}%)")
        p(f"  Revenue: £{ue['total_rev']:,.2f}  |  EBITDA: £{ue['ebitda']:,.2f}  "
          f"|  EBIT: £{ue['ebit']:,.2f}")
        p(f"  {'━'*90}")

        for ly in LOAN_YEARS:
            p()
            p(f"    ── {ly}-YEAR LOAN PAYOFF  (annual principal £{BAT_COST/ly:,.2f}) ──")
            p()
            p("    PROFIT & LOSS")
            pl_disp = all_pl[sk][ly].drop(columns=["Loan Yrs"])
            p(pl_disp.to_string(index=False))
            p()
            p("    CASH FLOW STATEMENT")
            cf_disp = all_cf[sk][ly].drop(columns=["Loan Yrs", "CF Positive"])
            p(cf_disp.to_string(index=False))
            p()
            p("    LOAN AMORTISATION SCHEDULE")
            ls_disp = all_loan[sk][ly].drop(columns=["Loan Yrs"])
            p(ls_disp.to_string(index=False))
            neg_yrs   = all_cf[sk][ly][~all_cf[sk][ly]["CF Positive"]]["Year"].tolist()
            cum_final = all_cf[sk][ly]["Cumulative Cash (£)"].iloc[-1]
            tot_int   = all_pl[sk][ly]["Interest (£)"].sum()
            p()
            p(f"    Cashflow-negative years : {neg_yrs if neg_yrs else 'none'}")
            p(f"    Total interest cost     : £{tot_int:,.2f}")
            p(f"    {MODEL_YEARS}-yr cumulative cash   : £{cum_final:,.2f}")

    # ── cross-scenario summary ──
    h("3. CROSS-SCENARIO SUMMARY TABLES")

    sh("Year-1 Free Cashflow (£) — how much cash is consumed or generated in year 1")
    p()
    p(f"  {'Scenario':<36} " + "  ".join(f"{ly}-yr loan" for ly in LOAN_YEARS))
    p("  " + "─" * 72)
    for sk in ["A", "B", "C"]:
        sc  = SCENARIOS[sk]
        row = f"  {sc['pkwh']}p+£{sc['pmo']}/mo ({sc['saving_target']} saving)     "
        for ly in LOAN_YEARS:
            v = all_cf[sk][ly]["Free Cash (£)"].iloc[0]
            row += f"  {v:>9,.0f}"
        p(row)

    sh("Total Interest Paid (£) over full loan term")
    p()
    p(f"  {'Scenario':<36} " + "  ".join(f"{ly}-yr loan" for ly in LOAN_YEARS))
    p("  " + "─" * 72)
    for sk in ["A", "B", "C"]:
        sc  = SCENARIOS[sk]
        row = f"  {sc['pkwh']}p+£{sc['pmo']}/mo ({sc['saving_target']} saving)     "
        for ly in LOAN_YEARS:
            v = all_pl[sk][ly]["Interest (£)"].sum()
            row += f"  {v:>9,.0f}"
        p(row)

    sh(f"{MODEL_YEARS}-Year Cumulative Free Cash (£)")
    p()
    p(f"  {'Scenario':<36} " + "  ".join(f"{ly}-yr loan" for ly in LOAN_YEARS))
    p("  " + "─" * 72)
    for sk in ["A", "B", "C"]:
        sc  = SCENARIOS[sk]
        row = f"  {sc['pkwh']}p+£{sc['pmo']}/mo ({sc['saving_target']} saving)     "
        for ly in LOAN_YEARS:
            v = all_cf[sk][ly]["Cumulative Cash (£)"].iloc[-1]
            row += f"  {v:>9,.0f}"
        p(row)

    sh("Cashflow-Negative Years (where fixed principal repayment exceeds OCF)")
    p()
    for sk in ["A", "B", "C"]:
        sc = SCENARIOS[sk]
        p(f"  {sc['pkwh']}p+£{sc['pmo']}/mo ({sc['saving_target']} saving)")
        for ly in LOAN_YEARS:
            neg = all_cf[sk][ly][~all_cf[sk][ly]["CF Positive"]]["Year"].tolist()
            p(f"    {ly}-yr loan: {neg if neg else 'none — cashflow positive throughout'}")

    # ── limitations ──
    h("4. MODEL LIMITATIONS & ASSUMPTIONS")

    limitations = [
        ("1. Single daily cycle",
         "The model assumes one complete charge–discharge cycle every day of the "
         "year. Real dispatch depends on price spread availability. Days with "
         "near-flat price curves produce little or no arbitrage value."),
        ("2. Perfect foresight on prices",
         "Arbitrage revenue is computed with hindsight — selecting the cheapest "
         "charging hours and most expensive discharging hours from the full day's "
         "actual price series. A real dispatch algorithm operating in advance "
         "would achieve a lower spread, potentially 10–30% below this estimate."),
        ("3. No battery degradation",
         "Capacity and efficiency are held constant across all 10–12 modelled "
         "years. Real lithium batteries degrade ~2–3% per year; by year 10 "
         "arbitrage revenue and grid services capacity could be materially lower. "
         "A degradation-adjusted model would reduce revenue in later years."),
        ("4. Static tariff throughout",
         "Customer revenue is fixed at the modelled p/kWh + monthly fee for the "
         "entire period. In practice tariffs are reviewed annually against "
         "changing wholesale prices, network charges, and competitor rates."),
        ("5. Grid services assumed contractually secured",
         "Frequency response and capacity market revenues are modelled as certain "
         "annual income. In practice both require competitive tender wins. Capacity "
         "market payments are also subject to interconnector and demand-side response "
         "competition, and may not be available at current rates."),
        ("6. Balancing Mechanism estimated, not modelled",
         "BM revenue is a simple percentage uplift on arbitrage net rather than "
         "an explicit dispatch model. Actual BM participation is volume-based, "
         "gate-closure dependent, and highly variable year-to-year."),
        ("7. Fixed equal-principal loan repayment",
         "The model uses equal annual principal instalments. In practice a term "
         "loan would typically use equal total payments (annuity / amortising "
         "structure), meaning higher early payments and lower later ones, giving "
         "slightly different interest and cashflow profiles."),
        ("8. Interest rate held constant",
         "Loan interest is modelled at a flat 7% p.a. on declining balance. "
         "This is reasonable for a fixed-rate facility but does not capture "
         "floating-rate risk or refinancing at maturity."),
        ("9. Corporation tax simplified",
         "Tax at 25% on each year's standalone EBT. Group relief, prior-year "
         "losses, R&D credits, Full Expensing capital allowances, and ring-fencing "
         "are all excluded. In a real entity the effective tax rate is likely "
         "lower in early years due to capital allowances."),
        ("10. No working capital",
         "The cash flow model omits receivables, payables, and cash reserves. "
         "A real business would require operating capital, increasing the "
         "effective funding need above the battery installation cost alone."),
        ("11. Single customer, single battery",
         "All results are per-customer. Portfolio effects — load diversification, "
         "operational leverage, shared grid service contracts — are not captured "
         "and would improve unit economics at scale."),
        ("12. EUR/GBP exchange rate fixed",
         f"Wholesale prices denominated in EUR are converted at a fixed {EUR_TO_GBP}. "
         "GBP appreciation relative to EUR would reduce both supply cost and "
         "arbitrage revenue; depreciation would increase both, with uncertain net "
         "effect depending on pass-through arrangements."),
        ("13. UK use profile applied universally",
         "Annual demand of 3,604 kWh is based on the UK household use profile "
         "(use_profiles.csv). Customer demand shape affects the timing of "
         "battery use relative to customer consumption but not the modelled "
         "annual totals."),
        ("14. Aggressive pricing and viability",
         "Scenario C (18p/kWh + £5/mo) gives EBIT of ~£239/yr, representing a "
         "thin margin. Any adverse movement in wholesale prices, network charges, "
         "or lower-than-expected arbitrage spread could make the unit economics "
         "loss-making. Scenario A provides a more conservative cushion."),
    ]

    for title, body in limitations:
        p()
        p(f"  {title}")
        for line in textwrap.wrap(body, width=90):
            p(f"    {line}")

    p()
    p("=" * W2)
    p("END OF REPORT")
    p("=" * W2)

    out_path.write_text("\n".join(lines))
    print(f"  Report saved: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 90)
    print(f"  VLP Battery — Aggressive Offering  (Battery £{BAT_COST:,.0f})")
    print("=" * 90)
    print(f"  Arbitrage: {DAYS_USED} days UK price data  "
          f"| Net: £{ARB_NET:,.2f}/yr  | Grid svcs: £{GRID_SVCS:,.2f}/yr")
    print()

    all_ue   = {}
    all_pl   = {}
    all_cf   = {}
    all_loan = {}

    for sk, sc in SCENARIOS.items():
        ue = unit_economics(sc["pkwh"], sc["pmo"])
        all_ue[sk]   = ue
        all_pl[sk]   = {}
        all_cf[sk]   = {}
        all_loan[sk] = {}

        print(f"  Scenario {sk}: {sc['pkwh']}p/kWh + £{sc['pmo']}/mo  "
              f"| Bill £{ue['cust_bill']:,.0f}  Saving {ue['saving_pct']:.1f}%  "
              f"| EBIT £{ue['ebit']:,.2f}")

        for ly in LOAN_YEARS:
            all_pl[sk][ly]   = build_pl(ue, ly)
            all_cf[sk][ly]   = build_cashflow(ue, ly)
            all_loan[sk][ly] = build_loan_schedule(ly)
            yr1_fcf  = all_cf[sk][ly]["Free Cash (£)"].iloc[0]
            cum_last = all_cf[sk][ly]["Cumulative Cash (£)"].iloc[-1]
            neg_yrs  = all_cf[sk][ly][~all_cf[sk][ly]["CF Positive"]]["Year"].tolist()
            print(f"    {ly}-yr loan: Yr1 FCF £{yr1_fcf:>7,.0f}  "
                  f"| {MODEL_YEARS}-yr cum £{cum_last:>7,.0f}  "
                  f"| CF-neg yrs {neg_yrs if neg_yrs else 'none'}")

        # Stack all loan periods into one CSV per scenario
        pl_stack   = pd.concat(all_pl[sk].values(), ignore_index=True)
        cf_stack   = pd.concat(all_cf[sk].values(), ignore_index=True)
        loan_stack = pd.concat(all_loan[sk].values(), ignore_index=True)

        pl_stack.to_csv(OUT_DIR / f"pl_{sk}.csv", index=False)
        cf_stack.to_csv(OUT_DIR / f"cashflow_{sk}.csv", index=False)
        loan_stack.to_csv(OUT_DIR / f"loan_schedule_{sk}.csv", index=False)
        print(f"    → Saved pl_{sk}.csv  cashflow_{sk}.csv  loan_schedule_{sk}.csv")
        print()

    # Charts
    for sk, sc in SCENARIOS.items():
        chart_path = OUT_DIR / f"chart_{sk}.png"
        plot_scenario_chart(sk, sc, all_cf[sk], chart_path)
        print(f"  Chart saved: chart_{sk}.png")

    summary_path = OUT_DIR / "chart_summary.png"
    plot_summary_chart(all_cf, summary_path)
    print(f"  Chart saved: chart_summary.png")

    # Report
    write_report(all_ue, all_pl, all_cf, all_loan, OUT_DIR / "report.txt")

    print()
    print(f"  All files written to: {OUT_DIR}")
    print("=" * 90)


if __name__ == "__main__":
    main()
