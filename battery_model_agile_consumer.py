#!/usr/bin/env python3
"""
Battery Storage Optimization Model - Agile Consumer + VPP Joint Optimization

This model optimizes the battery for TOTAL income across all revenue streams:
1. Consumer savings (Agile retail spread - charge low, discharge saves consumer high Agile rate)
2. Wholesale arbitrage (wholesale spread - charge low, sell high to grid)
3. Frequency Response (FR) - capacity commitment
4. Capacity Market (CM) - annual de-rated capacity payment
5. Balancing Mechanism (BM) - multiplier on wholesale arbitrage

The optimization jointly decides hour-by-hour:
- When to charge (at low prices)
- When to discharge to consumer (saves them Agile rate)
- When to discharge to grid (earns wholesale rate)
- When to commit to FR (can't charge/discharge)

Uses:
- Agile retail rates from agilebuddy.uk for consumer savings
- Wholesale rates from United Kingdom.csv for VPP arbitrage
- UK consumption profile from use_profiles.csv

================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import re

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    print("WARNING: PuLP not installed. Using simplified optimization.")

# Configuration
DATA_DIR = Path(__file__).parent
AGILE_CACHE_FILE = DATA_DIR / "agile_rates_2025_cache.csv"

# Battery parameters (can be overridden)
DEFAULT_BATTERY_KWH = 10.0
DEFAULT_BATTERY_KW = 5.0
DEFAULT_EFFICIENCY = 0.9

# EUR to GBP conversion
EUR_TO_GBP = 0.85


def load_model_inputs():
    """Load model parameters from CSV."""
    df = pd.read_csv(DATA_DIR / "Week plan batteries sprint 1 - model_input.csv")
    df = df.set_index("input")

    p = {}
    col = "UK"

    for idx in df.index.unique():
        val = df.loc[idx, col]
        if isinstance(val, pd.Series):
            for v in val.values[::-1]:
                if not (pd.isna(v) or v == ""):
                    val = v
                    break
            else:
                val = val.values[-1]

        if pd.isna(val) or val == "":
            p[idx] = 0.0
        elif isinstance(val, str) and "%" in val:
            p[idx] = float(val.replace("%", "")) / 100
        else:
            try:
                p[idx] = float(val)
            except:
                p[idx] = 0.0

    return p


def load_use_profiles():
    """Load hourly consumption profiles for UK."""
    df = pd.read_csv(DATA_DIR / "Week plan batteries sprint 1 - use_profiles.csv")
    uk_profile = df[df["Nation"] == "UK"].set_index("hour")["consumption_kwh"]
    return uk_profile.to_dict()


def scrape_agile_day(date_str):
    """Scrape Agile rates for a single day from agilebuddy.uk."""
    url = f"https://agilebuddy.uk/historic/agile/{date_str}"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')
        tables = soup.find_all('table')
        if len(tables) < 2:
            return None

        rates = []
        price_table = tables[1]

        for row in price_table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 2:
                time_text = cells[0].get_text().strip()
                price_text = cells[1].get_text().strip()

                if re.match(r'\d{2}:\d{2}', time_text):
                    price_match = re.search(r'([-]?\d+\.?\d*)', price_text)
                    if price_match:
                        try:
                            rate = float(price_match.group(1))
                            rates.append({'time': time_text, 'rate': rate})
                        except ValueError:
                            continue

        rates.sort(key=lambda x: x['time'])
        return rates if len(rates) >= 24 else None

    except Exception:
        return None


def scrape_agile_year(year=2025, use_cache=True, days_limit=None):
    """Scrape all Agile rates for a year."""
    if use_cache and AGILE_CACHE_FILE.exists():
        print(f"Loading cached Agile data from {AGILE_CACHE_FILE}")
        return pd.read_csv(AGILE_CACHE_FILE, parse_dates=['date'])

    print(f"Scraping Agile rates for {year}...")
    if days_limit:
        print(f"Limited to first {days_limit} days...")
    else:
        print("This may take a while (365 requests with delays)...")

    all_data = []
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    current_date = start_date
    days_scraped = 0
    total_days = 0

    while current_date <= end_date:
        if days_limit and total_days >= days_limit:
            break
        total_days += 1
        date_str = current_date.strftime("%Y/%m/%d")

        if days_scraped % 30 == 0:
            print(f"  Scraping {current_date.strftime('%B %Y')}...")

        rates = scrape_agile_day(date_str)

        if rates:
            for r in rates:
                hour = int(r['time'].split(':')[0])
                minute = int(r['time'].split(':')[1])
                all_data.append({
                    'date': current_date.date(),
                    'hour': hour,
                    'half_hour': 0 if minute == 0 else 1,
                    'time': r['time'],
                    'rate_p_kwh': r['rate']
                })
            days_scraped += 1

        time.sleep(0.5)
        current_date += timedelta(days=1)

    print(f"Scraped {days_scraped} days successfully")

    if not all_data:
        return None

    df = pd.DataFrame(all_data)
    df.to_csv(AGILE_CACHE_FILE, index=False)
    print(f"Saved cache to {AGILE_CACHE_FILE}")

    return df


def get_hourly_agile_rates(agile_df):
    """Convert 30-min Agile rates to hourly averages."""
    hourly = agile_df.groupby(['date', 'hour'])['rate_p_kwh'].mean().reset_index()
    # Convert date to datetime.date for consistent matching
    hourly['date'] = pd.to_datetime(hourly['date']).dt.date
    return hourly


def load_wholesale_prices(year=2025):
    """Load wholesale market prices from United Kingdom.csv."""
    df = pd.read_csv(DATA_DIR / "United Kingdom.csv")
    df["datetime"] = pd.to_datetime(df["Datetime (UTC)"])
    df["price_eur_mwh"] = df["Price (EUR/MWhe)"]
    df = df[df["datetime"].dt.year == year]
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    return df[["datetime", "date", "hour", "price_eur_mwh"]]


# =============================================================================
# DE-RATING AND CM/BM CALCULATIONS
# =============================================================================

UK_CM_DERATING_FACTORS = {
    1: 0.1047,
    2: 0.2094,
    4: 0.37,
    8: 0.92,
}


def get_battery_duration_hours(battery_kwh, battery_kw):
    """Calculate battery duration for de-rating lookup."""
    if battery_kw <= 0:
        return 1
    duration = battery_kwh / battery_kw
    if duration < 1.5:
        return 1
    elif duration < 3:
        return 2
    elif duration < 6:
        return 4
    else:
        return 8


def get_derating_factor(battery_kwh, battery_kw):
    """Get the appropriate de-rating factor."""
    duration = get_battery_duration_hours(battery_kwh, battery_kw)
    return UK_CM_DERATING_FACTORS.get(duration, 0.2)


def calculate_capacity_market(params, battery_kwh, battery_kw):
    """Calculate annual CM income."""
    cm_rate = params.get("Capacity market /kW/year", 0)
    if cm_rate == 0:
        return 0
    derating = get_derating_factor(battery_kwh, battery_kw)
    return cm_rate * derating * battery_kw


def calculate_bm_multiplier(params):
    """Calculate BM multiplier for arbitrage."""
    win_rate = params.get("Balancing mechanism (BM) avg win rate", 0)
    uplift = params.get("Avg £/kwh uplift for BM", 0)
    return 1 + (win_rate * uplift)


# =============================================================================
# DAILY MILP OPTIMIZATION
# =============================================================================

def solve_daily_milp(agile_prices_24h, wholesale_prices_24h, consumer_demand_24h,
                     battery_kwh, battery_kw, efficiency, fr_rate_hourly):
    """
    Solve daily MILP optimization for joint consumer + VPP income.

    Decision variables per hour:
    - c[h]: charge power (kW)
    - d_consumer[h]: discharge to consumer (kW)
    - d_grid[h]: discharge to grid for arbitrage (kW)
    - fr[h]: binary FR commitment

    Objective: Maximize consumer_savings + wholesale_arbitrage + FR_income

    Args:
        agile_prices_24h: list of 24 Agile prices (p/kWh)
        wholesale_prices_24h: list of 24 wholesale prices (EUR/MWh)
        consumer_demand_24h: list of 24 consumer demand values (kWh)
        battery_kwh: capacity
        battery_kw: power
        efficiency: round-trip efficiency
        fr_rate_hourly: FR rate per hour (£/kW/h)

    Returns: dict with profit breakdown and schedules
    """
    if not HAS_PULP:
        return solve_daily_simple(agile_prices_24h, wholesale_prices_24h, consumer_demand_24h,
                                  battery_kwh, battery_kw, efficiency, fr_rate_hourly)

    eta = np.sqrt(efficiency)
    hours = range(24)
    E_start = battery_kwh * 0.5  # Start at 50% SoC

    # Create problem
    prob = pulp.LpProblem("Daily_Battery_Optimization", pulp.LpMaximize)

    # Decision variables
    c = {h: pulp.LpVariable(f"charge_{h}", 0, battery_kw) for h in hours}
    d_consumer = {h: pulp.LpVariable(f"discharge_consumer_{h}", 0, battery_kw) for h in hours}
    d_grid = {h: pulp.LpVariable(f"discharge_grid_{h}", 0, battery_kw) for h in hours}
    fr = {h: pulp.LpVariable(f"fr_{h}", cat='Binary') for h in hours}
    E = {h: pulp.LpVariable(f"SoC_{h}", 0, battery_kwh) for h in range(25)}

    # Objective components
    # Key insight: battery charges once, but value depends on WHERE energy goes
    # - Consumer discharge: saves them Agile rate (vs what they would have paid)
    # - Grid discharge: earns wholesale rate
    # - Charging cost: paid at wholesale rate (we buy from wholesale market)

    consumer_value = []  # Value of serving consumer (saves them Agile rate)
    grid_revenue = []    # Revenue from selling to grid (wholesale rate)
    charge_cost = []     # Cost of charging (wholesale rate)
    fr_income = []       # FR commitment income

    for h in hours:
        agile_rate_gbp = agile_prices_24h[h] / 100  # p/kWh to £/kWh
        wholesale_rate_gbp = (wholesale_prices_24h[h] / 1000) * EUR_TO_GBP  # EUR/MWh to £/kWh

        # Consumer value: discharge saves them from paying Agile rate at this hour
        # NOTE: Efficiency is already in energy balance - we can only discharge what we stored
        # Financial calculation is simply: delivered kWh × rate
        consumer_value.append(d_consumer[h] * agile_rate_gbp)

        # Grid revenue: sell to wholesale market
        grid_revenue.append(d_grid[h] * wholesale_rate_gbp)

        # Charge cost: buy from wholesale market
        # We pay for actual kWh bought from grid (not adjusted for efficiency)
        charge_cost.append(c[h] * wholesale_rate_gbp)

        # FR income
        fr_income.append(fr[h] * fr_rate_hourly * battery_kw)

    # Objective: maximize total value
    # Total = Consumer_Value + Grid_Revenue + FR - Charge_Cost
    prob += (pulp.lpSum(consumer_value) + pulp.lpSum(grid_revenue) +
             pulp.lpSum(fr_income) - pulp.lpSum(charge_cost)), "Total_Value"

    # Constraints

    # 1. Initial and final SoC
    prob += E[0] == E_start, "Initial_SoC"
    prob += E[24] == E_start, "Final_SoC"

    # 2. Energy balance
    for h in hours:
        prob += E[h + 1] == E[h] + c[h] * eta - (d_consumer[h] + d_grid[h]) / eta, f"Energy_Balance_{h}"

    # 3. Consumer demand limit: can't discharge more to consumer than they need
    for h in hours:
        prob += d_consumer[h] <= consumer_demand_24h[h], f"Consumer_Demand_Limit_{h}"

    # 4. Total discharge limit per hour
    for h in hours:
        prob += d_consumer[h] + d_grid[h] <= battery_kw, f"Total_Discharge_Limit_{h}"

    # 5. FR blocks charging and discharging
    for h in hours:
        prob += c[h] <= battery_kw * (1 - fr[h]), f"FR_Block_Charge_{h}"
        prob += d_consumer[h] + d_grid[h] <= battery_kw * (1 - fr[h]), f"FR_Block_Discharge_{h}"

    # 6. FR requires 50% SoC (big-M)
    M = battery_kwh
    for h in hours:
        prob += E[h] >= E_start - M * (1 - fr[h]), f"FR_SoC_Lower_{h}"
        prob += E[h] <= E_start + M * (1 - fr[h]), f"FR_SoC_Upper_{h}"

    # 7. CRITICAL: Limit daily throughput to ONE cycle per day
    # This prevents unrealistic multiple charge/discharge cycles
    # Total discharge (consumer + grid) limited to battery capacity
    prob += pulp.lpSum([d_consumer[h] + d_grid[h] for h in hours]) <= battery_kwh, "Daily_Discharge_Limit"
    # Total charge also limited to battery capacity
    prob += pulp.lpSum([c[h] for h in hours]) <= battery_kwh, "Daily_Charge_Limit"

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        return {
            "status": "Failed",
            "consumer_savings": 0,
            "wholesale_arbitrage": 0,
            "fr_income": 0,
            "total": 0,
            "fr_hours": 0,
        }

    # Extract results
    # Consumer value: what consumer saves by getting electricity from battery
    # (delivered kWh × Agile rate - efficiency already in energy balance)
    total_consumer_value = sum(
        (d_consumer[h].varValue or 0) * (agile_prices_24h[h] / 100)
        for h in hours
    )

    # Grid revenue: what we earn selling to wholesale market
    total_grid_revenue = sum(
        (d_grid[h].varValue or 0) * ((wholesale_prices_24h[h] / 1000) * EUR_TO_GBP)
        for h in hours
    )

    # Total charge cost (actual kWh bought from grid)
    total_charge_cost = sum(
        (c[h].varValue or 0) * ((wholesale_prices_24h[h] / 1000) * EUR_TO_GBP)
        for h in hours
    )

    # FR income
    total_fr = sum(
        (fr[h].varValue or 0) * fr_rate_hourly * battery_kw
        for h in hours
    )

    # Allocate charge cost proportionally between consumer and grid
    total_discharge = sum((d_consumer[h].varValue or 0) + (d_grid[h].varValue or 0) for h in hours)
    total_consumer_discharge = sum((d_consumer[h].varValue or 0) for h in hours)
    total_grid_discharge = sum((d_grid[h].varValue or 0) for h in hours)

    if total_discharge > 0:
        consumer_share = total_consumer_discharge / total_discharge
        grid_share = total_grid_discharge / total_discharge
    else:
        consumer_share = 0.5
        grid_share = 0.5

    consumer_charge_cost = total_charge_cost * consumer_share
    grid_charge_cost = total_charge_cost * grid_share

    # Net values
    net_consumer_savings = total_consumer_value - consumer_charge_cost
    net_wholesale_profit = total_grid_revenue - grid_charge_cost

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


def solve_daily_simple(agile_prices_24h, wholesale_prices_24h, consumer_demand_24h,
                       battery_kwh, battery_kw, efficiency, fr_rate_hourly):
    """
    Simplified daily optimization without MILP.
    Greedy approach: serve consumer first, then arbitrage with remaining capacity.
    """
    eta = np.sqrt(efficiency)
    hours_to_cycle = int(np.ceil(battery_kwh / battery_kw))

    # Sort hours by price
    wholesale_sorted = sorted(range(24), key=lambda h: wholesale_prices_24h[h])
    agile_sorted = sorted(range(24), key=lambda h: agile_prices_24h[h])

    # Charge at lowest wholesale hours
    charge_hours = wholesale_sorted[:hours_to_cycle]
    discharge_hours = wholesale_sorted[-hours_to_cycle:]

    # Calculate consumer savings (serve all consumer demand from battery)
    daily_consumer_demand = sum(consumer_demand_24h)
    consumer_kwh = min(battery_kwh, daily_consumer_demand)

    # Average Agile rate at consumption times vs lowest charging rate
    avg_consumption_rate = np.mean([agile_prices_24h[h] for h in range(24)
                                    if consumer_demand_24h[h] > 0]) if daily_consumer_demand > 0 else 0
    avg_charge_rate = np.mean([agile_prices_24h[h] for h in agile_sorted[:hours_to_cycle]])

    consumer_savings = consumer_kwh * (avg_consumption_rate - avg_charge_rate) / 100 * eta

    # Wholesale arbitrage with remaining capacity
    remaining_kwh = max(0, battery_kwh - consumer_kwh)
    if remaining_kwh > 0:
        avg_buy = np.mean([wholesale_prices_24h[h] for h in charge_hours])
        avg_sell = np.mean([wholesale_prices_24h[h] for h in discharge_hours])
        wholesale_arb = remaining_kwh * ((avg_sell - avg_buy) / 1000) * EUR_TO_GBP * eta
    else:
        wholesale_arb = 0

    # FR comparison
    fr_24h = fr_rate_hourly * battery_kw * 24

    return {
        "status": "Simplified",
        "consumer_savings": max(0, consumer_savings),
        "wholesale_arbitrage": max(0, wholesale_arb),
        "fr_income": 0,  # Simplified doesn't optimize FR
        "total": max(0, consumer_savings) + max(0, wholesale_arb),
        "fr_hours": 0,
    }


# =============================================================================
# ANNUAL OPTIMIZATION
# =============================================================================

def optimize_annual(agile_hourly, wholesale_df, use_profile, params,
                    battery_kwh, battery_kw, annual_consumption):
    """
    Run daily optimization across the year and aggregate.

    Args:
        agile_hourly: DataFrame with date, hour, rate_p_kwh
        wholesale_df: DataFrame with date, hour, price_eur_mwh
        use_profile: dict hour -> consumption weight
        params: model parameters
        battery_kwh: battery capacity
        battery_kw: battery power
        annual_consumption: annual consumption in kWh

    Returns: dict with annual totals
    """
    efficiency = params.get("battery charge recharge efficiency %", DEFAULT_EFFICIENCY)
    fr_rate_annual = params.get("Frequency response per kw per year", 0)
    fr_rate_hourly = fr_rate_annual / (365 * 24)  # £/kW/hour

    # Normalize consumption profile
    total_profile = sum(use_profile.values())
    profile_norm = {h: v / total_profile for h, v in use_profile.items()}
    daily_consumption = annual_consumption / 365

    # Get unique dates present in both datasets
    agile_dates = set(agile_hourly['date'].unique())
    wholesale_dates = set(wholesale_df['date'].unique())
    common_dates = agile_dates & wholesale_dates

    total_consumer_savings = 0
    total_wholesale_arb = 0
    total_fr_income = 0
    days_processed = 0

    for date in sorted(common_dates):
        # Get day's prices
        agile_day = agile_hourly[agile_hourly['date'] == date].sort_values('hour')
        wholesale_day = wholesale_df[wholesale_df['date'] == date].sort_values('hour')

        if len(agile_day) < 24 or len(wholesale_day) < 24:
            continue

        agile_prices = agile_day['rate_p_kwh'].values.tolist()
        wholesale_prices = wholesale_day['price_eur_mwh'].values.tolist()

        # Consumer demand per hour
        consumer_demand = [daily_consumption * profile_norm.get(h, 0) for h in range(24)]

        # Solve daily optimization
        result = solve_daily_milp(
            agile_prices, wholesale_prices, consumer_demand,
            battery_kwh, battery_kw, efficiency, fr_rate_hourly
        )

        total_consumer_savings += result['consumer_savings']
        total_wholesale_arb += result['wholesale_arbitrage']
        total_fr_income += result['fr_income']
        days_processed += 1

    # Add CM and BM
    cm_income = calculate_capacity_market(params, battery_kwh, battery_kw)
    bm_multiplier = calculate_bm_multiplier(params)
    bm_income = total_wholesale_arb * (bm_multiplier - 1)

    return {
        'days_processed': days_processed,
        'consumer_savings': total_consumer_savings,
        'wholesale_arbitrage': total_wholesale_arb,
        'bm_income': bm_income,
        'wholesale_with_bm': total_wholesale_arb + bm_income,
        'fr_income': total_fr_income,
        'cm_income': cm_income,
        'total_vpp': total_wholesale_arb + bm_income + total_fr_income + cm_income,
        'total_value': total_consumer_savings + total_wholesale_arb + bm_income + total_fr_income + cm_income,
    }


# =============================================================================
# MAIN MODEL
# =============================================================================

def run_model(agile_hourly, wholesale_df, params, use_profile,
              battery_kwh=None, battery_kw=None, override_consumption=None):
    """Run the joint optimization model."""
    if battery_kwh is None:
        battery_kwh = params.get("battery capacity kwh", DEFAULT_BATTERY_KWH)
    if battery_kw is None:
        battery_kw = params.get("battery power kw", DEFAULT_BATTERY_KW)

    if override_consumption is not None:
        annual_consumption = override_consumption
    else:
        annual_consumption = params.get("Mean electricity residential consumption annual (kWH)", 3500)

    efficiency = params.get("battery charge recharge efficiency %", DEFAULT_EFFICIENCY)

    print(f"\n{'='*70}")
    print("JOINT OPTIMIZATION: Consumer Savings + VPP Income")
    print(f"{'='*70}")
    print(f"\nBattery: {battery_kwh} kWh / {battery_kw} kW")
    print(f"Annual consumption: {annual_consumption} kWh")
    print(f"Efficiency: {efficiency*100:.0f}%")

    # Run annual optimization
    result = optimize_annual(
        agile_hourly, wholesale_df, use_profile, params,
        battery_kwh, battery_kw, annual_consumption
    )

    # Display results
    print(f"\n{'-'*70}")
    print(f"ANNUAL RESULTS ({result['days_processed']} days)")
    print(f"{'-'*70}")

    print(f"\n1. CONSUMER SAVINGS (Agile spread):")
    print(f"   Annual savings: £{result['consumer_savings']:.2f}")

    print(f"\n2. WHOLESALE ARBITRAGE:")
    print(f"   Base arbitrage: £{result['wholesale_arbitrage']:.2f}")
    print(f"   BM uplift:      £{result['bm_income']:.2f}")
    print(f"   With BM:        £{result['wholesale_with_bm']:.2f}")

    print(f"\n3. FREQUENCY RESPONSE:")
    print(f"   FR income:      £{result['fr_income']:.2f}")

    print(f"\n4. CAPACITY MARKET:")
    derating = get_derating_factor(battery_kwh, battery_kw)
    print(f"   De-rating:      {derating*100:.1f}%")
    print(f"   CM income:      £{result['cm_income']:.2f}")

    print(f"\n{'-'*70}")
    print(f"TOTALS")
    print(f"{'-'*70}")
    print(f"  Consumer savings:    £{result['consumer_savings']:>10.2f}")
    print(f"  VPP income:          £{result['total_vpp']:>10.2f}")
    print(f"    (Arb+BM: £{result['wholesale_with_bm']:.2f}, FR: £{result['fr_income']:.2f}, CM: £{result['cm_income']:.2f})")
    print(f"  {'='*35}")
    print(f"  TOTAL VALUE:         £{result['total_value']:>10.2f}")

    result['battery_kwh'] = battery_kwh
    result['battery_kw'] = battery_kw
    result['annual_consumption'] = annual_consumption

    return result


def compare_battery_sizes(agile_hourly, wholesale_df, params, use_profile,
                          sizes=None, override_consumption=None):
    """Compare different battery sizes."""
    if sizes is None:
        sizes = [5, 7.5, 10, 12.5, 15, 20]

    results = []
    kw_per_kwh = params.get("kw per kwh of battery capacity", 0.5)

    print(f"\n{'='*100}")
    print("BATTERY SIZE COMPARISON")
    print(f"{'='*100}")

    for size in sizes:
        battery_kw = size * kw_per_kwh
        result = run_model(
            agile_hourly, wholesale_df, params, use_profile,
            battery_kwh=size, battery_kw=battery_kw,
            override_consumption=override_consumption
        )
        results.append(result)

    # Summary table
    print(f"\n{'='*120}")
    print("SUMMARY TABLE")
    print(f"{'='*120}")
    print(f"{'Size':>6} | {'Consumer':>10} | {'Arb+BM':>10} | {'FR':>8} | {'CM':>8} | {'VPP Tot':>10} | {'TOTAL':>12}")
    print(f"{'(kWh)':>6} | {'Savings':>10} | {'':>10} | {'':>8} | {'':>8} | {'':>10} | {'VALUE':>12}")
    print("-"*120)

    for r in results:
        print(f"{r['battery_kwh']:>6.1f} | £{r['consumer_savings']:>8.2f} | £{r['wholesale_with_bm']:>8.2f} | "
              f"£{r['fr_income']:>6.2f} | £{r['cm_income']:>6.2f} | £{r['total_vpp']:>8.2f} | £{r['total_value']:>10.2f}")

    return results


def main(test_mode=False, override_consumption=None):
    """Run the model."""
    print("="*70)
    print("BATTERY MODEL - JOINT CONSUMER + VPP OPTIMIZATION")
    print("="*70)

    print("\nLoading model inputs...")
    params = load_model_inputs()
    use_profile = load_use_profiles()

    print("\nLoading Agile rate data...")
    days_limit = 30 if test_mode else None
    agile_df = scrape_agile_year(2025, use_cache=True, days_limit=days_limit)

    if agile_df is None or len(agile_df) == 0:
        print("ERROR: No Agile data available.")
        return

    print(f"Loaded {len(agile_df)} Agile records")
    agile_hourly = get_hourly_agile_rates(agile_df)

    print("\nLoading wholesale market prices...")
    wholesale_df = load_wholesale_prices(2025)
    print(f"Loaded {len(wholesale_df)} wholesale records")

    # Run model
    result = run_model(agile_hourly, wholesale_df, params, use_profile,
                       override_consumption=override_consumption)

    # Compare sizes
    compare_battery_sizes(agile_hourly, wholesale_df, params, use_profile,
                          sizes=[5, 7.5, 10, 12.5, 15, 20],
                          override_consumption=override_consumption)

    print(f"\n{'='*70}")
    print("NOTES")
    print(f"{'='*70}")
    print("""
1. JOINT OPTIMIZATION: Battery is optimized for total value across all streams.
   Each kWh discharged goes to either consumer (saves Agile rate) or grid (earns wholesale rate).

2. Consumer savings = discharge to consumer × Agile rate saved

3. VPP income = Wholesale arbitrage + BM uplift + FR + CM

4. The optimizer decides hour-by-hour the best allocation.

5. Data sources:
   - Agile rates: agilebuddy.uk (retail tariff)
   - Wholesale rates: United Kingdom.csv (EUR/MWh)
""")


if __name__ == "__main__":
    import sys

    test_mode = "--test" in sys.argv

    override_consumption = None
    for arg in sys.argv:
        if arg.startswith("--consumption="):
            override_consumption = float(arg.split("=")[1])

    main(test_mode=test_mode, override_consumption=override_consumption)
