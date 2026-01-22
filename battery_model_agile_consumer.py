#!/usr/bin/env python3
"""
Battery Storage Optimization Model - Agile Consumer-First Version

This model calculates VPP income AFTER ensuring the customer gets the best possible
electricity rate. It uses Octopus Agile tariff data scraped from agilebuddy.uk.

The key insight:
1. First, calculate consumer savings by comparing:
   - What they'd pay at Agile rates during their consumption hours
   - What they'd pay if all electricity came from battery charged at lowest rate
2. Then, calculate remaining VPP income from leftover battery capacity

Uses UK consumption profile from use_profiles.csv.
Scrapes Agile data for 2025 from agilebuddy.uk.

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
import json
import os

# Configuration
DATA_DIR = Path(__file__).parent
AGILE_CACHE_FILE = DATA_DIR / "agile_rates_2025_cache.csv"

# Battery parameters (can be overridden from model_input.csv)
DEFAULT_BATTERY_KWH = 10.0
DEFAULT_BATTERY_KW = 5.0
DEFAULT_EFFICIENCY = 0.9


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


def load_wholesale_prices(year=2025):
    """
    Load wholesale market prices from United Kingdom.csv for VPP arbitrage.

    Returns DataFrame with date, hour, price_eur_mwh
    """
    df = pd.read_csv(DATA_DIR / "United Kingdom.csv")
    df["datetime"] = pd.to_datetime(df["Datetime (UTC)"])
    df["price_eur_mwh"] = df["Price (EUR/MWhe)"]

    # Filter to the year
    df = df[df["datetime"].dt.year == year]

    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour

    return df[["datetime", "date", "hour", "price_eur_mwh"]]


def calculate_wholesale_arbitrage(wholesale_df, battery_kwh, battery_kw, efficiency, consumer_reserved_kwh=0):
    """
    Calculate arbitrage income using WHOLESALE market rates (not Agile retail).

    This is what VPP actually earns from trading on wholesale market.
    Prices are in EUR/MWh, converted to GBP.

    Args:
        wholesale_df: DataFrame with date, hour, price_eur_mwh
        battery_kwh: Total battery capacity
        battery_kw: Battery power
        efficiency: Round-trip efficiency
        consumer_reserved_kwh: kWh reserved for consumer (reduces available for arb)

    Returns:
        dict with arbitrage income breakdown in GBP
    """
    EUR_TO_GBP = 0.85  # Approximate conversion

    # Available capacity for arbitrage
    available_kwh = max(0, battery_kwh - consumer_reserved_kwh)
    available_kw = available_kwh * (battery_kw / battery_kwh) if battery_kwh > 0 else 0

    if available_kwh <= 0:
        return {
            'annual_income_gbp': 0,
            'available_kwh': 0,
            'avg_daily_profit_eur': 0,
            'days_profitable': 0
        }

    efficiency_factor = np.sqrt(efficiency)

    # Hours needed to cycle
    hours_to_cycle = int(np.ceil(available_kwh / available_kw)) if available_kw > 0 else 0
    hours_to_cycle = min(hours_to_cycle, 12)

    total_profit_eur = 0
    days = wholesale_df['date'].unique()
    daily_profits = []

    for date in days:
        day_rates = wholesale_df[wholesale_df['date'] == date]
        if len(day_rates) < 24:
            continue

        # Sort by rate
        day_rates_sorted = day_rates.sort_values('price_eur_mwh')

        # Charge at lowest rates
        charge_hours = day_rates_sorted.head(hours_to_cycle)
        avg_charge_rate = charge_hours['price_eur_mwh'].mean()

        # Discharge at highest rates
        discharge_hours = day_rates_sorted.tail(hours_to_cycle)
        avg_discharge_rate = discharge_hours['price_eur_mwh'].mean()

        # Profit calculation (EUR/MWh -> EUR/kWh = /1000)
        charge_cost = available_kwh * (avg_charge_rate / 1000) / efficiency_factor
        discharge_revenue = available_kwh * (avg_discharge_rate / 1000) * efficiency_factor
        daily_profit = discharge_revenue - charge_cost

        if daily_profit > 0:
            total_profit_eur += daily_profit
            daily_profits.append(daily_profit)

    return {
        'annual_income_gbp': total_profit_eur * EUR_TO_GBP,
        'annual_income_eur': total_profit_eur,
        'available_kwh': available_kwh,
        'avg_daily_profit_eur': np.mean(daily_profits) if daily_profits else 0,
        'days_profitable': len(daily_profits)
    }


def scrape_agile_day(date_str):
    """
    Scrape Agile rates for a single day from agilebuddy.uk.

    Args:
        date_str: Date in format 'YYYY/MM/DD'

    Returns:
        list of dicts with 'time' (HH:MM) and 'rate' (p/kWh) for each 30-min slot
    """
    url = f"https://agilebuddy.uk/historic/agile/{date_str}"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all tables - Table 1 has the time/price pairs
        tables = soup.find_all('table')
        if len(tables) < 2:
            return None

        rates = []
        price_table = tables[1]  # Second table has the hourly rates

        for row in price_table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 2:
                time_text = cells[0].get_text().strip()
                price_text = cells[1].get_text().strip()

                # Validate time format (HH:MM)
                if re.match(r'\d{2}:\d{2}', time_text):
                    # Extract numeric price (remove 'p' suffix)
                    price_match = re.search(r'([-]?\d+\.?\d*)', price_text)
                    if price_match:
                        try:
                            rate = float(price_match.group(1))
                            rates.append({'time': time_text, 'rate': rate})
                        except ValueError:
                            continue

        # Sort by time
        rates.sort(key=lambda x: x['time'])

        if len(rates) >= 24:  # At least 24 half-hour slots expected
            return rates
        else:
            return None

    except Exception as e:
        return None


def scrape_agile_year(year=2025, use_cache=True, days_limit=None):
    """
    Scrape all Agile rates for a year.

    Args:
        year: Year to scrape
        use_cache: If True, use cached data if available
        days_limit: If set, only scrape this many days (for testing)

    Returns:
        DataFrame with columns: date, hour, rate_p_kwh
    """
    # Check for cache
    if use_cache and AGILE_CACHE_FILE.exists():
        print(f"Loading cached Agile data from {AGILE_CACHE_FILE}")
        return pd.read_csv(AGILE_CACHE_FILE, parse_dates=['date'])

    print(f"Scraping Agile rates for {year}...")
    if days_limit:
        print(f"Limited to first {days_limit} days for testing...")
    else:
        print("This may take a while (365 requests with delays)...")

    all_data = []

    # Generate all dates for the year
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    current_date = start_date

    days_scraped = 0
    days_failed = 0
    total_days = 0

    while current_date <= end_date:
        # Check days limit
        if days_limit and total_days >= days_limit:
            break
        total_days += 1
        date_str = current_date.strftime("%Y/%m/%d")

        # Progress indicator
        if days_scraped % 30 == 0:
            print(f"  Scraping {current_date.strftime('%B %Y')}...")

        rates = scrape_agile_day(date_str)

        if rates:
            for r in rates:
                # Convert 30-min time to hour (0-23)
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
        else:
            days_failed += 1

        # Rate limiting - be polite to the server
        time.sleep(0.5)
        current_date += timedelta(days=1)

    print(f"Scraped {days_scraped} days successfully, {days_failed} failed")

    if not all_data:
        print("ERROR: No data scraped!")
        return None

    df = pd.DataFrame(all_data)

    # Save cache
    df.to_csv(AGILE_CACHE_FILE, index=False)
    print(f"Saved cache to {AGILE_CACHE_FILE}")

    return df


def get_hourly_agile_rates(agile_df):
    """
    Convert 30-min Agile rates to hourly averages.

    Returns DataFrame with date, hour, rate_p_kwh (hourly average)
    """
    # Average the two half-hours for each hour
    hourly = agile_df.groupby(['date', 'hour'])['rate_p_kwh'].mean().reset_index()
    return hourly


def calculate_consumer_cost_at_consumption(agile_hourly, use_profile, annual_consumption_kwh):
    """
    Calculate what consumer would pay if buying electricity at Agile rates
    during their consumption hours.

    Args:
        agile_hourly: DataFrame with date, hour, rate_p_kwh
        use_profile: Dict mapping hour -> consumption weight
        annual_consumption_kwh: Total annual consumption

    Returns:
        Total annual cost in GBP
    """
    # Normalize profile to sum to 1
    total_profile = sum(use_profile.values())
    profile_normalized = {h: v / total_profile for h, v in use_profile.items()}

    # Daily consumption
    daily_consumption = annual_consumption_kwh / 365

    # For each day, calculate cost based on consumption profile
    total_cost_pence = 0
    days = agile_hourly['date'].unique()

    for date in days:
        day_rates = agile_hourly[agile_hourly['date'] == date]
        if len(day_rates) < 24:
            continue

        day_rates = day_rates.set_index('hour')

        for hour, weight in profile_normalized.items():
            if hour in day_rates.index:
                rate = day_rates.loc[hour, 'rate_p_kwh']
                consumption = daily_consumption * weight
                total_cost_pence += consumption * rate

    return total_cost_pence / 100  # Convert to GBP


def calculate_consumer_cost_from_battery(agile_hourly, annual_consumption_kwh, battery_kwh, battery_kw, efficiency):
    """
    Calculate what consumer would pay if all electricity came from battery
    charged at the lowest daily Agile rate.

    The battery charges at the cheapest hours each day and provides all
    consumer electricity.

    Args:
        agile_hourly: DataFrame with date, hour, rate_p_kwh
        annual_consumption_kwh: Total annual consumption
        battery_kwh: Battery capacity
        battery_kw: Battery power
        efficiency: Round-trip efficiency

    Returns:
        dict with cost breakdown
    """
    daily_consumption = annual_consumption_kwh / 365
    efficiency_factor = np.sqrt(efficiency)

    # How much needs to be charged (accounting for efficiency)
    daily_charge_needed = daily_consumption / efficiency_factor

    # Hours needed to charge
    hours_to_charge = int(np.ceil(daily_charge_needed / battery_kw))
    hours_to_charge = min(hours_to_charge, 12)  # Max half the day

    total_cost_pence = 0
    days = agile_hourly['date'].unique()
    days_analyzed = 0

    lowest_rates = []

    for date in days:
        day_rates = agile_hourly[agile_hourly['date'] == date]
        if len(day_rates) < 24:
            continue

        # Sort by rate to find cheapest hours
        day_rates_sorted = day_rates.sort_values('rate_p_kwh')

        # Take the cheapest hours for charging
        charge_hours = day_rates_sorted.head(hours_to_charge)
        avg_charge_rate = charge_hours['rate_p_kwh'].mean()
        lowest_rates.append(avg_charge_rate)

        # Cost to charge enough for daily consumption
        charge_cost_pence = daily_charge_needed * avg_charge_rate
        total_cost_pence += charge_cost_pence
        days_analyzed += 1

    return {
        'total_cost_gbp': total_cost_pence / 100,
        'avg_lowest_rate': np.mean(lowest_rates) if lowest_rates else 0,
        'days_analyzed': days_analyzed,
        'daily_charge_needed_kwh': daily_charge_needed,
        'hours_to_charge': hours_to_charge
    }


def calculate_daily_agile_spread(agile_hourly):
    """
    Calculate daily spread (max - min rate) from Agile data.

    Returns DataFrame with date, min_rate, max_rate, spread, avg_rate
    """
    daily_stats = agile_hourly.groupby('date').agg({
        'rate_p_kwh': ['min', 'max', 'mean']
    }).reset_index()

    daily_stats.columns = ['date', 'min_rate', 'max_rate', 'avg_rate']
    daily_stats['spread'] = daily_stats['max_rate'] - daily_stats['min_rate']

    return daily_stats


def calculate_arbitrage_from_agile(agile_hourly, battery_kwh, battery_kw, efficiency, consumer_reserved_kwh=0):
    """
    Calculate arbitrage income using actual Agile rate data.

    After reserving capacity for consumer, uses remaining battery for arbitrage.

    Args:
        agile_hourly: DataFrame with date, hour, rate_p_kwh
        battery_kwh: Total battery capacity
        battery_kw: Battery power
        efficiency: Round-trip efficiency
        consumer_reserved_kwh: kWh reserved for consumer (reduces available for arb)

    Returns:
        dict with arbitrage income breakdown
    """
    # Available capacity for arbitrage
    available_kwh = max(0, battery_kwh - consumer_reserved_kwh)
    available_kw = available_kwh * (battery_kw / battery_kwh) if battery_kwh > 0 else 0

    if available_kwh <= 0:
        return {
            'annual_income_gbp': 0,
            'available_kwh': 0,
            'avg_daily_profit_pence': 0,
            'days_profitable': 0
        }

    efficiency_factor = np.sqrt(efficiency)

    # Hours needed to cycle
    hours_to_cycle = int(np.ceil(available_kwh / available_kw)) if available_kw > 0 else 0
    hours_to_cycle = min(hours_to_cycle, 12)

    total_profit_pence = 0
    days = agile_hourly['date'].unique()
    daily_profits = []

    for date in days:
        day_rates = agile_hourly[agile_hourly['date'] == date]
        if len(day_rates) < 24:
            continue

        # Sort by rate
        day_rates_sorted = day_rates.sort_values('rate_p_kwh')

        # Charge at lowest rates
        charge_hours = day_rates_sorted.head(hours_to_cycle)
        avg_charge_rate = charge_hours['rate_p_kwh'].mean()

        # Discharge at highest rates
        discharge_hours = day_rates_sorted.tail(hours_to_cycle)
        avg_discharge_rate = discharge_hours['rate_p_kwh'].mean()

        # Profit calculation
        charge_cost = available_kwh * avg_charge_rate / efficiency_factor
        discharge_revenue = available_kwh * avg_discharge_rate * efficiency_factor
        daily_profit = discharge_revenue - charge_cost

        if daily_profit > 0:
            total_profit_pence += daily_profit
            daily_profits.append(daily_profit)

    return {
        'annual_income_gbp': total_profit_pence / 100,
        'available_kwh': available_kwh,
        'avg_daily_profit_pence': np.mean(daily_profits) if daily_profits else 0,
        'days_profitable': len(daily_profits)
    }


def calculate_frequency_response(params, battery_kw, hours_per_day=24):
    """Calculate FR income. Same as original model."""
    rate = params.get("Frequency response per kw per year", 0)
    fraction_of_day = hours_per_day / 24
    return rate * battery_kw * fraction_of_day


# UK Capacity Market de-rating factors (Source: EMR DB 2024/25 Scaled EFC methodology)
UK_CM_DERATING_FACTORS = {
    1: 0.1047,  # 1-hour battery: 10.47%
    2: 0.2094,  # 2-hour battery: 20.94%
    4: 0.37,    # 4-hour battery: ~37%
    8: 0.92,    # 8-hour battery: ~92%
}


def get_battery_duration_hours(battery_kwh, battery_kw):
    """Calculate battery duration in hours for de-rating factor lookup."""
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
    """Get the appropriate de-rating factor based on battery duration."""
    duration = get_battery_duration_hours(battery_kwh, battery_kw)
    return UK_CM_DERATING_FACTORS.get(duration, 0.2)


def calculate_capacity_market(params, battery_kwh, battery_kw):
    """
    Calculate annual income from Capacity Market with proper de-rating.

    Uses "Capacity market /kW/year" from model_input.csv as the clearing price.
    Applies de-rating factors based on battery duration.

    Revenue = Clearing_Price * De_Rating_Factor * Battery_kW

    Returns: annual income in GBP
    """
    cm_rate = params.get("Capacity market /kW/year", 0)

    if cm_rate == 0:
        return 0

    derating = get_derating_factor(battery_kwh, battery_kw)
    annual_revenue = cm_rate * derating * battery_kw

    return annual_revenue


def calculate_balancing_mechanism_multiplier(params):
    """
    Calculate BM multiplier for arbitrage revenue.
    UK: 10% win rate, 5% uplift on winning trades.

    BM acts as a multiplier: when you win a BM bid, you get an uplift on that trade.
    Effective multiplier = 1 + (win_rate * uplift)

    Returns: multiplier (>= 1.0)
    """
    win_rate = params.get("Balancing mechanism (BM) avg win rate", 0)
    uplift = params.get("Avg £/kwh uplift for BM", 0)
    return 1 + (win_rate * uplift)


def run_consumer_first_vpp_model(agile_hourly, params, use_profile, wholesale_df, battery_kwh=None, battery_kw=None):
    """
    Run the consumer-first VPP model.

    1. Calculate consumer savings from optimal battery charging (uses Agile retail rates)
    2. Calculate remaining VPP income from leftover capacity (uses wholesale market rates)

    Args:
        agile_hourly: Hourly Agile rate data (for consumer savings)
        params: Model parameters from model_input.csv
        use_profile: Hourly consumption profile
        wholesale_df: Hourly wholesale market rates (for VPP arbitrage)
        battery_kwh: Battery capacity (or from params)
        battery_kw: Battery power (or from params)

    Returns:
        dict with all income/savings breakdown
    """
    # Get battery specs
    if battery_kwh is None:
        battery_kwh = params.get("battery capacity kwh", DEFAULT_BATTERY_KWH)
    if battery_kw is None:
        battery_kw = params.get("battery power kw", DEFAULT_BATTERY_KW)

    efficiency = params.get("battery charge recharge efficiency %", DEFAULT_EFFICIENCY)
    annual_consumption = params.get("Mean electricity residential consumption annual (kWH)", 3500)
    daily_consumption = annual_consumption / 365

    print(f"\n{'='*70}")
    print("CONSUMER-FIRST VPP MODEL (Octopus Agile Tariff)")
    print(f"{'='*70}")
    print(f"\nBattery: {battery_kwh} kWh / {battery_kw} kW")
    print(f"Annual consumption: {annual_consumption} kWh")
    print(f"Daily consumption: {daily_consumption:.2f} kWh")
    print(f"Efficiency: {efficiency*100:.0f}%")

    # =========================================
    # STEP 1: CONSUMER SAVINGS
    # =========================================
    print(f"\n{'-'*70}")
    print("STEP 1: CONSUMER SAVINGS FROM OPTIMAL BATTERY CHARGING")
    print(f"{'-'*70}")

    # Cost if paying Agile rates at consumption time
    cost_at_consumption = calculate_consumer_cost_at_consumption(
        agile_hourly, use_profile, annual_consumption
    )

    # Cost if charging battery at lowest rates
    battery_cost_result = calculate_consumer_cost_from_battery(
        agile_hourly, annual_consumption, battery_kwh, battery_kw, efficiency
    )
    cost_from_battery = battery_cost_result['total_cost_gbp']

    # Consumer savings
    consumer_savings = cost_at_consumption - cost_from_battery

    print(f"\nWithout optimization (pay at consumption time):")
    print(f"  Annual cost: £{cost_at_consumption:.2f}")

    print(f"\nWith battery optimization (charge at lowest rate):")
    print(f"  Average charge rate: {battery_cost_result['avg_lowest_rate']:.2f} p/kWh")
    print(f"  Daily charge needed: {battery_cost_result['daily_charge_needed_kwh']:.2f} kWh")
    print(f"  Hours charging: {battery_cost_result['hours_to_charge']}")
    print(f"  Annual cost: £{cost_from_battery:.2f}")

    print(f"\n>>> CONSUMER ANNUAL SAVINGS: £{consumer_savings:.2f}")

    # =========================================
    # STEP 2: BATTERY ALLOCATION FOR VPP
    # =========================================
    print(f"\n{'-'*70}")
    print("STEP 2: BATTERY ALLOCATION FOR VPP INCOME")
    print(f"{'-'*70}")

    # Battery capacity reserved for consumer
    # The battery must store enough for daily consumption (plus efficiency losses)
    efficiency_factor = np.sqrt(efficiency)
    consumer_reserved_kwh = min(battery_kwh, daily_consumption / efficiency_factor)
    leftover_kwh = max(0, battery_kwh - consumer_reserved_kwh)
    leftover_kw = leftover_kwh * (battery_kw / battery_kwh) if battery_kwh > 0 else 0

    print(f"\nBattery allocation:")
    print(f"  Reserved for consumer: {consumer_reserved_kwh:.2f} kWh")
    print(f"  Leftover for VPP:      {leftover_kwh:.2f} kWh / {leftover_kw:.2f} kW")

    # =========================================
    # STEP 3: VPP INCOME FROM LEFTOVER CAPACITY
    # =========================================
    print(f"\n{'-'*70}")
    print("STEP 3: VPP INCOME FROM LEFTOVER CAPACITY")
    print(f"{'-'*70}")
    print("Revenue streams: Arbitrage, Frequency Response (FR), Capacity Market (CM), Balancing Mechanism (BM)")

    # --- ARBITRAGE (using WHOLESALE market rates, not Agile retail) ---
    arb_result = calculate_wholesale_arbitrage(
        wholesale_df, battery_kwh, battery_kw, efficiency, consumer_reserved_kwh
    )
    arbitrage_income = arb_result['annual_income_gbp']

    print(f"\n1. Arbitrage (from {leftover_kwh:.2f} kWh leftover) - WHOLESALE RATES:")
    print(f"   Available capacity: {arb_result['available_kwh']:.2f} kWh")
    print(f"   Days profitable: {arb_result['days_profitable']}")
    print(f"   Annual income: £{arbitrage_income:.2f} (€{arb_result.get('annual_income_eur', 0):.2f})")

    # --- BALANCING MECHANISM (multiplier on arbitrage) ---
    bm_multiplier = calculate_balancing_mechanism_multiplier(params)
    arbitrage_with_bm = arbitrage_income * bm_multiplier
    bm_income = arbitrage_with_bm - arbitrage_income

    win_rate = params.get("Balancing mechanism (BM) avg win rate", 0)
    uplift = params.get("Avg £/kwh uplift for BM", 0)

    print(f"\n2. Balancing Mechanism (BM) - multiplier on arbitrage:")
    print(f"   Win rate: {win_rate*100:.1f}%, Uplift: {uplift*100:.1f}%")
    print(f"   Multiplier: {bm_multiplier:.4f}")
    print(f"   Arbitrage with BM: £{arbitrage_income:.2f} × {bm_multiplier:.4f} = £{arbitrage_with_bm:.2f}")
    print(f"   BM uplift: £{bm_income:.2f}")

    # --- FREQUENCY RESPONSE ---
    fr_income = calculate_frequency_response(params, leftover_kw, hours_per_day=24)
    fr_rate = params.get("Frequency response per kw per year", 0)

    print(f"\n3. Frequency Response (from {leftover_kw:.2f} kW leftover):")
    print(f"   Rate: £{fr_rate}/kW/year")
    print(f"   Annual income: £{fr_income:.2f}")

    # --- CAPACITY MARKET ---
    cm_income = calculate_capacity_market(params, leftover_kwh, leftover_kw)
    cm_rate = params.get("Capacity market /kW/year", 0)
    derating = get_derating_factor(leftover_kwh, leftover_kw) if leftover_kwh > 0 else 0
    duration = get_battery_duration_hours(leftover_kwh, leftover_kw) if leftover_kwh > 0 else 0

    print(f"\n4. Capacity Market (from {leftover_kw:.2f} kW leftover):")
    print(f"   Clearing price: £{cm_rate}/kW/year")
    print(f"   Duration: {duration}hr → De-rating: {derating*100:.1f}%")
    print(f"   De-rated capacity: {leftover_kw:.2f} kW × {derating*100:.1f}% = {leftover_kw * derating:.2f} kW")
    print(f"   Annual income: £{cm_income:.2f}")

    # --- CHOOSE BEST COMBINATION ---
    # Option A: Arbitrage + BM + CM (can do arbitrage AND earn CM)
    # Option B: FR + CM (FR is exclusive with arbitrage, but can stack with CM)

    option_arb = arbitrage_with_bm + cm_income
    option_fr = fr_income + cm_income

    print(f"\n--- VPP Strategy Options ---")
    print(f"   Option A (Arbitrage + BM + CM): £{arbitrage_with_bm:.2f} + £{cm_income:.2f} = £{option_arb:.2f}")
    print(f"   Option B (FR + CM):             £{fr_income:.2f} + £{cm_income:.2f} = £{option_fr:.2f}")

    if option_fr > option_arb:
        vpp_income = option_fr
        vpp_strategy = "FR + CM"
        vpp_breakdown = {'fr': fr_income, 'cm': cm_income, 'arbitrage': 0, 'bm': 0}
    else:
        vpp_income = option_arb
        vpp_strategy = "Arb + BM + CM"
        vpp_breakdown = {'fr': 0, 'cm': cm_income, 'arbitrage': arbitrage_with_bm, 'bm': bm_income}

    print(f"\n>>> Best strategy: {vpp_strategy}")
    print(f">>> VPP ANNUAL INCOME: £{vpp_income:.2f}")

    # =========================================
    # TOTAL VALUE
    # =========================================
    print(f"\n{'-'*70}")
    print("TOTAL ANNUAL VALUE")
    print(f"{'-'*70}")

    total_value = consumer_savings + vpp_income

    print(f"\n  Consumer savings:           £{consumer_savings:>10.2f}")
    print(f"  VPP income ({vpp_strategy}):  £{vpp_income:>10.2f}")
    if vpp_strategy == "FR + CM":
        print(f"    - Frequency Response:     £{fr_income:>10.2f}")
        print(f"    - Capacity Market:        £{cm_income:>10.2f}")
    else:
        print(f"    - Arbitrage (with BM):    £{arbitrage_with_bm:>10.2f}")
        print(f"    - Capacity Market:        £{cm_income:>10.2f}")
    print(f"  {'-'*40}")
    print(f"  TOTAL VALUE:                £{total_value:>10.2f}")

    # =========================================
    # AGILE SPREAD ANALYSIS
    # =========================================
    print(f"\n{'-'*70}")
    print("AGILE RATE STATISTICS (2025)")
    print(f"{'-'*70}")

    daily_stats = calculate_daily_agile_spread(agile_hourly)

    print(f"\n  Average daily min rate:    {daily_stats['min_rate'].mean():.2f} p/kWh")
    print(f"  Average daily max rate:    {daily_stats['max_rate'].mean():.2f} p/kWh")
    print(f"  Average daily spread:      {daily_stats['spread'].mean():.2f} p/kWh")
    print(f"  Average rate overall:      {daily_stats['avg_rate'].mean():.2f} p/kWh")
    print(f"\n  Max spread day:            {daily_stats['spread'].max():.2f} p/kWh")
    print(f"  Min spread day:            {daily_stats['spread'].min():.2f} p/kWh")

    return {
        'battery_kwh': battery_kwh,
        'battery_kw': battery_kw,
        'annual_consumption': annual_consumption,
        'cost_at_consumption': cost_at_consumption,
        'cost_from_battery': cost_from_battery,
        'consumer_savings': consumer_savings,
        'consumer_reserved_kwh': consumer_reserved_kwh,
        'leftover_kwh': leftover_kwh,
        'leftover_kw': leftover_kw,
        'arbitrage_income': arbitrage_income,
        'arbitrage_with_bm': arbitrage_with_bm,
        'bm_income': bm_income,
        'fr_income': fr_income,
        'cm_income': cm_income,
        'vpp_income': vpp_income,
        'vpp_strategy': vpp_strategy,
        'vpp_breakdown': vpp_breakdown,
        'total_value': total_value,
        'agile_stats': {
            'avg_min': daily_stats['min_rate'].mean(),
            'avg_max': daily_stats['max_rate'].mean(),
            'avg_spread': daily_stats['spread'].mean(),
        }
    }


def compare_battery_sizes(agile_hourly, params, use_profile, wholesale_df, sizes=None):
    """
    Compare different battery sizes for consumer-first VPP model.
    """
    if sizes is None:
        sizes = [5, 7.5, 10, 12.5, 15, 17.5, 20]

    results = []
    kw_per_kwh = params.get("kw per kwh of battery capacity", 0.5)

    print(f"\n{'='*90}")
    print("BATTERY SIZE COMPARISON")
    print(f"{'='*90}")

    for size in sizes:
        battery_kw = size * kw_per_kwh
        result = run_consumer_first_vpp_model(
            agile_hourly, params, use_profile, wholesale_df,
            battery_kwh=size, battery_kw=battery_kw
        )
        results.append(result)

    # Summary table
    print(f"\n{'='*120}")
    print("SUMMARY TABLE")
    print(f"{'='*120}")
    print(f"{'Size':>6} | {'Consumer':>10} | {'Arb+BM':>10} | {'FR':>8} | {'CM':>8} | {'VPP Tot':>10} | {'Strategy':>12} | {'Total':>10} | {'Left':>6}")
    print(f"{'(kWh)':>6} | {'Savings':>10} | {'':>10} | {'':>8} | {'':>8} | {'':>10} | {'':>12} | {'Value':>10} | {'(kWh)':>6}")
    print("-"*120)

    for r in results:
        arb_bm = r.get('arbitrage_with_bm', 0)
        fr = r.get('fr_income', 0)
        cm = r.get('cm_income', 0)
        print(f"{r['battery_kwh']:>6.1f} | £{r['consumer_savings']:>8.2f} | £{arb_bm:>8.2f} | £{fr:>6.2f} | £{cm:>6.2f} | £{r['vpp_income']:>8.2f} | {r['vpp_strategy']:>12} | £{r['total_value']:>8.2f} | {r['leftover_kwh']:>6.2f}")

    return results


def main(test_mode=False):
    """
    Run the consumer-first VPP model.

    Args:
        test_mode: If True, only scrape 30 days for quick testing
    """
    print("="*70)
    print("BATTERY MODEL - AGILE CONSUMER-FIRST VERSION")
    print("="*70)

    # Load model inputs
    print("\nLoading model inputs...")
    params = load_model_inputs()
    use_profile = load_use_profiles()

    # Load or scrape Agile data
    print("\nLoading Agile rate data...")
    days_limit = 30 if test_mode else None
    agile_df = scrape_agile_year(2025, use_cache=True, days_limit=days_limit)

    if agile_df is None or len(agile_df) == 0:
        print("ERROR: No Agile data available. Please check internet connection or cache file.")
        return

    print(f"Loaded {len(agile_df)} Agile rate records")

    # Convert to hourly
    agile_hourly = get_hourly_agile_rates(agile_df)
    print(f"Converted to {len(agile_hourly)} hourly Agile records")

    # Load wholesale market prices for VPP arbitrage
    print("\nLoading wholesale market prices for VPP arbitrage...")
    wholesale_df = load_wholesale_prices(2025)
    print(f"Loaded {len(wholesale_df)} wholesale price records")

    # Run single analysis with default battery
    result = run_consumer_first_vpp_model(agile_hourly, params, use_profile, wholesale_df)

    # Compare different battery sizes
    compare_results = compare_battery_sizes(
        agile_hourly, params, use_profile, wholesale_df,
        sizes=[5, 7.5, 10, 12.5, 15, 20]
    )

    # Final notes
    print(f"\n{'='*70}")
    print("NOTES")
    print(f"{'='*70}")
    print("""
1. Consumer savings = what they'd pay at AGILE RETAIL rates during consumption hours
   MINUS what they'd pay if battery charged at lowest daily Agile rates.
   (Uses Agile tariff data scraped from agilebuddy.uk)

2. VPP income comes from LEFTOVER capacity after serving consumer needs.
   Battery must store enough for daily consumption first.

3. VPP income sources (all from leftover capacity):
   - Arbitrage: Buy low, sell high on WHOLESALE MARKET (United Kingdom.csv, EUR/MWh)
   - Balancing Mechanism (BM): Multiplier on arbitrage (UK: 10% win rate × 5% uplift)
   - Frequency Response (FR): Payment per kW for grid frequency support
   - Capacity Market (CM): Payment per de-rated kW for capacity availability

   NOTE: Arbitrage uses WHOLESALE rates (not Agile retail) - this is what VPP
   actually earns from trading on the wholesale electricity market.

4. VPP Strategy options:
   - Option A: Arbitrage + BM + CM (can stack)
   - Option B: FR + CM (FR is exclusive with arbitrage)
   Best option is chosen based on total income.

5. CM de-rating factors (UK): 1hr=10.47%, 2hr=20.94%, 4hr=37%, 8hr=92%

6. Agile rates scraped from agilebuddy.uk for 2025 (cached locally).

7. Using UK consumption profile (from use_profiles.csv).
""")


if __name__ == "__main__":
    import sys
    test_mode = "--test" in sys.argv
    main(test_mode=test_mode)
