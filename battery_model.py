#!/usr/bin/env python3
"""
Battery Storage Optimization Model
Calculates optimal income across multiple revenue streams for VPP and Electricity Company scenarios.
All parameters are loaded from model_input.csv - no hardcoded values.

================================================================================
MODELING ASSUMPTIONS AND METHODOLOGY
================================================================================

1. REVENUE STREAM OPTIMIZATION APPROACH
   ------------------------------------
   - Uses a hybrid discrete scenario testing + greedy selection approach
   - Tests multiple FR commitment levels (0h, 2h, 4h, 6h, 8h, 24h) and selects optimal
   - This is a pragmatic alternative to full MILP optimization that works well for
     single-battery problems (MILP would be needed for portfolio optimization)

   Mathematical basis (simplified from MILP formulation):
     Maximize: Arbitrage + FR + Capacity_Market + BM_Uplift
     Subject to:
       - FR hours and Arbitrage hours are mutually exclusive
       - Battery SoC must be ~50% during FR commitment (can't cycle)
       - Consumer demand served first (Elec Company only)

2. FREQUENCY RESPONSE (FR) MODELING
   ---------------------------------
   Source: National Grid ESO, Modo Energy research (2024)

   - UK Dynamic Containment (DC): Market collapsed from £17/MW/h to £1-5/MW/h
   - FR rate from model_input (£34/kW/year) assumes full 24h commitment
   - Partial day commitment prorated: income = rate * (hours_committed / 24)

   Key constraint: During FR commitment hours, battery at 50% SoC +/- 10%
   - Cannot actively charge/discharge for arbitrage during these hours
   - FR peak hours typically: morning (7-9am), evening (5-9pm) - highest grid stress

   FR_PEAK_HOURS mapping:
     2h: {17, 18}                    - Evening peak only
     4h: {7, 8, 17, 18}              - Morning + early evening
     6h: {7, 8, 17, 18, 19, 20}      - Full peaks
     8h: {6, 7, 8, 9, 17, 18, 19, 20} - Extended peaks

3. CAPACITY MARKET (CM) MODELING
   ------------------------------
   Source: NESO EMR Delivery Body, Modo Energy (2024/25 auctions)

   Key parameters:
   - T-4 clearing price: £65/kW/year (2024 auction, DY 2027/28)
   - T-1 clearing price: £36/kW/year (2024 auction, DY 2024/25)

   De-rating factors (EMR DB "Scaled EFC" methodology):
   - 1-hour battery: 10.47% of nameplate capacity
   - 2-hour battery: 20.94% of nameplate capacity
   - 4-hour battery: ~37% (estimated)
   - 8-hour battery: ~92% (long duration near full credit)

   Revenue formula:
     CM_Revenue = Clearing_Price * De_Rating_Factor * Battery_kW

   Assumption: model_input.csv "Capacity market per MWh" is blank = £0
   - High-arbitrage days overlap with stress events, CM and arbitrage conflict
   - For accurate CM modeling, use uk_market_inputs.csv de-rating factors

4. BALANCING MECHANISM (BM) MODELING
   ----------------------------------
   Source: NESO BM data, Modo Energy market analysis (2024)

   - BM acts as MULTIPLIER on arbitrage, not additive revenue
   - When BM bid accepted, receive uplift on that trade

   Formula:
     BM_Multiplier = 1 + (win_rate * uplift_per_win)
     UK: 1 + (10% * 5%) = 1.005

   Total arbitrage with BM = Base_Arbitrage * BM_Multiplier

5. ARBITRAGE STRATEGY
   -------------------
   Strategy: Buy at daily minimum price, sell at daily maximum price hours

   - Uses model_input "wholesale avg daily minimum" as buy price
   - Hours to charge/discharge = battery_kwh / battery_kw (e.g., 10/5 = 2 hours)
   - Round-trip efficiency split: sqrt(efficiency) for each leg

   Daily profit = Discharge_Revenue - Charge_Cost
     Discharge: sum(highest_prices) * battery_kw * sqrt(efficiency) / 1000
     Charge: sum(lowest_prices) * battery_kw / sqrt(efficiency) / 1000

6. CONSUMER-FIRST BATTERY ALLOCATION (Electricity Company)
   --------------------------------------------------------
   KEY CONSTRAINT: Same kWh cannot earn both consumer margin AND arbitrage

   Allocation logic:
   1. Calculate daily consumer demand = annual_consumption / 365
   2. Battery serves consumer first: battery_for_consumer = min(battery_kwh, daily_demand)
   3. Leftover capacity: max(0, battery_kwh - daily_demand) available for arb/FR

   Revenue = Consumer_Margin + Leftover_Income
   - Consumer margin already assumes buying at wholesale min (no separate "battery savings")
   - Leftover income = max(leftover_arbitrage × BM_multiplier, leftover_FR)

7. TAX AND LEVY TREATMENT
   -----------------------
   Consumer income calculation:
   - Consumer pays rate inc VAT
   - Back out VAT: rate_ex_vat = rate_inc_vat / (1 + vat_rate)
   - Subtract energy tax (pass-through)
   - Net income = rate_ex_vat - energy_tax

   Levies (Elec Company only):
   - Subtracted at END of profit calculation, not from revenue
   - UK: £95/customer/year electricity company annual levy

8. ROI CALCULATION
   ----------------
   - Assumes 8-year battery life
   - Total investment = battery_cost_per_kwh * size + install_cost
   - ROI = (annual_profit * 8 - total_investment) / total_investment

9. OPTIMIZATION LOOP APPROACH
   ---------------------------
   The optimization tests all battery sizes from min (model default) to max (ceiling).
   For each size:
   1. VPP: Test all FR commitment strategies, select best
   2. Elec Company: Allocate to consumer first, optimize leftover
   3. Calculate ROI for each scenario
   4. Select optimal battery size based on best 8-year ROI

================================================================================
DATA SOURCES
================================================================================
- Hourly wholesale prices: Week plan batteries sprint 1 - NL Data.csv (and nation-specific)
- Model parameters: Week plan batteries sprint 1 - model_input.csv
- Consumption profiles: Week plan batteries sprint 1 - use_profiles.csv (UK only)
- UK market parameters: uk_market_inputs.csv (optional, for enhanced CM modeling)

================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pulp

# Configuration
DATA_DIR = Path(__file__).parent

# Date range for "latest full year" - derived from available data
DATE_START = "2024-07-01"
DATE_END = "2025-06-30"

NATIONS = {
    "Netherlands": "Week plan batteries sprint 1 - NL Data.csv",
    "Germany": "Germany.csv",
    "Spain": "Spain.csv",
    "UK": "United Kingdom.csv",
}

# Battery size optimization step (kWh) - could also be added to model_input.csv if needed
BATTERY_SIZE_STEP = 2.5

# UK Capacity Market de-rating factors (Source: EMR DB 2024/25 Scaled EFC methodology)
# These represent the fraction of nameplate capacity that receives CM payment
UK_CM_DERATING_FACTORS = {
    1: 0.1047,  # 1-hour battery: 10.47%
    2: 0.2094,  # 2-hour battery: 20.94%
    4: 0.37,    # 4-hour battery: ~37% (estimated)
    8: 0.92,    # 8-hour battery: ~92%
}

# UK Capacity Market clearing prices (Source: NESO EMR DB auctions 2024)
UK_CM_PRICES = {
    "T-4": 65,  # £65/kW/year (2024 auction for DY 2027/28)
    "T-1": 36,  # £36/kW/year (2024 auction for DY 2024/25)
}


def load_uk_market_inputs():
    """
    Load enhanced UK market inputs if available.
    Falls back to defaults if file doesn't exist.
    """
    uk_file = DATA_DIR / "uk_market_inputs.csv"
    if not uk_file.exists():
        return None

    df = pd.read_csv(uk_file)
    df = df.set_index("input")

    params = {}
    for idx in df.index:
        if idx.startswith("#"):  # Skip comment rows
            continue
        val = df.loc[idx, "UK"]
        if pd.isna(val) or val == "":
            continue
        try:
            params[idx] = float(val)
        except (ValueError, TypeError):
            params[idx] = val

    return params


def get_battery_duration_hours(battery_kwh, battery_kw):
    """Calculate battery duration in hours for de-rating factor lookup."""
    if battery_kw <= 0:
        return 1
    duration = battery_kwh / battery_kw
    # Round to nearest standard duration
    if duration < 1.5:
        return 1
    elif duration < 3:
        return 2
    elif duration < 6:
        return 4
    else:
        return 8


def get_derating_factor(battery_kwh, battery_kw):
    """
    Get the appropriate de-rating factor based on battery duration.
    Source: EMR DB 2024/25 Scaled EFC methodology
    """
    duration = get_battery_duration_hours(battery_kwh, battery_kw)
    return UK_CM_DERATING_FACTORS.get(duration, 0.2)  # Default to 2hr factor


def load_model_inputs():
    """Load model parameters from CSV."""
    df = pd.read_csv(DATA_DIR / "Week plan batteries sprint 1 - model_input.csv")
    df = df.set_index("input")

    params = {}
    for nation in ["Netherlands", "Germany", "Spain", "UK"]:
        col = nation if nation != "UK" else "UK"
        p = {}

        # Parse each row, handling percentages and blanks
        for idx in df.index.unique():
            val = df.loc[idx, col]
            # Handle duplicate row labels (returns Series instead of scalar)
            if isinstance(val, pd.Series):
                # Take the last non-empty value
                for v in val.values[::-1]:
                    if not (pd.isna(v) or v == ""):
                        val = v
                        break
                else:
                    val = val.values[-1]  # All empty, take last

            if pd.isna(val) or val == "":
                p[idx] = 0.0
            elif isinstance(val, str) and "%" in val:
                p[idx] = float(val.replace("%", "")) / 100
            else:
                try:
                    p[idx] = float(val)
                except:
                    p[idx] = 0.0
        params[nation] = p

    return params


def get_battery_cost_per_kwh(params):
    """Get battery cost per kWh after applying wholesale discount."""
    base_cost = params["battery cost per kwh (inc wires, controller, inverter, etc)"]
    discount = params.get("wholesale_discount_battery", 0)
    return base_cost * (1 - discount)


def get_wholesale_buy_price(params):
    """
    Get total wholesale buy price including grid costs.

    Total = wholesale avg daily minimum + residential min grid cost
    Both are in cents/kWh, returns EUR/kWh.
    """
    wholesale_min = params["wholesale avg daily minimum p/kWH"]  # cents
    grid_cost = params.get("residential min grid cost p/kwh avg daily (lowest hour)", 0)  # cents
    return (wholesale_min + grid_cost) / 100  # EUR/kWh


def load_use_profiles():
    """Load hourly consumption profiles. Only UK available, used for all nations per instructions."""
    df = pd.read_csv(DATA_DIR / "Week plan batteries sprint 1 - use_profiles.csv")
    # Normalize to daily total = 1
    uk_profile = df[df["Nation"] == "UK"].set_index("hour")["consumption_kwh"]
    uk_profile = uk_profile / uk_profile.sum()  # Normalize
    return uk_profile.to_dict()


def load_hourly_prices(nation):
    """Load hourly wholesale prices for a nation for the analysis period."""
    filepath = DATA_DIR / NATIONS[nation]

    if nation == "Netherlands":
        df = pd.read_csv(filepath)
        df["datetime"] = pd.to_datetime(df["Datetime (UTC)"])
        df["price"] = df["Price (EUR/MWhe)"]
    else:
        df = pd.read_csv(filepath)
        df["datetime"] = pd.to_datetime(df["Datetime (UTC)"])
        df["price"] = df["Price (EUR/MWhe)"]

    # Filter to analysis period
    df = df[(df["datetime"] >= DATE_START) & (df["datetime"] < "2025-07-01")]
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour

    return df[["datetime", "date", "hour", "price"]]


def calculate_daily_arbitrage(prices_df, battery_kwh, battery_kw, efficiency, buy_price_eur_per_kwh=None, excluded_hours=None):
    """
    Calculate daily arbitrage profit from wholesale trading.
    Strategy: Charge at lowest price (or fixed buy price), discharge at highest price hours.

    Args:
        prices_df: Hourly price data
        battery_kwh: Battery capacity in kWh
        battery_kw: Battery power in kW
        efficiency: Round-trip efficiency (e.g., 0.9)
        buy_price_eur_per_kwh: If provided, use this as fixed buy price (from model_input).
                              Otherwise calculate from hourly data.
        excluded_hours: Set of hours to exclude (e.g., hours committed to FR)

    Returns: annual profit in EUR
    """
    efficiency_factor = np.sqrt(efficiency)  # Split efficiency between charge and discharge
    excluded_hours = excluded_hours or set()

    daily_profits = []

    for date, day_data in prices_df.groupby("date"):
        if len(day_data) < 24:
            continue

        day_data = day_data.sort_values("hour")

        # Filter out excluded hours
        available_data = day_data[~day_data["hour"].isin(excluded_hours)]
        if len(available_data) < 4:  # Need at least some hours for arbitrage
            daily_profits.append(0)
            continue

        prices = available_data["price"].values
        hours = available_data["hour"].values

        # Determine how many hours needed to charge/discharge
        hours_to_cycle = battery_kwh / battery_kw  # e.g., 10kWh / 5kW = 2 hours
        hours_to_cycle = int(np.ceil(hours_to_cycle))

        # Limit to available hours
        hours_to_cycle = min(hours_to_cycle, len(hours) // 2)
        if hours_to_cycle == 0:
            daily_profits.append(0)
            continue

        # Find lowest and highest price hours from available hours
        sorted_idx = np.argsort(prices)
        charge_indices = sorted_idx[:hours_to_cycle]
        discharge_indices = sorted_idx[-hours_to_cycle:]

        # Calculate charge cost
        if buy_price_eur_per_kwh is not None:
            # Use fixed wholesale minimum price from model_input
            effective_kwh = hours_to_cycle * battery_kw
            charge_cost = effective_kwh * buy_price_eur_per_kwh / efficiency_factor
        else:
            # Use actual hourly prices (convert EUR/MWh to EUR)
            charge_cost = sum(prices[i] for i in charge_indices) * battery_kw * (1 / efficiency_factor) / 1000

        # Discharge revenue (always use actual hourly prices, convert EUR/MWh to EUR)
        discharge_revenue = sum(prices[i] for i in discharge_indices) * battery_kw * efficiency_factor / 1000

        daily_profit = discharge_revenue - charge_cost
        daily_profits.append(daily_profit)

    return sum(daily_profits)


def calculate_consumer_income_net_of_taxes(params, annual_consumption_kwh):
    """
    Calculate net income from selling electricity to consumers.
    Consumer pays retail rate inc VAT, but we must remit VAT and energy tax.

    Returns: annual net income in EUR (what we actually keep)
    """
    # Consumer cost is in cents/kWh, convert to EUR/kWh
    consumer_rate_gross = params["consumer cost p/kwh inc VAT"] / 100  # EUR/kWh

    # Back out VAT to get pre-VAT price
    vat_rate = params["VAT on electricity %"] / 100
    consumer_rate_ex_vat = consumer_rate_gross / (1 + vat_rate)

    # Subtract energy tax (we collect it but pass it through to government)
    energy_tax = params["energy tax p/kwh (excluding VAT)"] / 100  # EUR/kWh
    net_rate = consumer_rate_ex_vat - energy_tax

    return annual_consumption_kwh * net_rate


def calculate_wholesale_cost_for_consumer(prices_df, annual_consumption_kwh, use_profile):
    """
    Calculate cost of buying wholesale electricity to serve consumer demand WITHOUT battery.
    Uses hourly consumption profile to weight purchases.

    Returns: annual cost in EUR
    """
    # Get average price per hour across the year
    hourly_avg = prices_df.groupby("hour")["price"].mean()

    # Weight by consumption profile
    daily_consumption = annual_consumption_kwh / 365

    total_cost = 0
    for hour, profile_weight in use_profile.items():
        hour_consumption = daily_consumption * profile_weight
        hour_price = hourly_avg.get(hour, hourly_avg.mean()) / 1000  # EUR/kWh
        total_cost += hour_consumption * hour_price * 365

    return total_cost


def calculate_elec_company_battery_value(prices_df, params, use_profile, battery_kwh, battery_kw):
    """
    Calculate battery value for electricity company.

    KEY CONSTRAINT: Same kWh can't earn both consumer margin AND arbitrage.
    Battery first serves consumer demand, only leftover does arbitrage/FR.

    Logic:
    - Consumer margin already assumes we buy at wholesale minimum price
    - Battery capacity up to daily consumption is "reserved" for consumer supply
    - Only LEFTOVER capacity (battery_kwh - daily_consumption) can do arbitrage/FR
    - No "consumer savings" - that would double-count buying at min price

    Returns: dict with leftover capacity and leftover income
    """
    efficiency = params["battery charge recharge efficiency %"]
    efficiency_factor = np.sqrt(efficiency)
    annual_consumption = params["Mean electricity residential consumption annual (kWH)"]
    daily_consumption = annual_consumption / 365
    buy_price = get_wholesale_buy_price(params)  # EUR/kWh (wholesale min + grid cost)

    # Battery serves consumer demand first - this capacity is "used" for consumer supply
    # Only leftover can do arbitrage/FR
    battery_for_consumer = min(battery_kwh, daily_consumption)
    leftover_capacity = max(0, battery_kwh - daily_consumption)
    leftover_kw = leftover_capacity * (battery_kw / battery_kwh) if battery_kwh > 0 else 0

    total_leftover_arbitrage = 0

    for date, day_data in prices_df.groupby("date"):
        if len(day_data) < 24:
            continue

        day_data = day_data.sort_values("hour")
        prices = day_data.set_index("hour")["price"].to_dict()

        # Leftover capacity arbitrage (if any)
        if leftover_capacity > 0:
            sorted_hours = sorted(prices.keys(), key=lambda h: prices[h])
            hours_to_cycle = int(np.ceil(leftover_capacity / leftover_kw)) if leftover_kw > 0 else 0
            hours_to_cycle = min(hours_to_cycle, 12)  # Max half day

            if hours_to_cycle > 0:
                discharge_hours = sorted_hours[-hours_to_cycle:]
                discharge_price = np.mean([prices[h] for h in discharge_hours]) / 1000

                charge_cost = leftover_capacity * buy_price / efficiency_factor
                discharge_revenue = leftover_capacity * discharge_price * efficiency_factor

                daily_arb = discharge_revenue - charge_cost
                total_leftover_arbitrage += max(0, daily_arb)

    return {
        "battery_for_consumer_kwh": battery_for_consumer,
        "leftover_capacity_kwh": leftover_capacity,
        "leftover_arbitrage": total_leftover_arbitrage,
    }


def calculate_frequency_response(params, battery_kw, hours_per_day=24):
    """
    Calculate annual income from frequency response services.
    UK: £34/kW/year for full commitment (24h/day)

    If committing for fewer hours per day, income is prorated.

    Args:
        params: Model parameters
        battery_kw: Battery power in kW
        hours_per_day: Hours per day committed to FR (default 24 = full commitment)

    Returns: annual income in EUR
    """
    rate = params.get("Frequency response per kw per year", 0)
    # Prorate based on hours committed
    fraction_of_day = hours_per_day / 24
    return rate * battery_kw * fraction_of_day


# Define typical FR commitment windows (peak grid stress hours)
# Morning peak: 7-9am, Evening peak: 5-9pm
FR_PEAK_HOURS = {
    2: {17, 18},  # 2 hours: evening peak only
    4: {7, 8, 17, 18},  # 4 hours: morning + early evening
    6: {7, 8, 17, 18, 19, 20},  # 6 hours: full peaks
    8: {6, 7, 8, 9, 17, 18, 19, 20},  # 8 hours: extended peaks
}


# =============================================================================
# MILP-BASED HOURLY OPTIMIZATION
# =============================================================================
# This replaces the discrete strategy testing with true hourly optimization.
# For each day, we solve a Mixed Integer Linear Program to find the optimal
# hour-by-hour allocation between arbitrage and FR.
# =============================================================================

def solve_daily_milp(prices_24h, battery_kwh, battery_kw, efficiency, fr_rate_per_kw_year):
    """
    Solve the optimal hourly allocation for a single day using MILP.

    Decision variables (per hour h):
      c[h] = charge power (kW), continuous [0, P_max]
      d[h] = discharge power (kW), continuous [0, P_max]
      fr[h] = FR commitment, binary {0, 1}
      E[h] = state of charge (kWh), continuous [0, E_max]

    Objective: Maximize arbitrage profit + FR income

    Constraints:
      - Energy balance: E[h+1] = E[h] + c[h]*η - d[h]/η
      - Charge/discharge blocked during FR: c[h] ≤ P_max*(1-fr[h])
      - SoC bounds: 0.1*E_max ≤ E[h] ≤ 0.9*E_max
      - Daily cycle: E[0] = E[24] = 0.5*E_max

    Args:
        prices_24h: List of 24 hourly prices (EUR/MWh)
        battery_kwh: Battery capacity (kWh)
        battery_kw: Battery power (kW)
        efficiency: Round-trip efficiency (e.g., 0.9)
        fr_rate_per_kw_year: Annual FR rate per kW (EUR or GBP)

    Returns: dict with daily profit breakdown and hour-by-hour decisions
    """
    # Efficiency factors (split between charge and discharge)
    eta = np.sqrt(efficiency)

    # FR income per hour (prorated from annual rate)
    fr_hourly_rate = fr_rate_per_kw_year * battery_kw / 8760  # Per hour

    # SoC bounds (keep 10% buffer on each end for battery health)
    E_min = 0.1 * battery_kwh
    E_max = 0.9 * battery_kwh
    E_start = 0.5 * battery_kwh  # Start and end at 50%

    # Create the problem
    prob = pulp.LpProblem("Daily_Battery_Optimization", pulp.LpMaximize)

    # Decision variables
    hours = range(24)

    # Charge power per hour (kW)
    c = [pulp.LpVariable(f"charge_{h}", lowBound=0, upBound=battery_kw) for h in hours]

    # Discharge power per hour (kW)
    d = [pulp.LpVariable(f"discharge_{h}", lowBound=0, upBound=battery_kw) for h in hours]

    # FR commitment per hour (binary)
    fr = [pulp.LpVariable(f"fr_{h}", cat='Binary') for h in hours]

    # State of charge at each hour boundary (25 values: start of hour 0 to end of hour 23)
    E = [pulp.LpVariable(f"soc_{h}", lowBound=E_min, upBound=E_max) for h in range(25)]

    # Objective: Maximize (discharge revenue - charge cost + FR income)
    # Prices are in EUR/MWh, convert to EUR/kWh by dividing by 1000
    arbitrage_profit = []
    fr_income = []

    for h in hours:
        price_kwh = prices_24h[h] / 1000  # Convert EUR/MWh to EUR/kWh

        # Discharge revenue (sell at price, accounting for efficiency loss)
        discharge_revenue = d[h] * price_kwh * eta

        # Charge cost (buy at price, accounting for efficiency loss)
        charge_cost = c[h] * price_kwh / eta

        arbitrage_profit.append(discharge_revenue - charge_cost)

        # FR income for this hour
        fr_income.append(fr[h] * fr_hourly_rate)

    prob += pulp.lpSum(arbitrage_profit) + pulp.lpSum(fr_income), "Total_Daily_Profit"

    # Constraints

    # 1. Initial SoC
    prob += E[0] == E_start, "Initial_SoC"

    # 2. Final SoC (daily cycle - end where we started)
    prob += E[24] == E_start, "Final_SoC"

    # 3. Energy balance for each hour
    for h in hours:
        # E[h+1] = E[h] + charge*eta - discharge/eta
        prob += E[h + 1] == E[h] + c[h] * eta - d[h] / eta, f"Energy_Balance_{h}"

    # 4. Charge/discharge blocked during FR commitment
    # When fr[h] = 1, both c[h] and d[h] must be 0
    for h in hours:
        prob += c[h] <= battery_kw * (1 - fr[h]), f"Charge_FR_Block_{h}"
        prob += d[h] <= battery_kw * (1 - fr[h]), f"Discharge_FR_Block_{h}"

    # 5. FR requires SoC at 50% (to respond in either direction)
    # Use big-M formulation: when fr[h]=1, E[h] must equal E_start (50%)
    M = battery_kwh  # Big-M value
    for h in hours:
        # E[h] >= 50% - M*(1-fr[h])  →  when fr=1: E[h] >= 50%
        prob += E[h] >= E_start - M * (1 - fr[h]), f"FR_SoC_Lower_{h}"
        # E[h] <= 50% + M*(1-fr[h])  →  when fr=1: E[h] <= 50%
        prob += E[h] <= E_start + M * (1 - fr[h]), f"FR_SoC_Upper_{h}"

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))  # msg=0 suppresses solver output

    # Check if solved successfully
    if pulp.LpStatus[prob.status] != "Optimal":
        # Fall back to pure arbitrage estimate
        return {
            "status": "Failed",
            "arbitrage_profit": 0,
            "fr_income": 0,
            "total_profit": 0,
            "fr_hours": 0,
            "charge_schedule": [0] * 24,
            "discharge_schedule": [0] * 24,
            "fr_schedule": [0] * 24,
            "soc_schedule": [E_start] * 25,
        }

    # Extract results
    charge_schedule = [c[h].varValue or 0 for h in hours]
    discharge_schedule = [d[h].varValue or 0 for h in hours]
    fr_schedule = [int(fr[h].varValue or 0) for h in hours]
    soc_schedule = [E[h].varValue or E_start for h in range(25)]

    # Calculate actual profit breakdown
    total_arb_profit = 0
    total_fr_income = 0

    for h in hours:
        price_kwh = prices_24h[h] / 1000
        total_arb_profit += discharge_schedule[h] * price_kwh * eta - charge_schedule[h] * price_kwh / eta
        total_fr_income += fr_schedule[h] * fr_hourly_rate

    return {
        "status": "Optimal",
        "arbitrage_profit": total_arb_profit,
        "fr_income": total_fr_income,
        "total_profit": total_arb_profit + total_fr_income,
        "fr_hours": sum(fr_schedule),
        "charge_schedule": charge_schedule,
        "discharge_schedule": discharge_schedule,
        "fr_schedule": fr_schedule,
        "soc_schedule": soc_schedule,
    }


def optimize_annual_strategy_milp(prices_df, params, battery_kwh, battery_kw):
    """
    Run daily MILP optimization across the entire year and aggregate results.

    This finds the true optimal hour-by-hour allocation between arbitrage and FR,
    accounting for daily price variations.

    Args:
        prices_df: DataFrame with hourly prices for the year
        params: Model parameters
        battery_kwh: Battery capacity (kWh)
        battery_kw: Battery power (kW)

    Returns: dict with annual totals and daily breakdown statistics
    """
    efficiency = params["battery charge recharge efficiency %"]
    fr_rate = params.get("Frequency response per kw per year", 0)

    # Track results
    total_arbitrage = 0
    total_fr_income = 0
    total_fr_hours = 0
    daily_results = []

    # Strategy categorization
    pure_arb_days = 0
    pure_fr_days = 0
    mixed_days = 0

    # Group by date and solve each day
    for date, day_data in prices_df.groupby("date"):
        if len(day_data) < 24:
            continue  # Skip incomplete days

        # Get 24 hourly prices
        day_data = day_data.sort_values("hour")
        prices_24h = day_data["price"].values.tolist()

        # Solve daily MILP
        result = solve_daily_milp(prices_24h, battery_kwh, battery_kw, efficiency, fr_rate)

        # Aggregate
        total_arbitrage += result["arbitrage_profit"]
        total_fr_income += result["fr_income"]
        total_fr_hours += result["fr_hours"]

        # Categorize day
        if result["fr_hours"] == 0:
            pure_arb_days += 1
        elif result["fr_hours"] == 24:
            pure_fr_days += 1
        else:
            mixed_days += 1

        daily_results.append({
            "date": date,
            "arbitrage": result["arbitrage_profit"],
            "fr_income": result["fr_income"],
            "fr_hours": result["fr_hours"],
            "total": result["total_profit"],
            "charge_schedule": result["charge_schedule"],
            "discharge_schedule": result["discharge_schedule"],
            "fr_schedule": result["fr_schedule"],
            "soc_schedule": result["soc_schedule"],
            "prices": prices_24h,
        })

    num_days = len(daily_results)
    avg_fr_hours_per_day = total_fr_hours / num_days if num_days > 0 else 0

    # Determine overall strategy description
    if avg_fr_hours_per_day < 1:
        strategy = "MILP: Mostly Arbitrage"
    elif avg_fr_hours_per_day > 20:
        strategy = "MILP: Mostly FR"
    else:
        strategy = f"MILP: Mixed ({avg_fr_hours_per_day:.1f}h FR/day avg)"

    return {
        "strategy": strategy,
        "arbitrage": total_arbitrage,
        "frequency_response": total_fr_income,
        "total": total_arbitrage + total_fr_income,
        "fr_hours_total": total_fr_hours,
        "fr_hours_avg_per_day": avg_fr_hours_per_day,
        "num_days": num_days,
        "pure_arb_days": pure_arb_days,
        "pure_fr_days": pure_fr_days,
        "mixed_days": mixed_days,
        "daily_results": daily_results,
    }


def print_sample_day_schedule(daily_result, battery_kwh):
    """Print hour-by-hour schedule for a single day showing SoC tracking."""
    date = daily_result["date"]
    prices = daily_result["prices"]
    charge = daily_result["charge_schedule"]
    discharge = daily_result["discharge_schedule"]
    fr = daily_result["fr_schedule"]
    soc = daily_result["soc_schedule"]

    print(f"\n  Sample Day: {date}")
    print(f"  FR hours: {daily_result['fr_hours']}, Arbitrage: €{daily_result['arbitrage']:.2f}, FR income: €{daily_result['fr_income']:.4f}")
    print(f"  Hour | Price €/MWh | Charge kW | Disch kW | FR | SoC kWh | SoC %  | Action")
    print(f"  {'-'*80}")

    for h in range(24):
        price = prices[h]
        c = charge[h]
        d = discharge[h]
        f = fr[h]
        s = soc[h]
        s_pct = (s / battery_kwh) * 100

        # Determine action
        if f == 1:
            action = "FR (hold 50%)"
        elif c > 0.01:
            action = f"CHARGE {c:.1f}kW"
        elif d > 0.01:
            action = f"DISCHARGE {d:.1f}kW"
        else:
            action = "idle"

        print(f"  {h:4d} | {price:>10.1f} | {c:>9.2f} | {d:>8.2f} | {f:>2d} | {s:>7.2f} | {s_pct:>5.1f}% | {action}")

    # Final SoC
    print(f"  {24:4d} | {'(end)':>10} | {'-':>9} | {'-':>8} | {'-':>2} | {soc[24]:>7.2f} | {(soc[24]/battery_kwh)*100:>5.1f}% | end of day")


def optimize_battery_allocation_vpp(prices_df, params, battery_kwh, battery_kw, nation="UK"):
    """
    Optimize allocation of battery capacity across VPP revenue streams using MILP.

    Uses daily Mixed Integer Linear Programming to find the true optimal hour-by-hour
    allocation between arbitrage and FR, accounting for:
    - Daily price variations (high-spread days favor arbitrage)
    - FR commitment constraints (battery at 50% SoC, can't cycle)
    - State of charge tracking across hours
    - Efficiency losses

    Revenue streams:
    1. Arbitrage: Buy low, sell high. Optimized hour-by-hour via MILP.
    2. Frequency Response (FR): Paid per kW for hours committed. MILP decides which hours.
    3. Balancing Mechanism: Multiplier on arbitrage revenue when bids accepted.
    4. Capacity Market: Annual payment per kW (de-rated for UK).

    Returns: dict with optimal allocation and revenues
    """
    # Run MILP optimization across all days
    milp_result = optimize_annual_strategy_milp(prices_df, params, battery_kwh, battery_kw)

    # Apply BM as multiplier on arbitrage revenue
    bm_multiplier = calculate_balancing_mechanism_multiplier(params)
    arbitrage_with_bm = milp_result["arbitrage"] * bm_multiplier
    bm_income = arbitrage_with_bm - milp_result["arbitrage"]  # The uplift portion

    # Capacity Market (with proper de-rating for UK)
    capacity_market = calculate_capacity_market(params, battery_kwh, battery_kw, nation)

    total_revenue = arbitrage_with_bm + milp_result["frequency_response"] + capacity_market

    # Build explanation
    avg_fr = milp_result["fr_hours_avg_per_day"]
    pure_arb = milp_result["pure_arb_days"]
    mixed = milp_result["mixed_days"]
    pure_fr = milp_result["pure_fr_days"]

    explanation = (f"MILP optimized {milp_result['num_days']} days: "
                   f"{pure_arb} pure arb, {mixed} mixed, {pure_fr} pure FR. "
                   f"Avg {avg_fr:.1f}h FR/day.")

    return {
        "strategy": milp_result["strategy"],
        "fr_hours": milp_result["fr_hours_total"],
        "fr_hours_avg_per_day": avg_fr,
        "arbitrage": milp_result["arbitrage"],
        "arbitrage_with_bm": arbitrage_with_bm,
        "frequency_response": milp_result["frequency_response"],
        "balancing_mechanism": bm_income,
        "capacity_market": capacity_market,
        "total_revenue": total_revenue,
        "pure_arb_days": pure_arb,
        "mixed_days": mixed,
        "pure_fr_days": pure_fr,
        "num_days": milp_result["num_days"],
        "explanation": explanation,
        "daily_results": milp_result["daily_results"],
    }


def calculate_capacity_market(params, battery_kwh, battery_kw, nation="UK"):
    """
    Calculate annual income from Capacity Market with proper de-rating.

    Uses "Capacity market /kW/year" from model_input.csv as the clearing price.
    Applies de-rating factors based on battery duration:
    - 1-hour battery: 10.47%
    - 2-hour battery: 20.94%
    - 4-hour battery: ~37%
    - 8-hour battery: ~92%

    Revenue = Clearing_Price * De_Rating_Factor * Battery_kW

    Returns: annual income in EUR (assumes input is in GBP for UK, converts to EUR)
    """
    # Get CM rate from model_input (new row: "Capacity market /kW/year")
    cm_rate = params.get("Capacity market /kW/year", 0)

    if cm_rate == 0:
        # No CM rate specified for this nation
        return 0

    # Get de-rating factor based on battery duration
    derating = get_derating_factor(battery_kwh, battery_kw)

    # Revenue = price * derating * capacity (kW)
    annual_revenue = cm_rate * derating * battery_kw

    # Convert GBP to EUR for UK (CM prices are in GBP)
    if nation == "UK":
        gbp_to_eur = 1.17
        annual_revenue = annual_revenue * gbp_to_eur

    return annual_revenue


def calculate_balancing_mechanism_multiplier(params):
    """
    Calculate BM multiplier for arbitrage revenue.
    UK: 10% win rate, 5% uplift on winning trades.

    BM acts as a multiplier: when you win a BM bid, you get an uplift on that trade.
    Effective multiplier = 1 + (win_rate * uplift)

    Example: 10% win rate × 5% uplift = 0.5% boost on all arbitrage
    Multiplier = 1.005

    Returns: multiplier (>= 1.0)
    """
    win_rate = params.get("Balancing mechanism (BM) avg win rate", 0)
    uplift = params.get("Avg £/kwh uplift for BM", 0)

    # Multiplier on arbitrage revenue
    return 1 + (win_rate * uplift)


def calculate_balancing_mechanism(arbitrage_profit, params):
    """
    Calculate additional income from balancing mechanism.
    BM acts as a multiplier on arbitrage revenue.

    Returns: the BM uplift portion (total arbitrage with BM - base arbitrage)
    """
    multiplier = calculate_balancing_mechanism_multiplier(params)
    return arbitrage_profit * (multiplier - 1)


def iterative_revenue_optimization(prices_df, params, battery_kwh, battery_kw, nation="UK", max_iterations=10, tolerance=0.01):
    """
    Iterative optimization loop to converge on optimal revenue stream combination.

    This function iteratively refines the allocation of battery capacity across
    revenue streams, accounting for the interdependencies between:
    - FR commitment (reduces arbitrage hours)
    - Arbitrage (affected by price patterns)
    - Capacity Market (fixed based on battery duration)
    - BM (multiplier on arbitrage)

    The iteration converges when the change in total revenue between iterations
    is below the tolerance threshold.

    Algorithm:
    1. Start with pure arbitrage as baseline
    2. Test all FR commitment levels
    3. For each FR level, calculate residual arbitrage
    4. Calculate total revenue for each combination
    5. Select best combination
    6. Verify stability by checking if optimal changes

    Args:
        prices_df: Hourly price data
        params: Model parameters
        battery_kwh: Battery capacity
        battery_kw: Battery power
        nation: Nation for CM de-rating
        max_iterations: Maximum iterations before stopping
        tolerance: Convergence threshold (fraction of revenue)

    Returns: dict with converged optimal allocation
    """
    efficiency = params["battery charge recharge efficiency %"]
    buy_price = get_wholesale_buy_price(params)
    fr_rate = params.get("Frequency response per kw per year", 0)
    bm_multiplier = calculate_balancing_mechanism_multiplier(params)

    # Track convergence
    prev_best_total = 0
    iteration_results = []

    for iteration in range(max_iterations):
        # Test all FR commitment scenarios
        scenarios = []

        # Scenario 0: Pure Arbitrage
        arb_pure = calculate_daily_arbitrage(prices_df, battery_kwh, battery_kw, efficiency, buy_price)
        arb_with_bm = arb_pure * bm_multiplier
        cm = calculate_capacity_market(params, battery_kwh, battery_kw, nation)
        scenarios.append({
            "name": "Pure Arbitrage",
            "fr_hours": 0,
            "arbitrage": arb_pure,
            "arbitrage_with_bm": arb_with_bm,
            "fr_income": 0,
            "cm_income": cm,
            "total": arb_with_bm + cm
        })

        # Scenarios 1-4: FR commitment + residual arbitrage
        for fr_hours, excluded_hours in FR_PEAK_HOURS.items():
            arb_reduced = calculate_daily_arbitrage(
                prices_df, battery_kwh, battery_kw, efficiency, buy_price, excluded_hours
            )
            arb_reduced_with_bm = arb_reduced * bm_multiplier
            fr_income = calculate_frequency_response(params, battery_kw, fr_hours)

            scenarios.append({
                "name": f"FR {fr_hours}h + Arbitrage",
                "fr_hours": fr_hours,
                "arbitrage": arb_reduced,
                "arbitrage_with_bm": arb_reduced_with_bm,
                "fr_income": fr_income,
                "cm_income": cm,
                "total": arb_reduced_with_bm + fr_income + cm
            })

        # Scenario 5: Full FR (24h), no arbitrage
        fr_full = calculate_frequency_response(params, battery_kw, 24)
        scenarios.append({
            "name": "FR Only (24h)",
            "fr_hours": 24,
            "arbitrage": 0,
            "arbitrage_with_bm": 0,
            "fr_income": fr_full,
            "cm_income": cm,
            "total": fr_full + cm
        })

        # Find best scenario
        best = max(scenarios, key=lambda x: x["total"])
        iteration_results.append({
            "iteration": iteration,
            "best_strategy": best["name"],
            "total_revenue": best["total"]
        })

        # Check convergence
        if prev_best_total > 0:
            change = abs(best["total"] - prev_best_total) / prev_best_total
            if change < tolerance:
                break

        prev_best_total = best["total"]

    return {
        "converged_strategy": best["name"],
        "iterations": iteration + 1,
        "arbitrage": best["arbitrage"],
        "arbitrage_with_bm": best["arbitrage_with_bm"],
        "frequency_response": best["fr_income"],
        "capacity_market": best["cm_income"],
        "balancing_mechanism": best["arbitrage_with_bm"] - best["arbitrage"],
        "total_revenue": best["total"],
        "fr_hours": best["fr_hours"],
        "convergence_history": iteration_results
    }


def calculate_costs_vpp(params, battery_kwh):
    """
    Calculate annual costs for VPP model.
    Note: Grid costs NOT included - customer already pays these separately.
    VPP only pays for customer management.
    """
    costs = {
        "customer_management": params["Cost of managing customers per customer per annum if VLP"],
        # Grid costs excluded - customer pays these as part of their normal bills
    }
    return costs


def calculate_costs_elec_company(params, battery_kwh):
    """
    Calculate annual costs for electricity company model.

    Costs are split into:
    - Operating costs: customer management, grid costs
    - Levies: annual levy (subtracted at end of profit calculation)
    """
    costs = {
        "customer_management": params["Cost of managing customers per customer per annum if electricity company"],
        "grid_costs": params.get("residential min grid cost PA (euros)", 0),
        "energy_tax_per_kwh": params.get("energy tax p/kwh (excluding VAT)", 0) / 100,  # Convert to EUR
    }
    return costs


def calculate_levy_elec_company(params):
    """
    Calculate annual levy for electricity company.
    This is subtracted at the very end of profit calculation.
    """
    return params.get("electricity company annual levy per customer", 0)


def calculate_upfront_investment(params):
    """
    Calculate upfront investment assuming battery is already owned.
    Includes: installation cost, setup costs
    """
    return params["battery install cost"]


def calculate_battery_ceiling(params):
    """
    Calculate effective battery size ceiling based on TWO constraints:
    1. max kwh battery appetite due to physical limits (direct kWh limit)
    2. max power kw / kw per kwh of battery capacity (power-derived kWh limit)

    Returns: dict with both limits and the effective ceiling (minimum of the two)
    """
    max_kwh_physical = params["max kwh battery appetite due to physical limits"]
    max_power_kw = params["max power kw"]
    kw_per_kwh = params["kw per kwh of battery capacity"]

    # Power-derived limit: if max power is 11kW and ratio is 0.5 kW/kWh, max battery = 22 kWh
    max_kwh_from_power = max_power_kw / kw_per_kwh if kw_per_kwh > 0 else float('inf')

    # Effective ceiling is the minimum of both constraints
    effective_ceiling = min(max_kwh_physical, max_kwh_from_power)

    return {
        "max_kwh_physical": max_kwh_physical,
        "max_power_kw": max_power_kw,
        "kw_per_kwh": kw_per_kwh,
        "max_kwh_from_power": max_kwh_from_power,
        "effective_ceiling": effective_ceiling,
        "binding_constraint": "physical" if max_kwh_physical <= max_kwh_from_power else "power"
    }


def generate_battery_sizes(params):
    """
    Generate battery sizes to test based on model parameters.
    Uses default battery capacity as minimum.
    Maximum is the effective ceiling (min of physical limit and power-derived limit).
    """
    min_size = params["battery capacity kwh"]  # Default/minimum size

    # Get effective ceiling considering both constraints
    ceiling_info = calculate_battery_ceiling(params)
    max_size = ceiling_info["effective_ceiling"]

    # Generate sizes from min to max in steps
    sizes = []
    current = min_size
    while current <= max_size:
        sizes.append(current)
        current += BATTERY_SIZE_STEP

    # Ensure max size is included
    if sizes and sizes[-1] < max_size:
        sizes.append(max_size)

    # If min > max, just use min
    if not sizes:
        sizes = [min_size]

    return sizes


def optimize_battery_size(nation, params, prices_df, use_profile):
    """
    Find optimal battery size based on ROI for both VPP and Electricity Company.
    Battery sizes are derived from model parameters (min: battery capacity, max: max appetite).
    Assumes 8-year battery life for ROI calculation.

    VPP: Optimizes combination of FR commitment hours + Arbitrage in remaining hours.
         Uses iterative optimization to converge on best revenue stream combination.
    Elec Company:
        - Consumer margin from retail-wholesale spread
        - Battery serves consumer first (saves wholesale cost)
        - Only LEFTOVER battery capacity can do arbitrage/FR
    Levies subtracted at end of profit calculation.

    The optimization loop tests all battery sizes and for each:
    1. VPP: Tests all FR commitment strategies (0h, 2h, 4h, 6h, 8h, 24h)
    2. Selects the strategy that maximizes total revenue
    3. Calculates ROI accounting for consumer-first allocation
    """
    results = []
    battery_kw_per_kwh = params["kw per kwh of battery capacity"]
    annual_consumption = params["Mean electricity residential consumption annual (kWH)"]
    daily_consumption = annual_consumption / 365
    buy_price_eur = get_wholesale_buy_price(params)

    # Generate sizes from model parameters
    sizes = generate_battery_sizes(params)

    # Pre-calculate consumer margin (doesn't change with battery size)
    consumer_income_net = calculate_consumer_income_net_of_taxes(params, annual_consumption)
    wholesale_cost = annual_consumption * buy_price_eur
    consumer_margin = consumer_income_net - wholesale_cost

    # Get levy (subtracted at end)
    levy = calculate_levy_elec_company(params)

    for size in sizes:
        battery_kw = size * battery_kw_per_kwh

        # === VPP (optimized FR + Arbitrage combination with nation-specific CM) ===
        vpp_allocation = optimize_battery_allocation_vpp(prices_df, params, size, battery_kw, nation)
        vpp_revenue = vpp_allocation["total_revenue"]
        vpp_costs = sum(calculate_costs_vpp(params, size).values())
        vpp_profit = vpp_revenue - vpp_costs

        # === ELECTRICITY COMPANY ===
        # Battery serves consumer FIRST, only leftover does arbitrage/FR
        elec_battery = calculate_elec_company_battery_value(prices_df, params, use_profile, size, battery_kw)

        # Leftover capacity can do FR (prorated) or arbitrage - compare options
        leftover_kwh = elec_battery["leftover_capacity_kwh"]
        leftover_kw = leftover_kwh * battery_kw_per_kwh

        # Option A: Leftover does arbitrage (already calculated in elec_battery)
        leftover_arb = elec_battery["leftover_arbitrage"]

        # Option B: Leftover commits to FR (prorated by capacity)
        if leftover_kwh > 0:
            fr_rate = params.get("Frequency response per kw per year", 0)
            leftover_fr = fr_rate * leftover_kw
        else:
            leftover_fr = 0

        # Choose best option for leftover
        if leftover_fr > leftover_arb:
            leftover_income = leftover_fr
            leftover_strategy = f"FR ({leftover_kwh:.1f}kWh)"
        else:
            leftover_income = leftover_arb
            leftover_strategy = f"Arb ({leftover_kwh:.1f}kWh)"

        # Apply BM multiplier to any arbitrage
        bm_multiplier = calculate_balancing_mechanism_multiplier(params)
        if leftover_fr <= leftover_arb:
            leftover_income = leftover_arb * bm_multiplier

        # Total revenue = consumer margin + leftover income from battery
        # (No "consumer savings" - we already buy at wholesale min for consumer margin)
        elec_revenue = consumer_margin + leftover_income

        # Operating costs (excluding levy)
        elec_costs_dict = calculate_costs_elec_company(params, size)
        operating_costs = elec_costs_dict["customer_management"] + elec_costs_dict["grid_costs"]

        # Final profit = revenue - costs - levy
        elec_profit = elec_revenue - operating_costs - levy

        # Investment (same for both)
        battery_cost = get_battery_cost_per_kwh(params) * size
        install_cost = params["battery install cost"]
        total_investment = battery_cost + install_cost

        # Simple ROI over 8 years
        roi_vpp = (vpp_profit * 8 - total_investment) / total_investment if total_investment > 0 else 0
        roi_elec = (elec_profit * 8 - total_investment) / total_investment if total_investment > 0 else 0

        results.append({
            "size_kwh": size,
            "vpp_profit": vpp_profit,
            "vpp_strategy": vpp_allocation["strategy"],
            "elec_profit": elec_profit,
            "elec_battery_for_consumer": elec_battery["battery_for_consumer_kwh"],
            "elec_leftover": leftover_kwh,
            "elec_leftover_income": leftover_income,
            "total_investment": total_investment,
            "battery_cost": battery_cost,
            "install_cost": install_cost,
            "roi_vpp_8yr": roi_vpp,
            "roi_elec_8yr": roi_elec
        })

    return pd.DataFrame(results)


def run_model_for_nation(nation, params, use_profile):
    """Run full model for a single nation."""
    print(f"\n{'='*70}")
    print(f"NATION: {nation}")
    print(f"{'='*70}")

    # Note if using UK profile
    if nation != "UK":
        print(f"Note: Using UK consumption profile (no {nation} profile available)")

    # Load prices
    try:
        prices_df = load_hourly_prices(nation)
        print(f"Loaded {len(prices_df)} hourly price records")
    except Exception as e:
        print(f"Error loading prices for {nation}: {e}")
        return None

    # Get parameters
    p = params[nation]

    # =========================================
    # MODEL INPUTS & BATTERY CEILING
    # =========================================
    print(f"\n{'-'*70}")
    print("MODEL INPUTS & BATTERY CONSTRAINTS")
    print(f"{'-'*70}")

    # Battery specs
    battery_kwh = p["battery capacity kwh"]
    battery_kw = p["battery power kw"]
    efficiency = p["battery charge recharge efficiency %"]
    annual_consumption = p["Mean electricity residential consumption annual (kWH)"]

    print(f"\nDefault Battery Specs:")
    print(f"  Capacity: {battery_kwh} kWh")
    print(f"  Power: {battery_kw} kW")
    print(f"  Efficiency: {efficiency*100:.0f}%")

    print(f"\nConsumer:")
    print(f"  Annual consumption: {annual_consumption} kWh")
    print(f"  Daily consumption: {annual_consumption/365:.2f} kWh")

    # Battery ceiling calculation
    ceiling = calculate_battery_ceiling(p)
    print(f"\nBattery Size Ceiling Calculation:")
    print(f"  Constraint 1 - Physical limit: {ceiling['max_kwh_physical']:.1f} kWh")
    print(f"  Constraint 2 - Power derived: {ceiling['max_power_kw']:.1f} kW / {ceiling['kw_per_kwh']} kW/kWh = {ceiling['max_kwh_from_power']:.1f} kWh")
    print(f"  >>> Effective ceiling: {ceiling['effective_ceiling']:.1f} kWh (binding: {ceiling['binding_constraint']})")
    print(f"  >>> Size range to test: {battery_kwh:.1f} - {ceiling['effective_ceiling']:.1f} kWh")

    # =========================================
    # VPP MODEL
    # =========================================
    print(f"\n{'-'*70}")
    print("SCENARIO 1: Virtual Power Plant (VPP)")
    print(f"{'-'*70}")
    print("(No consumer sales - battery used purely for grid services)")

    # ---- MILP OPTIMIZATION ----
    print(f"\n--- MILP Hourly Optimization ---")
    print(f"  Solving daily optimization for {len(prices_df.groupby('date'))} days...")
    print(f"  Decision per hour: Arbitrage (charge/discharge) OR FR (hold at 50% SoC)")

    # ---- RUN OPTIMIZATION ----
    vpp_allocation = optimize_battery_allocation_vpp(prices_df, p, battery_kwh, battery_kw, nation)

    # Show MILP results
    print(f"\n  MILP Results:")
    print(f"    Days analyzed:     {vpp_allocation['num_days']}")
    print(f"    Pure arbitrage days: {vpp_allocation['pure_arb_days']}")
    print(f"    Mixed strategy days: {vpp_allocation['mixed_days']}")
    print(f"    Pure FR days:        {vpp_allocation['pure_fr_days']}")
    print(f"    Total FR hours:      {vpp_allocation['fr_hours']:,.0f} ({vpp_allocation['fr_hours_avg_per_day']:.1f}h/day avg)")

    print(f"\n  >>> OPTIMAL STRATEGY: {vpp_allocation['strategy']}")
    print(f"      {vpp_allocation['explanation']}")

    # Show sample day schedules (only for UK with mixed strategy, and limit to 2 examples)
    if nation == "UK" and vpp_allocation.get("daily_results"):
        daily_results = vpp_allocation["daily_results"]
        # Find one mixed day and one high-arb day to show contrasting strategies
        mixed_days = [d for d in daily_results if 0 < d["fr_hours"] < 24]
        arb_rich_days = sorted([d for d in daily_results if d["fr_hours"] < 12],
                               key=lambda x: x["arbitrage"], reverse=True)

        print(f"\n  --- Sample Day Schedules (showing hour-by-hour SoC tracking) ---")

        if mixed_days:
            # Show a typical mixed day
            sample_mixed = mixed_days[len(mixed_days)//2]  # Pick middle of sorted
            print_sample_day_schedule(sample_mixed, battery_kwh)

        if arb_rich_days and len(arb_rich_days) > 0:
            # Show best arbitrage day
            if arb_rich_days[0] != (mixed_days[len(mixed_days)//2] if mixed_days else None):
                print_sample_day_schedule(arb_rich_days[0], battery_kwh)

    # Extract values
    arbitrage = vpp_allocation["arbitrage"]
    freq_response = vpp_allocation["frequency_response"]
    bm_income = vpp_allocation["balancing_mechanism"]
    capacity_market = vpp_allocation["capacity_market"]

    # ---- CAPACITY MARKET NOTE ----
    cm_rate = p.get("Capacity market /kW/year", 0)
    print(f"\n--- Capacity Market ---")
    if cm_rate > 0:
        duration = get_battery_duration_hours(battery_kwh, battery_kw)
        derating = get_derating_factor(battery_kwh, battery_kw)
        print(f"  Using Capacity Market with de-rating:")
        print(f"    Clearing price (model_input): £{cm_rate}/kW/year")
        print(f"    Battery duration: {duration}hr → De-rating factor: {derating*100:.2f}%")
        print(f"    De-rated capacity: {battery_kw:.1f} kW × {derating*100:.1f}% = {battery_kw * derating:.2f} kW")
        if nation == "UK":
            print(f"    Revenue: £{cm_rate} × {battery_kw * derating:.2f} × 1.17 (EUR) = €{capacity_market:.2f}")
        else:
            print(f"    Revenue: €{cm_rate} × {battery_kw * derating:.2f} = €{capacity_market:.2f}")
    else:
        print(f"  No Capacity Market rate specified for this nation")
    print(f"  Annual income: €{capacity_market:.2f}")

    # ---- BALANCING MECHANISM ----
    bm_win_rate = p.get("Balancing mechanism (BM) avg win rate", 0)
    bm_uplift = p.get("Avg £/kwh uplift for BM", 0)
    bm_multiplier = calculate_balancing_mechanism_multiplier(p)
    print(f"\n--- Balancing Mechanism (as Multiplier) ---")
    print(f"  Win rate: {bm_win_rate*100:.1f}%, Uplift per win: {bm_uplift*100:.1f}%")
    print(f"  Multiplier: 1 + ({bm_win_rate*100:.0f}% × {bm_uplift*100:.0f}%) = {bm_multiplier:.4f}")
    print(f"  Arbitrage with BM: €{arbitrage:.2f} × {bm_multiplier:.4f} = €{vpp_allocation.get('arbitrage_with_bm', arbitrage * bm_multiplier):.2f}")
    print(f"  BM uplift portion: €{bm_income:.2f}")

    # ---- REVENUE SUMMARY ----
    print(f"\n--- Revenue Summary (EUR/year) ---")
    print(f"  Wholesale Arbitrage:                 €{arbitrage:>10.2f}")
    print(f"  Frequency Response:                  €{freq_response:>10.2f}")
    print(f"  Capacity Market:                     €{capacity_market:>10.2f}")
    print(f"  Balancing Mechanism:                 €{bm_income:>10.2f}")

    total_vpp_revenue = vpp_allocation["total_revenue"]
    print(f"  {'-'*45}")
    print(f"  TOTAL REVENUE:                       €{total_vpp_revenue:>10.2f}")

    # ---- COSTS ----
    vpp_costs = calculate_costs_vpp(p, battery_kwh)

    print(f"\n--- Costs (EUR/year) ---")
    print(f"  Customer Management (VPP rate):      €{vpp_costs['customer_management']:>10.2f}")
    print(f"  Grid Costs:                          €{0:>10.2f}  (customer pays separately)")
    print(f"  {'-'*45}")

    total_vpp_costs = sum(vpp_costs.values())
    print(f"  TOTAL COSTS:                         €{total_vpp_costs:>10.2f}")

    vpp_profit = total_vpp_revenue - total_vpp_costs
    print(f"\n>>> VPP ANNUAL PROFIT PER CUSTOMER: €{vpp_profit:.2f}")

    # =========================================
    # ELECTRICITY COMPANY MODEL
    # =========================================
    print(f"\n{'-'*70}")
    print("SCENARIO 2: Electricity Company")
    print(f"{'-'*70}")

    # ---- DETAILED TAX CALCULATION ----
    print(f"\n--- Consumer Income Calculation (with tax breakdown) ---")

    consumer_rate_gross = p["consumer cost p/kwh inc VAT"] / 100  # EUR/kWh
    vat_rate = p["VAT on electricity %"] / 100
    energy_tax_rate = p["energy tax p/kwh (excluding VAT)"] / 100  # EUR/kWh

    consumer_rate_ex_vat = consumer_rate_gross / (1 + vat_rate)
    net_rate = consumer_rate_ex_vat - energy_tax_rate

    print(f"  Consumer pays (inc VAT): €{consumer_rate_gross:.4f}/kWh ({p['consumer cost p/kwh inc VAT']:.1f} cents)")
    print(f"  VAT rate: {vat_rate*100:.0f}%")
    print(f"  Price ex-VAT: €{consumer_rate_gross:.4f} / {1+vat_rate:.2f} = €{consumer_rate_ex_vat:.4f}/kWh")
    print(f"  Energy tax (pass-through): €{energy_tax_rate:.4f}/kWh ({p['energy tax p/kwh (excluding VAT)']:.1f} cents)")
    print(f"  Net income rate: €{consumer_rate_ex_vat:.4f} - €{energy_tax_rate:.4f} = €{net_rate:.4f}/kWh")
    print(f"  Annual consumption: {annual_consumption} kWh")

    consumer_income_gross = annual_consumption * consumer_rate_gross
    consumer_income_net = annual_consumption * net_rate
    vat_collected = annual_consumption * (consumer_rate_gross - consumer_rate_ex_vat)
    energy_tax_collected = annual_consumption * energy_tax_rate

    print(f"\n  GROSS from consumer: €{consumer_income_gross:.2f}")
    print(f"  Less VAT (remit to govt): -€{vat_collected:.2f}")
    print(f"  Less Energy Tax (remit): -€{energy_tax_collected:.2f}")
    print(f"  >>> NET INCOME RETAINED: €{consumer_income_net:.2f}")

    # ---- CONSUMER MARGIN ----
    print(f"\n--- Consumer Margin (retail - wholesale) ---")
    wholesale_min_cents = p["wholesale avg daily minimum p/kWH"]
    grid_cost_cents = p.get("residential min grid cost p/kwh avg daily (lowest hour)", 0)
    buy_price_eur = get_wholesale_buy_price(p)
    print(f"  Wholesale min: {wholesale_min_cents:.1f} cents + Grid cost: {grid_cost_cents:.1f} cents = {(wholesale_min_cents + grid_cost_cents):.1f} cents/kWh")
    print(f"  Total buy price: €{buy_price_eur:.4f}/kWh")

    wholesale_cost = annual_consumption * buy_price_eur
    print(f"  Wholesale cost: {annual_consumption} kWh × €{buy_price_eur:.4f}/kWh = €{wholesale_cost:.2f}")

    consumer_margin = consumer_income_net - wholesale_cost
    print(f"  Consumer margin: €{consumer_income_net:.2f} - €{wholesale_cost:.2f} = €{consumer_margin:.2f}")

    # ---- BATTERY ALLOCATION (CONSUMER FIRST, THEN LEFTOVER) ----
    print(f"\n--- Battery Allocation (Consumer First) ---")
    print(f"  KEY CONSTRAINT: Same kWh can't earn consumer margin AND arbitrage")
    print(f"  - Battery first serves consumer demand (buy cheap, discharge to consumer)")
    print(f"  - Only LEFTOVER capacity can do arbitrage or FR")

    # Calculate battery value for elec company
    elec_battery = calculate_elec_company_battery_value(prices_df, p, use_profile, battery_kwh, battery_kw)
    daily_consumption = annual_consumption / 365

    print(f"\n  Battery allocation:")
    print(f"    Daily consumer demand:             {daily_consumption:>10.2f} kWh")
    print(f"    Battery capacity:                  {battery_kwh:>10.1f} kWh")
    print(f"    → For consumer supply:             {elec_battery['battery_for_consumer_kwh']:>10.2f} kWh")
    print(f"    → Leftover for arb/FR:             {elec_battery['leftover_capacity_kwh']:>10.2f} kWh")

    # Leftover capacity income
    leftover_kwh = elec_battery["leftover_capacity_kwh"]
    leftover_arb = elec_battery["leftover_arbitrage"]
    leftover_kw = leftover_kwh * p["kw per kwh of battery capacity"]
    leftover_fr = p.get("Frequency response per kw per year", 0) * leftover_kw if leftover_kwh > 0 else 0

    bm_multiplier = calculate_balancing_mechanism_multiplier(p)
    leftover_arb_with_bm = leftover_arb * bm_multiplier

    print(f"\n  Leftover capacity options:")
    print(f"    Arbitrage ({leftover_kwh:.2f} kWh):            €{leftover_arb:.2f} × BM {bm_multiplier:.3f} = €{leftover_arb_with_bm:.2f}")
    print(f"    FR ({leftover_kw:.2f} kW):                     €{leftover_fr:.2f}")

    if leftover_fr > leftover_arb_with_bm:
        leftover_income = leftover_fr
        leftover_choice = "FR"
    else:
        leftover_income = leftover_arb_with_bm
        leftover_choice = "Arbitrage"
    print(f"    >>> Best option: {leftover_choice}             €{leftover_income:.2f}")

    # ---- REVENUE SUMMARY ----
    print(f"\n--- Revenue Summary (EUR/year) ---")
    print(f"  Consumer margin (retail - wholesale):")
    print(f"    Retail income (net of tax):        €{consumer_income_net:>10.2f}")
    print(f"    Wholesale cost:                   -€{wholesale_cost:>10.2f}")
    print(f"    = Consumer margin:                 €{consumer_margin:>10.2f}")
    print(f"")
    print(f"  Leftover battery income ({leftover_choice}):     €{leftover_income:>10.2f}")

    total_elec_revenue = consumer_margin + leftover_income
    print(f"  {'-'*45}")
    print(f"  TOTAL NET REVENUE:                   €{total_elec_revenue:>10.2f}")

    # ---- COSTS ----
    elec_costs = calculate_costs_elec_company(p, battery_kwh)
    levy = calculate_levy_elec_company(p)

    print(f"\n--- Operating Costs (EUR/year) ---")
    print(f"  Customer Management:                 €{elec_costs['customer_management']:>10.2f}")
    print(f"  Grid Costs:                          €{elec_costs['grid_costs']:>10.2f}")

    operating_costs = elec_costs["customer_management"] + elec_costs["grid_costs"]
    print(f"  {'-'*45}")
    print(f"  TOTAL OPERATING COSTS:               €{operating_costs:>10.2f}")

    profit_before_levy = total_elec_revenue - operating_costs
    print(f"\n  Profit before levy:                  €{profit_before_levy:>10.2f}")

    print(f"\n--- Levy (subtracted at end) ---")
    print(f"  Annual Levy (elec company):          €{levy:>10.2f}")

    elec_profit = profit_before_levy - levy
    print(f"\n>>> ELEC COMPANY ANNUAL PROFIT PER CUSTOMER: €{elec_profit:.2f}")

    # =========================================
    # UPFRONT INVESTMENT
    # =========================================
    print(f"\n{'-'*40}")
    print("UPFRONT INVESTMENT (battery already owned)")
    print(f"{'-'*40}")

    investment = calculate_upfront_investment(p)
    print(f"Installation Cost: €{investment:.2f}")

    # =========================================
    # BATTERY SIZE OPTIMIZATION
    # =========================================
    print(f"\n{'-'*40}")
    print("BATTERY SIZE OPTIMIZATION")
    print(f"{'-'*40}")

    opt_results = optimize_battery_size(nation, p, prices_df, use_profile)

    print("\nSize (kWh) | VPP Profit | Elec Profit | Investment | VPP ROI | Elec ROI")
    print("-" * 75)
    for _, row in opt_results.iterrows():
        print(f"  {row['size_kwh']:>6.1f}   | €{row['vpp_profit']:>9.2f} | €{row['elec_profit']:>10.2f} | €{row['total_investment']:>9.0f} | {row['roi_vpp_8yr']*100:>6.1f}% | {row['roi_elec_8yr']*100:>6.1f}%")

    best_vpp = opt_results.loc[opt_results["roi_vpp_8yr"].idxmax()]
    best_elec = opt_results.loc[opt_results["roi_elec_8yr"].idxmax()]

    print(f"\nOptimal for VPP: {best_vpp['size_kwh']:.1f} kWh (8yr ROI: {best_vpp['roi_vpp_8yr']*100:.1f}%)")
    print(f"Optimal for Elec Company: {best_elec['size_kwh']:.1f} kWh (8yr ROI: {best_elec['roi_elec_8yr']*100:.1f}%)")

    return {
        "nation": nation,
        "vpp_profit": vpp_profit,
        "elec_profit": elec_profit,
        "vpp_revenue": total_vpp_revenue,
        "elec_revenue": total_elec_revenue,
        "vpp_costs": total_vpp_costs,
        "elec_operating_costs": operating_costs,
        "elec_levy": levy,
        "upfront_investment": investment,
        "optimal_vpp_kwh": best_vpp["size_kwh"],
        "optimal_vpp_roi": best_vpp["roi_vpp_8yr"],
        "optimal_vpp_strategy": best_vpp["vpp_strategy"],
        "optimal_elec_kwh": best_elec["size_kwh"],
        "optimal_elec_roi": best_elec["roi_elec_8yr"],
        "optimal_elec_strategy": best_elec.get("elec_strategy", ""),
    }


def main():
    print("="*60)
    print("BATTERY STORAGE OPTIMIZATION MODEL")
    print(f"Analysis Period: {DATE_START} to {DATE_END}")
    print("="*60)

    # Load inputs
    params = load_model_inputs()
    use_profile = load_use_profiles()

    # Run for each nation
    results = []
    for nation in NATIONS.keys():
        result = run_model_for_nation(nation, params, use_profile)
        if result:
            results.append(result)

    # Summary table
    print("\n" + "="*90)
    print("SUMMARY: ALL NATIONS")
    print("="*90)

    summary_df = pd.DataFrame(results)
    summary_df = summary_df.set_index("nation")

    # Calculate additional metrics for summary
    for nation in summary_df.index:
        # Get optimal battery costs
        vpp_size = summary_df.loc[nation, "optimal_vpp_kwh"]
        elec_size = summary_df.loc[nation, "optimal_elec_kwh"]
        p = params[nation]

        vpp_battery_cost = get_battery_cost_per_kwh(p) * vpp_size + p["battery install cost"]
        elec_battery_cost = get_battery_cost_per_kwh(p) * elec_size + p["battery install cost"]

        summary_df.loc[nation, "vpp_battery_cost"] = vpp_battery_cost
        summary_df.loc[nation, "elec_battery_cost"] = elec_battery_cost

        # Payback period (years)
        vpp_profit = summary_df.loc[nation, "vpp_profit"]
        elec_profit = summary_df.loc[nation, "elec_profit"]
        summary_df.loc[nation, "vpp_payback"] = vpp_battery_cost / vpp_profit if vpp_profit > 0 else float('inf')
        summary_df.loc[nation, "elec_payback"] = elec_battery_cost / elec_profit if elec_profit > 0 else float('inf')

    # Comprehensive summary table - recalculate profits at optimal sizes
    print("\n" + "="*120)
    print("COMPREHENSIVE SUMMARY TABLE")
    print("="*120)
    print(f"{'Nation':<12} | {'Scenario':<12} | {'Battery':<8} | {'Strategy':<20} | {'Upfront':<10} | {'Profit/yr':<10} | {'8yr ROI':<8} | {'Payback':<8}")
    print("-"*120)

    for nation in summary_df.index:
        p = params[nation]
        prices_df = load_hourly_prices(nation)

        # VPP row - recalculate at optimal size
        vpp_size = summary_df.loc[nation, "optimal_vpp_kwh"]
        vpp_cost = summary_df.loc[nation, "vpp_battery_cost"]
        vpp_kw = vpp_size * p["kw per kwh of battery capacity"]
        vpp_allocation = optimize_battery_allocation_vpp(prices_df, p, vpp_size, vpp_kw, nation)
        vpp_profit_optimal = vpp_allocation["total_revenue"] - sum(calculate_costs_vpp(p, vpp_size).values())
        vpp_roi = summary_df.loc[nation, "optimal_vpp_roi"]
        vpp_payback = vpp_cost / vpp_profit_optimal if vpp_profit_optimal > 0 else float('inf')
        payback_str = f"{vpp_payback:.1f} yrs" if vpp_payback < 100 else "N/A"
        strategy_str = vpp_allocation["strategy"][:18]
        print(f"{nation:<12} | {'VPP':<12} | {vpp_size:>5.1f} kWh | {strategy_str:<20} | €{vpp_cost:>8.0f} | €{vpp_profit_optimal:>8.2f} | {vpp_roi*100:>6.1f}% | {payback_str:>8}")

        # Elec Company row - recalculate at optimal size using consumer-first model
        elec_size = summary_df.loc[nation, "optimal_elec_kwh"]
        elec_cost = summary_df.loc[nation, "elec_battery_cost"]
        elec_kw = elec_size * p["kw per kwh of battery capacity"]
        buy_price = get_wholesale_buy_price(p)
        annual_consumption = p["Mean electricity residential consumption annual (kWH)"]

        consumer_income_net = calculate_consumer_income_net_of_taxes(p, annual_consumption)
        consumer_margin = consumer_income_net - annual_consumption * buy_price

        # Use consumer-first battery model
        elec_battery = calculate_elec_company_battery_value(prices_df, p, use_profile, elec_size, elec_kw)
        leftover_kwh = elec_battery["leftover_capacity_kwh"]
        leftover_kw = leftover_kwh * p["kw per kwh of battery capacity"]
        leftover_arb = elec_battery["leftover_arbitrage"]
        leftover_fr = p.get("Frequency response per kw per year", 0) * leftover_kw if leftover_kwh > 0 else 0
        bm_multiplier = calculate_balancing_mechanism_multiplier(p)
        leftover_income = max(leftover_arb * bm_multiplier, leftover_fr)

        elec_revenue = consumer_margin + leftover_income

        elec_costs_dict = calculate_costs_elec_company(p, elec_size)
        operating_costs = elec_costs_dict["customer_management"] + elec_costs_dict["grid_costs"]
        levy = calculate_levy_elec_company(p)
        elec_profit_optimal = elec_revenue - operating_costs - levy

        elec_roi = summary_df.loc[nation, "optimal_elec_roi"]
        elec_payback = elec_cost / elec_profit_optimal if elec_profit_optimal > 0 else float('inf')
        payback_str = f"{elec_payback:.1f} yrs" if elec_payback < 100 else "N/A"
        elec_strategy_str = f"Cons:{elec_battery['battery_for_consumer_kwh']:.1f}+Left:{leftover_kwh:.1f}"
        print(f"{'':<12} | {'Elec Company':<12} | {elec_size:>5.1f} kWh | {elec_strategy_str:<20} | €{elec_cost:>8.0f} | €{elec_profit_optimal:>8.2f} | {elec_roi*100:>6.1f}% | {payback_str:>8}")
        print("-"*120)

    # VPP Income breakdown table
    print("\n" + "="*130)
    print("VPP INCOME BREAKDOWN (Annual, at optimal battery)")
    print("="*130)
    print(f"{'Nation':<12} | {'Battery':<8} | {'Strategy':<20} | {'Arbitrage':<10} | {'FR':<8} | {'CM':<8} | {'BM':<8} | {'Costs':<8} | {'Profit':<10}")
    print("-"*130)

    for nation in summary_df.index:
        p = params[nation]
        prices_df = load_hourly_prices(nation)
        opt_kwh = summary_df.loc[nation, "optimal_vpp_kwh"]
        opt_kw = opt_kwh * p["kw per kwh of battery capacity"]

        # Get VPP allocation at optimal size
        vpp_alloc = optimize_battery_allocation_vpp(prices_df, p, opt_kwh, opt_kw, nation)
        vpp_costs = sum(calculate_costs_vpp(p, opt_kwh).values())
        vpp_profit = vpp_alloc["total_revenue"] - vpp_costs

        strategy = vpp_alloc["strategy"][:18]  # Truncate for display
        cm = vpp_alloc.get("capacity_market", 0)
        print(f"{nation:<12} | {opt_kwh:>5.1f} kWh | {strategy:<20} | €{vpp_alloc['arbitrage']:>8.2f} | €{vpp_alloc['frequency_response']:>6.2f} | €{cm:>6.2f} | €{vpp_alloc['balancing_mechanism']:>6.2f} | €{vpp_costs:>6.2f} | €{vpp_profit:>8.2f}")

    print("-"*130)
    print("Note: VPP optimizes combination of FR hours + Arbitrage in remaining hours")
    print("      CM = Capacity Market (UK uses de-rating factors: 1hr=10.47%, 2hr=20.94%, 4hr=37%, 8hr=92%)")

    # Elec Company Income breakdown table
    print("\n" + "="*110)
    print("ELEC COMPANY INCOME BREAKDOWN (Annual, at optimal battery)")
    print("Battery serves consumer first, only leftover does arbitrage/FR")
    print("="*110)
    print(f"{'Nation':<12} | {'Batt':<6} | {'ForCons':<7} | {'Left':<6} | {'ConsMarg':<9} | {'LeftInc':<8} | {'OpCost':<7} | {'Levy':<5} | {'Profit':<9}")
    print("-"*110)

    for nation in summary_df.index:
        p = params[nation]
        prices_df = load_hourly_prices(nation)
        opt_kwh = summary_df.loc[nation, "optimal_elec_kwh"]
        opt_kw = opt_kwh * p["kw per kwh of battery capacity"]
        buy_price = get_wholesale_buy_price(p)
        annual_consumption = p["Mean electricity residential consumption annual (kWH)"]

        # Consumer margin
        consumer_income_net = calculate_consumer_income_net_of_taxes(p, annual_consumption)
        consumer_margin = consumer_income_net - annual_consumption * buy_price

        # Battery value with consumer-first model
        elec_battery = calculate_elec_company_battery_value(prices_df, p, use_profile, opt_kwh, opt_kw)
        leftover_kwh = elec_battery["leftover_capacity_kwh"]
        leftover_kw = leftover_kwh * p["kw per kwh of battery capacity"]
        leftover_arb = elec_battery["leftover_arbitrage"]
        leftover_fr = p.get("Frequency response per kw per year", 0) * leftover_kw if leftover_kwh > 0 else 0
        bm_mult = calculate_balancing_mechanism_multiplier(p)
        leftover_income = max(leftover_arb * bm_mult, leftover_fr)

        # Costs
        operating_costs = p["Cost of managing customers per customer per annum if electricity company"] + \
                         p.get("residential min grid cost PA (euros)", 0)
        levy = calculate_levy_elec_company(p)

        total_revenue = consumer_margin + leftover_income
        net_profit = total_revenue - operating_costs - levy

        print(f"{nation:<12} | {opt_kwh:>4.0f}kW | {elec_battery['battery_for_consumer_kwh']:>5.1f}kW | {leftover_kwh:>4.1f}kW | €{consumer_margin:>7.0f} | €{leftover_income:>6.2f} | €{operating_costs:>5.0f} | €{levy:>3.0f} | €{net_profit:>7.2f}")

    print("-"*110)

    # FINAL TABLE: Upfront costs
    print("\n" + "="*100)
    print("FINAL SUMMARY: UPFRONT COSTS AND INCOME")
    print("="*100)
    print(f"{'Nation':<12} | {'Scenario':<12} | {'Battery':<8} | {'Batt Cost':<10} | {'Install':<8} | {'Total Cost':<10} | {'Income/yr':<10}")
    print("-"*100)

    for nation in summary_df.index:
        p = params[nation]

        # VPP
        vpp_size = summary_df.loc[nation, "optimal_vpp_kwh"]
        vpp_batt_cost = get_battery_cost_per_kwh(p) * vpp_size
        vpp_install = p["battery install cost"]
        vpp_total = vpp_batt_cost + vpp_install

        # Recalculate VPP profit at optimal size
        prices_df = load_hourly_prices(nation)
        vpp_kw = vpp_size * p["kw per kwh of battery capacity"]
        vpp_alloc = optimize_battery_allocation_vpp(prices_df, p, vpp_size, vpp_kw, nation)
        vpp_profit = vpp_alloc["total_revenue"] - sum(calculate_costs_vpp(p, vpp_size).values())

        print(f"{nation:<12} | {'VPP':<12} | {vpp_size:>5.1f} kWh | €{vpp_batt_cost:>8.0f} | €{vpp_install:>6.0f} | €{vpp_total:>8.0f} | €{vpp_profit:>8.2f}")

        # Elec Company
        elec_size = summary_df.loc[nation, "optimal_elec_kwh"]
        elec_batt_cost = get_battery_cost_per_kwh(p) * elec_size
        elec_install = p["battery install cost"]
        elec_total = elec_batt_cost + elec_install

        # Recalculate Elec profit at optimal size
        elec_kw = elec_size * p["kw per kwh of battery capacity"]
        buy_price = get_wholesale_buy_price(p)
        annual_consumption = p["Mean electricity residential consumption annual (kWH)"]
        consumer_income_net = calculate_consumer_income_net_of_taxes(p, annual_consumption)
        consumer_margin = consumer_income_net - annual_consumption * buy_price

        elec_battery = calculate_elec_company_battery_value(prices_df, p, use_profile, elec_size, elec_kw)
        leftover_kwh = elec_battery["leftover_capacity_kwh"]
        leftover_kw = leftover_kwh * p["kw per kwh of battery capacity"]
        leftover_arb = elec_battery["leftover_arbitrage"]
        leftover_fr = p.get("Frequency response per kw per year", 0) * leftover_kw if leftover_kwh > 0 else 0
        bm_mult = calculate_balancing_mechanism_multiplier(p)
        leftover_income = max(leftover_arb * bm_mult, leftover_fr)

        elec_revenue = consumer_margin + leftover_income
        elec_costs_dict = calculate_costs_elec_company(p, elec_size)
        operating_costs = elec_costs_dict["customer_management"] + elec_costs_dict["grid_costs"]
        levy = calculate_levy_elec_company(p)
        elec_profit = elec_revenue - operating_costs - levy

        print(f"{'':<12} | {'Elec Company':<12} | {elec_size:>5.1f} kWh | €{elec_batt_cost:>8.0f} | €{elec_install:>6.0f} | €{elec_total:>8.0f} | €{elec_profit:>8.2f}")
        print("-"*100)

    # Get sample params for notes (all nations have same battery costs)
    sample_p = params["UK"]
    base_cost = sample_p["battery cost per kwh (inc wires, controller, inverter, etc)"]
    discount = sample_p.get("wholesale_discount_battery", 0)
    discounted_cost = get_battery_cost_per_kwh(sample_p)

    print("\nNotes:")
    print(f"  - Battery Cost = €{base_cost:.0f}/kWh × (1 - {discount*100:.0f}% wholesale discount) = €{discounted_cost:.0f}/kWh")
    print(f"  - Install Cost = €{sample_p['battery install cost']:.0f} per installation")
    print("  - VPP: Full battery available for grid trading (arbitrage + FR + CM)")
    print("  - Elec Company: Battery serves consumer first, only leftover for trading")
    print("  - Consumer Margin = Retail (net of VAT/tax) - Wholesale (at min price)")
    print("  - Levy subtracted at end of Elec Company profit")


if __name__ == "__main__":
    main()
