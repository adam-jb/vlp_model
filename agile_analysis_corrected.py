"""
Corrected battery savings calculation that properly accounts for:
- Consumption during off-peak hours uses grid directly (no battery needed)
- Battery only provides value by shifting expensive consumption to cheap charging times
"""

import pandas as pd

# UK consumption profile (hourly kWh)
UK_CONSUMPTION_PROFILE = {
    0: 0.275, 1: 0.275, 2: 0.275, 3: 0.275, 4: 0.275, 5: 0.275,
    6: 0.375, 7: 0.550, 8: 0.550, 9: 0.375, 10: 0.375, 11: 0.375,
    12: 0.425, 13: 0.425, 14: 0.375, 15: 0.375, 16: 0.375,
    17: 0.675, 18: 0.675, 19: 0.550, 20: 0.550,
    21: 0.400, 22: 0.400, 23: 0.400
}

DAILY_CONSUMPTION = sum(UK_CONSUMPTION_PROFILE.values())  # 9.875 kWh
BATTERY_CAPACITY = 10.0  # kWh

def calculate_optimal_battery_strategy(hourly_rates):
    """
    Proper calculation accounting for off-peak consumption.

    Strategy:
    1. For each hour, we have consumption and a rate
    2. WITHOUT battery: pay consumption × rate for each hour
    3. WITH battery:
       - Can charge up to 10kWh at any hours
       - Can discharge to cover consumption at any hours
       - Optimal: charge at cheapest hours, discharge during most expensive consumption hours
       - But if consumption occurs during a cheap hour, just use grid directly

    The optimal strategy with unlimited charge/discharge rate:
    - Sort all hours by rate
    - Charge during the N cheapest hours
    - For consumption: use grid directly if it's a cheap hour, use battery if expensive
    """

    # Cost without battery - straightforward
    cost_without = sum(
        UK_CONSUMPTION_PROFILE[h] * hourly_rates[h]
        for h in range(24)
    )

    # With battery - optimal arbitrage strategy
    # Create list of (hour, rate, consumption)
    hour_data = [(h, hourly_rates[h], UK_CONSUMPTION_PROFILE[h]) for h in range(24)]

    # Sort by rate to find cheap vs expensive hours
    sorted_by_rate = sorted(hour_data, key=lambda x: x[1])

    # Identify the "threshold rate" - the rate at which we're indifferent
    # between using battery or grid

    # Greedy approach:
    # 1. Find cheapest hours for charging (need to charge DAILY_CONSUMPTION kWh)
    # 2. For consumption, use battery if rate > charging rate, else use grid

    # Determine charging cost if we charge all consumption at cheapest times
    # Assume we can charge up to 2.5 kWh per hour
    CHARGE_RATE = 2.5  # kWh per hour

    energy_to_charge = DAILY_CONSUMPTION
    charging_cost = 0
    charging_hours = []

    for hour, rate, _ in sorted_by_rate:
        if energy_to_charge <= 0:
            break
        charge_amount = min(CHARGE_RATE, energy_to_charge)
        charging_cost += charge_amount * rate
        charging_hours.append((hour, rate, charge_amount))
        energy_to_charge -= charge_amount

    # The "break-even rate" is roughly the weighted average charging rate
    if charging_hours:
        total_charged = sum(c[2] for c in charging_hours)
        avg_charging_rate = charging_cost / total_charged
    else:
        avg_charging_rate = 0

    # Now calculate cost WITH battery using optimal strategy:
    # For each consumption hour:
    # - If grid rate <= avg charging rate: use grid (no benefit from battery)
    # - If grid rate > avg charging rate: use battery (pay avg charging rate instead)

    cost_with_battery = 0
    energy_from_grid = 0
    energy_from_battery = 0

    for hour in range(24):
        consumption = UK_CONSUMPTION_PROFILE[hour]
        grid_rate = hourly_rates[hour]

        if grid_rate <= avg_charging_rate:
            # Cheaper to use grid directly
            cost_with_battery += consumption * grid_rate
            energy_from_grid += consumption
        else:
            # Use battery (charged at avg_charging_rate)
            cost_with_battery += consumption * avg_charging_rate
            energy_from_battery += consumption

    return {
        'cost_without': cost_without,
        'cost_with': cost_with_battery,
        'saving': cost_without - cost_with_battery,
        'avg_charging_rate': avg_charging_rate,
        'energy_from_grid': energy_from_grid,
        'energy_from_battery': energy_from_battery
    }


def calculate_realistic_battery_strategy(hourly_rates):
    """
    Even more realistic: proper hour-by-hour simulation with battery state.

    Each hour:
    - Decide to charge, discharge, or do nothing
    - Track battery state of charge (SOC)
    - Goal: minimize total cost

    Simplified optimal strategy:
    - Charge during the cheapest N hours until battery full
    - Use battery during expensive hours when consuming
    - Use grid during cheap hours even if battery has charge
    """

    CHARGE_RATE = 2.5  # kWh per hour max
    DISCHARGE_RATE = 5.0  # kWh per hour max (typically higher than charge)

    # Create hour info
    hours = [(h, hourly_rates[h], UK_CONSUMPTION_PROFILE[h]) for h in range(24)]

    # Find cheapest hours for charging
    sorted_hours = sorted(hours, key=lambda x: x[1])
    cheapest_hours = set(h for h, _, _ in sorted_hours[:6])  # ~6 hours to fully charge at 2.5kW

    # Find threshold - the rate above which we prefer battery
    # Use the most expensive charging hour rate as threshold
    charging_rates = [r for h, r, _ in sorted_hours[:6]]
    threshold_rate = max(charging_rates) if charging_rates else 15

    # Simulate day
    battery_soc = 5.0  # Start half charged
    total_cost = 0
    energy_from_grid = 0
    energy_from_battery = 0

    # Two passes: first figure out what we need, then optimize

    # Simple strategy:
    cost_without = sum(UK_CONSUMPTION_PROFILE[h] * hourly_rates[h] for h in range(24))

    # With battery: charge at cheap times, discharge at expensive times
    battery_soc = 0
    cost_with = 0

    # Sort hours and process in rate order for optimal charging
    # But we need chronological for realistic simulation...

    # Let's use a practical daily strategy:
    # 1. Overnight (cheap): charge battery
    # 2. Morning/evening peak: use battery
    # 3. Off-peak daytime: use grid if cheap enough, else battery

    for hour in range(24):
        rate = hourly_rates[hour]
        consumption = UK_CONSUMPTION_PROFILE[hour]

        # Charging decision: charge if rate is in bottom 25% and battery not full
        if rate <= sorted_hours[5][1] and battery_soc < BATTERY_CAPACITY:
            charge = min(CHARGE_RATE, BATTERY_CAPACITY - battery_soc)
            battery_soc += charge
            cost_with += charge * rate

        # Consumption decision: use battery if rate > threshold and battery has charge
        if rate > threshold_rate and battery_soc >= consumption:
            # Use battery
            battery_soc -= consumption
            energy_from_battery += consumption
            # Cost already paid when charging
        else:
            # Use grid
            cost_with += consumption * rate
            energy_from_grid += consumption

    return {
        'cost_without': cost_without,
        'cost_with': cost_with,
        'saving': cost_without - cost_with,
        'energy_from_grid': energy_from_grid,
        'energy_from_battery': energy_from_battery
    }


# Read the scraped data and recalculate
print("Reading scraped Agile data...")

# We need to re-scrape to get all rates, not just summary stats
# For now, let's use the summary data and estimate

df = pd.read_csv('agile_analysis_2025.csv')

print("\n" + "="*80)
print("CORRECTED ANALYSIS - Accounting for Off-Peak Consumption")
print("="*80)

print(f"\nUK Daily Consumption Profile: {DAILY_CONSUMPTION:.3f} kWh")
print("\nHourly breakdown:")
print("-" * 50)

# Show consumption by time period
overnight = sum(UK_CONSUMPTION_PROFILE[h] for h in range(0, 6))
morning = sum(UK_CONSUMPTION_PROFILE[h] for h in range(6, 12))
afternoon = sum(UK_CONSUMPTION_PROFILE[h] for h in range(12, 17))
evening = sum(UK_CONSUMPTION_PROFILE[h] for h in range(17, 24))

print(f"Overnight (00:00-06:00):  {overnight:.3f} kWh ({overnight/DAILY_CONSUMPTION*100:.1f}%)")
print(f"Morning (06:00-12:00):    {morning:.3f} kWh ({morning/DAILY_CONSUMPTION*100:.1f}%)")
print(f"Afternoon (12:00-17:00):  {afternoon:.3f} kWh ({afternoon/DAILY_CONSUMPTION*100:.1f}%)")
print(f"Evening (17:00-24:00):    {evening:.3f} kWh ({evening/DAILY_CONSUMPTION*100:.1f}%)")

print("\n" + "="*80)
print("KEY INSIGHT: Off-peak consumption already at cheap rates")
print("="*80)

# Typical Agile pattern:
# - Overnight (00:00-05:00): ~5-12p (CHEAP)
# - Morning peak (07:00-09:00): ~25-35p
# - Daytime (10:00-16:00): ~15-25p
# - Evening peak (16:00-19:00): ~30-45p (EXPENSIVE)
# - Late evening (20:00-23:00): ~15-25p

print("""
Typical Agile rate pattern:
- Overnight (00:00-05:00): 5-12p    ← Consumption: 1.65 kWh (17%)
- Morning (06:00-09:00): 20-35p     ← Consumption: 1.85 kWh (19%)
- Daytime (10:00-16:00): 15-25p     ← Consumption: 2.35 kWh (24%)
- Evening peak (17:00-19:00): 30-45p ← Consumption: 1.90 kWh (19%) ← BATTERY HELPS HERE
- Late evening (20:00-23:00): 15-25p ← Consumption: 2.10 kWh (21%)

The overnight consumption (1.65 kWh, 17%) is ALREADY at cheap rates.
Battery savings come mainly from shifting the 19% evening peak consumption.
""")

# Estimate corrected savings
# Previous calc: ALL consumption from battery charged at cheapest rates
# Corrected: Only expensive-hour consumption benefits

# Consumption during cheap hours (say, rate < 15p typically overnight)
cheap_hour_consumption = overnight  # ~1.65 kWh

# Consumption during expensive hours (evening peak primarily)
expensive_consumption = evening + morning  # ~4.00 kWh where battery really helps

# Midday - moderate benefit
moderate_consumption = afternoon  # ~2.35 kWh

print("\n" + "="*80)
print("CORRECTED SAVINGS ESTIMATE")
print("="*80)

# Using average rates from data
avg_low = df['lowest_rate_p'].mean()  # ~11p
avg_high = df['highest_rate_p'].mean()  # ~38.5p
avg_rate = df['average_rate_p'].mean()  # ~20p

print(f"\nAverage rates from 2025 data:")
print(f"  Lowest (charging target): {avg_low:.1f}p/kWh")
print(f"  Average: {avg_rate:.1f}p/kWh")
print(f"  Highest (peak): {avg_high:.1f}p/kWh")

# Estimate typical rates by time period
overnight_rate = avg_low + 3  # ~14p
morning_rate = avg_rate + 5   # ~25p
afternoon_rate = avg_rate     # ~20p
evening_rate = avg_high - 5   # ~33p
late_evening_rate = avg_rate - 2  # ~18p

print(f"\nEstimated typical rates by period:")
print(f"  Overnight: ~{overnight_rate:.0f}p")
print(f"  Morning: ~{morning_rate:.0f}p")
print(f"  Afternoon: ~{afternoon_rate:.0f}p")
print(f"  Evening peak: ~{evening_rate:.0f}p")
print(f"  Late evening: ~{late_evening_rate:.0f}p")

# WITHOUT BATTERY: pay at time of use
cost_without_daily = (
    overnight * overnight_rate +
    morning * morning_rate +
    afternoon * afternoon_rate +
    (evening - 2.1) * evening_rate +  # 17:00-19:00 peak
    2.1 * late_evening_rate  # 20:00-23:00
)

# WITH BATTERY:
# - Overnight consumption: still pay overnight rate (already cheap, no benefit)
# - Other consumption: pay avg charging rate instead
charging_rate = avg_low + 2  # ~13p (slightly higher than absolute min)

cost_with_daily = (
    overnight * overnight_rate +  # No change - already cheap
    (morning + afternoon + evening) * charging_rate  # Shifted to cheap charging
)

# But wait - we're limited by battery capacity!
# Can only shift 10 kWh, and we need some for overnight too
# More realistic: battery covers ~8 kWh of the expensive periods

battery_useful_capacity = 8.0  # Practical daily discharge
grid_consumption = DAILY_CONSUMPTION - battery_useful_capacity

# Refined calculation
expensive_hours_consumption = morning + afternoon + evening - overnight
# ~7.35 kWh that could benefit from battery

if expensive_hours_consumption <= battery_useful_capacity:
    shifted_consumption = expensive_hours_consumption
else:
    shifted_consumption = battery_useful_capacity

# Without battery
without_daily = DAILY_CONSUMPTION * avg_rate  # Simple estimate using avg

# With battery - shifted consumption pays charging rate, rest pays time-of-use
# This is complex, let's use the weighted approach

# Realistic savings rate:
# Average rate paid without battery: ~20p
# Average rate paid with battery: ~14p (mix of charging rate and cheap direct use)
# Savings: ~6p per kWh shifted

savings_per_kwh = avg_rate - charging_rate  # ~7p per kWh
daily_saving_corrected = shifted_consumption * savings_per_kwh

print(f"\n" + "-"*50)
print("CORRECTED DAILY SAVINGS CALCULATION")
print("-"*50)
print(f"Energy shifted to cheap rates: {shifted_consumption:.2f} kWh/day")
print(f"Average rate saving per kWh: {savings_per_kwh:.1f}p")
print(f"Daily saving: {daily_saving_corrected:.1f}p")

annual_saving_corrected = daily_saving_corrected * 365

print(f"\n" + "="*50)
print("CORRECTED ANNUAL RESULTS")
print("="*50)

# Original estimate
original_annual = df['daily_saving_p'].sum()

print(f"\nOriginal estimate (all from battery): £{original_annual/100:.2f}")
print(f"Corrected estimate (accounting for off-peak): £{annual_saving_corrected/100:.2f}")
print(f"Reduction factor: {annual_saving_corrected/original_annual*100:.0f}%")

print(f"""
\nThe corrected savings of ~£{annual_saving_corrected/100:.0f} accounts for:
1. Overnight consumption (17%) already at cheap rates - no battery benefit
2. Battery capacity limits how much can be shifted
3. Only ~{shifted_consumption:.1f} kWh/day actually benefits from arbitrage
""")

# Also show sensitivity
print("\n" + "="*50)
print("SENSITIVITY ANALYSIS")
print("="*50)
print("\nAnnual savings at different spread levels:")
for spread in [15, 20, 25, 30, 35]:
    savings = shifted_consumption * spread * 0.3 * 365 / 100  # 30% of spread captured
    print(f"  {spread}p average spread: £{savings:.0f}")
