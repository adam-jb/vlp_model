"""
Battery savings calculation with 5kW charge rate and 10kWh battery.
Properly accounts for off-peak consumption already at cheap rates.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import csv

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
CHARGE_RATE = 5.0  # kW - can charge 5kWh per hour, so full battery in 2 hours

def scrape_day_rates(date):
    """Scrape all half-hourly rates for a specific date."""
    url = f"https://agilebuddy.uk/historic/agile/{date.year}/{date.month:02d}/{date.day:02d}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()

        # Extract all rate values with times
        rates = []
        rate_patterns = re.findall(r'(\d{1,2}:\d{2})[^\d]*?(-?\d+\.?\d*)p', text)

        for time_str, rate in rate_patterns:
            try:
                rates.append(float(rate))
            except ValueError:
                continue

        return rates if len(rates) >= 48 else None

    except Exception as e:
        return None

def calculate_savings_5kw(half_hourly_rates):
    """
    Calculate savings with 5kW charger and 10kWh battery.

    With 5kW charging:
    - Can charge 5kWh per hour (2.5kWh per half-hour slot)
    - Full 10kWh battery in 2 hours
    - Can target the 2 absolute cheapest hours

    Strategy:
    - Find cheapest 2 hours to charge (4 half-hour slots)
    - For consumption: use battery if rate > charging rate, else use grid
    """

    if not half_hourly_rates or len(half_hourly_rates) < 48:
        return None

    # Convert to hourly rates (average of two half-hours)
    hourly_rates = []
    for i in range(24):
        hourly_rates.append((half_hourly_rates[i*2] + half_hourly_rates[i*2+1]) / 2)

    # Cost WITHOUT battery - pay at time of consumption
    cost_without = sum(
        UK_CONSUMPTION_PROFILE[h] * hourly_rates[h]
        for h in range(24)
    )

    # Find cheapest hours for charging
    # With 5kW rate, need 2 hours to charge 10kWh
    hours_sorted = sorted(range(24), key=lambda h: hourly_rates[h])
    cheapest_2_hours = hours_sorted[:2]

    # Calculate charging cost
    charging_cost = sum(hourly_rates[h] * CHARGE_RATE for h in cheapest_2_hours)
    avg_charging_rate = charging_cost / BATTERY_CAPACITY  # p per kWh in battery

    # Cost WITH battery
    # For each hour: use grid if rate <= charging rate, else use battery
    cost_with = 0
    energy_from_grid = 0
    energy_from_battery = 0
    battery_used = 0

    for hour in range(24):
        consumption = UK_CONSUMPTION_PROFILE[hour]
        grid_rate = hourly_rates[hour]

        # If grid is cheaper than what we charged at, use grid
        # If grid is more expensive AND we have battery capacity, use battery
        if grid_rate <= avg_charging_rate:
            # Use grid - it's cheap enough
            cost_with += consumption * grid_rate
            energy_from_grid += consumption
        elif battery_used + consumption <= BATTERY_CAPACITY:
            # Use battery - we charged cheaper
            cost_with += consumption * avg_charging_rate
            energy_from_battery += consumption
            battery_used += consumption
        else:
            # Battery depleted, must use grid
            remaining_battery = max(0, BATTERY_CAPACITY - battery_used)
            if remaining_battery > 0:
                cost_with += remaining_battery * avg_charging_rate
                cost_with += (consumption - remaining_battery) * grid_rate
                energy_from_battery += remaining_battery
                energy_from_grid += (consumption - remaining_battery)
                battery_used = BATTERY_CAPACITY
            else:
                cost_with += consumption * grid_rate
                energy_from_grid += consumption

    return {
        'cost_without': cost_without,
        'cost_with': cost_with,
        'saving': cost_without - cost_with,
        'avg_charging_rate': avg_charging_rate,
        'cheapest_hours': cheapest_2_hours,
        'energy_from_grid': energy_from_grid,
        'energy_from_battery': energy_from_battery,
        'hourly_rates': hourly_rates
    }

def main():
    print("="*80)
    print("OCTOPUS AGILE ANALYSIS - 5kW CHARGER / 10kWh BATTERY")
    print("="*80)
    print(f"\nBattery: {BATTERY_CAPACITY} kWh")
    print(f"Charge rate: {CHARGE_RATE} kW (full charge in {BATTERY_CAPACITY/CHARGE_RATE:.0f} hours)")
    print(f"Daily consumption (UK profile): {DAILY_CONSUMPTION:.3f} kWh")

    print("\n" + "-"*50)
    print("Scraping 2025 Agile rates with full half-hourly data...")
    print("-"*50)

    # Generate all dates for 2025
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    # Scrape with parallelism
    all_results = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_date = {executor.submit(scrape_day_rates, date): date for date in dates}

        completed = 0
        for future in as_completed(future_to_date):
            date = future_to_date[future]
            rates = future.result()

            if rates:
                result = calculate_savings_5kw(rates)
                if result:
                    result['date'] = date.strftime('%Y-%m-%d')
                    all_results.append(result)

            completed += 1
            if completed % 50 == 0:
                print(f"Progress: {completed}/{len(dates)} days...")

            time.sleep(0.05)

    # Sort by date
    all_results.sort(key=lambda x: x['date'])

    print(f"\nSuccessfully processed {len(all_results)} days")

    # Calculate totals
    total_cost_without = sum(r['cost_without'] for r in all_results)
    total_cost_with = sum(r['cost_with'] for r in all_results)
    total_saving = total_cost_without - total_cost_with
    total_from_grid = sum(r['energy_from_grid'] for r in all_results)
    total_from_battery = sum(r['energy_from_battery'] for r in all_results)

    # Save detailed results
    output_file = 'agile_analysis_2025_5kw.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'cost_without_p', 'cost_with_p', 'saving_p',
                        'avg_charge_rate_p', 'energy_from_grid_kwh', 'energy_from_battery_kwh'])
        for r in all_results:
            writer.writerow([
                r['date'],
                round(r['cost_without'], 2),
                round(r['cost_with'], 2),
                round(r['saving'], 2),
                round(r['avg_charging_rate'], 2),
                round(r['energy_from_grid'], 3),
                round(r['energy_from_battery'], 3)
            ])

    print(f"\nDetailed results saved to {output_file}")

    # Summary
    print("\n" + "="*80)
    print("ANNUAL SUMMARY (5kW charger)")
    print("="*80)

    print(f"\nCost WITHOUT battery:  £{total_cost_without/100:.2f}")
    print(f"Cost WITH battery:     £{total_cost_with/100:.2f}")
    print(f"ANNUAL SAVING:         £{total_saving/100:.2f}")
    print(f"Savings percentage:    {(total_saving/total_cost_without)*100:.1f}%")

    print(f"\n" + "-"*50)
    print("Energy source breakdown (annual)")
    print("-"*50)
    print(f"From grid (cheap hours):  {total_from_grid:.1f} kWh ({total_from_grid/(total_from_grid+total_from_battery)*100:.1f}%)")
    print(f"From battery (shifted):   {total_from_battery:.1f} kWh ({total_from_battery/(total_from_grid+total_from_battery)*100:.1f}%)")

    # Daily averages
    days = len(all_results)
    print(f"\n" + "-"*50)
    print("Daily averages")
    print("-"*50)
    print(f"Avg daily cost without battery: {total_cost_without/days:.1f}p")
    print(f"Avg daily cost with battery:    {total_cost_with/days:.1f}p")
    print(f"Avg daily saving:               {total_saving/days:.1f}p")
    print(f"Avg charging rate achieved:     {sum(r['avg_charging_rate'] for r in all_results)/days:.1f}p/kWh")

    # Monthly breakdown
    print("\n" + "="*80)
    print("MONTHLY BREAKDOWN")
    print("="*80)

    # Group by month
    monthly = {}
    for r in all_results:
        month = r['date'][:7]
        if month not in monthly:
            monthly[month] = {'without': 0, 'with': 0, 'saving': 0, 'days': 0}
        monthly[month]['without'] += r['cost_without']
        monthly[month]['with'] += r['cost_with']
        monthly[month]['saving'] += r['saving']
        monthly[month]['days'] += 1

    print(f"\n{'Month':<10} {'Without':<12} {'With':<12} {'Saving':<12} {'Avg Save/Day':<12}")
    print("-"*60)

    for month in sorted(monthly.keys()):
        m = monthly[month]
        print(f"{month:<10} £{m['without']/100:>8.2f}    £{m['with']/100:>8.2f}    £{m['saving']/100:>8.2f}    {m['saving']/m['days']:>8.1f}p")

    print("-"*60)
    print(f"{'TOTAL':<10} £{total_cost_without/100:>8.2f}    £{total_cost_with/100:>8.2f}    £{total_saving/100:>8.2f}")

    # Compare with 2.5kW
    print("\n" + "="*80)
    print("COMPARISON: 5kW vs 2.5kW CHARGING")
    print("="*80)
    print("""
With 5kW charging (2 hours to full):
- Can target the 2 absolute cheapest hours each day
- More selective = lower average charging cost
- Better arbitrage opportunity

With 2.5kW charging (4 hours to full):
- Must use 4 cheapest hours
- Less selective = slightly higher average charging cost
""")

    # Best/worst days
    best = max(all_results, key=lambda x: x['saving'])
    worst = min(all_results, key=lambda x: x['saving'])

    print(f"\nBest day: {best['date']} - saved {best['saving']:.0f}p (£{best['saving']/100:.2f})")
    print(f"  Charged at avg {best['avg_charging_rate']:.1f}p, consumption would have cost {best['cost_without']:.0f}p")

    print(f"\nWorst day: {worst['date']} - saved {worst['saving']:.0f}p")
    print(f"  Charged at avg {worst['avg_charging_rate']:.1f}p")

if __name__ == '__main__':
    main()
