"""
Octopus Agile rate scraper and battery savings calculator.
Scrapes rates from agilebuddy.uk for 2025 and calculates savings with a 10kWh battery.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import re
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

# UK consumption profile (hourly kWh - normalized from watts)
UK_CONSUMPTION_PROFILE = {
    0: 0.275, 1: 0.275, 2: 0.275, 3: 0.275, 4: 0.275, 5: 0.275,
    6: 0.375, 7: 0.550, 8: 0.550, 9: 0.375, 10: 0.375, 11: 0.375,
    12: 0.425, 13: 0.425, 14: 0.375, 15: 0.375, 16: 0.375,
    17: 0.675, 18: 0.675, 19: 0.550, 20: 0.550,
    21: 0.400, 22: 0.400, 23: 0.400
}

DAILY_CONSUMPTION = sum(UK_CONSUMPTION_PROFILE.values())  # ~9.875 kWh
BATTERY_CAPACITY = 10.0  # kWh

def scrape_day(date):
    """Scrape Agile rates for a specific date."""
    url = f"https://agilebuddy.uk/historic/agile/{date.year}/{date.month:02d}/{date.day:02d}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract rates from the page - look for the rate data
        text = soup.get_text()

        # Look for lowest and highest rates
        lowest_match = re.search(r'Lowest[:\s]*(\d+\.?\d*)p', text, re.IGNORECASE)
        highest_match = re.search(r'Highest[:\s]*(\d+\.?\d*)p', text, re.IGNORECASE)
        avg_match = re.search(r'Average[:\s]*Rate[:\s]*(\d+\.?\d*)p', text, re.IGNORECASE)

        if not avg_match:
            avg_match = re.search(r'Average[:\s]*(\d+\.?\d*)p', text, re.IGNORECASE)

        # Also try to find all the rate values in the table/list
        rates = []
        rate_patterns = re.findall(r'(\d{1,2}:\d{2})[^\d]*?(-?\d+\.?\d*)p', text)

        for time_str, rate in rate_patterns:
            try:
                rates.append(float(rate))
            except ValueError:
                continue

        result = {
            'date': date.strftime('%Y-%m-%d'),
            'lowest': float(lowest_match.group(1)) if lowest_match else None,
            'highest': float(highest_match.group(1)) if highest_match else None,
            'average': float(avg_match.group(1)) if avg_match else None,
            'all_rates': rates if rates else None
        }

        # If we got all rates, calculate our own stats
        if rates and len(rates) >= 24:
            result['lowest'] = min(rates)
            result['highest'] = max(rates)
            result['average'] = sum(rates) / len(rates)

        return result

    except Exception as e:
        return {
            'date': date.strftime('%Y-%m-%d'),
            'lowest': None,
            'highest': None,
            'average': None,
            'all_rates': None,
            'error': str(e)
        }

def calculate_costs_with_rates(all_rates):
    """
    Calculate costs with and without battery given half-hourly rates.

    Without battery: pay at time of consumption
    With battery: charge during cheapest periods, use battery for consumption
    """
    if not all_rates or len(all_rates) < 48:
        return None, None

    # Convert 48 half-hourly rates to 24 hourly rates (average)
    hourly_rates = []
    for i in range(24):
        if i*2+1 < len(all_rates):
            hourly_rates.append((all_rates[i*2] + all_rates[i*2+1]) / 2)
        elif i*2 < len(all_rates):
            hourly_rates.append(all_rates[i*2])
        else:
            hourly_rates.append(all_rates[-1])  # fallback

    # Cost WITHOUT battery - pay at time of consumption
    cost_without_battery = 0
    for hour, consumption in UK_CONSUMPTION_PROFILE.items():
        cost_without_battery += consumption * hourly_rates[hour]

    # Cost WITH battery - charge at cheapest rates
    # Battery can store 10kWh, need to charge ~10kWh per day
    # Find cheapest hours to charge

    # We need to charge enough to cover daily consumption
    consumption_needed = DAILY_CONSUMPTION

    # Sort hours by rate to find cheapest
    hours_by_rate = sorted(range(24), key=lambda h: hourly_rates[h])

    # Charge during cheapest hours (assume we can charge 2.5kWh per hour)
    # With 10kWh battery and ~10kWh daily use, need about 4-5 hours charging
    charge_rate_per_hour = 2.5  # kWh
    energy_charged = 0
    cost_with_battery = 0

    for hour in hours_by_rate:
        if energy_charged >= consumption_needed:
            break
        charge_amount = min(charge_rate_per_hour, consumption_needed - energy_charged)
        cost_with_battery += charge_amount * hourly_rates[hour]
        energy_charged += charge_amount

    return cost_without_battery, cost_with_battery

def main():
    print("Starting Octopus Agile scraping for 2025...")

    # Generate all dates for 2025
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    print(f"Scraping {len(dates)} days...")

    results = []

    # Scrape with some parallelism but be respectful
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_date = {executor.submit(scrape_day, date): date for date in dates}

        completed = 0
        for future in as_completed(future_to_date):
            result = future.result()
            results.append(result)
            completed += 1

            if completed % 30 == 0:
                print(f"Progress: {completed}/{len(dates)} days scraped...")

            # Small delay between batches
            time.sleep(0.1)

    # Sort by date
    results.sort(key=lambda x: x['date'])

    # Calculate statistics and costs
    print("\nCalculating costs...")

    output_data = []
    total_cost_without = 0
    total_cost_with = 0
    valid_days = 0

    for result in results:
        row = {
            'date': result['date'],
            'lowest_rate_p': result.get('lowest'),
            'highest_rate_p': result.get('highest'),
            'average_rate_p': result.get('average'),
            'spread_p': None,
            'cost_without_battery_p': None,
            'cost_with_battery_p': None,
            'daily_saving_p': None
        }

        if result.get('lowest') and result.get('highest'):
            row['spread_p'] = result['highest'] - result['lowest']

        if result.get('all_rates'):
            cost_without, cost_with = calculate_costs_with_rates(result['all_rates'])
            if cost_without and cost_with:
                row['cost_without_battery_p'] = round(cost_without, 2)
                row['cost_with_battery_p'] = round(cost_with, 2)
                row['daily_saving_p'] = round(cost_without - cost_with, 2)
                total_cost_without += cost_without
                total_cost_with += cost_with
                valid_days += 1
        elif result.get('average') and result.get('lowest'):
            # Estimate using average and lowest
            cost_without = DAILY_CONSUMPTION * result['average']
            cost_with = DAILY_CONSUMPTION * result['lowest']
            row['cost_without_battery_p'] = round(cost_without, 2)
            row['cost_with_battery_p'] = round(cost_with, 2)
            row['daily_saving_p'] = round(cost_without - cost_with, 2)
            total_cost_without += cost_without
            total_cost_with += cost_with
            valid_days += 1

        output_data.append(row)

    # Save to CSV
    output_file = '/Users/adambricknail/Desktop/elec/agile_analysis_2025.csv'

    fieldnames = ['date', 'lowest_rate_p', 'highest_rate_p', 'average_rate_p',
                  'spread_p', 'cost_without_battery_p', 'cost_with_battery_p', 'daily_saving_p']

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_data)

    print(f"\nResults saved to {output_file}")

    # Summary statistics
    if valid_days > 0:
        print(f"\n{'='*60}")
        print("SUMMARY FOR 2025 (based on {valid_days} days with data)")
        print('='*60)
        print(f"Daily consumption (UK profile): {DAILY_CONSUMPTION:.3f} kWh")
        print(f"Battery capacity: {BATTERY_CAPACITY} kWh")
        print()
        print(f"Total cost WITHOUT battery: {total_cost_without/100:.2f} GBP")
        print(f"Total cost WITH battery: {total_cost_with/100:.2f} GBP")
        print(f"Annual savings: {(total_cost_without - total_cost_with)/100:.2f} GBP")
        print()
        print(f"Avg daily cost without battery: {total_cost_without/valid_days:.1f}p")
        print(f"Avg daily cost with battery: {total_cost_with/valid_days:.1f}p")
        print(f"Avg daily saving: {(total_cost_without - total_cost_with)/valid_days:.1f}p")
        print()

        # Calculate spread stats
        spreads = [r['spread_p'] for r in output_data if r['spread_p'] is not None]
        if spreads:
            print(f"Average daily spread: {sum(spreads)/len(spreads):.1f}p")
            print(f"Max spread: {max(spreads):.1f}p")
            print(f"Min spread: {min(spreads):.1f}p")

if __name__ == '__main__':
    main()
