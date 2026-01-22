"""
Optimal battery arbitrage with proper strategy:

1. Self-consumption only: Single daily cycle - charge cheap, use for consumption
2. Full arbitrage: Multiple cycles per day
   - Cycle 1: Overnight charge → morning consumption
   - Cycle 2: Midday recharge (if cheap) → evening peak export

With 5kW charge/discharge and 10kWh battery:
- Can do up to 2 full cycles per day if rate windows allow

Key insight: Self-consumption comes FIRST, then excess capacity for export
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
CHARGE_RATE = 5.0  # kW
DISCHARGE_RATE = 5.0  # kW
BATTERY_EFFICIENCY = 0.90  # 90% round-trip efficiency
EXPORT_RATE_FACTOR = 0.55  # Agile Outgoing ~55% of import


def scrape_day_rates(date):
    """Scrape all half-hourly rates for a specific date."""
    url = f"https://agilebuddy.uk/historic/agile/{date.year}/{date.month:02d}/{date.day:02d}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        rates = []
        rate_patterns = re.findall(r'(\d{1,2}:\d{2})[^\d]*?(-?\d+\.?\d*)p', text)
        for time_str, rate in rate_patterns:
            try:
                rates.append(float(rate))
            except ValueError:
                continue
        return rates if len(rates) >= 48 else None
    except:
        return None


def calculate_optimal_arbitrage(half_hourly_rates):
    """
    Calculate optimal arbitrage with proper priority:
    1. Self-consumption (use battery to avoid expensive grid imports)
    2. Export arbitrage (use EXCESS/additional cycles for grid trading)
    """

    if not half_hourly_rates or len(half_hourly_rates) < 48:
        return None

    # Convert to hourly rates
    hourly_import = [(half_hourly_rates[i*2] + half_hourly_rates[i*2+1]) / 2 for i in range(24)]
    hourly_export = [max(0, r * EXPORT_RATE_FACTOR) for r in hourly_import]

    # ========================================
    # BASELINE: No battery
    # ========================================
    baseline_cost = sum(UK_CONSUMPTION_PROFILE[h] * hourly_import[h] for h in range(24))

    # ========================================
    # SCENARIO 1: Self-consumption only
    # ========================================
    # Single daily cycle: charge at cheapest, use for consumption

    # Find 2 cheapest hours for charging
    hours_sorted = sorted(range(24), key=lambda h: hourly_import[h])
    charge_hours = hours_sorted[:2]
    charge_cost = sum(hourly_import[h] * CHARGE_RATE for h in charge_hours)
    usable_energy = BATTERY_CAPACITY * BATTERY_EFFICIENCY
    avg_charge_rate = charge_cost / usable_energy

    # Hour-by-hour: use battery when grid is expensive
    sc_cost = charge_cost  # Start with charging cost
    battery_soc = usable_energy

    for hour in range(24):
        consumption = UK_CONSUMPTION_PROFILE[hour]
        grid_rate = hourly_import[hour]

        if battery_soc >= consumption and grid_rate > avg_charge_rate:
            # Use battery
            battery_soc -= consumption
        else:
            # Use grid
            sc_cost += consumption * grid_rate

    sc_saving = baseline_cost - sc_cost

    # ========================================
    # SCENARIO 2: Self-consumption + Export excess
    # ========================================
    # After covering consumption, export remaining battery at peak

    sc_export_cost = charge_cost
    battery_soc = usable_energy
    export_revenue = 0

    # First priority: cover consumption
    for hour in range(24):
        consumption = UK_CONSUMPTION_PROFILE[hour]
        grid_rate = hourly_import[hour]

        if battery_soc >= consumption and grid_rate > avg_charge_rate:
            battery_soc -= consumption
        else:
            sc_export_cost += consumption * grid_rate

    # Second priority: export any remaining at best rates
    if battery_soc > 0:
        # Find best export hour (highest rate)
        best_export_hour = max(range(24), key=lambda h: hourly_export[h])
        export_revenue = battery_soc * hourly_export[best_export_hour]

    sc_export_total = sc_export_cost - export_revenue
    sc_export_saving = baseline_cost - sc_export_total
    excess_exported = usable_energy - (usable_energy - battery_soc) if battery_soc > 0 else 0

    # ========================================
    # SCENARIO 3: Dual cycle arbitrage
    # ========================================
    # Cycle 1: Overnight charge → morning/day consumption
    # Cycle 2: Midday charge (if cheap) → evening peak export

    # Find optimal windows for dual cycle
    # Overnight charging: hours 0-5 typically cheapest
    # Midday charging: hours 10-14 often have solar dip
    # Evening peak export: hours 16-19

    overnight_hours = [0, 1, 2, 3, 4, 5]
    midday_hours = [10, 11, 12, 13, 14]
    evening_hours = [16, 17, 18, 19]

    # Cycle 1: Overnight charge
    overnight_sorted = sorted(overnight_hours, key=lambda h: hourly_import[h])
    c1_charge_hours = overnight_sorted[:2]
    c1_charge_cost = sum(hourly_import[h] * CHARGE_RATE for h in c1_charge_hours)
    c1_avg_rate = c1_charge_cost / usable_energy

    # Cycle 2: Midday charge
    midday_sorted = sorted(midday_hours, key=lambda h: hourly_import[h])
    c2_charge_hours = midday_sorted[:2]
    c2_charge_cost = sum(hourly_import[h] * CHARGE_RATE for h in c2_charge_hours)
    c2_avg_rate = c2_charge_cost / usable_energy

    # Best evening export rates
    evening_sorted = sorted(evening_hours, key=lambda h: hourly_export[h], reverse=True)
    best_export_hours = evening_sorted[:2]
    export_rate_avg = sum(hourly_export[h] for h in best_export_hours) / 2

    # Dual cycle simulation
    dual_total_cost = 0
    dual_export_revenue = 0

    # Morning: battery from overnight charge
    battery_soc = usable_energy
    dual_total_cost += c1_charge_cost

    # 00:00-12:00: Use battery for consumption
    for hour in range(0, 12):
        consumption = UK_CONSUMPTION_PROFILE[hour]
        grid_rate = hourly_import[hour]

        if battery_soc >= consumption and grid_rate > c1_avg_rate:
            battery_soc -= consumption
        else:
            dual_total_cost += consumption * grid_rate

    # 12:00-14:00: Midday recharge if beneficial
    # Check if midday charge + evening export is profitable
    midday_to_evening_profit = (export_rate_avg * usable_energy) - c2_charge_cost

    if midday_to_evening_profit > 0:
        # Do cycle 2
        dual_total_cost += c2_charge_cost
        battery_soc = usable_energy  # Recharged

        # 14:00-16:00: Use for consumption
        for hour in range(14, 16):
            consumption = UK_CONSUMPTION_PROFILE[hour]
            grid_rate = hourly_import[hour]

            if battery_soc >= consumption and grid_rate > c2_avg_rate:
                battery_soc -= consumption
            else:
                dual_total_cost += consumption * grid_rate

        # 16:00-19:00: Export at peak (minus any consumption)
        for hour in range(16, 19):
            consumption = UK_CONSUMPTION_PROFILE[hour]
            grid_rate = hourly_import[hour]
            export_rate = hourly_export[hour]

            # Can we export AND cover consumption?
            # Export capacity this hour: DISCHARGE_RATE - consumption
            if battery_soc >= consumption:
                battery_soc -= consumption  # Cover consumption from battery
                # Export remaining capacity
                export_amount = min(DISCHARGE_RATE - consumption, battery_soc)
                if export_amount > 0 and export_rate > c2_avg_rate:
                    dual_export_revenue += export_amount * export_rate
                    battery_soc -= export_amount
            else:
                dual_total_cost += consumption * grid_rate

        # 19:00-24:00: Use remaining battery for evening consumption
        for hour in range(19, 24):
            consumption = UK_CONSUMPTION_PROFILE[hour]
            grid_rate = hourly_import[hour]

            if battery_soc >= consumption and grid_rate > c2_avg_rate:
                battery_soc -= consumption
            else:
                dual_total_cost += consumption * grid_rate

    else:
        # Single cycle only - no midday recharge
        # Continue using overnight battery and grid for rest of day
        for hour in range(12, 24):
            consumption = UK_CONSUMPTION_PROFILE[hour]
            grid_rate = hourly_import[hour]

            if battery_soc >= consumption and grid_rate > c1_avg_rate:
                battery_soc -= consumption
            else:
                dual_total_cost += consumption * grid_rate

    dual_net_cost = dual_total_cost - dual_export_revenue
    dual_saving = baseline_cost - dual_net_cost

    return {
        'baseline_cost': baseline_cost,

        # Scenario 1: Self-consumption only
        'sc_cost': sc_cost,
        'sc_saving': sc_saving,

        # Scenario 2: Self-consumption + export excess
        'sc_export_cost': sc_export_total,
        'sc_export_saving': sc_export_saving,
        'sc_export_revenue': export_revenue,

        # Scenario 3: Dual cycle
        'dual_cost': dual_net_cost,
        'dual_saving': dual_saving,
        'dual_export_revenue': dual_export_revenue,
        'dual_profitable': midday_to_evening_profit > 0,

        # Supporting data
        'avg_charge_rate': avg_charge_rate,
        'c2_charge_rate': c2_avg_rate,
        'export_rate_avg': export_rate_avg,
        'midday_profit_potential': midday_to_evening_profit,
    }


def main():
    print("="*80)
    print("OPTIMAL BATTERY ARBITRAGE ANALYSIS")
    print("5kW Charge/Discharge, 10kWh Battery")
    print("="*80)

    print(f"\nStrategy comparison:")
    print(f"  1. Self-consumption only: Charge cheap → use for consumption")
    print(f"  2. SC + Export excess: Same + export any leftover at peak")
    print(f"  3. Dual cycle: Overnight→consumption + Midday→evening export")

    print(f"\nParameters:")
    print(f"  Battery: {BATTERY_CAPACITY} kWh, {CHARGE_RATE} kW charge/discharge")
    print(f"  Efficiency: {BATTERY_EFFICIENCY*100:.0f}% round-trip")
    print(f"  Export rate: ~{EXPORT_RATE_FACTOR*100:.0f}% of import (Agile Outgoing)")
    print(f"  Daily consumption: {DAILY_CONSUMPTION:.2f} kWh")

    print("\n" + "-"*50)
    print("Scraping 2025 data...")
    print("-"*50)

    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    all_results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_date = {executor.submit(scrape_day_rates, d): d for d in dates}
        completed = 0
        for future in as_completed(future_to_date):
            date = future_to_date[future]
            rates = future.result()
            if rates:
                result = calculate_optimal_arbitrage(rates)
                if result:
                    result['date'] = date.strftime('%Y-%m-%d')
                    all_results.append(result)
            completed += 1
            if completed % 50 == 0:
                print(f"Progress: {completed}/{len(dates)}...")
            time.sleep(0.05)

    all_results.sort(key=lambda x: x['date'])
    days = len(all_results)
    print(f"\nProcessed {days} days")

    # Totals
    baseline = sum(r['baseline_cost'] for r in all_results)
    sc_saving = sum(r['sc_saving'] for r in all_results)
    sc_export_saving = sum(r['sc_export_saving'] for r in all_results)
    sc_export_rev = sum(r['sc_export_revenue'] for r in all_results)
    dual_saving = sum(r['dual_saving'] for r in all_results)
    dual_export_rev = sum(r['dual_export_revenue'] for r in all_results)
    dual_days = sum(1 for r in all_results if r['dual_profitable'])

    # Save CSV
    with open('agile_optimal_arbitrage_2025.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['date', 'baseline_p', 'sc_saving_p', 'sc_export_saving_p',
                        'sc_export_rev_p', 'dual_saving_p', 'dual_export_rev_p', 'dual_profitable'])
        for r in all_results:
            writer.writerow([
                r['date'], round(r['baseline_cost'], 2),
                round(r['sc_saving'], 2), round(r['sc_export_saving'], 2),
                round(r['sc_export_revenue'], 2), round(r['dual_saving'], 2),
                round(r['dual_export_revenue'], 2), r['dual_profitable']
            ])

    # Results
    print("\n" + "="*80)
    print("ANNUAL RESULTS - 2025")
    print("="*80)

    print(f"\n{'BASELINE (no battery):':<45} £{baseline/100:>8.2f}")

    print(f"\n{'SCENARIO 1: Self-consumption only':<45}")
    print(f"  {'Annual saving:':<40} £{sc_saving/100:>8.2f}")

    print(f"\n{'SCENARIO 2: Self-consumption + Export excess':<45}")
    print(f"  {'Annual saving:':<40} £{sc_export_saving/100:>8.2f}")
    print(f"  {'Export revenue included:':<40} £{sc_export_rev/100:>8.2f}")
    print(f"  {'Extra vs SC only:':<40} £{(sc_export_saving-sc_saving)/100:>8.2f}")

    print(f"\n{'SCENARIO 3: Dual cycle arbitrage':<45}")
    print(f"  {'Annual saving:':<40} £{dual_saving/100:>8.2f}")
    print(f"  {'Export revenue included:':<40} £{dual_export_rev/100:>8.2f}")
    print(f"  {'Days where 2nd cycle profitable:':<40} {dual_days}/{days}")
    print(f"  {'Extra vs SC only:':<40} £{(dual_saving-sc_saving)/100:>8.2f}")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"""
    ┌────────────────────────────────────────────────────────────────┐
    │ Strategy                          │ Annual Saving │ Extra     │
    ├────────────────────────────────────────────────────────────────┤
    │ 1. Self-consumption only          │ £{sc_saving/100:>7.2f}       │    -      │
    │ 2. SC + Export excess             │ £{sc_export_saving/100:>7.2f}       │ +£{(sc_export_saving-sc_saving)/100:>5.2f}   │
    │ 3. Dual cycle arbitrage           │ £{dual_saving/100:>7.2f}       │ +£{(dual_saving-sc_saving)/100:>5.2f}   │
    └────────────────────────────────────────────────────────────────┘
    """)

    additional_from_arbitrage = dual_saving - sc_saving

    print(f"\n{'='*60}")
    print("ANSWER TO YOUR QUESTION")
    print('='*60)
    print(f"""
    WITHOUT ARBITRAGE (self-consumption only):
      Annual income/saving: £{sc_saving/100:.2f}

    WITH FULL ARBITRAGE (dual cycle trading):
      Annual income/saving: £{dual_saving/100:.2f}
      - Export revenue:     £{dual_export_rev/100:.2f}

    ADDITIONAL VALUE FROM ARBITRAGE: £{additional_from_arbitrage/100:.2f}
    """)

    # Monthly breakdown
    print("\n" + "="*80)
    print("MONTHLY BREAKDOWN")
    print("="*80)
    print(f"\n{'Month':<10} {'SC Only':<12} {'SC+Export':<12} {'Dual Cycle':<12} {'Export Rev':<12}")
    print("-"*60)

    monthly = {}
    for r in all_results:
        m = r['date'][:7]
        if m not in monthly:
            monthly[m] = {'sc': 0, 'sc_exp': 0, 'dual': 0, 'exp_rev': 0}
        monthly[m]['sc'] += r['sc_saving']
        monthly[m]['sc_exp'] += r['sc_export_saving']
        monthly[m]['dual'] += r['dual_saving']
        monthly[m]['exp_rev'] += r['dual_export_revenue']

    for m in sorted(monthly.keys()):
        d = monthly[m]
        print(f"{m:<10} £{d['sc']/100:>8.2f}    £{d['sc_exp']/100:>8.2f}    £{d['dual']/100:>8.2f}    £{d['exp_rev']/100:>8.2f}")

    print("-"*60)
    print(f"{'TOTAL':<10} £{sc_saving/100:>8.2f}    £{sc_export_saving/100:>8.2f}    £{dual_saving/100:>8.2f}    £{dual_export_rev/100:>8.2f}")


if __name__ == '__main__':
    main()
