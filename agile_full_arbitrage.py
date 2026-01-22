"""
Full battery arbitrage model with:
1. Self-consumption savings (charge cheap, use during expensive hours)
2. Grid export revenue (sell excess to grid during peak)

5kW charge/discharge, 10kWh battery
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

# Export rate is typically ~50-60% of import rate for Agile Outgoing
EXPORT_RATE_FACTOR = 0.55  # Conservative estimate


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

    except Exception as e:
        return None


def calculate_full_arbitrage(half_hourly_rates):
    """
    Full arbitrage calculation:

    Scenario 1: Self-consumption only (no export)
    - Charge at cheapest hours
    - Use battery for consumption during expensive hours
    - No grid export

    Scenario 2: Full arbitrage (with export)
    - Charge at cheapest hours
    - Use battery for consumption during expensive hours
    - Export excess battery capacity during peak prices

    Optimal daily strategy with 5kW/10kWh:
    - Identify 2 cheapest hours for charging (10kWh at 5kW)
    - Identify peak export hours (usually 16:00-19:00)
    - Use battery for consumption when rate > charging rate
    - Export remaining capacity during peak prices
    """

    if not half_hourly_rates or len(half_hourly_rates) < 48:
        return None

    # Convert to hourly rates (average of two half-hours)
    hourly_rates = []
    for i in range(24):
        hourly_rates.append((half_hourly_rates[i*2] + half_hourly_rates[i*2+1]) / 2)

    # Export rates (Agile Outgoing is typically ~55% of import)
    export_rates = [max(0, r * EXPORT_RATE_FACTOR) for r in hourly_rates]

    # ========================================
    # BASELINE: Cost without any battery
    # ========================================
    cost_without_battery = sum(
        UK_CONSUMPTION_PROFILE[h] * hourly_rates[h]
        for h in range(24)
    )

    # ========================================
    # SCENARIO 1: Self-consumption only
    # ========================================

    # Find 2 cheapest hours for charging
    hours_by_import_rate = sorted(range(24), key=lambda h: hourly_rates[h])
    charging_hours = hours_by_import_rate[:2]

    charging_cost = sum(hourly_rates[h] * CHARGE_RATE for h in charging_hours)
    usable_energy = BATTERY_CAPACITY * BATTERY_EFFICIENCY  # Account for losses
    avg_charging_rate = charging_cost / usable_energy

    # Use battery for consumption when beneficial
    sc_cost = 0
    sc_battery_used = 0

    for hour in range(24):
        consumption = UK_CONSUMPTION_PROFILE[hour]
        grid_rate = hourly_rates[hour]

        if grid_rate <= avg_charging_rate or sc_battery_used >= usable_energy:
            # Use grid - cheaper or battery empty
            sc_cost += consumption * grid_rate
        else:
            # Use battery
            available = min(consumption, usable_energy - sc_battery_used)
            sc_cost += available * avg_charging_rate
            sc_battery_used += available
            if consumption > available:
                sc_cost += (consumption - available) * grid_rate

    sc_total_cost = charging_cost + sc_cost - (sc_battery_used * avg_charging_rate)
    # Simplify: total cost = charging cost + grid consumption
    # Actually: we pay charging_cost upfront, then either use battery (no extra cost) or grid

    # Recalculate properly:
    sc_grid_cost = 0
    sc_battery_used = 0

    for hour in range(24):
        consumption = UK_CONSUMPTION_PROFILE[hour]
        grid_rate = hourly_rates[hour]

        if grid_rate <= avg_charging_rate:
            sc_grid_cost += consumption * grid_rate
        elif sc_battery_used + consumption <= usable_energy:
            sc_battery_used += consumption
            # Cost already paid in charging
        else:
            remaining = max(0, usable_energy - sc_battery_used)
            sc_battery_used += remaining
            sc_grid_cost += (consumption - remaining) * grid_rate

    sc_total_cost = charging_cost + sc_grid_cost
    sc_saving = cost_without_battery - sc_total_cost

    # ========================================
    # SCENARIO 2: Full arbitrage with export
    # ========================================

    # Strategy:
    # 1. Charge at 2 cheapest hours (same as before)
    # 2. During day: use battery for consumption when rate > charging rate
    # 3. During peak export hours: export excess capacity at export rate

    # Find best export hours (highest export rates)
    hours_by_export_rate = sorted(range(24), key=lambda h: export_rates[h], reverse=True)

    # Simulation hour by hour
    battery_soc = 0  # Start empty
    fa_grid_cost = 0
    fa_export_revenue = 0
    fa_charging_cost = 0

    # Two-pass approach:
    # Pass 1: Determine charging and consumption schedule
    # Pass 2: Determine export opportunity

    # For optimal strategy, we need to balance:
    # - Charging at cheapest times
    # - Using battery for consumption (saving import cost)
    # - Exporting at peak times (earning export revenue)

    # Simplified optimal approach:
    # Morning: Charge during cheapest overnight/early hours
    # Day: Use for consumption if needed
    # Evening peak (16-19): Export any excess at high rates
    # Night: Prepare for next cycle

    # Hour-by-hour simulation
    battery_soc = 0
    total_charged = 0
    total_discharged_consumption = 0
    total_exported = 0

    # Pre-calculate optimal charging windows (2 cheapest consecutive or near hours)
    charge_schedule = {h: 0 for h in range(24)}
    for h in charging_hours:
        charge_schedule[h] = CHARGE_RATE  # Charge 5kWh this hour

    # Pre-calculate peak export window (find 2 highest export rate hours)
    export_hours = sorted(range(24), key=lambda h: export_rates[h], reverse=True)[:3]

    for hour in range(24):
        consumption = UK_CONSUMPTION_PROFILE[hour]
        import_rate = hourly_rates[hour]
        export_rate = export_rates[hour]

        # Charging
        if charge_schedule[hour] > 0 and battery_soc < BATTERY_CAPACITY:
            charge_amount = min(charge_schedule[hour], BATTERY_CAPACITY - battery_soc)
            battery_soc += charge_amount * BATTERY_EFFICIENCY  # Efficiency loss on charge
            fa_charging_cost += charge_amount * import_rate
            total_charged += charge_amount

        # Consumption - decide: use grid or battery?
        if import_rate > avg_charging_rate and battery_soc >= consumption:
            # Use battery - it's cheaper
            battery_soc -= consumption
            total_discharged_consumption += consumption
        else:
            # Use grid
            fa_grid_cost += consumption * import_rate

        # Export opportunity - if peak export hour and battery has charge
        if hour in export_hours and battery_soc > 0:
            # Export as much as possible at 5kW rate
            export_amount = min(DISCHARGE_RATE, battery_soc)
            # Only export if profitable: export_rate > charging_rate
            if export_rate > avg_charging_rate:
                battery_soc -= export_amount
                fa_export_revenue += export_amount * export_rate
                total_exported += export_amount

    fa_total_cost = fa_charging_cost + fa_grid_cost - fa_export_revenue
    fa_saving = cost_without_battery - fa_total_cost

    # ========================================
    # SCENARIO 3: Maximum export (grid trading focus)
    # ========================================
    # Charge full 10kWh at cheapest, export maximum at peak
    # Minimal self-consumption consideration

    # Find absolute best 2 hours for charging
    best_charge_hours = hours_by_import_rate[:2]
    max_charge_cost = sum(hourly_rates[h] * CHARGE_RATE for h in best_charge_hours)

    # Find best 2 hours for export
    best_export_hours = hours_by_export_rate[:2]

    # After charging 10kWh, we have 9kWh usable (90% efficiency)
    # Export up to 5kWh per hour = 10kWh in 2 hours
    max_export_amount = min(BATTERY_CAPACITY * BATTERY_EFFICIENCY, DISCHARGE_RATE * 2)
    max_export_revenue = sum(export_rates[h] * DISCHARGE_RATE for h in best_export_hours[:2])

    # But we also have consumption to cover...
    # For pure trading: pay grid rate for all consumption
    grid_only_consumption_cost = sum(
        UK_CONSUMPTION_PROFILE[h] * hourly_rates[h] for h in range(24)
    )

    max_trading_profit = max_export_revenue - max_charge_cost
    max_trading_total = grid_only_consumption_cost - max_trading_profit

    return {
        'hourly_rates': hourly_rates,
        'export_rates': export_rates,
        'cost_without_battery': cost_without_battery,

        # Scenario 1: Self-consumption only
        'sc_charging_cost': charging_cost,
        'sc_grid_cost': sc_grid_cost,
        'sc_total_cost': sc_total_cost,
        'sc_saving': sc_saving,
        'sc_battery_used': sc_battery_used,

        # Scenario 2: Full arbitrage
        'fa_charging_cost': fa_charging_cost,
        'fa_grid_cost': fa_grid_cost,
        'fa_export_revenue': fa_export_revenue,
        'fa_total_cost': fa_total_cost,
        'fa_saving': fa_saving,
        'fa_exported': total_exported,

        # Scenario 3: Max trading
        'mt_charge_cost': max_charge_cost,
        'mt_export_revenue': max_export_revenue,
        'mt_trading_profit': max_trading_profit,

        # Additional arbitrage value
        'additional_from_export': fa_saving - sc_saving,

        'avg_charging_rate': avg_charging_rate,
        'charging_hours': charging_hours,
    }


def main():
    print("="*80)
    print("OCTOPUS AGILE FULL ARBITRAGE ANALYSIS")
    print("5kW Charge/Discharge, 10kWh Battery, 90% Efficiency")
    print("="*80)

    print(f"\nParameters:")
    print(f"  Battery capacity: {BATTERY_CAPACITY} kWh")
    print(f"  Charge/discharge rate: {CHARGE_RATE} kW")
    print(f"  Round-trip efficiency: {BATTERY_EFFICIENCY*100:.0f}%")
    print(f"  Export rate factor: {EXPORT_RATE_FACTOR*100:.0f}% of import rate")
    print(f"  Daily consumption: {DAILY_CONSUMPTION:.3f} kWh")

    print("\n" + "-"*50)
    print("Scraping 2025 Agile rates...")
    print("-"*50)

    # Generate dates
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    # Scrape
    all_results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_date = {executor.submit(scrape_day_rates, date): date for date in dates}

        completed = 0
        for future in as_completed(future_to_date):
            date = future_to_date[future]
            rates = future.result()

            if rates:
                result = calculate_full_arbitrage(rates)
                if result:
                    result['date'] = date.strftime('%Y-%m-%d')
                    all_results.append(result)

            completed += 1
            if completed % 50 == 0:
                print(f"Progress: {completed}/{len(dates)} days...")

            time.sleep(0.05)

    all_results.sort(key=lambda x: x['date'])
    print(f"\nProcessed {len(all_results)} days")

    # Aggregate results
    days = len(all_results)

    total_without = sum(r['cost_without_battery'] for r in all_results)

    # Scenario 1 totals
    sc_total_charging = sum(r['sc_charging_cost'] for r in all_results)
    sc_total_grid = sum(r['sc_grid_cost'] for r in all_results)
    sc_total_cost = sum(r['sc_total_cost'] for r in all_results)
    sc_total_saving = sum(r['sc_saving'] for r in all_results)

    # Scenario 2 totals
    fa_total_charging = sum(r['fa_charging_cost'] for r in all_results)
    fa_total_grid = sum(r['fa_grid_cost'] for r in all_results)
    fa_total_export = sum(r['fa_export_revenue'] for r in all_results)
    fa_total_cost = sum(r['fa_total_cost'] for r in all_results)
    fa_total_saving = sum(r['fa_saving'] for r in all_results)
    fa_total_exported = sum(r['fa_exported'] for r in all_results)

    # Additional value from export
    additional_export_value = fa_total_saving - sc_total_saving

    # Save detailed CSV
    output_file = 'agile_full_arbitrage_2025.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'date', 'cost_no_battery_p',
            'sc_saving_p', 'sc_cost_p',
            'fa_saving_p', 'fa_cost_p', 'fa_export_revenue_p',
            'additional_export_value_p'
        ])
        for r in all_results:
            writer.writerow([
                r['date'],
                round(r['cost_without_battery'], 2),
                round(r['sc_saving'], 2),
                round(r['sc_total_cost'], 2),
                round(r['fa_saving'], 2),
                round(r['fa_total_cost'], 2),
                round(r['fa_export_revenue'], 2),
                round(r['fa_saving'] - r['sc_saving'], 2)
            ])

    print(f"\nDetailed results saved to {output_file}")

    # ========================================
    # RESULTS OUTPUT
    # ========================================

    print("\n" + "="*80)
    print("ANNUAL RESULTS COMPARISON")
    print("="*80)

    print(f"\n{'BASELINE (No Battery)':<40}")
    print(f"  Annual electricity cost: £{total_without/100:.2f}")

    print(f"\n{'SCENARIO 1: Self-Consumption Only':<40}")
    print(f"  Charging cost:           £{sc_total_charging/100:.2f}")
    print(f"  Grid consumption cost:   £{sc_total_grid/100:.2f}")
    print(f"  Total annual cost:       £{sc_total_cost/100:.2f}")
    print(f"  ANNUAL SAVING:           £{sc_total_saving/100:.2f}")

    print(f"\n{'SCENARIO 2: Full Arbitrage (with Export)':<40}")
    print(f"  Charging cost:           £{fa_total_charging/100:.2f}")
    print(f"  Grid consumption cost:   £{fa_total_grid/100:.2f}")
    print(f"  Export revenue:          £{fa_total_export/100:.2f}")
    print(f"  Net annual cost:         £{fa_total_cost/100:.2f}")
    print(f"  ANNUAL SAVING:           £{fa_total_saving/100:.2f}")
    print(f"  Energy exported:         {fa_total_exported:.0f} kWh")

    print(f"\n" + "-"*50)
    print("ADDITIONAL VALUE FROM EXPORT ARBITRAGE")
    print("-"*50)
    print(f"  Self-consumption saving:    £{sc_total_saving/100:.2f}")
    print(f"  + Export revenue:           £{fa_total_export/100:.2f}")
    print(f"  = Total with arbitrage:     £{fa_total_saving/100:.2f}")
    print(f"\n  EXTRA from export:          £{additional_export_value/100:.2f}")

    # Monthly breakdown
    print("\n" + "="*80)
    print("MONTHLY COMPARISON")
    print("="*80)
    print(f"\n{'Month':<10} {'No Battery':<12} {'Self-Cons':<12} {'Full Arb':<12} {'Export Rev':<12}")
    print("-"*60)

    monthly = {}
    for r in all_results:
        month = r['date'][:7]
        if month not in monthly:
            monthly[month] = {'without': 0, 'sc': 0, 'fa': 0, 'export': 0}
        monthly[month]['without'] += r['cost_without_battery']
        monthly[month]['sc'] += r['sc_saving']
        monthly[month]['fa'] += r['fa_saving']
        monthly[month]['export'] += r['fa_export_revenue']

    for month in sorted(monthly.keys()):
        m = monthly[month]
        print(f"{month:<10} £{m['without']/100:>8.2f}    £{m['sc']/100:>8.2f}    £{m['fa']/100:>8.2f}    £{m['export']/100:>8.2f}")

    print("-"*60)
    print(f"{'TOTAL':<10} £{total_without/100:>8.2f}    £{sc_total_saving/100:>8.2f}    £{fa_total_saving/100:>8.2f}    £{fa_total_export/100:>8.2f}")

    # Summary box
    print("\n" + "="*80)
    print("SUMMARY: 10kWh BATTERY / 5kW INVERTER ANNUAL VALUE")
    print("="*80)
    print(f"""
    ┌─────────────────────────────────────────────────────────┐
    │  WITHOUT EXPORT (self-consumption only)                 │
    │  Annual saving: £{sc_total_saving/100:>6.2f}                                │
    ├─────────────────────────────────────────────────────────┤
    │  WITH EXPORT ARBITRAGE                                  │
    │  Annual saving: £{fa_total_saving/100:>6.2f}                                │
    │  (includes £{fa_total_export/100:.2f} export revenue)                    │
    ├─────────────────────────────────────────────────────────┤
    │  ADDITIONAL VALUE FROM ARBITRAGE: £{additional_export_value/100:>6.2f}              │
    └─────────────────────────────────────────────────────────┘
    """)

    # Daily averages
    print(f"Daily averages:")
    print(f"  Self-consumption saving: {sc_total_saving/days:.1f}p/day")
    print(f"  Full arbitrage saving:   {fa_total_saving/days:.1f}p/day")
    print(f"  Export revenue:          {fa_total_export/days:.1f}p/day")


if __name__ == '__main__':
    main()
