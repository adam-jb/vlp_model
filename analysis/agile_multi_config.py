"""
Battery arbitrage analysis for multiple configurations.
Runs analysis for different battery/inverter sizes.
"""

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import csv
import sys

# UK consumption profile (hourly kWh)
UK_CONSUMPTION_PROFILE = {
    0: 0.275, 1: 0.275, 2: 0.275, 3: 0.275, 4: 0.275, 5: 0.275,
    6: 0.375, 7: 0.550, 8: 0.550, 9: 0.375, 10: 0.375, 11: 0.375,
    12: 0.425, 13: 0.425, 14: 0.375, 15: 0.375, 16: 0.375,
    17: 0.675, 18: 0.675, 19: 0.550, 20: 0.550,
    21: 0.400, 22: 0.400, 23: 0.400
}

DAILY_CONSUMPTION = sum(UK_CONSUMPTION_PROFILE.values())  # 9.875 kWh
BATTERY_EFFICIENCY = 0.90
EXPORT_RATE_FACTOR = 0.55

# Cache for scraped rates
_rates_cache = {}


def scrape_day_rates(date):
    """Scrape all half-hourly rates for a specific date (with caching)."""
    date_str = date.strftime('%Y-%m-%d')
    if date_str in _rates_cache:
        return _rates_cache[date_str]

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
        result = rates if len(rates) >= 48 else None
        _rates_cache[date_str] = result
        return result
    except:
        return None


def calculate_arbitrage(half_hourly_rates, battery_kwh, power_kw):
    """
    Calculate arbitrage for given battery size and power rating.
    """
    if not half_hourly_rates or len(half_hourly_rates) < 48:
        return None

    # Convert to hourly rates
    hourly_import = [(half_hourly_rates[i*2] + half_hourly_rates[i*2+1]) / 2 for i in range(24)]
    hourly_export = [max(0, r * EXPORT_RATE_FACTOR) for r in hourly_import]

    # Baseline: No battery
    baseline_cost = sum(UK_CONSUMPTION_PROFILE[h] * hourly_import[h] for h in range(24))

    # Calculate hours needed to charge
    hours_to_charge = battery_kwh / power_kw  # e.g., 15kWh / 7.5kW = 2 hours

    # Find cheapest hours for charging
    hours_sorted = sorted(range(24), key=lambda h: hourly_import[h])
    charge_hours = hours_sorted[:int(hours_to_charge) + 1]  # Round up

    # Calculate charging cost
    energy_to_charge = battery_kwh
    charge_cost = 0
    for h in charge_hours:
        charge_amount = min(power_kw, energy_to_charge)
        charge_cost += hourly_import[h] * charge_amount
        energy_to_charge -= charge_amount
        if energy_to_charge <= 0:
            break

    usable_energy = battery_kwh * BATTERY_EFFICIENCY
    avg_charge_rate = charge_cost / usable_energy if usable_energy > 0 else 0

    # ========================================
    # SCENARIO 1: Self-consumption only
    # ========================================
    sc_cost = charge_cost
    battery_soc = usable_energy

    for hour in range(24):
        consumption = UK_CONSUMPTION_PROFILE[hour]
        grid_rate = hourly_import[hour]

        if battery_soc >= consumption and grid_rate > avg_charge_rate:
            battery_soc -= consumption
        else:
            sc_cost += consumption * grid_rate

    sc_saving = baseline_cost - sc_cost
    sc_battery_used = usable_energy - battery_soc

    # ========================================
    # SCENARIO 2: Self-consumption + Export excess
    # ========================================
    sc_exp_cost = charge_cost
    battery_soc = usable_energy
    export_revenue = 0

    # First: cover consumption
    for hour in range(24):
        consumption = UK_CONSUMPTION_PROFILE[hour]
        grid_rate = hourly_import[hour]

        if battery_soc >= consumption and grid_rate > avg_charge_rate:
            battery_soc -= consumption
        else:
            sc_exp_cost += consumption * grid_rate

    # Second: export excess at best rate
    if battery_soc > 0:
        # Find best export hours based on power rating
        hours_by_export = sorted(range(24), key=lambda h: hourly_export[h], reverse=True)
        remaining = battery_soc
        for h in hours_by_export:
            if remaining <= 0:
                break
            export_amount = min(power_kw, remaining)
            export_revenue += export_amount * hourly_export[h]
            remaining -= export_amount

    sc_exp_total = sc_exp_cost - export_revenue
    sc_exp_saving = baseline_cost - sc_exp_total
    excess_exported = usable_energy - sc_battery_used

    # ========================================
    # SCENARIO 3: Dual cycle (for larger batteries)
    # ========================================
    # Only makes sense if battery > consumption AND export profitable

    dual_saving = 0
    dual_export_rev = 0
    dual_profitable = False

    # With larger batteries, we have excess capacity for trading
    excess_capacity = usable_energy - DAILY_CONSUMPTION

    if excess_capacity > 0:
        # We can potentially do arbitrage with excess capacity
        # Find cheapest midday hours and best evening export hours
        midday_hours = [10, 11, 12, 13, 14]
        evening_hours = [16, 17, 18, 19]

        midday_sorted = sorted(midday_hours, key=lambda h: hourly_import[h])
        evening_sorted = sorted(evening_hours, key=lambda h: hourly_export[h], reverse=True)

        # Cost to charge excess capacity at midday
        mid_charge_cost = 0
        to_charge = min(excess_capacity, power_kw * 2)  # Max 2 hours midday
        temp_charge = to_charge
        for h in midday_sorted:
            if temp_charge <= 0:
                break
            amt = min(power_kw, temp_charge)
            mid_charge_cost += amt * hourly_import[h]
            temp_charge -= amt

        # Revenue from exporting at evening peak
        evening_export_rev = 0
        to_export = to_charge * BATTERY_EFFICIENCY
        temp_export = to_export
        for h in evening_sorted:
            if temp_export <= 0:
                break
            amt = min(power_kw, temp_export)
            evening_export_rev += amt * hourly_export[h]
            temp_export -= amt

        # Is it profitable?
        trade_profit = evening_export_rev - mid_charge_cost

        if trade_profit > 0:
            dual_profitable = True
            dual_export_rev = export_revenue + evening_export_rev
            dual_saving = sc_exp_saving + trade_profit
        else:
            dual_saving = sc_exp_saving
            dual_export_rev = export_revenue

    else:
        dual_saving = sc_exp_saving
        dual_export_rev = export_revenue

    return {
        'baseline_cost': baseline_cost,
        'sc_saving': sc_saving,
        'sc_exp_saving': sc_exp_saving,
        'sc_export_revenue': export_revenue,
        'dual_saving': dual_saving,
        'dual_export_revenue': dual_export_rev,
        'dual_profitable': dual_profitable,
        'excess_capacity': max(0, usable_energy - DAILY_CONSUMPTION),
        'avg_charge_rate': avg_charge_rate,
    }


def run_analysis(battery_kwh, power_kw, rates_data=None):
    """Run full year analysis for a battery configuration."""

    print(f"\n{'='*70}")
    print(f"CONFIGURATION: {battery_kwh}kWh Battery / {power_kw}kW Inverter")
    print(f"{'='*70}")

    print(f"\nParameters:")
    print(f"  Battery capacity: {battery_kwh} kWh")
    print(f"  Charge/discharge rate: {power_kw} kW")
    print(f"  Hours to full charge: {battery_kwh/power_kw:.1f} hours")
    print(f"  Usable energy (90% eff): {battery_kwh * BATTERY_EFFICIENCY:.1f} kWh")
    print(f"  Daily consumption: {DAILY_CONSUMPTION:.2f} kWh")
    print(f"  Excess capacity: {max(0, battery_kwh * BATTERY_EFFICIENCY - DAILY_CONSUMPTION):.1f} kWh")

    # Get or scrape rates
    if rates_data is None:
        print("\nScraping 2025 rates...")
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 12, 31)
        dates = [start_date + timedelta(days=i) for i in range(365)]

        rates_data = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_date = {executor.submit(scrape_day_rates, d): d for d in dates}
            completed = 0
            for future in as_completed(future_to_date):
                date = future_to_date[future]
                rates = future.result()
                if rates:
                    rates_data.append((date, rates))
                completed += 1
                if completed % 100 == 0:
                    print(f"  Progress: {completed}/365...")
                time.sleep(0.03)

        rates_data.sort(key=lambda x: x[0])

    # Calculate for each day
    results = []
    for date, rates in rates_data:
        result = calculate_arbitrage(rates, battery_kwh, power_kw)
        if result:
            result['date'] = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date
            results.append(result)

    days = len(results)

    # Totals
    baseline = sum(r['baseline_cost'] for r in results)
    sc_saving = sum(r['sc_saving'] for r in results)
    sc_exp_saving = sum(r['sc_exp_saving'] for r in results)
    sc_exp_rev = sum(r['sc_export_revenue'] for r in results)
    dual_saving = sum(r['dual_saving'] for r in results)
    dual_exp_rev = sum(r['dual_export_revenue'] for r in results)
    dual_days = sum(1 for r in results if r['dual_profitable'])

    # Results
    print(f"\n{'-'*50}")
    print("ANNUAL RESULTS")
    print(f"{'-'*50}")

    print(f"\n{'Baseline (no battery):':<40} £{baseline/100:>8.2f}")

    print(f"\n{'1. Self-consumption only:':<40} £{sc_saving/100:>8.2f} saving")

    print(f"\n{'2. Self-consumption + Export excess:':<40}")
    print(f"   {'Saving:':<37} £{sc_exp_saving/100:>8.2f}")
    print(f"   {'Export revenue:':<37} £{sc_exp_rev/100:>8.2f}")
    print(f"   {'Extra vs SC only:':<37} £{(sc_exp_saving - sc_saving)/100:>8.2f}")

    if battery_kwh * BATTERY_EFFICIENCY > DAILY_CONSUMPTION:
        print(f"\n{'3. With additional arbitrage trading:':<40}")
        print(f"   {'Saving:':<37} £{dual_saving/100:>8.2f}")
        print(f"   {'Total export revenue:':<37} £{dual_exp_rev/100:>8.2f}")
        print(f"   {'Days with profitable trades:':<37} {dual_days}/{days}")
        print(f"   {'Extra vs SC+Export:':<37} £{(dual_saving - sc_exp_saving)/100:>8.2f}")

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")

    best_saving = max(sc_saving, sc_exp_saving, dual_saving)
    best_strategy = "Self-consumption only" if best_saving == sc_saving else \
                   "SC + Export excess" if best_saving == sc_exp_saving else \
                   "SC + Export + Trading"

    print(f"""
    ┌──────────────────────────────────────────────────┐
    │ {battery_kwh}kWh / {power_kw}kW Configuration{' '*(25-len(str(battery_kwh))-len(str(power_kw)))}│
    ├──────────────────────────────────────────────────┤
    │ Self-consumption only:      £{sc_saving/100:>7.2f}            │
    │ SC + Export excess:         £{sc_exp_saving/100:>7.2f}            │""")

    if battery_kwh * BATTERY_EFFICIENCY > DAILY_CONSUMPTION:
        print(f"    │ SC + Export + Trading:      £{dual_saving/100:>7.2f}            │")

    print(f"""    ├──────────────────────────────────────────────────┤
    │ BEST STRATEGY: {best_strategy:<20} │
    │ ANNUAL VALUE:  £{best_saving/100:>7.2f}                       │
    └──────────────────────────────────────────────────┘
    """)

    return {
        'battery_kwh': battery_kwh,
        'power_kw': power_kw,
        'sc_saving': sc_saving,
        'sc_exp_saving': sc_exp_saving,
        'sc_exp_rev': sc_exp_rev,
        'dual_saving': dual_saving,
        'dual_exp_rev': dual_exp_rev,
        'best_saving': best_saving,
        'rates_data': rates_data
    }


def main():
    print("="*70)
    print("OCTOPUS AGILE BATTERY ANALYSIS - MULTIPLE CONFIGURATIONS")
    print("="*70)

    configs = [
        (10, 5),      # 10kWh / 5kW
        (15, 7.5),    # 15kWh / 7.5kW
        (20, 10),     # 20kWh / 10kW
    ]

    all_results = []
    rates_data = None  # Will be populated on first run and reused

    for battery, power in configs:
        result = run_analysis(battery, power, rates_data)
        all_results.append(result)
        rates_data = result['rates_data']  # Reuse cached rates

    # Final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON - ALL CONFIGURATIONS")
    print("="*70)

    print(f"""
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Config          │ SC Only    │ SC+Export  │ Full Arb   │ Export Rev   │
    ├─────────────────────────────────────────────────────────────────────────┤""")

    for r in all_results:
        config = f"{r['battery_kwh']}kWh/{r['power_kw']}kW"
        print(f"    │ {config:<15} │ £{r['sc_saving']/100:>7.2f}   │ £{r['sc_exp_saving']/100:>7.2f}   │ £{r['dual_saving']/100:>7.2f}   │ £{r['dual_exp_rev']/100:>7.2f}     │")

    print(f"""    └─────────────────────────────────────────────────────────────────────────┘
    """)

    # Incremental value
    print("INCREMENTAL VALUE OF LARGER BATTERIES:")
    print("-"*50)

    base = all_results[0]
    for r in all_results[1:]:
        extra = r['best_saving'] - base['best_saving']
        extra_capacity = r['battery_kwh'] - base['battery_kwh']
        print(f"  {base['battery_kwh']}kWh → {r['battery_kwh']}kWh: +£{extra/100:.2f}/year (+{extra_capacity}kWh capacity)")
        base = r

    print("\n" + "="*70)
    print("SUMMARY: ANNUAL VALUE BY CONFIGURATION")
    print("="*70)

    for r in all_results:
        extra_cap = max(0, r['battery_kwh'] * 0.9 - DAILY_CONSUMPTION)
        print(f"""
    {r['battery_kwh']}kWh / {r['power_kw']}kW:
      WITHOUT extra arbitrage: £{r['sc_exp_saving']/100:.2f}
      WITH extra arbitrage:    £{r['dual_saving']/100:.2f}
      Export revenue total:    £{r['dual_exp_rev']/100:.2f}
      Excess capacity/day:     {extra_cap:.1f} kWh""")


if __name__ == '__main__':
    main()
