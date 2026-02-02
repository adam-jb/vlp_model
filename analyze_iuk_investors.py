#!/usr/bin/env python3
"""
Analyze IUK Business Connect Investors for Battery VLP Business
Filter and rank by suitability
"""

import json

# Load the raw data
with open('/Users/adambricknail/Desktop/elec/iuk_investors_raw.json', 'r') as f:
    investors = json.load(f)

print(f"Total investors loaded: {len(investors)}")

# Field mapping (from sample record):
# field_59: Investor Name
# field_118: Investor Type (VC, Angel, etc.)
# field_63: Investment Stage (Pre-Revenue, Early Revenue, etc.)
# field_117: Sector Focus

# Target criteria for battery VLP business:
# 1. Sector: Clean Energy Industries (primary), also: Digital and Technologies
# 2. Stage: Pre-Revenue (required), also good: Early Revenue
# 3. Type: Any (VC, Angel, Family Office, etc.)

def get_field(record, field_num, raw=False):
    """Get field value from record"""
    key = f"field_{field_num}_raw" if raw else f"field_{field_num}"
    value = record.get(key, '')
    if isinstance(value, list):
        return value
    return value

def score_investor(record):
    """Score investor suitability for battery VLP business"""
    score = 0
    reasons = []

    name = get_field(record, 59)
    inv_type = get_field(record, 118)
    stages = get_field(record, 63, raw=True)
    sectors = get_field(record, 117, raw=True)

    if isinstance(stages, str):
        stages = [stages]
    if isinstance(sectors, str):
        sectors = [sectors]

    # Sector scoring (most important)
    sector_keywords = {
        'Clean Energy Industries': 50,
        'Clean Energy': 50,
        'Energy': 40,
        'Climate': 40,
        'Sustainability': 30,
        'Infrastructure': 20,
        'Digital and Technologies': 15,
        'Technology': 10,
        'Consumer': 10,
        'Financial Services': 5,
    }

    for sector in sectors:
        for keyword, points in sector_keywords.items():
            if keyword.lower() in sector.lower():
                score += points
                reasons.append(f"Sector: {sector} (+{points})")
                break

    # Stage scoring
    stage_scores = {
        'Pre-Revenue': 30,
        'Early Revenue': 20,
        'Seed': 25,
        'Revenue': 10,
        'Growth': 5,
    }

    for stage in stages:
        for keyword, points in stage_scores.items():
            if keyword.lower() in stage.lower():
                score += points
                reasons.append(f"Stage: {stage} (+{points})")
                break

    # Type scoring
    type_scores = {
        'Angel': 20,  # Angels often more flexible for early stage
        'VC': 15,
        'Family Office': 15,
        'Corporate': 10,
        'Accelerator': 20,
        'Grant': 10,
    }

    if isinstance(inv_type, list):
        inv_type = ', '.join(inv_type)

    for t, points in type_scores.items():
        if t.lower() in inv_type.lower():
            score += points
            reasons.append(f"Type: {inv_type} (+{points})")
            break

    return score, reasons, name, inv_type, stages, sectors

# Score all investors
scored_investors = []
for record in investors:
    score, reasons, name, inv_type, stages, sectors = score_investor(record)
    scored_investors.append({
        'name': name,
        'type': inv_type,
        'stages': stages,
        'sectors': sectors,
        'score': score,
        'reasons': reasons,
        'id': record.get('id')
    })

# Sort by score
scored_investors.sort(key=lambda x: x['score'], reverse=True)

# Print results
print("\n" + "=" * 80)
print("RANKED INVESTORS FOR BATTERY VLP BUSINESS")
print("Filtered: Clean Energy Industries + Pre-Revenue")
print("=" * 80)

# Filter for truly relevant investors (Clean Energy + Pre-Revenue)
tier1 = []
tier2 = []
tier3 = []

for inv in scored_investors:
    sectors_str = ', '.join(inv['sectors']) if isinstance(inv['sectors'], list) else inv['sectors']
    stages_str = ', '.join(inv['stages']) if isinstance(inv['stages'], list) else inv['stages']

    has_clean_energy = any('clean energy' in s.lower() or 'energy' in s.lower()
                          for s in (inv['sectors'] if isinstance(inv['sectors'], list) else [inv['sectors']]))
    has_pre_revenue = any('pre-revenue' in s.lower() or 'pre revenue' in s.lower()
                         for s in (inv['stages'] if isinstance(inv['stages'], list) else [inv['stages']]))
    has_early_revenue = any('early revenue' in s.lower()
                           for s in (inv['stages'] if isinstance(inv['stages'], list) else [inv['stages']]))

    if has_clean_energy and has_pre_revenue:
        tier1.append(inv)
    elif has_clean_energy and has_early_revenue:
        tier2.append(inv)
    elif has_clean_energy:
        tier3.append(inv)

print(f"\n📊 SUMMARY:")
print(f"   Total investors in database: {len(investors)}")
print(f"   TIER 1 (Clean Energy + Pre-Revenue): {len(tier1)}")
print(f"   TIER 2 (Clean Energy + Early Revenue): {len(tier2)}")
print(f"   TIER 3 (Clean Energy, other stages): {len(tier3)}")

print("\n" + "=" * 80)
print("🏆 TIER 1: CLEAN ENERGY + PRE-REVENUE (BEST FIT)")
print("   These investors explicitly target pre-revenue clean energy companies")
print("=" * 80)

for i, inv in enumerate(tier1, 1):
    sectors_str = ', '.join(inv['sectors']) if isinstance(inv['sectors'], list) else inv['sectors']
    stages_str = ', '.join(inv['stages']) if isinstance(inv['stages'], list) else inv['stages']
    print(f"\n{i}. {inv['name']}")
    print(f"   Type: {inv['type']}")
    print(f"   Stages: {stages_str}")
    print(f"   Sectors: {sectors_str}")
    print(f"   Score: {inv['score']}")

print("\n" + "=" * 80)
print("🥈 TIER 2: CLEAN ENERGY + EARLY REVENUE")
print("   Good fit if you have some traction/revenue")
print("=" * 80)

for i, inv in enumerate(tier2, 1):
    sectors_str = ', '.join(inv['sectors']) if isinstance(inv['sectors'], list) else inv['sectors']
    stages_str = ', '.join(inv['stages']) if isinstance(inv['stages'], list) else inv['stages']
    print(f"\n{i}. {inv['name']}")
    print(f"   Type: {inv['type']}")
    print(f"   Stages: {stages_str}")
    print(f"   Sectors: {sectors_str}")
    print(f"   Score: {inv['score']}")

print("\n" + "=" * 80)
print("🥉 TIER 3: CLEAN ENERGY (Other stages)")
print("   May consider pre-revenue despite stated preference")
print("=" * 80)

for i, inv in enumerate(tier3[:20], 1):  # Show top 20
    sectors_str = ', '.join(inv['sectors']) if isinstance(inv['sectors'], list) else inv['sectors']
    stages_str = ', '.join(inv['stages']) if isinstance(inv['stages'], list) else inv['stages']
    print(f"\n{i}. {inv['name']}")
    print(f"   Type: {inv['type']}")
    print(f"   Stages: {stages_str}")
    print(f"   Sectors: {sectors_str}")
    print(f"   Score: {inv['score']}")

if len(tier3) > 20:
    print(f"\n   ... and {len(tier3) - 20} more")

# Save ranked list to CSV
import csv

with open('/Users/adambricknail/Desktop/elec/iuk_investors_ranked.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Rank', 'Tier', 'Name', 'Type', 'Stages', 'Sectors', 'Score'])

    rank = 1
    for inv in tier1:
        sectors_str = ', '.join(inv['sectors']) if isinstance(inv['sectors'], list) else inv['sectors']
        stages_str = ', '.join(inv['stages']) if isinstance(inv['stages'], list) else inv['stages']
        writer.writerow([rank, 'TIER 1', inv['name'], inv['type'], stages_str, sectors_str, inv['score']])
        rank += 1

    for inv in tier2:
        sectors_str = ', '.join(inv['sectors']) if isinstance(inv['sectors'], list) else inv['sectors']
        stages_str = ', '.join(inv['stages']) if isinstance(inv['stages'], list) else inv['stages']
        writer.writerow([rank, 'TIER 2', inv['name'], inv['type'], stages_str, sectors_str, inv['score']])
        rank += 1

    for inv in tier3:
        sectors_str = ', '.join(inv['sectors']) if isinstance(inv['sectors'], list) else inv['sectors']
        stages_str = ', '.join(inv['stages']) if isinstance(inv['stages'], list) else inv['stages']
        writer.writerow([rank, 'TIER 3', inv['name'], inv['type'], stages_str, sectors_str, inv['score']])
        rank += 1

print(f"\n✓ Saved ranked list to iuk_investors_ranked.csv")
