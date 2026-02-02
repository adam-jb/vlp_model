#!/usr/bin/env python3
"""
Get contact details for top investors from IUK Business Connect
"""

import requests
import json
import time

# Knack API credentials
APP_ID = "67f664e4ef6776029638f4d1"
API_KEY = "5c4559c4-7c42-4a1b-81af-857017edeea2"
BASE_URL = "https://api.knack.com/v1"

# Load the raw data
with open('/Users/adambricknail/Desktop/elec/iuk_investors_raw.json', 'r') as f:
    investors = json.load(f)

# First, let's see ALL fields in a record to find contact info
print("=" * 80)
print("FULL RECORD STRUCTURE (First Investor)")
print("=" * 80)

sample = investors[0]
for key, value in sorted(sample.items()):
    print(f"{key}: {value}")

print("\n" + "=" * 80)
print("CHECKING FOR CONTACT FIELDS IN ALL RECORDS")
print("=" * 80)

# Check what fields exist
all_fields = set()
for inv in investors:
    all_fields.update(inv.keys())

print(f"\nAll fields found: {sorted(all_fields)}")

# Look for fields that might contain contact info
print("\n" + "=" * 80)
print("TOP INVESTORS WITH ALL AVAILABLE DATA")
print("=" * 80)

# Target investors we want details for
target_names = [
    "Octopus Ventures",
    "Green angel Ventures",
    "Carbon13",
    "Zero Carbon Capital",
    "Scottish National Investment Bank",
    "ETF Partners",
    "Sustainable Venture Development Partners",
    "ONE PLANET CAPITAL",
    "Cambridge Angels",
    "Northern Gritstone",
    "Foresight Group",
    "SUPERSEED VENTURES",
    "Clean Growth Fund",
    "IQ Capital Partners",
    "Mercia Asset Management",
    "Northstar Ventures",
    "IP Group",
    "Extantia",
    "CPT Capital",
    "Aer Ventures",
]

for inv in investors:
    name = inv.get('field_59', '')

    # Check if this is one of our targets
    is_target = any(t.lower() in name.lower() for t in target_names)

    if is_target:
        print(f"\n{'='*60}")
        print(f"📌 {name}")
        print(f"{'='*60}")

        for key, value in sorted(inv.items()):
            if value and str(value).strip() and key != 'id':
                # Clean up field names
                field_name = key.replace('field_', 'F').replace('_raw', ' (raw)')
                print(f"  {field_name}: {value}")
