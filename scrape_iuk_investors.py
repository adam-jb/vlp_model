#!/usr/bin/env python3
"""
Scrape IUK Business Connect Investor Directory
Uses the Knack API directly to bypass JavaScript rendering
"""

import requests
import json
import time
import csv

# Knack API credentials from the page source
APP_ID = "67f664e4ef6776029638f4d1"
API_KEY = "5c4559c4-7c42-4a1b-81af-857017edeea2"

# API endpoints
BASE_URL = "https://api.knack.com/v1"

def get_knack_records(object_id, page=1, rows_per_page=100):
    """Fetch records from Knack API"""
    url = f"{BASE_URL}/objects/object_{object_id}/records"

    headers = {
        "X-Knack-Application-Id": APP_ID,
        "X-Knack-REST-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    params = {
        "page": page,
        "rows_per_page": rows_per_page
    }

    response = requests.get(url, headers=headers, params=params)
    return response

def get_view_records(scene_id, view_id, page=1, rows_per_page=100):
    """Fetch records from a specific view"""
    url = f"{BASE_URL}/pages/scene_{scene_id}/views/view_{view_id}/records"

    headers = {
        "X-Knack-Application-Id": APP_ID,
        "X-Knack-REST-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    params = {
        "page": page,
        "rows_per_page": rows_per_page
    }

    response = requests.get(url, headers=headers, params=params)
    return response

def try_multiple_approaches():
    """Try different API approaches to get investor data"""

    print("=" * 70)
    print("IUK BUSINESS CONNECT INVESTOR DIRECTORY SCRAPER")
    print("=" * 70)

    # From the page source:
    # sceneId: 115, viewId: 122, objectId: 11

    results = []

    # Approach 1: Try the view endpoint (most likely to work for filtered data)
    print("\n[1] Trying view endpoint (scene_115/view_122)...")
    try:
        resp = get_view_records(115, 122, page=1, rows_per_page=100)
        print(f"    Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"    Success! Found {data.get('total_records', 'unknown')} total records")
            results.append(("view_115_122", data))
        else:
            print(f"    Response: {resp.text[:500]}")
    except Exception as e:
        print(f"    Error: {e}")

    # Approach 2: Try the object endpoint directly
    print("\n[2] Trying object endpoint (object_11)...")
    try:
        resp = get_knack_records(11, page=1, rows_per_page=100)
        print(f"    Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"    Success! Found {data.get('total_records', 'unknown')} total records")
            results.append(("object_11", data))
        else:
            print(f"    Response: {resp.text[:500]}")
    except Exception as e:
        print(f"    Error: {e}")

    # Approach 3: Try other common object IDs
    print("\n[3] Trying other object IDs...")
    for obj_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]:
        try:
            resp = get_knack_records(obj_id, page=1, rows_per_page=10)
            if resp.status_code == 200:
                data = resp.json()
                total = data.get('total_records', 0)
                if total > 0:
                    print(f"    object_{obj_id}: {total} records found")
                    results.append((f"object_{obj_id}", data))
        except:
            pass
        time.sleep(0.1)

    return results

def fetch_all_pages(object_id=None, scene_id=None, view_id=None, rows_per_page=100):
    """Fetch all pages of records"""
    all_records = []
    page = 1

    while True:
        print(f"  Fetching page {page}...")

        if scene_id and view_id:
            resp = get_view_records(scene_id, view_id, page=page, rows_per_page=rows_per_page)
        else:
            resp = get_knack_records(object_id, page=page, rows_per_page=rows_per_page)

        if resp.status_code != 200:
            print(f"  Error on page {page}: {resp.status_code}")
            break

        data = resp.json()
        records = data.get('records', [])
        all_records.extend(records)

        total_pages = data.get('total_pages', 1)
        print(f"  Got {len(records)} records (page {page}/{total_pages})")

        if page >= total_pages:
            break

        page += 1
        time.sleep(0.3)  # Be nice to the API

    return all_records

def extract_investor_info(records):
    """Extract and format investor information from records"""
    investors = []

    for record in records:
        investor = {}

        # Try to extract common field patterns
        for key, value in record.items():
            if key == 'id':
                investor['id'] = value
            elif isinstance(value, str) and value.strip():
                # Clean up the key name
                clean_key = key.replace('field_', '').replace('_', ' ').title()
                investor[clean_key] = value
            elif isinstance(value, dict):
                # Handle nested objects (like connections)
                if 'identifier' in value:
                    clean_key = key.replace('field_', '').replace('_', ' ').title()
                    investor[clean_key] = value['identifier']
                elif 'raw' in value:
                    clean_key = key.replace('field_', '').replace('_', ' ').title()
                    investor[clean_key] = value['raw']
            elif isinstance(value, list) and len(value) > 0:
                # Handle arrays
                clean_key = key.replace('field_', '').replace('_', ' ').title()
                if isinstance(value[0], dict) and 'identifier' in value[0]:
                    investor[clean_key] = ', '.join([v.get('identifier', '') for v in value])
                else:
                    investor[clean_key] = ', '.join([str(v) for v in value])

        if investor:
            investors.append(investor)

    return investors

def main():
    # First, try to discover what's available
    results = try_multiple_approaches()

    if not results:
        print("\n❌ Could not access any Knack endpoints")
        print("The API may require authentication or the keys may have changed.")
        return

    # Process the most promising result
    print("\n" + "=" * 70)
    print("FETCHING ALL INVESTOR DATA")
    print("=" * 70)

    # Try view endpoint first (filtered for Clean Energy + Pre-Revenue)
    print("\nFetching from view endpoint (filtered data)...")
    all_records = fetch_all_pages(scene_id=115, view_id=122, rows_per_page=100)

    if not all_records:
        print("View endpoint failed, trying object endpoint...")
        all_records = fetch_all_pages(object_id=11, rows_per_page=100)

    if not all_records:
        print("❌ Could not fetch records")
        return

    print(f"\n✓ Fetched {len(all_records)} total records")

    # Extract and format investor info
    investors = extract_investor_info(all_records)

    # Save raw JSON
    with open('/Users/adambricknail/Desktop/elec/iuk_investors_raw.json', 'w') as f:
        json.dump(all_records, f, indent=2)
    print(f"\n✓ Saved raw data to iuk_investors_raw.json")

    # Print sample record structure
    if all_records:
        print("\n" + "=" * 70)
        print("SAMPLE RECORD STRUCTURE")
        print("=" * 70)
        sample = all_records[0]
        for key, value in sample.items():
            print(f"  {key}: {type(value).__name__} = {str(value)[:100]}")

    # Print investor summary
    print("\n" + "=" * 70)
    print("INVESTORS FOUND")
    print("=" * 70)

    for i, inv in enumerate(investors, 1):
        print(f"\n--- Investor {i} ---")
        for key, value in inv.items():
            if value and str(value).strip():
                print(f"  {key}: {value}")

    # Save to CSV
    if investors:
        # Get all unique keys
        all_keys = set()
        for inv in investors:
            all_keys.update(inv.keys())

        with open('/Users/adambricknail/Desktop/elec/iuk_investors.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(investors)
        print(f"\n✓ Saved {len(investors)} investors to iuk_investors.csv")

if __name__ == "__main__":
    main()
