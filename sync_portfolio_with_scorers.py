#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sync portfolio CSV files with Nordic scorer data
Ensures all Allsvenskan BTTS Yes and Eliteserien Under 9.5C signals are in portfolio
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import glob
import pandas as pd
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from nordic_config import PORTFOLIO_DIR, ALLSV_SCORER_DIR, ELITE_SCORER_DIR

STAKE = 100.0

def sync_portfolio():
    """Sync Nordic signals from scorers to portfolio CSVs"""

    # Collect all Nordic signals by date
    signals_by_date = defaultdict(list)

    # ── Extract Allsvenskan BTTS Yes ─────────────────────────────────────────────
    allsv_files = sorted(glob.glob(os.path.join(ALLSV_SCORER_DIR, "allsvenskan_scorer_*.csv")))
    print(f"📊 Found {len(allsv_files)} Allsvenskan scorer files\n")

    for file_path in allsv_files:
        try:
            basename = os.path.basename(file_path)
            date_str = basename.replace('allsvenskan_scorer_', '').replace('.csv', '')

            df = pd.read_csv(file_path, encoding='utf-8-sig')
            btts_rows = df[
                (df['Model_type'] == 'gpt_pred') &
                (df['Typ'] == 'BTTS Yes')
            ].copy()

            for _, row in btts_rows.iterrows():
                signal = {
                    'ID': int(row['ID']),
                    'Data': date_str,
                    'Godzina': str(row.get('Godzina', '')),
                    'Mecz': str(row.get('Mecz', '')),
                    'Liga': 'Allsvenskan',
                    'Signal_ID': 'allsv_btts_yes',
                    'Signal_Label': 'Allsvenskan GPT BTTS Yes',
                    'Tier': 'B',
                    'Source': 'scorer',
                    'Typ': 'BTTS Yes',
                    'Kurs': float(row.get('Kurs', 0)),
                    'Stake_PLN': STAKE,
                    'Wynik': row.get('Wynik', ''),
                    'Rezultat': row.get('Rezultat', ''),
                    'Corners': row.get('Corners', ''),
                    'Profit_PLN': row.get('Profit_PLN', ''),
                }
                signals_by_date[date_str].append(signal)

            if len(btts_rows) > 0:
                print(f"  {basename}: {len(btts_rows)} BTTS Yes")

        except Exception as e:
            print(f"  ❌ Error reading {basename}: {e}")

    # ── Extract Eliteserien Under 9.5 Corners ────────────────────────────────────
    elite_files = sorted(glob.glob(os.path.join(ELITE_SCORER_DIR, "eliteserien_scorer_*.csv")))
    print(f"\n📊 Found {len(elite_files)} Eliteserien scorer files\n")

    for file_path in elite_files:
        try:
            basename = os.path.basename(file_path)
            date_str = basename.replace('eliteserien_scorer_', '').replace('.csv', '')

            df = pd.read_csv(file_path, encoding='utf-8-sig')
            under_rows = df[
                (df['Model_type'] == 'liga') &
                (df['Typ'] == 'Under 9.5 corners')
            ].copy()

            for _, row in under_rows.iterrows():
                signal = {
                    'ID': int(row['ID']),
                    'Data': date_str,
                    'Godzina': str(row.get('Godzina', '')),
                    'Mecz': str(row.get('Mecz', '')),
                    'Liga': 'Eliteserien',
                    'Signal_ID': 'elite_under_corners',
                    'Signal_Label': 'Eliteserien ML Under 9.5C',
                    'Tier': 'B',
                    'Source': 'scorer',
                    'Typ': 'Under 9.5 corners',
                    'Kurs': float(row.get('Kurs', 0)),
                    'Stake_PLN': STAKE,
                    'Wynik': row.get('Wynik', ''),
                    'Rezultat': row.get('Rezultat', ''),
                    'Corners': row.get('Corners', ''),
                    'Profit_PLN': row.get('Profit_PLN', ''),
                }
                signals_by_date[date_str].append(signal)

            if len(under_rows) > 0:
                print(f"  {basename}: {len(under_rows)} Under 9.5C")

        except Exception as e:
            print(f"  ❌ Error reading {basename}: {e}")

    print(f"\n📋 Collected {sum(len(v) for v in signals_by_date.values())} Nordic signals across {len(signals_by_date)} dates")
    print(f"\n🔄 Syncing with portfolio files...\n")

    # Process each portfolio file
    portfolio_files = sorted(glob.glob(os.path.join(PORTFOLIO_DIR, "portfolio_*.csv")))
    updated_count = 0

    for portfolio_path in portfolio_files:
        basename = os.path.basename(portfolio_path)
        date_str = basename.replace('portfolio_', '').replace('.csv', '')

        # Load existing portfolio
        try:
            portfolio_df = pd.read_csv(portfolio_path, encoding='utf-8-sig')
        except Exception as e:
            print(f"❌ Error reading {basename}: {e}")
            continue

        # Get Nordic signals for this date
        new_signals = signals_by_date.get(date_str, [])

        if not new_signals:
            continue

        # Check which are already in portfolio
        portfolio_nordic_ids = set(portfolio_df[
            portfolio_df['Signal_ID'].isin(['allsv_btts_yes', 'elite_under_corners'])
        ]['ID'].unique())

        signals_to_add = [s for s in new_signals if s['ID'] not in portfolio_nordic_ids]

        if not signals_to_add:
            continue

        print(f"📄 {basename}")
        print(f"   Adding {len(signals_to_add)} Nordic signals")

        # Add signals
        for signal in signals_to_add:
            portfolio_df = pd.concat([portfolio_df, pd.DataFrame([signal])], ignore_index=True)

        # Remove duplicates
        portfolio_df = portfolio_df.drop_duplicates(subset=['ID', 'Signal_ID'], keep='first')

        # Sort
        tier_map = {'A': 0, 'B': 1}
        portfolio_df['Tier_sort'] = portfolio_df['Tier'].map(tier_map).fillna(2)
        portfolio_df = portfolio_df.sort_values(
            by=['Tier_sort', 'Liga', 'Godzina'],
            na_position='last'
        ).drop(columns=['Tier_sort']).reset_index(drop=True)

        # Save
        portfolio_df.to_csv(portfolio_path, index=False, encoding='utf-8-sig')
        updated_count += 1

    print(f"\n✅ Updated {updated_count} portfolio files")

    # Check if any dates with signals don't have portfolio files
    portfolio_dates = set(
        os.path.basename(f).replace('portfolio_', '').replace('.csv', '')
        for f in portfolio_files
    )
    signal_dates = set(signals_by_date.keys())
    missing_portfolio_dates = signal_dates - portfolio_dates

    if missing_portfolio_dates:
        print(f"\n⚠️  {len(missing_portfolio_dates)} dates have signals but NO portfolio file:")
        for date in sorted(missing_portfolio_dates):
            print(f"   - {date}: {len(signals_by_date[date])} signals")
        print(f"\n   These will be skipped. To include them, portfolio_scorer.py needs to run for these dates.")

if __name__ == "__main__":
    sync_portfolio()
