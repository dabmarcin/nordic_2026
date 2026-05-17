#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create portfolio CSV files for dates that have Nordic signals but no portfolio file
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

def create_missing_portfolio_files():
    """Create portfolio files for dates with Nordic signals but no portfolio CSV"""

    # Collect all Nordic signals by date
    signals_by_date = defaultdict(list)

    # ── Extract Allsvenskan BTTS Yes ─────────────────────────────────────────────
    allsv_files = sorted(glob.glob(os.path.join(ALLSV_SCORER_DIR, "allsvenskan_scorer_*.csv")))

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

        except Exception as e:
            print(f"❌ Error reading {basename}: {e}")

    # ── Extract Eliteserien Under 9.5 Corners ────────────────────────────────────
    elite_files = sorted(glob.glob(os.path.join(ELITE_SCORER_DIR, "eliteserien_scorer_*.csv")))

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

        except Exception as e:
            print(f"❌ Error reading {basename}: {e}")

    # Get existing portfolio dates
    portfolio_files = sorted(glob.glob(os.path.join(PORTFOLIO_DIR, "portfolio_*.csv")))
    existing_dates = set(
        os.path.basename(f).replace('portfolio_', '').replace('.csv', '')
        for f in portfolio_files
    )

    # Find missing dates
    missing_dates = set(signals_by_date.keys()) - existing_dates

    if not missing_dates:
        print("✅ All signal dates already have portfolio files")
        return

    print(f"📋 Creating portfolio files for {len(missing_dates)} missing dates:\n")

    created_count = 0
    for date_str in sorted(missing_dates):
        signals = signals_by_date[date_str]

        # Create DataFrame
        df = pd.DataFrame(signals)

        # Sort
        tier_map = {'A': 0, 'B': 1}
        df['Tier_sort'] = df['Tier'].map(tier_map).fillna(2)
        df = df.sort_values(
            by=['Tier_sort', 'Liga', 'Godzina'],
            na_position='last'
        ).drop(columns=['Tier_sort']).reset_index(drop=True)

        # Save
        portfolio_path = os.path.join(PORTFOLIO_DIR, f'portfolio_{date_str}.csv')
        df.to_csv(portfolio_path, index=False, encoding='utf-8-sig')
        created_count += 1

        allsv_count = len(df[df['Liga'] == 'Allsvenskan'])
        elite_count = len(df[df['Liga'] == 'Eliteserien'])

        print(f"✓ portfolio_{date_str}.csv")
        if allsv_count > 0:
            print(f"  - Allsvenskan BTTS Yes: {allsv_count}")
        if elite_count > 0:
            print(f"  - Eliteserien Under 9.5C: {elite_count}")

    print(f"\n✅ Created {created_count} portfolio files")

if __name__ == "__main__":
    create_missing_portfolio_files()
