#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill missing Allsvenskan/Eliteserien signals to portfolio CSV files
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import glob
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from nordic_config import PORTFOLIO_DIR, ALLSV_SCORER_DIR, ELITE_SCORER_DIR

STAKE = 100.0

def backfill_nordic_signals():
    """Backfill missing Nordic signals from scorers to portfolio"""

    # Get all portfolio files
    portfolio_files = sorted(glob.glob(os.path.join(PORTFOLIO_DIR, "portfolio_*.csv")))

    if not portfolio_files:
        print("❌ No portfolio files found")
        return

    print(f"📄 Processing {len(portfolio_files)} portfolio files...\n")

    # For each portfolio file, check what's missing from scorers
    for portfolio_path in portfolio_files:
        basename = os.path.basename(portfolio_path)
        date_str = basename.replace('portfolio_', '').replace('.csv', '')

        # Load portfolio
        try:
            portfolio_df = pd.read_csv(portfolio_path, encoding='utf-8-sig')
        except Exception as e:
            print(f"❌ Error reading {basename}: {e}")
            continue

        added_count = 0

        # ── Allsvenskan BTTS Yes ──────────────────────────────────────────────────
        allsv_path = os.path.join(ALLSV_SCORER_DIR, f'allsvenskan_scorer_{date_str}.csv')
        if os.path.exists(allsv_path):
            try:
                allsv_df = pd.read_csv(allsv_path, encoding='utf-8-sig')

                # Get all BTTS Yes predictions from scorer
                btts_all = allsv_df[
                    (allsv_df['Model_type'] == 'gpt_pred') &
                    (allsv_df['Typ'] == 'BTTS Yes')
                ].copy()

                # Check which ones are already in portfolio
                portfolio_ids = set(portfolio_df[
                    (portfolio_df['Liga'] == 'Allsvenskan') &
                    (portfolio_df['Signal_ID'] == 'allsv_btts_yes')
                ]['ID'].unique())

                scorer_ids = set(btts_all['ID'].unique())
                missing_ids = scorer_ids - portfolio_ids

                if len(missing_ids) > 0:
                    print(f"📄 {basename}")
                    print(f"   Allsvenskan BTTS Yes: {len(missing_ids)} missing (have {len(portfolio_ids)}, scorer has {len(scorer_ids)})")

                    # Add missing signals to portfolio
                    for mid in missing_ids:
                        row = btts_all[btts_all['ID'] == mid].iloc[0]
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
                        portfolio_df = pd.concat([portfolio_df, pd.DataFrame([signal])], ignore_index=True)
                        added_count += 1

                    print(f"   ✓ Added {added_count} signals")

            except Exception as e:
                print(f"   ⚠ Error reading scorer: {e}")

        # ── Eliteserien Under 9.5 Corners ────────────────────────────────────────
        elite_path = os.path.join(ELITE_SCORER_DIR, f'eliteserien_scorer_{date_str}.csv')
        if os.path.exists(elite_path):
            try:
                elite_df = pd.read_csv(elite_path, encoding='utf-8-sig')

                # Get all Under 9.5 Corners from scorer
                under_all = elite_df[
                    (elite_df['Model_type'] == 'liga') &
                    (elite_df['Typ'] == 'Under 9.5 corners')
                ].copy()

                # Check which ones are already in portfolio
                portfolio_ids = set(portfolio_df[
                    (portfolio_df['Liga'] == 'Eliteserien') &
                    (portfolio_df['Signal_ID'] == 'elite_under_corners')
                ]['ID'].unique())

                scorer_ids = set(under_all['ID'].unique())
                missing_ids = scorer_ids - portfolio_ids

                if len(missing_ids) > 0:
                    if added_count == 0:
                        print(f"📄 {basename}")
                    print(f"   Eliteserien Under 9.5C: {len(missing_ids)} missing (have {len(portfolio_ids)}, scorer has {len(scorer_ids)})")

                    # Add missing signals to portfolio
                    for mid in missing_ids:
                        row = under_all[under_all['ID'] == mid].iloc[0]
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
                        portfolio_df = pd.concat([portfolio_df, pd.DataFrame([signal])], ignore_index=True)
                        added_count += 1

                    print(f"   ✓ Added {added_count} signals total")

            except Exception as e:
                print(f"   ⚠ Error reading scorer: {e}")

        # Save if anything was added
        if added_count > 0:
            # Remove duplicates
            portfolio_df = portfolio_df.drop_duplicates(subset=['ID', 'Signal_ID'], keep='first')

            # Sort
            tier_map = {'A': 0, 'B': 1}
            portfolio_df['Tier_sort'] = portfolio_df['Tier'].map(tier_map).fillna(2)
            portfolio_df = portfolio_df.sort_values(
                by=['Tier_sort', 'Liga', 'Godzina'],
                na_position='last'
            ).drop(columns=['Tier_sort']).reset_index(drop=True)

            portfolio_df.to_csv(portfolio_path, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    backfill_nordic_signals()
    print("\n✅ Backfill complete")
