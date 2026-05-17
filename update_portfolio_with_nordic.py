#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update Portfolio - Dodaje sygnały Allsvenskan i Eliteserien do istniejących portfolio CSV
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

from nordic_config import (
    PORTFOLIO_DIR, ALLSV_SCORER_DIR, ELITE_SCORER_DIR,
)

STAKE = 100.0

def add_nordic_signals_to_portfolio(portfolio_path):
    """Dodaj sygnały allsvenskan i eliteserien do portfolio"""

    # Wyłącz datę z pliku
    basename = os.path.basename(portfolio_path)
    if not basename.startswith('portfolio_'):
        return 0

    date_str = basename.replace('portfolio_', '').replace('.csv', '')

    # Wczytaj portfolio
    try:
        portfolio_df = pd.read_csv(portfolio_path, encoding='utf-8-sig')
    except Exception as e:
        print(f"❌ Błąd wczytania {basename}: {e}")
        return 0

    added_signals = []

    # ── Allsvenskan BTTS Yes ──────────────────────────────────────────────────
    allsv_path = os.path.join(ALLSV_SCORER_DIR, f'allsvenskan_scorer_{date_str}.csv')
    if os.path.exists(allsv_path):
        try:
            allsv_df = pd.read_csv(allsv_path, encoding='utf-8-sig')
            btts_subset = allsv_df[
                (allsv_df['Model_type'] == 'gpt_pred') &
                (allsv_df['Typ'] == 'BTTS Yes')
            ].copy()

            for _, row in btts_subset.iterrows():
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
                added_signals.append(signal)

            print(f"  ✓ Allsvenskan BTTS Yes: {len(btts_subset)}")
        except Exception as e:
            print(f"  ⚠ Błąd czytania allsvenskan: {e}")

    # ── Eliteserien Under 9.5 Corners ────────────────────────────────────────
    elite_path = os.path.join(ELITE_SCORER_DIR, f'eliteserien_scorer_{date_str}.csv')
    if os.path.exists(elite_path):
        try:
            elite_df = pd.read_csv(elite_path, encoding='utf-8-sig')
            under_subset = elite_df[
                (elite_df['Model_type'] == 'liga') &
                (elite_df['Typ'] == 'Under 9.5 corners')
            ].copy()

            for _, row in under_subset.iterrows():
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
                added_signals.append(signal)

            print(f"  ✓ Eliteserien Under 9.5C: {len(under_subset)}")
        except Exception as e:
            print(f"  ⚠ Błąd czytania eliteserien: {e}")

    if not added_signals:
        return 0

    # Połącz z istniejącym portfolio
    new_signals_df = pd.DataFrame(added_signals)
    combined_df = pd.concat([portfolio_df, new_signals_df], ignore_index=True)

    # Usuń duplikaty
    combined_df = combined_df.drop_duplicates(subset=['ID', 'Signal_ID'], keep='first')

    # Sortuj
    tier_map = {'A': 0, 'B': 1}
    combined_df['Tier_sort'] = combined_df['Tier'].map(tier_map).fillna(2)
    combined_df = combined_df.sort_values(
        by=['Tier_sort', 'Liga', 'Godzina'],
        na_position='last'
    ).drop(columns=['Tier_sort']).reset_index(drop=True)

    # Zapisz
    combined_df.to_csv(portfolio_path, index=False, encoding='utf-8-sig')

    return len(added_signals)

def main():
    """Aktualizuj wszystkie pliki portfolio"""
    portfolio_files = sorted(glob.glob(os.path.join(PORTFOLIO_DIR, "portfolio_*.csv")))

    if not portfolio_files:
        print("❌ Brak plików portfolio")
        return

    print(f"🔧 Znaleziono {len(portfolio_files)} plik(i) portfolio\n")

    total_added = 0
    for file_path in portfolio_files:
        basename = os.path.basename(file_path)
        print(f"📄 {basename}")
        added = add_nordic_signals_to_portfolio(file_path)
        total_added += added

    print(f"\n✅ Dodano łącznie {total_added} sygnałów Nordic")

if __name__ == "__main__":
    main()
