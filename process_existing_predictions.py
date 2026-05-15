#!/usr/bin/env python3
"""
Skrypt do przetworzenia istniejących plików predykcji.
Łączy duplikaty: jeśli gpt_pred i liga mają ten sam Typ dla tego samego meczu,
tworzy nowy Model_type GPT+LIGA.
"""
import os
import glob
import sys
import io

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def merge_duplicate_predictions(all_rows):
    """Łączy predykcje gdy gpt_pred i liga mają ten sam Typ dla tego samego meczu."""
    if all_rows.empty:
        return all_rows

    # Grupuj po (ID, Mecz, Typ)
    grouped = {}
    for idx, row in all_rows.iterrows():
        key = (row['ID'], row['Mecz'], row['Typ'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append((idx, row))

    result = []
    for key, rows_with_same_type in grouped.items():
        model_types_set = {r[1].get('Model_type', '') for r in rows_with_same_type}

        # Jeśli mamy zarówno 'liga' i 'gpt_pred' z tym samym Typem
        if 'liga' in model_types_set and 'gpt_pred' in model_types_set:
            # Tworzymy nowy wiersz z Model_type 'GPT+LIGA'
            liga_row = next(r[1] for r in rows_with_same_type if r[1].get('Model_type') == 'liga')
            gpt_row = next(r[1] for r in rows_with_same_type if r[1].get('Model_type') == 'gpt_pred')

            merged = liga_row.copy()
            merged['Model_type'] = 'GPT+LIGA'
            merged['Model'] = 'Liga + GPT FootyStats'
            # Jeśli gpt ma pinnacle_odds, może być bardziej aktualne
            if pd.notna(gpt_row.get('Pinnacle_odds')) and gpt_row.get('Pinnacle_odds') != '':
                merged['Pinnacle_odds'] = gpt_row['Pinnacle_odds']
            result.append(merged)
        else:
            # Nie jest duplikatem, dodaj wszystkie jak są
            for _, r in rows_with_same_type:
                result.append(r)

    if result:
        return pd.DataFrame(result).reset_index(drop=True)
    return all_rows


def process_file(filepath):
    """Przetwórz jeden plik scorer."""
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        if df.empty:
            print(f'  SKIP (pusty): {os.path.basename(filepath)}')
            return False

        n_before = len(df)
        df_merged = merge_duplicate_predictions(df)
        n_after = len(df_merged)
        n_merged = n_before - n_after

        if n_merged > 0:
            df_merged.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f'  ✓ {os.path.basename(filepath):50} | '
                  f'{n_before:3} → {n_after:3} (merge: {n_merged})')
            return True
        else:
            print(f'  - {os.path.basename(filepath):50} | {n_before:3} (bez zmian)')
            return False

    except Exception as e:
        print(f'  ✗ {os.path.basename(filepath):50} | ERROR: {e}')
        return False


def main():
    scorer_dirs = [
        './data/telemetry/Allsvenskan scorer',
        './data/telemetry/Eliteserien scorer',
        './data/telemetry/Veikkausliiga scorer',
    ]

    total_processed = 0
    total_merged = 0

    for scorer_dir in scorer_dirs:
        if not os.path.exists(scorer_dir):
            print(f'Brak katalogu: {scorer_dir}')
            continue

        files = sorted(glob.glob(os.path.join(scorer_dir, '*_scorer_*.csv')))
        if not files:
            print(f'Brak plików w: {scorer_dir}')
            continue

        print(f'\n📂 {os.path.basename(scorer_dir)}:')
        for filepath in files:
            if process_file(filepath):
                total_merged += 1
            total_processed += 1

    print(f'\n══════════════════════════════════════════════════════')
    print(f'Przetworzono: {total_processed} plików')
    print(f'Zmienione:    {total_merged} plików')
    print(f'══════════════════════════════════════════════════════')


if __name__ == '__main__':
    main()
