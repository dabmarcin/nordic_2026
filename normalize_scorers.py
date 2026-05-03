import os
import glob
import pandas as pd
import re

def normalize_typ_in_csv(typ: str) -> str:
    """Normalizuje wartości kolumny Typ w CSV."""
    if not isinstance(typ, str) or not typ.strip():
        return typ

    t = typ.strip()
    t_lower = t.lower()

    # Usuń wiersze z Parlay (będą pominięte)
    if 'parlay' in t_lower:
        return 'REMOVE'

    # Reguła 1: Jeśli zawiera "Corners" -> "Over/Under X.X corners"
    if 'corners' in t_lower:
        m = re.match(r'(over|under)\s+([\d.]+)\s*corners?', t_lower, re.IGNORECASE)
        if m:
            direction = m.group(1).capitalize()
            value = m.group(2)
            return f'{direction} {value} corners'
        # Nawet bez liczby
        if 'over' in t_lower and 'corners' in t_lower:
            return 'Over 9.5 corners'
        if 'under' in t_lower and 'corners' in t_lower:
            return 'Under 9.5 corners'

    # Reguła 2: Jeśli samo "over"/"under" lub zawiera słowa golf/goli/bramek etc
    # -> "Over X.X" lub "Under X.X"
    m = re.match(
        r'(over|under)\s+([\d.]+)'
        r'(?:\s+(?:goli|goals|bramek|gola|bramki|mål|maalia|maalin))?',
        t_lower
    )
    if m:
        direction = m.group(1).capitalize()
        value = m.group(2)
        return f'{direction} {value}'

    return typ

def process_scorer_files():
    """Przetwórz wszystkie pliki scorer*.csv"""
    base_dir = r'c:\Projects\nordic_2026\data\telemetry'
    pattern = os.path.join(base_dir, '**', '*scorer*.csv')

    files = glob.glob(pattern, recursive=True)
    print(f'Znaleziono {len(files)} plików scorer')

    for fpath in sorted(files):
        print(f'\n[>] {os.path.basename(fpath)}')
        try:
            df = pd.read_csv(fpath, encoding='utf-8-sig')

            # Sprawdzenie czy istnieje kolumna Typ
            if 'Typ' not in df.columns:
                print(f'    [warn] Brak kolumny "Typ" — pominięto')
                continue

            initial_rows = len(df)

            # Normalizuj Typ
            df['Typ'] = df['Typ'].apply(normalize_typ_in_csv)

            # Usuń wiersze z REMOVE (Parlay)
            df = df[df['Typ'] != 'REMOVE'].copy()

            removed_rows = initial_rows - len(df)

            # Zapisz z powrotem
            df.to_csv(fpath, index=False, encoding='utf-8-sig')

            print(f'    [OK] Znormalizowano')
            print(f'    Wierszy: {initial_rows} → {len(df)} (usunięto {removed_rows} Parlay)')

        except Exception as e:
            print(f'    [ERR] Blad: {e}')

if __name__ == '__main__':
    print('=' * 60)
    print('NORMALIZACJA KOLUMNY "TYP" W PLIKACH SCORER')
    print('=' * 60)
    process_scorer_files()
    print('\n' + '=' * 60)
    print('GOTOWE')
    print('=' * 60)
