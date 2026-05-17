import sys
import io
import os
import glob
import json
import argparse
import datetime
from datetime import timezone
from collections import defaultdict

import numpy as np
import pandas as pd
import pytz

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from nordic_config import (
    PORTFOLIO_SIGNALS, PORTFOLIO_FLAT_STAKE, PORTFOLIO_DIR,
    ALLSV_SCORER_DIR, ELITE_SCORER_DIR, MLS_SCORER_DIR, CSL_SCORER_DIR,
    DAILY_DIR, CURRENT_DIR,
    ALLSVENSKAN_2026_ID, ELITESERIEN_2026_ID, VEIKKAUSLIIGA_2026_ID,
    MLS_2026_ID, CSL_2026_ID,
    LEAGUE_BY_SEASON_ID,
)

# ── STAŁE ────────────────────────────────────────────────────────────────────

STAKE = PORTFOLIO_FLAT_STAKE

LEAGUE_ABBR_MAP = {
    'allsvenskan':   'ALLSV',
    'eliteserien':   'ELITE',
    'veikkausliiga': 'VEIKK',
    'mls':           'MLS',
    'csl':           'CSL',
}

COMPETITION_TO_LEAGUE = {
    ALLSVENSKAN_2026_ID:   'allsvenskan',
    ELITESERIEN_2026_ID:   'eliteserien',
    VEIKKAUSLIIGA_2026_ID: 'veikkausliiga',
    MLS_2026_ID:           'mls',
    CSL_2026_ID:           'csl',
}

# ── SIGNAL VALIDATION ─────────────────────────────────────────────────────────

def validate_odds(signal_id, kurs):
    """Sprawdź czy odds spełniają warunki sygnału"""
    if signal_id not in PORTFOLIO_SIGNALS:
        return False

    signal = PORTFOLIO_SIGNALS[signal_id]

    # Sygnały scorer-based nie mają ograniczeń
    if signal.get('model_src') == 'scorer':
        return True

    # Parse condition
    condition = signal.get('condition', '')
    if not condition:
        return True

    # Obsługuj "X <= odds <= Y" i "odds >= X"
    if ">=" in condition and "<=" in condition:
        parts = condition.split("and")
        min_str = parts[0].strip().split(">=")[1].strip()
        max_str = parts[1].strip().split("<=")[1].strip()
        min_val, max_val = float(min_str), float(max_str)
        return min_val <= kurs <= max_val
    elif ">=" in condition:
        min_str = condition.split(">=")[1].strip()
        min_val = float(min_str)
        return kurs >= min_val

    return True

# ── KICKOFF FORMATTING ────────────────────────────────────────────────────────────

def format_kickoff(date_unix, tz_str):
    """Wyciągnij godzinę meczu z date_unix w lokalnej strefie czasowej."""
    try:
        dt_utc = datetime.datetime.fromtimestamp(int(date_unix), tz=timezone.utc)
        tz = pytz.timezone(tz_str)
        dt_local = dt_utc.astimezone(tz)
        return dt_local.strftime('%H:%M')
    except Exception:
        return ""

# ── DAILY FILE ────────────────────────────────────────────────────────────────

def find_daily_file(day_arg):
    today = datetime.date.today()
    if day_arg == 'today':
        target = today
        prefix = 'today_matches'
    else:
        target = today + datetime.timedelta(days=1)
        prefix = 'tomorrow_matches'
    date_str = target.strftime('%Y-%m-%d')
    path = os.path.join(DAILY_DIR, f'{prefix}_{date_str}.csv')
    if not os.path.exists(path):
        candidates = sorted(glob.glob(os.path.join(DAILY_DIR, f'{prefix}_*.csv')), reverse=True)
        if candidates:
            return candidates[0], date_str
        return None, date_str
    return path, date_str

# ── RULE-BASED SIGNALS ────────────────────────────────────────────────────────

def generate_rule_signals(date_str):
    """Wygeneruj sygnały rule-based dla MLS i CSL."""
    signals = []

    # Wczytaj daily matches
    daily_path = os.path.join(DAILY_DIR, f'today_matches_{date_str}.csv')
    if not os.path.exists(daily_path):
        daily_path = os.path.join(DAILY_DIR, f'tomorrow_matches_{date_str}.csv')
    if not os.path.exists(daily_path):
        return signals

    try:
        daily_df = pd.read_csv(daily_path, encoding='utf-8-sig')
    except Exception as e:
        print(f'[WARN] Błąd wczytania daily matches: {e}')
        return signals

    # Filtruj tylko MLS i CSL
    mls_csl = daily_df[daily_df['competition_id'].isin([MLS_2026_ID, CSL_2026_ID])].copy()

    for _, row in mls_csl.iterrows():
        try:
            match_id = int(row['id'])
            comp_id = int(row['competition_id'])
            league = COMPETITION_TO_LEAGUE.get(comp_id, 'unknown')
            home_name = str(row.get('home_name', ''))
            away_name = str(row.get('away_name', ''))

            # Wyciągnij godzinę z date_unix — MLS→NY, CSL→Shanghai
            date_unix = row.get('date_unix', 0)
            tz_for_league = 'America/New_York' if league == 'mls' else 'Asia/Shanghai'
            kickoff = format_kickoff(date_unix, tz_for_league)

            mecz = f'{home_name} vs {away_name}'
            liga_abbr = LEAGUE_ABBR_MAP.get(league, league.upper()[:3])

            # Sprawdź każdy sygnał rule-based
            for signal_id, signal_cfg in PORTFOLIO_SIGNALS.items():
                if signal_cfg.get('model_src') != 'rule':
                    continue

                if signal_cfg.get('league') != league:
                    continue

                # Wyciągnij odds z odpowiedniej kolumny
                odds_col = signal_cfg.get('odds_col', '')
                kurs = float(row.get(odds_col) or 0)

                # Waliduj odds
                if not validate_odds(signal_id, kurs) or kurs <= 0:
                    continue

                # Dodaj sygnał
                signals.append({
                    'ID': match_id,
                    'Data': date_str,
                    'Godzina': kickoff,
                    'Mecz': mecz,
                    'Liga': liga_abbr,
                    'Signal_ID': signal_id,
                    'Signal_Label': signal_cfg.get('label', ''),
                    'Tier': signal_cfg.get('tier', 'B'),
                    'Source': 'rule',
                    'Typ': signal_cfg.get('typ', ''),
                    'Kurs': round(kurs, 2),
                    'Stake_PLN': STAKE,
                    'Wynik': '',
                    'Rezultat': '',
                    'Corners': '',
                    'Profit_PLN': '',
                })

        except Exception as e:
            print(f'[WARN] Błąd przetwarzania meczu {match_id}: {e}')
            continue

    return signals

# ── SCORER SIGNALS ────────────────────────────────────────────────────────────

def generate_scorer_signals(date_str):
    """Wygeneruj sygnały ze scorerów nordyckich."""
    signals = []

    # elite_under_corners
    elite_path = os.path.join(ELITE_SCORER_DIR, f'eliteserien_scorer_{date_str}.csv')
    if os.path.exists(elite_path):
        try:
            elite_df = pd.read_csv(elite_path, encoding='utf-8-sig')
            elite_subset = elite_df[
                (elite_df['Model_type'] == 'liga') &
                (elite_df['Typ'] == 'Under 9.5 corners')
            ].copy()
            for _, row in elite_subset.iterrows():
                try:
                    wynik_val = row.get('Wynik', '')
                    if pd.isna(wynik_val):
                        wynik_val = ''
                    else:
                        wynik_val = str(wynik_val).strip()

                    signals.append({
                        'ID': int(row['ID']),
                        'Data': date_str,
                        'Godzina': str(row.get('Godzina', '')),
                        'Mecz': str(row.get('Mecz', '')),
                        'Liga': 'ELITE',
                        'Signal_ID': 'elite_under_corners',
                        'Signal_Label': 'Eliteserien ML Under 9.5C',
                        'Tier': 'B',
                        'Source': 'scorer',
                        'Typ': 'Under 9.5 corners',
                        'Kurs': float(row.get('Kurs', 0)),
                        'Stake_PLN': STAKE,
                        'Wynik': wynik_val,
                        'Rezultat': str(row.get('Rezultat', '')),
                        'Corners': row.get('Corners', ''),
                        'Profit_PLN': row.get('Profit_PLN', ''),
                    })
                except Exception:
                    continue
        except Exception as e:
            print(f'[WARN] Błąd wczytania elite_under_corners: {e}')

    # allsv_btts_yes
    allsv_path = os.path.join(ALLSV_SCORER_DIR, f'allsvenskan_scorer_{date_str}.csv')
    if os.path.exists(allsv_path):
        try:
            allsv_df = pd.read_csv(allsv_path, encoding='utf-8-sig')
            allsv_subset = allsv_df[
                (allsv_df['Model_type'] == 'gpt_pred') &
                (allsv_df['Typ'] == 'BTTS Yes')
            ].copy()
            for _, row in allsv_subset.iterrows():
                try:
                    wynik_val = row.get('Wynik', '')
                    if pd.isna(wynik_val):
                        wynik_val = ''
                    else:
                        wynik_val = str(wynik_val).strip()

                    signals.append({
                        'ID': int(row['ID']),
                        'Data': date_str,
                        'Godzina': str(row.get('Godzina', '')),
                        'Mecz': str(row.get('Mecz', '')),
                        'Liga': 'ALLSV',
                        'Signal_ID': 'allsv_btts_yes',
                        'Signal_Label': 'Allsvenskan GPT BTTS Yes',
                        'Tier': 'B',
                        'Source': 'scorer',
                        'Typ': 'BTTS Yes',
                        'Kurs': float(row.get('Kurs', 0)),
                        'Stake_PLN': STAKE,
                        'Wynik': wynik_val,
                        'Rezultat': str(row.get('Rezultat', '')),
                        'Corners': row.get('Corners', ''),
                        'Profit_PLN': row.get('Profit_PLN', ''),
                    })
                except Exception:
                    continue
        except Exception as e:
            print(f'[WARN] Błąd wczytania allsv_btts_yes: {e}')

    return signals

# ── SAVE PORTFOLIO ────────────────────────────────────────────────────────────

def save_portfolio(all_signals, date_str):
    """Zapisz sygnały do CSV, łącząc z istniejącymi."""
    os.makedirs(PORTFOLIO_DIR, exist_ok=True)

    csv_cols = ['ID', 'Data', 'Godzina', 'Mecz', 'Liga', 'Signal_ID', 'Signal_Label',
                'Tier', 'Source', 'Typ', 'Kurs', 'Stake_PLN', 'Wynik', 'Rezultat',
                'Corners', 'Profit_PLN']

    out_path = os.path.join(PORTFOLIO_DIR, f'portfolio_{date_str}.csv')

    # Wczytaj istniejący plik jeśli istnieje
    existing_rows = []
    if os.path.exists(out_path):
        try:
            existing_df = pd.read_csv(out_path, encoding='utf-8-sig')
            # Zachowaj tylko rozliczone (Wynik != '')
            settled = existing_df[existing_df['Wynik'] != ''].copy()
            existing_rows = settled.to_dict('records')
        except Exception as e:
            print(f'[WARN] Błąd wczytania istniejącego portfolio: {e}')

    # Połącz nowe sygnały z istniejącymi rozliczonymi
    combined = all_signals + existing_rows

    # Konwertuj na DataFrame i usuń duplikaty (ten sam ID + Signal_ID)
    out_df = pd.DataFrame(combined, columns=csv_cols)
    out_df = out_df.drop_duplicates(subset=['ID', 'Signal_ID'], keep='first')

    # Sortuj: Tier (A przed B), Liga, Godzina
    out_df['Tier_sort'] = out_df['Tier'].apply(lambda x: str(x) if pd.notna(x) else 'Z')
    out_df = out_df.sort_values(
        by=['Tier_sort', 'Liga', 'Godzina'],
        na_position='last'
    ).drop(columns=['Tier_sort']).reset_index(drop=True)

    # Zapisz
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')

    return len(out_df), out_path

# ── MAIN (daily) ──────────────────────────────────────────────────────────────

def main_daily(day_arg):
    """Główna funkcja dla trybu daily."""
    daily_path, date_str = find_daily_file(day_arg)
    if not daily_path:
        print(f'Brak pliku dla {day_arg} ({date_str}).')
        sys.exit(1)

    day_label = day_arg
    try:
        dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        day_name_pl = ['Poniedziałek','Wtorek','Środa','Czwartek','Piątek','Sobota','Niedziela'][dt.weekday()]
    except Exception:
        day_name_pl = day_label

    print('══════════════════════════════════════════════════════')
    print(f'PORTFOLIO SCORER — {date_str} ({day_name_pl})')
    print('══════════════════════════════════════════════════════')

    # Wygeneruj sygnały
    rule_signals = generate_rule_signals(date_str)
    scorer_signals = generate_scorer_signals(date_str)

    print('Sygnały rule-based:')
    mls_count = len([s for s in rule_signals if s['Liga'] == 'MLS'])
    csl_count = len([s for s in rule_signals if s['Liga'] == 'CSL'])
    print(f'  MLS mecze sprawdzone: {mls_count}')
    print(f'  CSL mecze sprawdzone: {csl_count}')

    print('Sygnały ze scorerów:')
    elite_count = len([s for s in scorer_signals if s['Liga'] == 'ELITE'])
    allsv_count = len([s for s in scorer_signals if s['Liga'] == 'ALLSV'])
    print(f'  Eliteserien scorer: {elite_count} sygnałów' if elite_count else '  Eliteserien scorer: brak')
    print(f'  Allsvenskan scorer: {allsv_count} sygnałów' if allsv_count else '  Allsvenskan scorer: brak')

    all_signals = rule_signals + scorer_signals

    print()
    if not all_signals:
        print('Brak sygnałów.')
    else:
        hdr = (f"{'#':>3} │ {'Liga':5} │ {'Tier':4} │ "
               f"{'Signal':27} │ {'Mecz':23} │ {'Kurs':5}")
        sep = '─' * len(hdr)
        print('══════════════════════════════════════════════════════')
        print(hdr)
        print(sep)
        for i, sig in enumerate(all_signals, 1):
            sig_label = sig['Signal_Label'][:25]
            mecz_short = sig['Mecz'][:23]
            print(f"{i:>3} │ {sig['Liga']:5} │ {sig['Tier']:4} │ "
                  f"{sig_label:27} │ {mecz_short:23} │ {sig['Kurs']:5.2f}")

    # Zapisz
    n_saved, out_path = save_portfolio(all_signals, date_str)

    print('══════════════════════════════════════════════════════')
    tier_a = len([s for s in all_signals if s['Tier'] == 'A'])
    tier_b = len([s for s in all_signals if s['Tier'] == 'B'])
    print(f'Łącznie: {len(all_signals)} sygnałów | Tier A: {tier_a} | Tier B: {tier_b}')
    print(f'Zapisano: {os.path.relpath(out_path)}')
    print('══════════════════════════════════════════════════════')

# ── HELPER FUNCTIONS FOR BACKFILL ────────────────────────────────────────────

SIGNAL_ODDS_COL = {
    "csl_draw":          "odds_ft_x",
    "mls_away_win_hi":   "odds_ft_2",
    "csl_under_corners": "odds_corners_under_95",
    "mls_over_corners":  "odds_corners_over_95",
}

def safe_float(val, default=0.0):
    """Bezpieczna konwersja na float."""
    try:
        return float(val) if pd.notna(val) else default
    except (TypeError, ValueError):
        return default

def safe_int(val, default=0):
    """Bezpieczna konwersja na int."""
    try:
        return int(float(val)) if pd.notna(val) else default
    except (TypeError, ValueError):
        return default

def check_signal(signal_id, row) -> float | None:
    """
    Sprawdza czy mecz spełnia warunki sygnału.
    Zwraca kurs jeśli tak, None jeśli nie.
    """
    if signal_id == "csl_draw":
        k = safe_float(row.get("odds_ft_x"))
        return k if 3.80 <= k <= 4.50 else None

    if signal_id == "mls_away_win_hi":
        k = safe_float(row.get("odds_ft_2"))
        return k if 3.80 <= k <= 5.00 else None

    if signal_id == "csl_under_corners":
        k = safe_float(row.get("odds_corners_under_95"))
        return k if k >= 2.20 else None

    if signal_id == "mls_over_corners":
        k = safe_float(row.get("odds_corners_over_95"))
        return k if k >= 2.00 else None

    return None

def settle_from_match(signal_id, row) -> tuple:
    """
    Oblicza (Wynik, Profit_PLN, Rezultat, Corners)
    z danych kompletnego meczu.
    Zwraca tuple: (wynik:int|None, profit:float|None, rezultat:str, corners:str|int)
    """
    hg = safe_int(row.get("homeGoalCount"))
    ag = safe_int(row.get("awayGoalCount"))
    tc = safe_int(row.get("totalCornerCount"), -1)
    rezultat = f"{hg}:{ag}"
    corners = str(tc) if tc >= 0 else ""

    wynik = None
    if signal_id == "csl_draw":
        wynik = 1 if hg == ag else 0
    elif signal_id == "mls_away_win_hi":
        wynik = 1 if ag > hg else 0
    elif signal_id == "csl_under_corners":
        if tc < 0:
            return None, None, rezultat, corners
        wynik = 1 if tc <= 9 else 0
    elif signal_id == "mls_over_corners":
        if tc < 0:
            return None, None, rezultat, corners
        wynik = 1 if tc > 9 else 0
    else:
        return None, None, rezultat, corners

    # Pobierz kurs z właściwej kolumny
    odds_col = SIGNAL_ODDS_COL.get(signal_id, "odds_ft_x")
    kurs = safe_float(row.get(odds_col), 0.0)

    if kurs <= 0 or wynik is None:
        return None, None, rezultat, corners

    profit = round((kurs - 1) * STAKE, 2) if wynik == 1 else -STAKE

    return wynik, profit, rezultat, corners

# ── BACKFILL ──────────────────────────────────────────────────────────────────

def main_backfill():
    """Backfill dla MLS i CSL z complete matches."""
    print('══════════════════════════════════════════════════════')
    print('BACKFILL — PORTFOLIO')
    print('══════════════════════════════════════════════════════')

    os.makedirs(PORTFOLIO_DIR, exist_ok=True)

    # MLS
    mls_path = os.path.join(CURRENT_DIR, 'mls_matches_2026.csv')
    mls_complete = pd.DataFrame()
    if os.path.exists(mls_path):
        try:
            mls_df = pd.read_csv(mls_path, encoding='utf-8-sig')
            mls_df = mls_df[mls_df['status'] == 'complete'].copy()
            mls_complete = mls_df
            print(f'MLS mecze przetworzone: {len(mls_df)} complete')
        except Exception as e:
            print(f'[WARN] Błąd MLS: {e}')
    else:
        print(f'[WARN] Brak {mls_path}')

    # CSL
    csl_path = os.path.join(CURRENT_DIR, 'csl_matches_2026.csv')
    csl_complete = pd.DataFrame()
    if os.path.exists(csl_path):
        try:
            csl_df = pd.read_csv(csl_path, encoding='utf-8-sig')
            csl_df = csl_df[csl_df['status'] == 'complete'].copy()
            csl_complete = csl_df
            print(f'CSL mecze przetworzone: {len(csl_df)} complete')
        except Exception as e:
            print(f'[WARN] Błąd CSL: {e}')
    else:
        print(f'[WARN] Brak {csl_path}')

    # Backfill rule signals from complete matches
    backfill_signals = []

    for league, df in [('mls', mls_complete), ('csl', csl_complete)]:
        if df.empty:
            continue
        for _, row in df.iterrows():
            try:
                match_id = int(row['id'])
                date_unix = float(row.get('date_unix', 0))

                # Oblicz date_str
                dt = datetime.datetime.fromtimestamp(date_unix, tz=datetime.timezone.utc)
                date_str = dt.strftime('%Y-%m-%d')
                kickoff = dt.strftime('%H:%M')

                home_name = str(row.get('home_name', ''))
                away_name = str(row.get('away_name', ''))
                mecz = f'{home_name} vs {away_name}'
                liga_abbr = 'MLS' if league == 'mls' else 'CSL'

                # Sprawdź każdy sygnał
                for signal_id, signal_cfg in PORTFOLIO_SIGNALS.items():
                    if signal_cfg.get('model_src') != 'rule':
                        continue
                    if signal_cfg.get('league') != league:
                        continue

                    # Sprawdź czy sygnał pasuje do kursów
                    kurs = check_signal(signal_id, row)
                    if kurs is None:
                        continue

                    # Oblicz Wynik i Profit z danych meczu
                    wynik, profit, rezultat, corners = settle_from_match(signal_id, row)

                    # Pomiń jeśli nie można obliczyć wyniku
                    if wynik is None:
                        continue

                    backfill_signals.append({
                        'ID': match_id,
                        'Data': date_str,
                        'Godzina': kickoff,
                        'Mecz': mecz,
                        'Liga': liga_abbr,
                        'Signal_ID': signal_id,
                        'Signal_Label': signal_cfg.get('label', ''),
                        'Tier': signal_cfg.get('tier', 'B'),
                        'Source': 'rule',
                        'Typ': signal_cfg.get('typ', ''),
                        'Kurs': round(kurs, 2),
                        'Stake_PLN': STAKE,
                        'Wynik': str(wynik),
                        'Rezultat': rezultat,
                        'Corners': str(corners) if corners else '',
                        'Profit_PLN': profit if profit is not None else '',
                    })

            except Exception as e:
                continue

    # Group by date and save per day
    by_date = defaultdict(list)
    for sig in backfill_signals:
        by_date[sig['Data']].append(sig)

    total_saved = 0
    for date_str, sigs in sorted(by_date.items()):
        # Deduplicate by ID + Signal_ID before saving
        sigs_df = pd.DataFrame(sigs)
        if not sigs_df.empty:
            sigs_df = sigs_df.drop_duplicates(
                subset=['ID', 'Signal_ID'],
                keep='first'
            )
            sigs_dedup = sigs_df.to_dict('records')
        else:
            sigs_dedup = []

        if sigs_dedup:
            n_saved, _ = save_portfolio(sigs_dedup, date_str)
            total_saved += n_saved

    # Statistics
    print('──────────────────────────────────────────────────────')
    print(f'Zakłady wygenerowane: {len(backfill_signals)}')
    tier_a = len([s for s in backfill_signals if s['Tier'] == 'A'])
    tier_b = len([s for s in backfill_signals if s['Tier'] == 'B'])
    print(f'  Tier A: {tier_a}')
    print(f'  Tier B: {tier_b}')

    settled = len([s for s in backfill_signals if s['Wynik'] != ''])
    print(f'  Rozliczone: {settled} ({100*settled/len(backfill_signals):.0f}%)' if backfill_signals else '  Rozliczone: 0')
    pending = len([s for s in backfill_signals if s['Wynik'] == ''])
    print(f'  Oczekujące: {pending}')

    # Per signal stats
    print('──────────────────────────────────────────────────────')
    print('Per sygnał:')
    for signal_id, signal_cfg in sorted(PORTFOLIO_SIGNALS.items()):
        if signal_cfg.get('model_src') != 'rule':
            continue
        sigs_for_id = [s for s in backfill_signals if s['Signal_ID'] == signal_id]
        if sigs_for_id:
            n = len(sigs_for_id)
            wins = len([s for s in sigs_for_id if s['Wynik'] == '1'])
            roi = 100 * sum(float(s.get('Profit_PLN', 0)) for s in sigs_for_id) / (n * STAKE) if n > 0 else 0
            print(f'  {signal_id:20} {n:2} zakładów, ROI {roi:+6.1f}%')

    print('══════════════════════════════════════════════════════')
    print(f'Zapisano: {len(by_date)} plików w data/portfolio/')
    print('══════════════════════════════════════════════════════')

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('day', nargs='?', choices=['today', 'tomorrow'], default='today')
    parser.add_argument('--backfill', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.backfill:
        main_backfill()
    else:
        main_daily(args.day)

if __name__ == '__main__':
    main()
