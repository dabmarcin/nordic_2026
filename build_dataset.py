import sys
import io
import os
import glob
import json
import argparse
import datetime
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from nordic_config import (
    ALLSVENSKAN_DIR, ELITESERIEN_DIR, VEIKKAUSLIIGA_DIR,
    DATA_DIR, REPORTS_DIR, H2H_CACHE,
    CURRENT_DIR,
    TEAMS_ALLSV_HIST, TEAMS_ELITE_HIST, TEAMS_VEIKK_HIST,
    ALLSVENSKAN_HISTORICAL, ELITESERIEN_HISTORICAL, VEIKKAUSLIIGA_HISTORICAL,
    ALLSVENSKAN_2026_ID, ELITESERIEN_2026_ID, VEIKKAUSLIIGA_2026_ID,
)

# ── POTENCJAŁY — próg niezerowego wypełnienia ─────────────────────────────────
FILL_THRESHOLD = 0.50  # poniżej 50% niezerowych → cecha niedostępna

POTENTIAL_COLS = [
    'btts_potential', 'o25_potential', 'corners_potential',
    'avg_potential', 'corners_o95_potential',
    'team_a_xg_prematch', 'team_b_xg_prematch',
    'pre_match_home_ppg', 'pre_match_away_ppg',
]

# ── WCZYTANIE DANYCH ──────────────────────────────────────────────────────────

def load_historical():
    league_dirs = {
        'allsvenskan':   ALLSVENSKAN_DIR,
        'eliteserien':   ELITESERIEN_DIR,
        'veikkausliiga': VEIKKAUSLIIGA_DIR,
    }
    dfs = []
    counts = {}
    for league, directory in league_dirs.items():
        files = glob.glob(os.path.join(directory, '*.csv'))
        league_dfs = []
        for f in files:
            df = pd.read_csv(f, encoding='utf-8-sig')
            df = df[df['status'] == 'complete'].copy()
            sid = os.path.basename(f).replace('advanced_league_matches_', '').replace('.csv', '')
            df['league'] = league
            df['season_id'] = sid
            league_dfs.append(df)
        if league_dfs:
            merged = pd.concat(league_dfs, ignore_index=True)
            counts[league] = len(merged)
            dfs.append(merged)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.drop_duplicates(subset='id')
    df_all = df_all.sort_values('date_unix').reset_index(drop=True)
    return df_all, counts

# ── SPRAWDZENIE POTENCJAŁÓW ───────────────────────────────────────────────────

def check_potentials(df):
    availability = {}
    for col in POTENTIAL_COLS:
        if col not in df.columns:
            availability[col] = {'pct_filled': 0.0, 'used': False}
            continue
        nonzero_rate = (df[col] != 0).mean()
        availability[col] = {
            'pct_filled': round(nonzero_rate * 100, 1),
            'used': nonzero_rate >= FILL_THRESHOLD,
        }
    return availability

# ── H2H FEATURES ─────────────────────────────────────────────────────────────

def load_h2h_features(match_id, home_id, away_id, cutoff_unix, cache_dir):
    path = os.path.join(cache_dir, f"match_{int(match_id)}.json")

    NEUTRAL = {
        "h2h_btts_pct":      0.5,
        "h2h_over25_pct":    0.5,
        "h2h_home_win_pct":  0.45,
        "h2h_avg_goals":     2.5,
        "h2h_avg_corners":   9.5,
        "h2h_matches_count": 0,
    }

    if not os.path.isfile(path):
        return NEUTRAL

    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        data = raw.get("data", {})
        h2h  = data.get("h2h", {})

        if not h2h:
            return NEUTRAL

        betting  = h2h.get("betting_stats", {})
        results  = h2h.get("previous_matches_results", {})
        prev_ids = h2h.get("previous_matches_ids", [])

        total = results.get("totalMatches", 0)
        if total < 2:
            return NEUTRAL

        prev_filtered = [
            m for m in prev_ids
            if m.get("date_unix", 0) < cutoff_unix
        ]
        n = len(prev_filtered)
        if n < 2:
            return NEUTRAL

        home_id = int(home_id)
        goals_list, corners_list = [], []
        btts_list, over25_list, home_win_list = [], [], []

        for m in prev_filtered:
            hg = m.get("team_a_goals", 0)
            ag = m.get("team_b_goals", 0)
            total_g = hg + ag
            goals_list.append(total_g)
            btts_list.append(1 if hg > 0 and ag > 0 else 0)
            over25_list.append(1 if total_g > 2 else 0)
            team_a_id = int(h2h.get("team_a_id", 0))
            if team_a_id == home_id:
                home_win_list.append(1 if hg > ag else 0)
            else:
                home_win_list.append(1 if ag > hg else 0)

        def safe_mean(lst, default):
            return sum(lst) / len(lst) if lst else default

        return {
            "h2h_btts_pct":      safe_mean(btts_list, 0.5),
            "h2h_over25_pct":    safe_mean(over25_list, 0.5),
            "h2h_home_win_pct":  safe_mean(home_win_list, 0.45),
            "h2h_avg_goals":     safe_mean(goals_list, 2.5),
            "h2h_avg_corners":   9.5,
            "h2h_matches_count": n,
        }

    except Exception:
        return NEUTRAL


# ── LAST-5 BEZ DATA LEAKAGE ───────────────────────────────────────────────────

def compute_last5(df):
    # Indeks historii per drużyna: team_id → [(date_unix, row_idx, is_home)]
    team_history = {}
    for idx, row in df.iterrows():
        for team_id, is_home in [(row['homeID'], True), (row['awayID'], False)]:
            team_history.setdefault(team_id, []).append((row['date_unix'], idx, is_home))
    for tid in team_history:
        team_history[tid].sort(key=lambda x: x[0])

    def past_matches(team_id, cutoff):
        hist = team_history.get(team_id, [])
        return [entry for entry in hist if entry[0] < cutoff][-5:]

    def compute_stats(matches):
        if len(matches) < 2:
            return None
        btts_v, o25_v, corners_v = [], [], []
        gs_v, gc_v, win_v, cards_v = [], [], [], []
        for _, idx, is_home in matches:
            row = df.loc[idx]
            btts_v.append(1 if row['btts'] else 0)
            o25_v.append(1 if row['over25'] else 0)
            corners_v.append(row['totalCornerCount'])
            if is_home:
                gs_v.append(row['homeGoalCount'])
                gc_v.append(row['awayGoalCount'])
                win_v.append(1 if row['homeGoalCount'] > row['awayGoalCount'] else 0)
                cards_v.append(row['team_a_yellow_cards'])
            else:
                gs_v.append(row['awayGoalCount'])
                gc_v.append(row['homeGoalCount'])
                win_v.append(1 if row['awayGoalCount'] > row['homeGoalCount'] else 0)
                cards_v.append(row['team_b_yellow_cards'])
        return {
            'btts_rate':       np.mean(btts_v),
            'over25_rate':     np.mean(o25_v),
            'corners_avg':     np.mean(corners_v),
            'goals_scored':    np.mean(gs_v),
            'goals_conceded':  np.mean(gc_v),
            'win_rate':        np.mean(win_v),
            'cards_avg':       np.mean(cards_v),
        }

    cols = [
        'home_last5_btts_rate', 'home_last5_over25_rate', 'home_last5_corners_avg',
        'home_last5_goals_scored', 'home_last5_goals_conceded',
        'home_last5_win_rate', 'home_last5_cards_avg',
        'away_last5_btts_rate', 'away_last5_over25_rate', 'away_last5_corners_avg',
        'away_last5_goals_scored', 'away_last5_goals_conceded',
        'away_last5_win_rate', 'away_last5_cards_avg',
        'diff_last5_goals_scored', 'diff_last5_win_rate', 'diff_last5_corners_avg',
    ]
    results = {c: [] for c in cols}
    coverage = []

    stat_keys = ['btts_rate', 'over25_rate', 'corners_avg',
                 'goals_scored', 'goals_conceded', 'win_rate', 'cards_avg']
    home_prefix = ['home_last5_btts_rate', 'home_last5_over25_rate', 'home_last5_corners_avg',
                   'home_last5_goals_scored', 'home_last5_goals_conceded',
                   'home_last5_win_rate', 'home_last5_cards_avg']
    away_prefix = ['away_last5_btts_rate', 'away_last5_over25_rate', 'away_last5_corners_avg',
                   'away_last5_goals_scored', 'away_last5_goals_conceded',
                   'away_last5_win_rate', 'away_last5_cards_avg']

    for idx, row in df.iterrows():
        cutoff = row['date_unix']
        h_matches = past_matches(row['homeID'], cutoff)
        a_matches = past_matches(row['awayID'], cutoff)
        h_stats = compute_stats(h_matches)
        a_stats = compute_stats(a_matches)

        coverage.append(h_stats is not None and a_stats is not None)

        for out_col, key in zip(home_prefix, stat_keys):
            results[out_col].append(h_stats[key] if h_stats else np.nan)
        for out_col, key in zip(away_prefix, stat_keys):
            results[out_col].append(a_stats[key] if a_stats else np.nan)

        if h_stats and a_stats:
            results['diff_last5_goals_scored'].append(h_stats['goals_scored'] - a_stats['goals_scored'])
            results['diff_last5_win_rate'].append(h_stats['win_rate'] - a_stats['win_rate'])
            results['diff_last5_corners_avg'].append(h_stats['corners_avg'] - a_stats['corners_avg'])
        else:
            results['diff_last5_goals_scored'].append(np.nan)
            results['diff_last5_win_rate'].append(np.nan)
            results['diff_last5_corners_avg'].append(np.nan)

    return results, coverage

# ── TEAMS LOOKUP ──────────────────────────────────────────────────────────────

def load_teams_lookup() -> dict:
    lookup = {}

    for liga, hist_dir, season_list in [
        ("allsvenskan",   TEAMS_ALLSV_HIST, ALLSVENSKAN_HISTORICAL),
        ("eliteserien",   TEAMS_ELITE_HIST, ELITESERIEN_HISTORICAL),
        ("veikkausliiga", TEAMS_VEIKK_HIST, VEIKKAUSLIIGA_HISTORICAL),
    ]:
        for season in season_list:
            sid = season["id"]
            path = os.path.join(hist_dir, f"advanced_league_teams_{sid}.csv")
            if not os.path.isfile(path):
                continue
            df = pd.read_csv(path, encoding='utf-8-sig')
            for _, row in df.iterrows():
                try:
                    tid = int(row.get("team_id", 0) or 0)
                except (TypeError, ValueError):
                    continue
                if tid:
                    lookup[(tid, int(sid))] = row.to_dict()

    for liga, filename, sid in [
        ("allsvenskan",   "allsvenskan_teams_2026.csv",   ALLSVENSKAN_2026_ID),
        ("eliteserien",   "eliteserien_teams_2026.csv",   ELITESERIEN_2026_ID),
        ("veikkausliiga", "veikkausliiga_teams_2026.csv", VEIKKAUSLIIGA_2026_ID),
    ]:
        path = os.path.join(CURRENT_DIR, filename)
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path, encoding='utf-8-sig')
        for _, row in df.iterrows():
            try:
                tid = int(row.get("team_id", 0) or 0)
            except (TypeError, ValueError):
                continue
            if tid:
                lookup[(tid, int(sid))] = row.to_dict()

    return lookup


def get_team_stats(team_id, season_id, role: str, lookup: dict) -> dict:
    prefix = f"{role}_team_"
    stats = lookup.get((int(team_id), int(season_id)), {})

    if not stats:
        return {
            f"{prefix}ppg_{role}":                   np.nan,
            f"{prefix}btts_pct_{role}":              np.nan,
            f"{prefix}over25_pct_{role}":            np.nan,
            f"{prefix}corners_avg_{role}":           np.nan,
            f"{prefix}corners_against_avg_{role}":   np.nan,
            f"{prefix}scored_avg_{role}":            np.nan,
            f"{prefix}conceded_avg_{role}":          np.nan,
            f"{prefix}xg_for_{role}":                np.nan,
            f"{prefix}xg_against_{role}":            np.nan,
            f"{prefix}win_pct_{role}":               np.nan,
            f"{prefix}matches_played":               0,
            f"{prefix}over95c_pct_{role}":           np.nan,
        }

    suffix = f"_{role}"

    def get_stat(key_base):
        val = stats.get(f"{key_base}{suffix}")
        if val is None:
            val = stats.get(f"{key_base}_overall")
        try:
            return float(val) if val is not None else np.nan
        except (TypeError, ValueError):
            return np.nan

    try:
        mp = int(stats.get(f"seasonMatchesPlayed{suffix}") or 0)
    except (TypeError, ValueError):
        mp = 0

    return {
        f"{prefix}ppg_{role}":                 get_stat("seasonPPG"),
        f"{prefix}btts_pct_{role}":            get_stat("seasonBTTSPercentage"),
        f"{prefix}over25_pct_{role}":          get_stat("seasonOver25Percentage"),
        f"{prefix}corners_avg_{role}":         get_stat("cornersAVG"),
        f"{prefix}corners_against_avg_{role}": get_stat("cornersAgainstAVG"),
        f"{prefix}scored_avg_{role}":          get_stat("seasonScoredAVG"),
        f"{prefix}conceded_avg_{role}":        get_stat("seasonConcededAVG"),
        f"{prefix}xg_for_{role}":              get_stat("xg_for_avg"),
        f"{prefix}xg_against_{role}":          get_stat("xg_against_avg"),
        f"{prefix}win_pct_{role}":             get_stat("winPercentage"),
        f"{prefix}matches_played":             mp,
        f"{prefix}over95c_pct_{role}":         get_stat("over95CornersPercentage"),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print("=== BUILD DATASET — Nordic 2026 ===\n")

    # ── KROK 1: wczytaj dane historyczne
    df_all, counts = load_historical()
    print("Wczytano historyczne:")
    for league in ['allsvenskan', 'eliteserien', 'veikkausliiga']:
        print(f"  {league.capitalize():15} {counts.get(league, 0):4} meczów")
    print(f"  {'Łącznie:':15} {len(df_all):4} meczów complete")

    # ── KROK 1b: dołącz current matches 2026
    current_counts = {}
    for liga, filename, sid in [
        ("allsvenskan",   "allsvenskan_matches_2026.csv",   ALLSVENSKAN_2026_ID),
        ("eliteserien",   "eliteserien_matches_2026.csv",   ELITESERIEN_2026_ID),
        ("veikkausliiga", "veikkausliiga_matches_2026.csv", VEIKKAUSLIIGA_2026_ID),
    ]:
        path = os.path.join(CURRENT_DIR, filename)
        if not os.path.isfile(path):
            continue
        df_cur = pd.read_csv(path, encoding='utf-8-sig')
        if df_cur.empty:
            continue
        df_cur = df_cur[df_cur['status'] == 'complete'].copy()
        if df_cur.empty:
            continue
        df_cur['league'] = liga
        df_cur['season_id'] = sid
        current_counts[liga] = len(df_cur)
        df_all = pd.concat([df_all, df_cur], ignore_index=True)
        print(f"  {liga.capitalize():15} 2026 (current): {len(df_cur)} meczów complete")

    df_all = df_all.drop_duplicates(subset='id')
    df_all = df_all.sort_values('date_unix').reset_index(drop=True)
    print(f"  {'Po dedup:':15} {len(df_all):4} meczów łącznie\n")

    # ── KROK 1c: teams lookup
    teams_lookup = load_teams_lookup()
    print(f"Teams w lookup: {len(teams_lookup)} wpisów\n")

    # ── KROK 2: konwersje
    df_all['btts']      = df_all['btts'].astype(int)
    df_all['over25']    = df_all['over25'].astype(int)
    df_all['date_unix'] = pd.to_numeric(df_all['date_unix'], errors='coerce')

    availability = check_potentials(df_all)
    unavailable = [col for col, info in availability.items() if not info['used']]
    available   = [col for col, info in availability.items() if info['used']]

    if unavailable:
        print("Cechy niedostępne (< 50% wypełnienia):")
        for col in unavailable:
            print(f"  {col}: {availability[col]['pct_filled']:.1f}%  → pomijam")
        print()
    else:
        print("Wszystkie potencjały dostępne (>= 50% niezerowych).")
        print()

    # ── KROK 3: feature engineering

    # Implied probability
    df_all['implied_home'] = 1.0 / df_all['odds_ft_1']
    df_all['implied_away'] = 1.0 / df_all['odds_ft_2']
    df_all['implied_draw'] = 1.0 / df_all['odds_ft_x']
    margin = df_all['implied_home'] + df_all['implied_away'] + df_all['implied_draw']
    df_all['implied_home_norm'] = df_all['implied_home'] / margin
    df_all['implied_away_norm'] = df_all['implied_away'] / margin
    df_all['odds_diff_home_away'] = df_all['odds_ft_2'] - df_all['odds_ft_1']
    df_all.drop(columns=['implied_home', 'implied_away', 'implied_draw'], inplace=True)

    # Liga one-hot
    df_all['is_allsvenskan']   = (df_all['league'] == 'allsvenskan').astype(int)
    df_all['is_eliteserien']   = (df_all['league'] == 'eliteserien').astype(int)
    df_all['is_veikkausliiga'] = (df_all['league'] == 'veikkausliiga').astype(int)

    # Pre-match potencjały (przepisane wprost z CSV — nie obliczane)
    # Kolumny: btts_potential, o25_potential, corners_potential, avg_potential,
    #          corners_o95_potential, team_a_xg_prematch, team_b_xg_prematch,
    #          pre_match_home_ppg, pre_match_away_ppg
    # Są już w df_all po wczytaniu CSV — nie wymagają dalszego przetwarzania.

    # Last-5 rolling bez leakage
    last5_results, coverage = compute_last5(df_all)
    for col, vals in last5_results.items():
        df_all[col] = vals

    # H2H features z cache
    h2h_cols = ['h2h_btts_pct', 'h2h_over25_pct', 'h2h_home_win_pct',
                'h2h_avg_goals', 'h2h_avg_corners', 'h2h_matches_count']
    h2h_rows = []
    h2h_with_cache = 0
    for _, row in df_all.iterrows():
        h2h = load_h2h_features(
            match_id    = row['id'],
            home_id     = row['homeID'],
            away_id     = row['awayID'],
            cutoff_unix = row['date_unix'],
            cache_dir   = H2H_CACHE,
        )
        h2h_rows.append(h2h)
        if h2h['h2h_matches_count'] >= 2:
            h2h_with_cache += 1
    h2h_df = pd.DataFrame(h2h_rows, index=df_all.index)
    for col in h2h_cols:
        df_all[col] = h2h_df[col]
    h2h_neutral = len(df_all) - h2h_with_cache

    # ── KROK 3b: team stats features
    team_stats_rows = []
    n_with_team_stats = 0
    for _, row in df_all.iterrows():
        sid = row.get('season_id', 0)
        try:
            sid = int(float(str(sid)))
        except (TypeError, ValueError):
            sid = 0
        h_stats = get_team_stats(row['homeID'], sid, 'home', teams_lookup)
        a_stats = get_team_stats(row['awayID'], sid, 'away', teams_lookup)
        has_data = h_stats.get('home_team_matches_played', 0) > 0 or \
                   a_stats.get('away_team_matches_played', 0) > 0
        if has_data:
            n_with_team_stats += 1
        feat = {}
        feat.update(h_stats)
        feat.update(a_stats)
        h_ppg = h_stats.get('home_team_ppg_home', np.nan)
        a_ppg = a_stats.get('away_team_ppg_away', np.nan)
        h_sc  = h_stats.get('home_team_scored_avg_home', np.nan)
        a_sc  = a_stats.get('away_team_scored_avg_away', np.nan)
        h_co  = h_stats.get('home_team_corners_avg_home', np.nan)
        a_co  = a_stats.get('away_team_corners_avg_away', np.nan)
        feat['diff_team_ppg']     = h_ppg - a_ppg if pd.notna(h_ppg) and pd.notna(a_ppg) else np.nan
        feat['diff_team_scored']  = h_sc  - a_sc  if pd.notna(h_sc)  and pd.notna(a_sc)  else np.nan
        feat['diff_team_corners'] = h_co  - a_co  if pd.notna(h_co)  and pd.notna(a_co)  else np.nan
        team_stats_rows.append(feat)

    team_stats_df = pd.DataFrame(team_stats_rows, index=df_all.index)
    for col in team_stats_df.columns:
        df_all[col] = team_stats_df[col]

    # ── KROK 4: targety
    df_all['target_btts']             = df_all['btts']
    df_all['target_over25']           = df_all['over25']
    df_all['target_corners_over95']   = (df_all['totalCornerCount'] > 9.5).astype(int)
    df_all['target_result_home']      = (df_all['homeGoalCount'] > df_all['awayGoalCount']).astype(int)
    df_all['target_result_away']      = (df_all['awayGoalCount'] > df_all['homeGoalCount']).astype(int)

    # ── KROK 5: imputacja per liga (mediany) — last5 + team_stats
    last5_cols = list(last5_results.keys())
    team_stat_cols = [c for c in team_stats_df.columns
                      if c in df_all.columns and df_all[c].isna().any()]
    imputation_values = {}
    for league in ['allsvenskan', 'eliteserien', 'veikkausliiga']:
        mask = df_all['league'] == league
        medians = {}
        for col in last5_cols + team_stat_cols:
            med = df_all.loc[mask, col].median()
            medians[col] = round(float(med), 4) if pd.notna(med) else 0.0
        imputation_values[league] = medians
        for col in last5_cols + team_stat_cols:
            fill_mask = mask & df_all[col].isna()
            df_all.loc[fill_mask, col] = medians[col]

    # ── OUTPUT
    targets = [
        'target_btts', 'target_over25', 'target_corners_over95',
        'target_result_home', 'target_result_away',
    ]
    print("Targety:")
    for t in targets:
        rate = df_all[t].mean() * 100
        print(f"  {t:25} {rate:.1f}% pozytywnych")

    n_with = sum(coverage)
    n_total = len(coverage)
    print(f"\nLast5 coverage:")
    print(f"  Mecze z >= 2 meczami historycznymi: {n_with} ({n_with/n_total*100:.1f}%)")

    print(f"\nH2H coverage:")
    print(f"  Mecze z cache (n>=2):   {h2h_with_cache} ({h2h_with_cache/n_total*100:.1f}%)")
    print(f"  Mecze neutral fallback: {h2h_neutral} ({h2h_neutral/n_total*100:.1f}%)")

    n_without_team = n_total - n_with_team_stats
    print(f"\nTeams coverage:")
    print(f"  Mecze z team stats:   {n_with_team_stats} ({n_with_team_stats/n_total*100:.1f}%)")
    print(f"  Mecze bez team stats: {n_without_team} ({n_without_team/n_total*100:.1f}%)")

    if current_counts:
        print(f"\nCurrent season complete:")
        for lg in ['allsvenskan', 'eliteserien', 'veikkausliiga']:
            if lg in current_counts:
                print(f"  {lg.capitalize():15} {current_counts[lg]} meczów")

    out_path = os.path.join(DATA_DIR, 'training_dataset.csv')
    df_all.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\nZapisano: data/training_dataset.csv")
    print(f"  ({len(df_all)} wierszy, {len(df_all.columns)} kolumn)")

    # ── DEBUG JSON
    if args.debug:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        debug = {
            'timestamp': ts,
            'n_matches': len(df_all),
            'n_by_league': {lg: counts.get(lg, 0) for lg in ['allsvenskan', 'eliteserien', 'veikkausliiga']},
            'feature_availability': availability,
            'target_stats': {
                t: {
                    'n_positive': int(df_all[t].sum()),
                    'rate': round(float(df_all[t].mean()), 4),
                }
                for t in targets
            },
            'last5_coverage': {
                'n_with_data':    n_with,
                'n_without_data': n_total - n_with,
                'coverage_pct':   round(n_with / n_total * 100, 1),
            },
            'h2h_coverage': {
                'n_with_cache':       h2h_with_cache,
                'n_neutral_fallback': h2h_neutral,
                'coverage_pct':       round(h2h_with_cache / n_total * 100, 1),
            },
            'imputation_values': imputation_values,
            'warnings': [],
        }
        debug_path = os.path.join(REPORTS_DIR, f'debug_dataset_{ts}.json')
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(debug, f, ensure_ascii=False, indent=2,
                      cls=NumpyEncoder)
        print(f"  Debug: {debug_path}")


if __name__ == '__main__':
    main()
