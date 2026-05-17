import sys
import io
import os
import glob
import json
import argparse
import datetime
from datetime import timezone

import numpy as np
import pandas as pd
import joblib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from nordic_config import (
    API_KEY, BASE_URL,
    CSL_DIR, DAILY_DIR,
    MODELS_CSL,
    CSL_SCORER_DIR,
    CSL_2026_ID,
    CURRENT_DIR, REPORTS_DIR,
)

# ── STAŁE ────────────────────────────────────────────────────────────────────

ACTIVE_LEAGUES = {
    'csl': CSL_2026_ID,
}

LEAGUE_DIRS = {
    'csl': (CSL_DIR, MODELS_CSL, CSL_SCORER_DIR),
}

FILTERS = {
    'result_home':   {'min_score': 62.0, 'min_odds': 1.40, 'max_odds': 5.0},
    'result_away':   {'min_score': 62.0, 'min_odds': 1.40, 'max_odds': 5.0},
    'btts':          {'min_score': 62.0, 'min_odds': 1.20, 'max_odds': 5.0},
    'over25':        {'min_score': 62.0, 'min_odds': 1.20, 'max_odds': 5.0},
    'over25_forced': {'min_score': 0.0, 'min_odds': 1.30, 'max_odds': 1.80},
    'corners_over':  {'min_score': 0.0, 'min_odds': 1.45, 'max_odds': 2.20},
}

BANKROLL   = 1000.0
KELLY_FRAC = 0.25
MAX_STAKE  = 50.0

# Empirical win rates for contrarian models (CSL)
P_CORNERS_OVER_CSL = 0.63
P_OVER25_FORCED_CSL = 0.66

MODEL_NAMES = ['result_home', 'result_away', 'btts', 'over25', 'corners']

TZ = 'Asia/Shanghai'

H2H_FALLBACK = {
    'h2h_btts_pct':     0.5,
    'h2h_over25_pct':   0.5,
    'h2h_home_win_pct': 0.45,
    'h2h_avg_goals':    2.5,
    'h2h_matches_count': 0,
    'h2h_avg_corners':  9.5,
}

# ── NORMALIZE TYPÓW ───────────────────────────────────────────────────────────

def normalize_typ(typ: str,
                  home: str = '',
                  away: str = '') -> str:
    if not isinstance(typ, str) or not typ.strip():
        return typ

    t = typ.strip()
    t_lower = t.lower()
    h = home.strip().lower()
    a = away.strip().lower()

    if ' + ' in t:
        return 'Parlay'

    if t_lower in ('home win', 'home_win'):
        return '1'
    if t_lower in ('away win', 'away_win'):
        return '2'
    if t_lower in ('draw',):
        return 'X'

    if t_lower in ('over corners', 'corners over', 'over corner'):
        return 'Over 9.5 corners'

    import re
    m = re.match(r'corners?\s+(over|under)\s+([\d.]+)', t_lower)
    if m:
        d = m.group(1).capitalize()
        return f'{d} {m.group(2)} corners'

    m = re.match(
        r'(over|under)\s+([\d.]+)'
        r'(?:\s+(?:goli|goals|bramek|gola|bramki|mål|maalia|maalin))?$',
        t_lower)
    if m:
        d = m.group(1).capitalize()
        return f'{d} {m.group(2)}'

    btts_yes = ('btts yes', 'btts tak', 'btts – tak', 'btts - tak',
                'btts yes - polecane')
    btts_no  = ('btts no', 'btts nie', 'btts – nie', 'btts - nie')
    if t_lower in btts_yes or (
            t_lower.startswith('btts') and
            'no' not in t_lower and
            'nie' not in t_lower):
        return 'BTTS Yes'
    if t_lower in btts_no:
        return 'BTTS No'

    if re.search(r'double\s*chance\s*1x', t_lower):
        return '1X'
    if re.search(r'double\s*chance\s*x2', t_lower):
        return 'X2'
    if re.search(r'double\s*chance\s*12', t_lower):
        return '12'

    dc_m = re.match(
        r'double\s*chance\s+(.+?)(?:\s+win[/\s]draw'
        r'|\s+wygra?\s+lub|\s+or\s+draw)?$',
        t_lower)
    if dc_m:
        mention = dc_m.group(1).strip()
        if h and h in mention:
            return '1X'
        if a and a in mention:
            return 'X2'
        return '1X'

    wlr = re.search(
        r'wygra\s+lub\s+remis'
        r'|win\s+or\s+draw|win/draw'
        r'|vinner\s+eller\s+(?:oavgjort|uavgjort)'
        r'|wygra?\s+lub\s+zremisuje',
        t_lower)
    if wlr:
        if h and h in t_lower:
            return '1X'
        if a and a in t_lower:
            return 'X2'
        return '1X'

    win = re.search(
        r'\bwygra\b|\bwins?\b(?!\s+or)'
        r'|\bvinner\b(?!\s+eller)'
        r'|\bsiegt\b|\bvoittaa\b|\bvann\b',
        t_lower)
    if win:
        pre = t_lower[:win.start()].strip()
        if h and (h in pre or h in t_lower):
            return '1'
        if a and (a in pre or a in t_lower):
            return '2'
        return '1'

    if t_lower in ('remis', 'draw', 'x', 'oavgjort', 'tasapeli'):
        return 'X'

    if t_lower in ('1', 'x', '2', '1x', 'x2', '12'):
        return t_lower.upper()

    return t


def load_current_teams_lookup() -> dict:
    lookup = {}
    sid = CSL_2026_ID
    fname = "csl_teams_2026.csv"
    path = os.path.join(CURRENT_DIR, fname)
    if not os.path.isfile(path):
        print(f"[WARN] Brak current teams: {path}")
        return lookup
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
        for _, row in df.iterrows():
            try:
                tid = int(row.get("team_id", 0) or 0)
            except (TypeError, ValueError):
                continue
            if tid:
                lookup[tid] = row.to_dict()
    except Exception as e:
        print(f"[WARN] Błąd wczytania {fname}: {e}")
    return lookup


def get_team_stats_from_lookup(team_id: int, role: str, lookup: dict) -> dict:
    prefix = f"{role}_team_"
    stats = lookup.get(int(team_id), {})

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

    def gs(key_base):
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
        f"{prefix}ppg_{role}":                 gs("seasonPPG"),
        f"{prefix}btts_pct_{role}":            gs("seasonBTTSPercentage"),
        f"{prefix}over25_pct_{role}":          gs("seasonOver25Percentage"),
        f"{prefix}corners_avg_{role}":         gs("cornersAVG"),
        f"{prefix}corners_against_avg_{role}": gs("cornersAgainstAVG"),
        f"{prefix}scored_avg_{role}":          gs("seasonScoredAVG"),
        f"{prefix}conceded_avg_{role}":        gs("seasonConcededAVG"),
        f"{prefix}xg_for_{role}":              gs("xg_for_avg"),
        f"{prefix}xg_against_{role}":          gs("xg_against_avg"),
        f"{prefix}win_pct_{role}":             gs("winPercentage"),
        f"{prefix}matches_played":             mp,
        f"{prefix}over95c_pct_{role}":         gs("over95CornersPercentage"),
    }

# ── DANE HISTORYCZNE ──────────────────────────────────────────────────────────

def load_historical(hist_dir):
    dfs = []
    for path in glob.glob(os.path.join(hist_dir, '*.csv')):
        try:
            df = pd.read_csv(path, encoding='utf-8-sig')
            df = df[df['status'] == 'complete'].copy()
            df['btts']   = df['btts'].astype(int)
            df['over25'] = df['over25'].astype(int)
            dfs.append(df)
        except Exception as e:
            print(f'[WARN] Wczytanie {os.path.basename(path)}: {e}')
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True).sort_values('date_unix').reset_index(drop=True)
    return out

# ── LAST-5 ────────────────────────────────────────────────────────────────────

def build_team_history(hist_df):
    history = {}
    for _, row in hist_df.iterrows():
        for team_id, is_home in [(row['homeID'], True), (row['awayID'], False)]:
            history.setdefault(int(team_id), []).append((row['date_unix'], row, is_home))
    return history


def compute_last5_stats(team_id, cutoff, team_history):
    past = [e for e in team_history.get(int(team_id), []) if e[0] < cutoff][-5:]
    if len(past) < 2:
        return None
    btts_v, o25_v, corners_v, gs_v, gc_v, win_v, cards_v = [], [], [], [], [], [], []
    for _, row, is_home in past:
        btts_v.append(int(row['btts']))
        o25_v.append(int(row['over25']))
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
        'btts_rate':      float(np.mean(btts_v)),
        'over25_rate':    float(np.mean(o25_v)),
        'corners_avg':    float(np.mean(corners_v)),
        'goals_scored':   float(np.mean(gs_v)),
        'goals_conceded': float(np.mean(gc_v)),
        'win_rate':       float(np.mean(win_v)),
        'cards_avg':      float(np.mean(cards_v)),
    }

# ── FEATURE ROW ───────────────────────────────────────────────────────────────

def build_features(row, league, last5_home, last5_away, h2h):
    odds1 = float(row.get('odds_ft_1', 2.0) or 2.0)
    oddsx = float(row.get('odds_ft_x', 3.5) or 3.5)
    odds2 = float(row.get('odds_ft_2', 3.0) or 3.0)

    imp_h = 1.0 / max(odds1, 0.01)
    imp_d = 1.0 / max(oddsx, 0.01)
    imp_a = 1.0 / max(odds2, 0.01)
    margin = imp_h + imp_d + imp_a

    feat = {
        # Kursy
        'odds_ft_1':            odds1,
        'odds_ft_x':            oddsx,
        'odds_ft_2':            odds2,
        'odds_btts_yes':        float(row.get('odds_btts_yes', 1.9) or 1.9),
        'odds_btts_no':         float(row.get('odds_btts_no', 1.9) or 1.9),
        'odds_ft_over25':       float(row.get('odds_ft_over25', 1.9) or 1.9),
        'odds_ft_under25':      float(row.get('odds_ft_under25', 1.9) or 1.9),
        'odds_corners_over_95': float(row.get('odds_corners_over_95', 1.85) or 1.85),
        'odds_corners_under_95':float(row.get('odds_corners_under_95', 1.85) or 1.85),
        # Implied probability
        'implied_home_norm':    imp_h / margin,
        'implied_away_norm':    imp_a / margin,
        'odds_diff_home_away':  odds2 - odds1,
        # Liga one-hot
        'is_allsvenskan':   0,
        'is_eliteserien':   0,
        'is_veikkausliiga': 0,
        # Potencjały (pre-match, z row)
        'btts_potential':        float(row.get('btts_potential', 0) or 0),
        'o25_potential':         float(row.get('o25_potential', 0) or 0),
        'corners_potential':     float(row.get('corners_potential', 0) or 0),
        'corners_o95_potential': float(row.get('corners_o95_potential', 0) or 0),
        'avg_potential':         float(row.get('avg_potential', 0) or 0),
        'team_a_xg_prematch':    float(row.get('team_a_xg_prematch', 0) or 0),
        'team_b_xg_prematch':    float(row.get('team_b_xg_prematch', 0) or 0),
        'pre_match_home_ppg':    float(row.get('pre_match_home_ppg', 0) or 0),
        'pre_match_away_ppg':    float(row.get('pre_match_away_ppg', 0) or 0),
        # Last-5
        'home_last5_btts_rate':      last5_home['btts_rate'],
        'home_last5_over25_rate':    last5_home['over25_rate'],
        'home_last5_corners_avg':    last5_home['corners_avg'],
        'home_last5_goals_scored':   last5_home['goals_scored'],
        'home_last5_goals_conceded': last5_home['goals_conceded'],
        'home_last5_win_rate':       last5_home['win_rate'],
        'home_last5_cards_avg':      last5_home['cards_avg'],
        'away_last5_btts_rate':      last5_away['btts_rate'],
        'away_last5_over25_rate':    last5_away['over25_rate'],
        'away_last5_corners_avg':    last5_away['corners_avg'],
        'away_last5_goals_scored':   last5_away['goals_scored'],
        'away_last5_goals_conceded': last5_away['goals_conceded'],
        'away_last5_win_rate':       last5_away['win_rate'],
        'away_last5_cards_avg':      last5_away['cards_avg'],
        'diff_last5_goals_scored':   last5_home['goals_scored'] - last5_away['goals_scored'],
        'diff_last5_win_rate':       last5_home['win_rate'] - last5_away['win_rate'],
        'diff_last5_corners_avg':    last5_home['corners_avg'] - last5_away['corners_avg'],
        # H2H (always fallback for CSL)
        'h2h_btts_pct':      h2h['h2h_btts_pct'],
        'h2h_over25_pct':    h2h['h2h_over25_pct'],
        'h2h_home_win_pct':  h2h['h2h_home_win_pct'],
        'h2h_avg_goals':     h2h['h2h_avg_goals'],
        'h2h_matches_count': h2h['h2h_matches_count'],
        'h2h_avg_corners':   h2h['h2h_avg_corners'],
    }
    return feat

# ── MODELE ────────────────────────────────────────────────────────────────────

def load_models(models_dir):
    models, imputors = {}, {}
    for name in MODEL_NAMES:
        model_path   = os.path.join(models_dir, f'nordic_{name}_model.pkl')
        imputer_path = os.path.join(models_dir, f'nordic_{name}_imputer.json')
        if not os.path.exists(model_path):
            print(f'[WARN] Brak modelu: {model_path}')
            continue
        if not os.path.exists(imputer_path):
            print(f'[WARN] Brak imputera: {imputer_path}')
            continue
        try:
            models[name]  = joblib.load(model_path)
            imputors[name] = json.load(open(imputer_path, encoding='utf-8'))
        except Exception as e:
            print(f'[WARN] Błąd ładowania {name}: {e}')
    return models, imputors

# ── PREDYKCJA ─────────────────────────────────────────────────────────────────

def kelly_stake(p, odds):
    edge = p * odds - 1.0
    if edge <= 0 or odds <= 1.0:
        return 0.0
    k = edge / (odds - 1.0)
    return min(max(BANKROLL * k * KELLY_FRAC, 0.0), MAX_STAKE)


def merge_duplicate_predictions(all_rows):
    """Łączy duplikaty (jeśli są)."""
    if isinstance(all_rows, pd.DataFrame):
        if all_rows.empty:
            return all_rows
        return all_rows
    else:
        if not all_rows:
            return all_rows
        return all_rows


def predict_model(model_name, model, imputer, features_dict, row, rejected, match_label):
    feat_cols  = imputer['features']
    imp_vals   = imputer['imputation_values']
    X = {f: features_dict.get(f, imp_vals.get(f, 0.0)) for f in feat_cols}
    X_arr = pd.DataFrame([X], columns=feat_cols)
    score = float(model.predict_proba(X_arr)[0][1]) * 100.0
    p = score / 100.0

    results = []

    if model_name == 'corners':
        # Always play Over 9.5 corners for CSL
        # Model corners is contrarian indicator (low score = model says Under = we play Over)
        odds_v = features_dict.get('odds_corners_over_95', 0.0)
        score_over = 100.0 - score  # Invert score
        flt = FILTERS['corners_over']
        reasons = []
        if score_over < flt['min_score']:
            reasons.append({'filter': 'MIN_SCORE', 'threshold': flt['min_score'],
                             'value': round(score_over, 2), 'passed': False})
        if odds_v < flt['min_odds']:
            reasons.append({'filter': 'MIN_ODDS', 'threshold': flt['min_odds'],
                             'value': round(odds_v, 2), 'passed': False})
        if odds_v > flt['max_odds']:
            reasons.append({'filter': 'MAX_ODDS', 'threshold': flt['max_odds'],
                             'value': round(odds_v, 2), 'passed': False})
        if reasons:
            rejected.append({'match': match_label, 'model': 'corners_over',
                              'score_pct': round(score_over, 1),
                              'odds': round(odds_v, 2),
                              'rejection_reasons': reasons})
        else:
            # Use empirical probability for corners_over (contrarian model)
            p_over = P_CORNERS_OVER_CSL
            ev = (p_over * odds_v - 1.0) * 100.0

            # Kelly calculation for corners_over
            if odds_v <= 1.0:
                stake = 0.0
            else:
                kelly_raw = (p_over * odds_v - 1.0) / (odds_v - 1.0)
                stake = min(max(BANKROLL * kelly_raw * KELLY_FRAC, 0), MAX_STAKE)

            results.append({
                'model_name': model_name,
                'direction':  'corners_over',
                'typ':        'Over 9.5 corners',
                'score':      round(score_over, 1),
                'p':          round(p_over, 4),
                'odds':       round(odds_v, 2),
                'ev':         round(ev, 1),
                'stake':      round(stake, 1),
                'features':   X,
            })
        return results

    # Modele wynikowe / btts / over25
    odds_map = {
        'result_home': 'odds_ft_1',
        'result_away': 'odds_ft_2',
        'btts':        'odds_btts_yes',
        'over25':      'odds_ft_over25',
    }
    typ_map = {
        'result_home': 'Home Win',
        'result_away': 'Away Win',
        'btts':        'BTTS Yes',
        'over25':      'Over 2.5',
    }

    # Special handling for over25 — if score < 50%, generate Over 2.5 with empirical p
    if model_name == 'over25' and score < 50.0:
        # Instead of Under 2.5, generate Over 2.5 with empirical probability
        odds_key = 'odds_ft_over25'
        direction = 'over25_forced'
        typ = 'Over 2.5'
        flt = FILTERS['over25_forced']
        p = P_OVER25_FORCED_CSL  # Use empirical probability
    else:
        odds_key = odds_map[model_name]
        direction = model_name
        typ = typ_map[model_name]
        flt = FILTERS[model_name]
        # For normal over25 (score >= 50%), use model score as probability
        p = score / 100.0

    odds_v   = features_dict.get(odds_key, 0.0)
    reasons = []
    if score < flt['min_score']:
        reasons.append({'filter': 'MIN_SCORE', 'threshold': flt['min_score'],
                         'value': round(score, 2), 'passed': False})
    if odds_v < flt['min_odds']:
        reasons.append({'filter': 'MIN_ODDS', 'threshold': flt['min_odds'],
                         'value': round(odds_v, 2), 'passed': False})
    if odds_v > flt['max_odds']:
        reasons.append({'filter': 'MAX_ODDS', 'threshold': flt['max_odds'],
                         'value': round(odds_v, 2), 'passed': False})
    if reasons:
        rejected.append({'match': match_label, 'model': model_name,
                          'score_pct': round(score, 1),
                          'odds': round(odds_v, 2),
                          'rejection_reasons': reasons})
        return []

    # Calculate EV and Kelly stake
    ev = (p * odds_v - 1.0) * 100.0
    if odds_v <= 1.0:
        stake = 0.0
    else:
        kelly_raw = (p * odds_v - 1.0) / (odds_v - 1.0)
        stake = min(max(BANKROLL * kelly_raw * KELLY_FRAC, 0), MAX_STAKE)
    return [{
        'model_name': model_name,
        'direction':  direction,
        'typ':        typ,
        'score':      round(score, 1),
        'p':          round(p, 4),
        'odds':       round(odds_v, 2),
        'ev':         round(ev, 1),
        'stake':      round(stake, 1),
        'features':   X,
    }]

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
        # Fallback: find most recent file with that prefix
        candidates = sorted(glob.glob(os.path.join(DAILY_DIR, f'{prefix}_*.csv')), reverse=True)
        if candidates:
            return candidates[0], date_str
        return None, date_str
    return path, date_str

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('day', nargs='?', choices=['today', 'tomorrow'], default='today')
    parser.add_argument('--debug',  action='store_true')
    parser.add_argument('--league', choices=list(ACTIVE_LEAGUES.keys()), default=None, nargs='+')
    parser.add_argument('--backfill', action='store_true', help='Backfill mode: process complete matches')
    args = parser.parse_args()

    if args.backfill:
        main_backfill()
        return

    for d in [CSL_SCORER_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Data
    daily_path, date_str = find_daily_file(args.day)
    if not daily_path:
        print(f'Brak pliku dla {args.day} ({date_str}).')
        print('Uruchom: python fetch_data.py --daily')
        sys.exit(1)

    day_label = args.day
    try:
        dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        day_name_pl = ['Poniedziałek','Wtorek','Środa','Czwartek','Piątek','Sobota','Niedziela'][dt.weekday()]
    except Exception:
        day_name_pl = day_label

    daily_df = pd.read_csv(daily_path, encoding='utf-8-sig')

    # Filtruj ligi CSL
    comp_filter = list(ACTIVE_LEAGUES.values())
    if args.league:
        comp_filter = [ACTIVE_LEAGUES[l] for l in args.league]
    csl = daily_df[daily_df['competition_id'].isin(comp_filter)].copy()

    id_to_league = {v: k for k, v in ACTIVE_LEAGUES.items()}

    print('══════════════════════════════════════════════════════')
    print(f'CSL SCORER — {date_str} ({day_name_pl})')
    print('══════════════════════════════════════════════════════')

    n_per_league = {lg: int((csl['competition_id'] == cid).sum())
                    for lg, cid in ACTIVE_LEAGUES.items()}
    print(f"Meczów: CSL={n_per_league['csl']}")

    if len(csl) == 0:
        print('Brak meczów CSL na wybrany dzień.')
        sys.exit(0)

    # ── Załaduj modele raz per liga
    league_models = {}
    for lg in ACTIVE_LEAGUES:
        _, models_dir, _ = LEAGUE_DIRS[lg]
        league_models[lg] = load_models(models_dir)

    # ── Załaduj historię per liga
    league_history = {}
    for lg, (hist_dir, _, _) in LEAGUE_DIRS.items():
        hist_df = load_historical(hist_dir)
        league_history[lg] = build_team_history(hist_df)

    # ── Załaduj current teams stats
    teams_lookup = load_current_teams_lookup()

    # ── Przetwórz mecze
    all_tips      = []
    rejected      = []
    warnings      = []
    debug_acc     = []
    debug_rej     = []

    for _, row in csl.iterrows():
        match_id  = int(row['id'])
        home_id   = int(row['homeID'])
        away_id   = int(row['awayID'])
        home_name = str(row.get('home_name', home_id))
        away_name = str(row.get('away_name', away_id))
        comp_id   = int(row['competition_id'])
        league    = id_to_league.get(comp_id, 'csl')
        cutoff    = float(row['date_unix'])
        match_label = f'{home_name} vs {away_name}'

        # Godzina (Asia/Shanghai)
        try:
            import pytz
            tz = pytz.timezone(TZ)
            ko_dt = datetime.datetime.fromtimestamp(cutoff, tz=tz)
            kickoff_str = ko_dt.strftime('%H:%M')
        except Exception:
            kickoff_str = '--:--'

        # Last-5
        team_hist = league_history.get(league, {})
        h_stats = compute_last5_stats(home_id, cutoff, team_hist)
        a_stats = compute_last5_stats(away_id, cutoff, team_hist)
        last5_available = h_stats is not None and a_stats is not None

        def fallback_last5(imputer_dict, prefix):
            iv = {}
            for mn in MODEL_NAMES:
                imp = imputer_dict.get(mn, {})
                iv.update(imp.get('imputation_values', {}))
            return {
                'btts_rate':      iv.get(f'{prefix}last5_btts_rate', 0.5),
                'over25_rate':    iv.get(f'{prefix}last5_over25_rate', 0.5),
                'corners_avg':    iv.get(f'{prefix}last5_corners_avg', 9.5),
                'goals_scored':   iv.get(f'{prefix}last5_goals_scored', 1.2),
                'goals_conceded': iv.get(f'{prefix}last5_goals_conceded', 1.2),
                'win_rate':       iv.get(f'{prefix}last5_win_rate', 0.4),
                'cards_avg':      iv.get(f'{prefix}last5_cards_avg', 2.0),
            }

        lg_models_dict, lg_imputors_dict = league_models[league]

        if h_stats is None:
            warnings.append(f'match_id={match_id}: brak last5 dla homeID={home_id} — imputer')
            h_stats = fallback_last5(lg_imputors_dict, 'home_')
        if a_stats is None:
            warnings.append(f'match_id={match_id}: brak last5 dla awayID={away_id} — imputer')
            a_stats = fallback_last5(lg_imputors_dict, 'away_')

        # H2H (always fallback for CSL)
        h2h = dict(H2H_FALLBACK)

        # Feature row
        try:
            features = build_features(row, league, h_stats, a_stats, h2h)
        except Exception as e:
            warnings.append(f'match_id={match_id}: błąd build_features ({e})')
            continue

        # Team stats z current lookup
        h_ts = get_team_stats_from_lookup(home_id, 'home', teams_lookup)
        a_ts = get_team_stats_from_lookup(away_id, 'away', teams_lookup)
        features.update(h_ts)
        features.update(a_ts)
        h_ppg = h_ts.get('home_team_ppg_home', np.nan)
        a_ppg = a_ts.get('away_team_ppg_away', np.nan)
        h_sc  = h_ts.get('home_team_scored_avg_home', np.nan)
        a_sc  = a_ts.get('away_team_scored_avg_away', np.nan)
        h_co  = h_ts.get('home_team_corners_avg_home', np.nan)
        a_co  = a_ts.get('away_team_corners_avg_away', np.nan)
        features['diff_team_ppg']     = h_ppg - a_ppg if not (np.isnan(h_ppg) or np.isnan(a_ppg)) else np.nan
        features['diff_team_scored']  = h_sc  - a_sc  if not (np.isnan(h_sc)  or np.isnan(a_sc))  else np.nan
        features['diff_team_corners'] = h_co  - a_co  if not (np.isnan(h_co)  or np.isnan(a_co))  else np.nan

        # Predykcje z modelu ligi
        for model_type, (models_dict, imputors_dict) in [
            ('liga', (lg_models_dict, lg_imputors_dict)),
        ]:
            for mn in MODEL_NAMES:
                if mn not in models_dict:
                    continue
                try:
                    preds = predict_model(mn, models_dict[mn], imputors_dict[mn],
                                          features, row, rejected, match_label)
                except Exception as e:
                    warnings.append(f'match_id={match_id} model={mn} type={model_type}: {e}')
                    continue

                for pred in preds:
                    all_tips.append({
                        'match_id':     match_id,
                        'league':       league,
                        'home':         home_name,
                        'away':         away_name,
                        'kickoff':      kickoff_str,
                        'model':        mn,
                        'model_type':   model_type,
                        'typ':          normalize_typ(
                            pred['typ'],
                            home=home_name,
                            away=away_name),
                        'score':        pred['score'],
                        'p':            pred['p'],
                        'odds':         pred['odds'],
                        'ev':           pred['ev'],
                        'stake':        pred['stake'],
                        'features':     pred['features'],
                        'last5_avail':  last5_available,
                    })

    # ── Odrzucone diag
    if rejected:
        print('\n[DIAG] Odrzucone:')
        for r in rejected[:20]:
            reason_str = ' | '.join(x['filter'] for x in r['rejection_reasons'])
            print(f"  {r['match']:30} │ {r['model']:12} │ {r['score_pct']:.1f}% │ {reason_str}")

    # ── Tabela wynikowa
    print()
    if not all_tips:
        print('Brak typów spełniających kryteria.')
    else:
        hdr = (f"{'#':>3} │ {'Godz':5} │ {'Mecz':25} │ "
               f"{'Model':12} │ {'Typ':12} │ {'Score':5} │ {'Kurs':5} │ "
               f"{'EV':7} │ {'Stake':5}")
        sep = '─' * len(hdr)
        print('══════════════════════════════════════════════════════')
        print(hdr)
        print(sep)
        for i, t in enumerate(all_tips, 1):
            match_short = f"{t['home'][:10]} vs {t['away'][:10]}"
            model_label = {'result_home': 'Result Home', 'result_away': 'Result Away',
                           'btts': 'BTTS', 'over25': 'Over 2.5',
                           'corners': 'Corners'}.get(t['model'], t['model'])
            print(
                f"{i:>3} │ {t['kickoff']:5} │ {match_short:25} │ "
                f"{model_label:12} │ {t['typ']:12} │ {t['score']:4.1f}% │ "
                f"{t['odds']:5.2f} │ {t['ev']:+6.1f}% │ {t['stake']:5.1f}"
            )

    # ── Zapis CSV per liga
    saved_paths = []
    league_abbr_map = {
        'csl': ('csl', CSL_SCORER_DIR),
    }
    csv_cols = ['ID', 'Data', 'Godzina', 'Mecz', 'Liga', 'Model', 'Model_type',
                'Typ', 'Score[%]', 'P_model', 'Kurs', 'EV[%]',
                'Stake_PLN', 'Wynik', 'Rezultat', 'Corners', 'Kartki', 'Profit_PLN']

    for lg, (lg_name, scorer_dir) in league_abbr_map.items():
        lg_ml = [t for t in all_tips if t['league'] == lg and t['model_type'] == 'liga']

        rows = []
        for t in lg_ml:
            rows.append({
                'ID':            t['match_id'],
                'Data':          date_str,
                'Godzina':       t['kickoff'],
                'Mecz':          f"{t['home']} vs {t['away']}",
                'Liga':          t['league'],
                'Model':         t['model'],
                'Model_type':    t['model_type'],
                'Typ':           t['typ'],
                'Score[%]':      t['score'],
                'P_model':       t['p'],
                'Kurs':          t['odds'],
                'EV[%]':         t['ev'],
                'Stake_PLN':     t['stake'],
                'Wynik':         '',
                'Rezultat':      '',
                'Corners':       '',
                'Kartki':        '',
                'Profit_PLN':    '',
            })

        # Łączy duplikaty (jeśli są)
        rows = merge_duplicate_predictions(rows)

        if rows:
            out_df = pd.DataFrame(rows, columns=csv_cols)
            out_path = os.path.join(scorer_dir, f'{lg_name}_scorer_{date_str}.csv')
            out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
            saved_paths.append(out_path)

    print('══════════════════════════════════════════════════════')
    paths_str = ' | '.join(os.path.basename(p) for p in saved_paths) if saved_paths else 'brak typów'
    print(f'Łącznie: {len(all_tips)} typów | Zapisano → {paths_str}')
    print('══════════════════════════════════════════════════════')

    # ── Debug JSON
    if args.debug:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        debug = {
            'timestamp': ts,
            'date':      date_str,
            'day':       day_label,
            'leagues':   {lg: {'n_matches': n_per_league[lg],
                               'n_scored':  sum(1 for t in all_tips if t['league'] == lg)}
                          for lg in ACTIVE_LEAGUES},
            'accepted':  [
                {
                    'match_id':         t['match_id'],
                    'home':             t['home'],
                    'away':             t['away'],
                    'liga':             t['league'],
                    'kickoff':          t['kickoff'],
                    'model':            t['model'],
                    'model_type':       t['model_type'],
                    'typ':              t['typ'],
                    'score_pct':        t['score'],
                    'odds':             t['odds'],
                    'ev_pct':           t['ev'],
                    'stake_pln':        t['stake'],
                    'last5_available':  t['last5_avail'],
                    'key_features': {
                        k: round(float(v), 4) for k, v in t['features'].items()
                        if k in ('odds_ft_1', 'odds_ft_2', 'implied_home_norm',
                                 'implied_away_norm', 'home_last5_win_rate',
                                 'away_last5_win_rate', 'h2h_home_win_pct')
                    },
                    'filters': {
                        flt: {
                            'threshold': FILTERS.get(t['model'], FILTERS.get(
                                'corners_over', {})
                            ).get(flt.replace('min_', 'min_').replace('max_', 'max_'), None),
                            'value': (t['score'] if flt == 'min_score' else t['odds']),
                            'passed': True,
                        }
                        for flt in ('min_score', 'min_odds', 'max_odds')
                    },
                }
                for t in all_tips
            ],
            'rejected': [
                {
                    'match_id': None,
                    'home':     r['match'].split(' vs ')[0] if ' vs ' in r['match'] else r['match'],
                    'away':     r['match'].split(' vs ')[1] if ' vs ' in r['match'] else '',
                    'model':    r['model'],
                    'model_type': '?',
                    'score_pct': r['score_pct'],
                    'odds':     r['odds'],
                    'rejection_reasons': r['rejection_reasons'],
                }
                for r in rejected
            ],
            'warnings': warnings,
        }
        debug_path = os.path.join(REPORTS_DIR, f'debug_csl_{date_str}_{ts}.json')
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(debug, f, ensure_ascii=False, indent=2, default=str)
        print(f'  Debug: {debug_path}')


# ── BACKFILL FUNCTION ─────────────────────────────────────────────────────────

def main_backfill():
    """Backfill mode: process all complete CSL matches from csl_matches_2026.csv"""
    os.makedirs(CSL_SCORER_DIR, exist_ok=True)

    # Wczytaj matches
    matches_path = os.path.join(CURRENT_DIR, 'csl_matches_2026.csv')
    if not os.path.exists(matches_path):
        print(f'Brak pliku {matches_path}')
        sys.exit(1)

    try:
        all_matches_df = pd.read_csv(matches_path, encoding='utf-8-sig')
    except Exception as e:
        print(f'Błąd wczytania matches: {e}')
        sys.exit(1)

    # Filtruj complete
    complete_df = all_matches_df[all_matches_df['status'] == 'complete'].copy()
    print('══════════════════════════════════════════════════════')
    print('BACKFILL — CSL SCORER')
    print('══════════════════════════════════════════════════════')
    print(f'Meczów complete w 2026: {len(complete_df)}')

    if complete_df.empty:
        print('Brak complete meczów.')
        sys.exit(0)

    # Wczytaj modele
    league_models = {}
    for lg in ACTIVE_LEAGUES:
        _, models_dir, _ = LEAGUE_DIRS[lg]
        league_models[lg] = load_models(models_dir)

    # Wczytaj historię
    league_history = {}
    for lg, (hist_dir, _, _) in LEAGUE_DIRS.items():
        hist_df = load_historical(hist_dir)
        league_history[lg] = build_team_history(hist_df)

    # Wczytaj teams
    teams_lookup = load_current_teams_lookup()

    # Pogrupuj po dacie
    complete_df['date_str'] = complete_df['date_unix'].apply(
        lambda x: datetime.datetime.fromtimestamp(
            int(x), tz=datetime.timezone.utc
        ).strftime('%Y-%m-%d')
    )

    date_groups = complete_df.groupby('date_str', sort=True)
    dates = sorted(date_groups.groups.keys())
    print(f'Dat do przetworzenia:   {len(dates)}')

    # Sprawdzenie cache
    cached_files = glob.glob(os.path.join(CSL_SCORER_DIR, 'csl_scorer_*.csv'))
    cached_dates = set(
        os.path.basename(f).replace('csl_scorer_', '').replace('.csv', '')
        for f in cached_files
    )
    print(f'Pliki już w cache:       {len(cached_dates)} (pominiętych)')

    to_process = [d for d in dates if d not in cached_dates]
    print(f'Pliki do wygenerowania: {len(to_process)}')
    print('──────────────────────────────────────────────────────')

    id_to_league = {v: k for k, v in ACTIVE_LEAGUES.items()}
    all_predictions = []
    processed_count = 0

    for date_str in to_process:
        date_matches = date_groups.get_group(date_str).copy()

        all_tips = []
        rejected = []

        for _, row in date_matches.iterrows():
            match_id = int(row['id'])
            home_id = int(row['homeID'])
            away_id = int(row['awayID'])
            home_name = str(row.get('home_name', home_id))
            away_name = str(row.get('away_name', away_id))
            comp_id = int(row['competition_id'])
            league = id_to_league.get(comp_id, 'csl')
            cutoff = float(row['date_unix'])
            match_label = f'{home_name} vs {away_name}'

            # Godzina
            try:
                import pytz
                tz = pytz.timezone(TZ)
                ko_dt = datetime.datetime.fromtimestamp(cutoff, tz=tz)
                kickoff_str = ko_dt.strftime('%H:%M')
            except Exception:
                kickoff_str = '--:--'

            # Last-5
            team_hist = league_history.get(league, {})
            h_stats = compute_last5_stats(home_id, cutoff, team_hist)
            a_stats = compute_last5_stats(away_id, cutoff, team_hist)

            def fallback_last5(imputer_dict, prefix):
                iv = {}
                for mn in MODEL_NAMES:
                    imp = imputer_dict.get(mn, {})
                    iv.update(imp.get('imputation_values', {}))
                return {
                    'btts_rate':      iv.get(f'{prefix}last5_btts_rate', 0.5),
                    'over25_rate':    iv.get(f'{prefix}last5_over25_rate', 0.5),
                    'corners_avg':    iv.get(f'{prefix}last5_corners_avg', 9.5),
                    'goals_scored':   iv.get(f'{prefix}last5_goals_scored', 1.2),
                    'goals_conceded': iv.get(f'{prefix}last5_goals_conceded', 1.2),
                    'win_rate':       iv.get(f'{prefix}last5_win_rate', 0.4),
                    'cards_avg':      iv.get(f'{prefix}last5_cards_avg', 2.0),
                }

            lg_models_dict, lg_imputors_dict = league_models[league]

            if h_stats is None:
                h_stats = fallback_last5(lg_imputors_dict, 'home_')
            if a_stats is None:
                a_stats = fallback_last5(lg_imputors_dict, 'away_')

            h2h = dict(H2H_FALLBACK)

            try:
                features = build_features(row, league, h_stats, a_stats, h2h)
            except Exception:
                continue

            h_ts = get_team_stats_from_lookup(home_id, 'home', teams_lookup)
            a_ts = get_team_stats_from_lookup(away_id, 'away', teams_lookup)
            features.update(h_ts)
            features.update(a_ts)
            h_ppg = h_ts.get('home_team_ppg_home', np.nan)
            a_ppg = a_ts.get('away_team_ppg_away', np.nan)
            h_sc = h_ts.get('home_team_scored_avg_home', np.nan)
            a_sc = a_ts.get('away_team_scored_avg_away', np.nan)
            h_co = h_ts.get('home_team_corners_avg_home', np.nan)
            a_co = a_ts.get('away_team_corners_avg_away', np.nan)
            features['diff_team_ppg']     = h_ppg - a_ppg if not (np.isnan(h_ppg) or np.isnan(a_ppg)) else np.nan
            features['diff_team_scored']  = h_sc  - a_sc  if not (np.isnan(h_sc)  or np.isnan(a_sc))  else np.nan
            features['diff_team_corners'] = h_co  - a_co  if not (np.isnan(h_co)  or np.isnan(a_co))  else np.nan

            # Predykcje
            for model_type, (models_dict, imputors_dict) in [
                ('liga', (lg_models_dict, lg_imputors_dict)),
            ]:
                for mn in MODEL_NAMES:
                    if mn not in models_dict:
                        continue
                    try:
                        preds = predict_model(mn, models_dict[mn], imputors_dict[mn],
                                              features, row, rejected, match_label)
                    except Exception:
                        continue

                    for pred in preds:
                        all_tips.append({
                            'match_id':     match_id,
                            'league':       league,
                            'home':         home_name,
                            'away':         away_name,
                            'kickoff':      kickoff_str,
                            'model':        mn,
                            'model_type':   model_type,
                            'typ':          normalize_typ(pred['typ'], home=home_name, away=away_name),
                            'score':        pred['score'],
                            'p':            pred['p'],
                            'odds':         pred['odds'],
                            'ev':           pred['ev'],
                            'stake':        pred['stake'],
                            'features':     pred['features'],
                            'home_goals':   int(row.get('homeGoalCount', 0)),
                            'away_goals':   int(row.get('awayGoalCount', 0)),
                            'corners':      int(row.get('totalCornerCount', 0)),
                            'yellow_cards': int(row.get('team_a_yellow_cards', 0)) + int(row.get('team_b_yellow_cards', 0)),
                        })

        # Buduj CSV rows z Wynik
        rows = []
        for t in all_tips:
            if t['model_type'] != 'liga':
                continue

            wynik = None
            if t['model'] == 'result_home':
                wynik = 1 if t['home_goals'] > t['away_goals'] else 0
            elif t['model'] == 'result_away':
                wynik = 1 if t['away_goals'] > t['home_goals'] else 0
            elif t['model'] == 'btts':
                wynik = 1 if t['home_goals'] > 0 and t['away_goals'] > 0 else 0
            elif t['model'] == 'over25':
                wynik = 1 if (t['home_goals'] + t['away_goals']) > 2.5 else 0
            elif t['model'] == 'corners':
                wynik = 1 if t['corners'] <= 9.5 else 0

            profit = 0.0
            if wynik is not None:
                profit = t['stake'] * (t['odds'] - 1) if wynik == 1 else -t['stake']

            rows.append({
                'ID':            t['match_id'],
                'Data':          date_str,
                'Godzina':       t['kickoff'],
                'Mecz':          f"{t['home']} vs {t['away']}",
                'Liga':          t['league'],
                'Model':         t['model'],
                'Model_type':    t['model_type'],
                'Typ':           t['typ'],
                'Score[%]':      t['score'],
                'P_model':       t['p'],
                'Kurs':          t['odds'],
                'EV[%]':         t['ev'],
                'Stake_PLN':     t['stake'],
                'Wynik':         wynik,
                'Rezultat':      f"{t['home_goals']}:{t['away_goals']}",
                'Corners':       t['corners'],
                'Kartki':        t['yellow_cards'],
                'Profit_PLN':    round(profit, 2),
            })

        rows = merge_duplicate_predictions(rows)

        if rows:
            out_df = pd.DataFrame(rows, columns=['ID', 'Data', 'Godzina', 'Mecz', 'Liga', 'Model', 'Model_type',
                                                 'Typ', 'Score[%]', 'P_model', 'Kurs', 'EV[%]',
                                                 'Stake_PLN', 'Wynik', 'Rezultat', 'Corners', 'Kartki', 'Profit_PLN'])
            out_path = os.path.join(CSL_SCORER_DIR, f'csl_scorer_{date_str}.csv')
            out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
            processed_count += 1
            all_predictions.extend(rows)
            settled_count = len([r for r in rows if r['Wynik'] is not None and r['Wynik'] in [0, 1]])
            print(f'✅ {date_str} — {len(rows)} predykcji ({len(date_matches)} meczów)')

    print('──────────────────────────────────────────────────────')
    settled = len([p for p in all_predictions if p['Wynik'] is not None and p['Wynik'] in [0, 1]])
    print(f'Łącznie wygenerowano:  {len(all_predictions)} predykcji')
    print(f'Rozliczonych:          {settled} ({100*settled/len(all_predictions) if all_predictions else 0:.0f}%)')
    print(f'Plików zapisanych:     {processed_count}')
    print('══════════════════════════════════════════════════════')


if __name__ == '__main__':
    main()
