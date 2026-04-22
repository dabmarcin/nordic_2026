import sys
import io
import os
import glob
import json
import argparse
import datetime

import numpy as np
import pandas as pd
import joblib

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from nordic_config import (
    API_KEY, BASE_URL,
    ALLSVENSKAN_DIR, ELITESERIEN_DIR, VEIKKAUSLIIGA_DIR,
    DAILY_DIR, H2H_CACHE, REPORTS_DIR,
    MODELS_ALLSV, MODELS_ELITE, MODELS_VEIKK, MODELS_COMBINED,
    ALLSV_SCORER_DIR, ELITE_SCORER_DIR, VEIKK_SCORER_DIR,
    ALLSVENSKAN_2026_ID, ELITESERIEN_2026_ID, VEIKKAUSLIIGA_2026_ID,
    MATCH_DETAILS_DIR,
)

# ── STAŁE ────────────────────────────────────────────────────────────────────

ACTIVE_LEAGUES = {
    'allsvenskan':   ALLSVENSKAN_2026_ID,
    'eliteserien':   ELITESERIEN_2026_ID,
    'veikkausliiga': VEIKKAUSLIIGA_2026_ID,
}

LEAGUE_DIRS = {
    'allsvenskan':   (ALLSVENSKAN_DIR,  MODELS_ALLSV,  ALLSV_SCORER_DIR),
    'eliteserien':   (ELITESERIEN_DIR,  MODELS_ELITE,  ELITE_SCORER_DIR),
    'veikkausliiga': (VEIKKAUSLIIGA_DIR, MODELS_VEIKK, VEIKK_SCORER_DIR),
}

FILTERS = {
    'result_home':   {'min_score': 62.0, 'min_odds': 1.40, 'max_odds': 5.0},
    'result_away':   {'min_score': 62.0, 'min_odds': 1.40, 'max_odds': 5.0},
    'btts':          {'min_score': 62.0, 'min_odds': 1.20, 'max_odds': 5.0},
    'over25':        {'min_score': 62.0, 'min_odds': 1.20, 'max_odds': 5.0},
    'corners_over':  {'min_score': 60.0, 'min_odds': 1.30, 'max_odds': 5.0},
    'corners_under': {'min_score': 60.0, 'min_odds': 1.30, 'max_odds': 5.0},
}

BANKROLL   = 1000.0
KELLY_FRAC = 0.25
MAX_STAKE  = 50.0

MODEL_NAMES = ['result_home', 'result_away', 'btts', 'over25', 'corners']

TZ = 'Europe/Helsinki'

H2H_FALLBACK = {
    'h2h_btts_pct':     0.5,
    'h2h_over25_pct':   0.5,
    'h2h_home_win_pct': 0.45,
    'h2h_avg_goals':    2.5,
    'h2h_matches_count': 0,
    'h2h_avg_corners':  9.5,
}

# ── PINNACLE ODDS ─────────────────────────────────────────────────────────────

def get_pinnacle_odds(match_id, model_name, typ=None):
    gpt_path = os.path.join(MATCH_DETAILS_DIR, f"gpt_{int(match_id)}.json")

    if not os.path.isfile(gpt_path):
        return None
    try:
        with open(gpt_path, encoding='utf-8') as f:
            gpt = json.load(f)
        pin = gpt.get("pinnacle_odds", {})

        if model_name == "corners":
            if typ and "under" in str(typ).lower():
                return pin.get("corners_under_95")
            else:
                return pin.get("corners_over_95")
        mapping = {
            "result_home": pin.get("home"),
            "result_away": pin.get("away"),
            "btts":        pin.get("btts_yes"),
            "over25":      pin.get("over25"),
        }
        return mapping.get(model_name)
    except Exception:
        return None


def get_match_potencjaly(match_id):
    gpt_path = os.path.join(MATCH_DETAILS_DIR, f'gpt_{int(match_id)}.json')
    if not os.path.isfile(gpt_path):
        return {}
    try:
        with open(gpt_path, encoding='utf-8') as f:
            return json.load(f).get('potencjaly', {})
    except Exception:
        return {}


def generate_gpt_predictions(match_id, home_name, away_name,
                              kickoff, liga, date_str) -> list:
    """Wczytuje gpt_tips z gpt_{match_id}.json i zwraca wiersze CSV z Model_type='gpt_pred'."""
    gpt_path = os.path.join(MATCH_DETAILS_DIR, f'gpt_{int(match_id)}.json')
    if not os.path.isfile(gpt_path):
        return []
    try:
        with open(gpt_path, encoding='utf-8') as f:
            gpt = json.load(f)
    except Exception:
        return []

    tips = gpt.get('gpt_tips', [])
    if not tips:
        return []

    pin = gpt.get('pinnacle_odds', {})
    pin_by_kierunek = {
        'home_win': pin.get('home'),
        'away_win': pin.get('away'),
        'draw':     pin.get('draw'),
        'btts_yes': pin.get('btts_yes'),
        'over':     pin.get('over25'),
    }

    rows = []
    for tip in tips:
        try:
            kurs = tip.get('kurs')
            if not kurs or float(kurs) <= 1.0:
                continue
            kurs = round(float(kurs), 2)

            pinnacle = pin_by_kierunek.get(tip.get('kierunek', ''))
            ev_str = ''
            if pinnacle and pinnacle > 1.0:
                p_pin = 1.0 / pinnacle
                ev_str = f'{(p_pin * kurs - 1.0) * 100:+.1f}%'

            rows.append({
                'ID':            match_id,
                'Data':          date_str,
                'Godzina':       kickoff,
                'Mecz':          f'{home_name} vs {away_name}',
                'Liga':          liga.upper()[:5],
                'Model':         'GPT FootyStats',
                'Model_type':    'gpt_pred',
                'Typ':           tip.get('typ', ''),
                'Score[%]':      '',
                'P_model':       '',
                'Kurs':          kurs,
                'Pinnacle_odds': round(float(pinnacle), 2) if pinnacle else '',
                'EV[%]':         ev_str,
                'Stake_PLN':     '',
                'BTTS_pot':      '',
                'Corners_pot':   '',
                'O25_pot':       '',
                'Wynik':         '',
                'Rezultat':      '',
                'Corners':       '',
                'Kartki':        '',
                'Profit_PLN':    '',
                '_liga_key':     liga,
            })
        except Exception:
            continue
    return rows


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

# ── H2H ───────────────────────────────────────────────────────────────────────

def load_h2h(match_id, home_id, away_id, cutoff_unix, warnings):
    cache_path = os.path.join(H2H_CACHE, f'match_{match_id}.json')
    if not os.path.exists(cache_path):
        warnings.append(f'match_id={match_id}: brak h2h_cache — neutral fallback')
        return dict(H2H_FALLBACK), False

    try:
        raw = json.load(open(cache_path, encoding='utf-8'))
        h2h = raw.get('data', {}).get('h2h', {})

        pmr = h2h.get('previous_matches_results', {})
        bs  = h2h.get('betting_stats', {})

        total = int(pmr.get('totalMatches', 0))
        if total == 0:
            warnings.append(f'match_id={match_id}: h2h totalMatches=0 — neutral fallback')
            return dict(H2H_FALLBACK), True

        team_a_id = h2h.get('team_a_id', home_id)
        if int(team_a_id) == int(home_id):
            home_win_pct = float(pmr.get('team_a_win_percent', 45)) / 100.0
        else:
            home_win_pct = float(pmr.get('team_b_win_percent', 45)) / 100.0

        btts_pct   = float(bs.get('bttsPercentage', 50)) / 100.0
        over25_pct = float(bs.get('over25Percentage', 50)) / 100.0
        avg_goals  = float(bs.get('avg_goals', 2.5))

        return {
            'h2h_btts_pct':      btts_pct,
            'h2h_over25_pct':    over25_pct,
            'h2h_home_win_pct':  home_win_pct,
            'h2h_avg_goals':     avg_goals,
            'h2h_matches_count': total,
            'h2h_avg_corners':   9.5,
        }, True

    except Exception as e:
        warnings.append(f'match_id={match_id}: błąd parsowania h2h ({e}) — neutral fallback')
        return dict(H2H_FALLBACK), False

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
        'is_allsvenskan':   1 if league == 'allsvenskan'   else 0,
        'is_eliteserien':   1 if league == 'eliteserien'   else 0,
        'is_veikkausliiga': 1 if league == 'veikkausliiga' else 0,
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
        # H2H
        'h2h_btts_pct':      h2h['h2h_btts_pct'],
        'h2h_over25_pct':    h2h['h2h_over25_pct'],
        'h2h_home_win_pct':  h2h['h2h_home_win_pct'],
        'h2h_avg_goals':     h2h['h2h_avg_goals'],
        'h2h_matches_count': h2h['h2h_matches_count'],
        'h2h_avg_corners':   h2h['h2h_avg_corners'],
    }
    return feat

# ── PREDYKCJA ─────────────────────────────────────────────────────────────────

def kelly_stake(p, odds):
    edge = p * odds - 1.0
    if edge <= 0 or odds <= 1.0:
        return 0.0
    k = edge / (odds - 1.0)
    return min(max(BANKROLL * k * KELLY_FRAC, 0.0), MAX_STAKE)


def predict_model(model_name, model, imputer, features_dict, row, rejected, match_label):
    feat_cols  = imputer['features']
    imp_vals   = imputer['imputation_values']
    X = {f: features_dict.get(f, imp_vals.get(f, 0.0)) for f in feat_cols}
    X_arr = pd.DataFrame([X], columns=feat_cols)
    score = float(model.predict_proba(X_arr)[0][1]) * 100.0
    p = score / 100.0

    results = []

    if model_name == 'corners':
        p_over  = p
        p_under = 1.0 - p

        for direction, p_dir, odds_key, fkey in [
            ('corners_over',  p_over,  'odds_corners_over_95',  'Over Corners'),
            ('corners_under', p_under, 'odds_corners_under_95', 'Under Corners'),
        ]:
            odds   = features_dict.get(odds_key.replace('fkey', odds_key), 0.0)
            odds_v = features_dict.get(odds_key, 0.0)
            score_dir = p_dir * 100.0
            flt = FILTERS[direction]
            reasons = []
            if score_dir < flt['min_score']:
                reasons.append({'filter': 'MIN_SCORE', 'threshold': flt['min_score'],
                                 'value': round(score_dir, 2), 'passed': False})
            if odds_v < flt['min_odds']:
                reasons.append({'filter': 'MIN_ODDS', 'threshold': flt['min_odds'],
                                 'value': round(odds_v, 2), 'passed': False})
            if odds_v > flt['max_odds']:
                reasons.append({'filter': 'MAX_ODDS', 'threshold': flt['max_odds'],
                                 'value': round(odds_v, 2), 'passed': False})
            if reasons:
                rejected.append({'match': match_label, 'model': direction,
                                  'score_pct': round(score_dir, 1),
                                  'odds': round(odds_v, 2),
                                  'rejection_reasons': reasons})
            else:
                ev = (p_dir * odds_v - 1.0) * 100.0
                stake = kelly_stake(p_dir, odds_v)
                results.append({
                    'model_name': model_name,
                    'direction':  direction,
                    'typ':        fkey,
                    'score':      round(score_dir, 1),
                    'p':          round(p_dir, 4),
                    'odds':       round(odds_v, 2),
                    'ev':         round(ev, 1),
                    'stake':      round(stake, 1),
                    'features':   X,
                })

        # Jeśli oba przeszły — wybierz wyższy EV
        if len(results) == 2:
            results = [max(results, key=lambda r: r['ev'])]
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
    odds_key = odds_map[model_name]
    odds_v   = features_dict.get(odds_key, 0.0)
    flt = FILTERS[model_name]
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

    ev    = (p * odds_v - 1.0) * 100.0
    stake = kelly_stake(p, odds_v)
    return [{
        'model_name': model_name,
        'direction':  model_name,
        'typ':        typ_map[model_name],
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
    parser.add_argument('day', choices=['today', 'tomorrow'])
    parser.add_argument('--debug',  action='store_true')
    parser.add_argument('--league', choices=list(ACTIVE_LEAGUES.keys()), default=None)
    args = parser.parse_args()

    for d in [ALLSV_SCORER_DIR, ELITE_SCORER_DIR, VEIKK_SCORER_DIR, REPORTS_DIR]:
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

    # Filtruj ligi nordic
    comp_filter = list(ACTIVE_LEAGUES.values())
    if args.league:
        comp_filter = [ACTIVE_LEAGUES[args.league]]
    nordic = daily_df[daily_df['competition_id'].isin(comp_filter)].copy()

    id_to_league = {v: k for k, v in ACTIVE_LEAGUES.items()}

    print('══════════════════════════════════════════════════════')
    print(f'NORDIC SCORER — {date_str} ({day_name_pl})')
    print('══════════════════════════════════════════════════════')

    n_per_league = {lg: int((nordic['competition_id'] == cid).sum())
                    for lg, cid in ACTIVE_LEAGUES.items()}
    print(f"Meczów: Allsvenskan={n_per_league['allsvenskan']} "
          f"Eliteserien={n_per_league['eliteserien']} "
          f"Veikkausliiga={n_per_league['veikkausliiga']}")

    if len(nordic) == 0:
        print('Brak meczów nordic na wybrany dzień.')
        sys.exit(0)

    # ── Załaduj modele raz per liga
    league_models = {}
    for lg in ACTIVE_LEAGUES:
        _, models_dir, _ = LEAGUE_DIRS[lg]
        league_models[lg] = load_models(models_dir)
    combined_models = load_models(MODELS_COMBINED)

    # ── Załaduj historię per liga
    league_history = {}
    for lg, (hist_dir, _, _) in LEAGUE_DIRS.items():
        hist_df = load_historical(hist_dir)
        league_history[lg] = build_team_history(hist_df)

    # Imputer fallback z combined (mediany globalne)
    combined_imp = combined_models[1] if combined_models[1] else {}

    # ── Przetwórz mecze
    all_tips      = []
    gpt_csv_rows  = []
    rejected      = []
    warnings      = []
    debug_acc     = []
    debug_rej     = []

    for _, row in nordic.iterrows():
        match_id  = int(row['id'])
        home_id   = int(row['homeID'])
        away_id   = int(row['awayID'])
        home_name = str(row.get('home_name', home_id))
        away_name = str(row.get('away_name', away_id))
        comp_id   = int(row['competition_id'])
        league    = id_to_league.get(comp_id, 'allsvenskan')
        cutoff    = float(row['date_unix'])
        match_label = f'{home_name} vs {away_name}'

        # Godzina (Europe/Helsinki)
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

        # H2H
        h2h, h2h_from_cache = load_h2h(match_id, home_id, away_id, cutoff, warnings)

        # Feature row
        try:
            features = build_features(row, league, h_stats, a_stats, h2h)
        except Exception as e:
            warnings.append(f'match_id={match_id}: błąd build_features ({e})')
            continue

        # Potencjały z gpt JSON (dla kolumn CSV)
        pot       = get_match_potencjaly(match_id)
        btts_pot  = pot.get('btts_potential', '')
        corn_pot  = pot.get('corners_o95_potential', '')
        o25_pot   = pot.get('o25_potential', '')

        # Predykcje z modelu ligi i combined
        for model_type, (models_dict, imputors_dict) in [
            ('liga',     (lg_models_dict, lg_imputors_dict)),
            ('combined', combined_models),
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
                    pinnacle_val = get_pinnacle_odds(match_id, mn, typ=pred['typ'])
                    all_tips.append({
                        'match_id':     match_id,
                        'league':       league,
                        'home':         home_name,
                        'away':         away_name,
                        'kickoff':      kickoff_str,
                        'model':        mn,
                        'model_type':   model_type,
                        'typ':          pred['typ'],
                        'score':        pred['score'],
                        'p':            pred['p'],
                        'odds':         pred['odds'],
                        'pinnacle_odds': round(pinnacle_val, 2) if pinnacle_val else '',
                        'ev':           pred['ev'],
                        'stake':        pred['stake'],
                        'features':     pred['features'],
                        'h2h_from_cache': h2h_from_cache,
                        'h2h_count':    h2h['h2h_matches_count'],
                        'last5_avail':  last5_available,
                        'btts_pot':     btts_pot,
                        'corners_pot':  corn_pot,
                        'o25_pot':      o25_pot,
                    })

        # GPT predictions
        gpt_rows = generate_gpt_predictions(
            match_id, home_name, away_name, kickoff_str, league, date_str)
        gpt_csv_rows.extend(gpt_rows)

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
        hdr = (f"{'#':>3} │ {'Godz':5} │ {'Mecz':25} │ {'Liga':5} │ "
               f"{'Model':12} │ {'Typ':12} │ {'Score':5} │ {'Kurs':5} │ "
               f"{'Pin':5} │ {'EV':7} │ {'Stake':5} │ MType")
        sep = '─' * len(hdr)
        print('══════════════════════════════════════════════════════')
        print(hdr)
        print(sep)
        for i, t in enumerate(all_tips, 1):
            lg_abbr = {'allsvenskan': 'ALLSV', 'eliteserien': 'ELITE',
                       'veikkausliiga': 'VEIKK'}.get(t['league'], t['league'][:5].upper())
            match_short = f"{t['home'][:10]} vs {t['away'][:10]}"
            model_label = {'result_home': 'Result Home', 'result_away': 'Result Away',
                           'btts': 'BTTS', 'over25': 'Over 2.5',
                           'corners': 'Corners'}.get(t['model'], t['model'])
            pin_str = f"{t['pinnacle_odds']:5.2f}" if t['pinnacle_odds'] != '' else '  —  '
            print(
                f"{i:>3} │ {t['kickoff']:5} │ {match_short:25} │ {lg_abbr:5} │ "
                f"{model_label:12} │ {t['typ']:12} │ {t['score']:4.1f}% │ "
                f"{t['odds']:5.2f} │ {pin_str} │ {t['ev']:+6.1f}% │ {t['stake']:5.1f} │ {t['model_type']}"
            )

    # ── GPT Predictions console
    if gpt_csv_rows:
        print()
        print('══════════════════════════════════════════════════════')
        print('[GPT PREDICTIONS]')
        gpt_hdr = (f"{'Godz':5} │ {'Mecz':28} │ {'Liga':5} │ "
                   f"{'Typ':22} │ {'Kurs':5} │ {'Pin':5} │ {'EV':7} │ {'Pewność'}")
        print(gpt_hdr)
        print('─' * len(gpt_hdr))
        for r in gpt_csv_rows:
            mecz_short = str(r['Mecz'])[:28]
            pin_str    = f"{r['Pinnacle_odds']:5.2f}" if r['Pinnacle_odds'] != '' else '  —  '
            ev_str     = str(r['EV[%]']) if r['EV[%]'] else '      '
            tip_short  = str(r['Typ'])[:22]
            print(f"{r['Godzina']:5} │ {mecz_short:28} │ {r['Liga']:5} │ "
                  f"{tip_short:22} │ {r['Kurs']:5.2f} │ {pin_str} │ {ev_str:7} │ {''}")

    # ── Zapis CSV per liga
    saved_paths = []
    league_abbr_map = {
        'allsvenskan':   ('allsvenskan', ALLSV_SCORER_DIR),
        'eliteserien':   ('eliteserien', ELITE_SCORER_DIR),
        'veikkausliiga': ('veikkausliiga', VEIKK_SCORER_DIR),
    }
    csv_cols = ['ID', 'Data', 'Godzina', 'Mecz', 'Liga', 'Model', 'Model_type',
                'Typ', 'Score[%]', 'P_model', 'Kurs', 'Pinnacle_odds', 'EV[%]',
                'Stake_PLN', 'BTTS_pot', 'Corners_pot', 'O25_pot',
                'Wynik', 'Rezultat', 'Corners', 'Kartki', 'Profit_PLN']

    for lg, (lg_name, scorer_dir) in league_abbr_map.items():
        # Tylko combined dla ML (liga nie trafia do CSV)
        lg_ml = [t for t in all_tips if t['league'] == lg and t['model_type'] == 'combined']
        # GPT rows dla tej ligi
        lg_gpt = [r for r in gpt_csv_rows if r.get('_liga_key') == lg]

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
                'Pinnacle_odds': t['pinnacle_odds'],
                'EV[%]':         t['ev'],
                'Stake_PLN':     t['stake'],
                'BTTS_pot':      t.get('btts_pot', ''),
                'Corners_pot':   t.get('corners_pot', ''),
                'O25_pot':       t.get('o25_pot', ''),
                'Wynik':         '',
                'Rezultat':      '',
                'Corners':       '',
                'Kartki':        '',
                'Profit_PLN':    '',
            })
        for r in lg_gpt:
            rows.append({k: r[k] for k in csv_cols})

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
                    'h2h_from_cache':   t['h2h_from_cache'],
                    'h2h_matches_count': t['h2h_count'],
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
                                'corners_over' if 'over' in t['typ'].lower() else 'corners_under', {})
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
        debug_path = os.path.join(REPORTS_DIR, f'debug_nordic_{date_str}_{ts}.json')
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(debug, f, ensure_ascii=False, indent=2, default=str)
        print(f'  Debug: {debug_path}')


if __name__ == '__main__':
    main()
