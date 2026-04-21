import sys
import io
import os
import json
import pickle
import argparse
import datetime
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss

from nordic_config import (
    DATA_DIR, REPORTS_DIR,
    MODELS_ALLSV, MODELS_ELITE, MODELS_VEIKK, MODELS_COMBINED,
)

# ── FEATURE SETS ──────────────────────────────────────────────────────────────

RESULT_FEATURES = [
    'odds_ft_1', 'odds_ft_x', 'odds_ft_2',
    'implied_home_norm', 'implied_away_norm',
    'odds_diff_home_away',
    'is_allsvenskan', 'is_eliteserien', 'is_veikkausliiga',
    'home_last5_win_rate', 'away_last5_win_rate',
    'home_last5_goals_scored', 'away_last5_goals_scored',
    'diff_last5_goals_scored', 'diff_last5_win_rate',
    'team_a_xg_prematch', 'team_b_xg_prematch',
    'pre_match_home_ppg', 'pre_match_away_ppg',
]

BTTS_FEATURES = [
    'odds_btts_yes', 'odds_btts_no',
    'implied_home_norm', 'implied_away_norm',
    'is_allsvenskan', 'is_eliteserien', 'is_veikkausliiga',
    'home_last5_btts_rate', 'away_last5_btts_rate',
    'home_last5_goals_scored', 'away_last5_goals_scored',
    'home_last5_goals_conceded', 'away_last5_goals_conceded',
    'diff_last5_goals_scored',
    'btts_potential', 'o25_potential', 'avg_potential',
    'team_a_xg_prematch', 'team_b_xg_prematch',
    'pre_match_home_ppg', 'pre_match_away_ppg',
]

OVER25_FEATURES = [
    'odds_ft_over25', 'odds_ft_under25',
    'implied_home_norm', 'implied_away_norm',
    'is_allsvenskan', 'is_eliteserien', 'is_veikkausliiga',
    'home_last5_over25_rate', 'away_last5_over25_rate',
    'home_last5_goals_scored', 'away_last5_goals_scored',
    'home_last5_goals_conceded', 'away_last5_goals_conceded',
    'diff_last5_goals_scored',
    'o25_potential', 'btts_potential', 'avg_potential',
    'team_a_xg_prematch', 'team_b_xg_prematch',
    'pre_match_home_ppg', 'pre_match_away_ppg',
]

CORNERS_FEATURES = [
    'odds_corners_over_95', 'odds_corners_under_95',
    'odds_ft_1', 'odds_ft_2',
    'implied_home_norm', 'implied_away_norm',
    'is_allsvenskan', 'is_eliteserien', 'is_veikkausliiga',
    'home_last5_corners_avg', 'away_last5_corners_avg',
    'diff_last5_corners_avg',
    'corners_potential', 'corners_o95_potential', 'avg_potential',
    'team_a_xg_prematch', 'team_b_xg_prematch',
]

MODELS_CONFIG = {
    'result_home': {'features': RESULT_FEATURES,  'target': 'target_result_home', 'arch': 'regularized'},
    'result_away': {'features': RESULT_FEATURES,  'target': 'target_result_away', 'arch': 'regularized'},
    'btts':        {'features': BTTS_FEATURES,    'target': 'target_btts',        'arch': 'regularized'},
    'over25':      {'features': OVER25_FEATURES,  'target': 'target_over25',      'arch': 'regularized'},
    'corners':     {'features': CORNERS_FEATURES, 'target': 'target_corners_over95', 'arch': 'regularized'},
}

MODELS_ARCH = {
    'standard': {
        'max_iter': 300, 'learning_rate': 0.05,
        'max_depth': 4, 'min_samples_leaf': 20,
        'l2_regularization': 0.0,
    },
    'regularized': {
        'max_iter': 100, 'learning_rate': 0.05,
        'max_depth': 3, 'min_samples_leaf': 40,
        'l2_regularization': 1.0,
    },
}

LEAGUE_SETS = {
    'allsvenskan':    ('allsvenskan',  MODELS_ALLSV),
    'eliteserien':    ('eliteserien',  MODELS_ELITE),
    'veikkausliiga':  ('veikkausliiga', MODELS_VEIKK),
    'nordic_combined': (None,          MODELS_COMBINED),
}

# ── POMOCNICZE ────────────────────────────────────────────────────────────────

def filter_features(feature_list, df):
    return [f for f in feature_list if f in df.columns]


def get_feature_importance(calibrated_model, feature_names):
    try:
        importances = np.mean(
            [est.estimator.feature_importances_
             for est in calibrated_model.calibrated_classifiers_],
            axis=0,
        )
        return {f: round(float(v), 4) for f, v in zip(feature_names, importances)}
    except Exception:
        return {}


def train_single(model_name, cfg, df, arch_override=None):
    arch_name = arch_override or cfg['arch']
    arch_params = MODELS_ARCH[arch_name]
    features = filter_features(cfg['features'], df)
    target = cfg['target']

    X = df[features].copy()
    y = df[target].copy()

    tscv = TimeSeriesSplit(n_splits=3)
    cv_aucs = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        base = HistGradientBoostingClassifier(**arch_params, random_state=42)
        base.fit(X_tr, y_tr)
        if len(y_val.unique()) < 2:
            continue
        cv_aucs.append(roc_auc_score(y_val, base.predict_proba(X_val)[:, 1]))

    # Finalny model kalibrowany na pełnym zbiorze
    base_final = HistGradientBoostingClassifier(**arch_params, random_state=42)
    calibrated = CalibratedClassifierCV(
        base_final, cv=TimeSeriesSplit(n_splits=3), method='isotonic',
    )
    calibrated.fit(X, y)

    proba_train = calibrated.predict_proba(X)[:, 1]
    auc_train = roc_auc_score(y, proba_train) if len(y.unique()) > 1 else float('nan')
    brier = brier_score_loss(y, proba_train)

    return {
        'model':       calibrated,
        'features':    features,
        'n_train':     len(X),
        'pos_rate':    float(y.mean()),
        'auc_train':   round(auc_train, 4),
        'cv_scores':   [round(v, 4) for v in cv_aucs],
        'cv_mean':     round(float(np.mean(cv_aucs)), 4) if cv_aucs else float('nan'),
        'cv_std':      round(float(np.std(cv_aucs)), 4)  if cv_aucs else float('nan'),
        'brier':       round(brier, 4),
        'importance':  get_feature_importance(calibrated, features),
        'imputation':  {f: round(float(X[f].median()), 4) for f in features},
    }


def save_model(result, model_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f'nordic_{model_name}_model.pkl')
    imputer_path = os.path.join(out_dir, f'nordic_{model_name}_imputer.json')
    with open(model_path, 'wb') as f:
        pickle.dump(result['model'], f)
    imputer_data = {
        'features': result['features'],
        'imputation_values': result['imputation'],
    }
    with open(imputer_path, 'w', encoding='utf-8') as f:
        json.dump(imputer_data, f, ensure_ascii=False, indent=2)
    return model_path


def print_league_table(league_label, n, rows):
    print(f"\n  === {league_label.upper()} (n={n}) ===")
    print(f"  {'Model':12} │ {'pos_rate':8} │ {'AUC_train':9} │ {'AUC_cv':14} │ {'Brier':6}")
    print(f"  {'─'*12}─┼─{'─'*8}─┼─{'─'*9}─┼─{'─'*14}─┼─{'─'*6}")
    for row in rows:
        print(
            f"  {row['name']:12} │ {row['pos_rate']*100:6.1f}%  │"
            f"    {row['auc_train']:.3f}  │"
            f" {row['cv_mean']:.3f}±{row['cv_std']:.3f}  │"
            f" {row['brier']:.3f}"
        )

# ── TRYB COMPARE ──────────────────────────────────────────────────────────────

def run_compare(df, args_model):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    report = {}
    print("\n╔══ STANDARD vs REGULARIZED ══════════════════════╗")
    print(f"  {'Model':12} │ {'AUC_cv standard':18} │ {'AUC_cv regular':16} │ Winner")
    print(f"  {'─'*12}─┼─{'─'*18}─┼─{'─'*16}─┼─{'─'*7}")

    for model_name, cfg in MODELS_CONFIG.items():
        if args_model and model_name != args_model:
            continue
        r_std  = train_single(model_name, cfg, df, arch_override='standard')
        r_reg  = train_single(model_name, cfg, df, arch_override='regularized')
        winner = 'standard' if r_std['cv_mean'] >= r_reg['cv_mean'] else 'regular'
        print(
            f"  {model_name:12} │"
            f" {r_std['cv_mean']:.3f} ± {r_std['cv_std']:.3f}     │"
            f" {r_reg['cv_mean']:.3f} ± {r_reg['cv_std']:.3f}  │ {winner}"
        )
        report[model_name] = {
            'standard':   {'auc_cv_mean': r_std['cv_mean'], 'auc_cv_std': r_std['cv_std']},
            'regularized': {'auc_cv_mean': r_reg['cv_mean'], 'auc_cv_std': r_reg['cv_std']},
            'winner': winner,
        }

    print("╚══════════════════════════════════════════════════╝")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out_path = os.path.join(REPORTS_DIR, f'regularization_{ts}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'timestamp': ts, 'results': report}, f, ensure_ascii=False, indent=2)
    print(f"\n  Zapisano: {out_path}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',   action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--model',   type=str, default=None,
                        choices=list(MODELS_CONFIG.keys()),
                        help='Trenuj tylko wybrany model')
    args = parser.parse_args()

    dataset_path = os.path.join(DATA_DIR, 'training_dataset.csv')
    if not os.path.exists(dataset_path):
        print("BŁĄD: brak data/training_dataset.csv — uruchom najpierw build_dataset.py")
        sys.exit(1)

    df_all = pd.read_csv(dataset_path, encoding='utf-8-sig')
    df_all = df_all.sort_values('date_unix').reset_index(drop=True)

    if args.compare:
        run_compare(df_all, args.model)
        return

    print("╔══ TRENING MODELI — Nordic 2026 ══════════════════╗")

    debug_report = {
        'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
        'datasets': {},
        'models': {},
    }

    for league_key, (league_filter, out_dir) in LEAGUE_SETS.items():
        if league_filter:
            df_league = df_all[df_all['league'] == league_filter].copy().reset_index(drop=True)
        else:
            df_league = df_all.copy().reset_index(drop=True)

        n = len(df_league)
        debug_report['datasets'][league_key] = {'n_train': n}
        debug_report['models'][league_key] = {}

        table_rows = []
        for model_name, cfg in MODELS_CONFIG.items():
            if args.model and model_name != args.model:
                continue
            result = train_single(model_name, cfg, df_league)
            model_path = save_model(result, model_name, out_dir)

            table_rows.append({
                'name':      model_name,
                'pos_rate':  result['pos_rate'],
                'auc_train': result['auc_train'],
                'cv_mean':   result['cv_mean'],
                'cv_std':    result['cv_std'],
                'brier':     result['brier'],
            })

            debug_report['models'][league_key][model_name] = {
                'n_train':            result['n_train'],
                'positive_rate_pct':  round(result['pos_rate'] * 100, 1),
                'auc_train':          result['auc_train'],
                'auc_cv_scores':      result['cv_scores'],
                'auc_cv_mean':        result['cv_mean'],
                'auc_cv_std':         result['cv_std'],
                'brier_score':        result['brier'],
                'feature_importance': result['importance'],
                'imputation_values':  result['imputation'],
                'model_path':         model_path,
                'warnings':           [],
            }

        label_map = {
            'allsvenskan':    'ALLSVENSKAN',
            'eliteserien':    'ELITESERIEN',
            'veikkausliiga':  'VEIKKAUSLIIGA',
            'nordic_combined': 'NORDIC COMBINED',
        }
        print_league_table(label_map[league_key], n, table_rows)

    print("\n╚══════════════════════════════════════════════════╝\n")

    if args.debug:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        ts = debug_report['timestamp']
        debug_path = os.path.join(REPORTS_DIR, f'debug_training_{ts}.json')
        with open(debug_path, 'w', encoding='utf-8') as f:
            json.dump(debug_report, f, ensure_ascii=False, indent=2)
        print(f"  Debug: {debug_path}")


if __name__ == '__main__':
    main()
