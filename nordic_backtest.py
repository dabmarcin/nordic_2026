# -*- coding: utf-8 -*-
"""
nordic_backtest.py — Backtest betting strategies on 2022–2025 historical data
Wrapper stdout UTF-8
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import glob
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from nordic_config import (
    ALLSVENSKAN_DIR, ELITESERIEN_DIR, VEIKKAUSLIIGA_DIR,
    REPORTS_DIR,
    ALL_HISTORICAL,
)

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

STRATEGIES = {
    'S1': {
        'name': '1 (Home Win)',
        'odds_col': 'odds_ft_1',
        'target_col': 'target_1',
        'min_odds': 1.10,
        'max_odds': 5.00,
    },
    'S2': {
        'name': '2 (Away Win)',
        'odds_col': 'odds_ft_2',
        'target_col': 'target_2',
        'min_odds': 1.10,
        'max_odds': 5.00,
    },
    'S3': {
        'name': 'X2 (Draw or Away)',
        'odds_col': 'odds_doublechance_x2',
        'target_col': 'target_x2',
        'min_odds': 1.10,
        'max_odds': 2.50,
    },
    'S4': {
        'name': '1X (Home or Draw)',
        'odds_col': 'odds_doublechance_1x',
        'target_col': 'target_1x',
        'min_odds': 1.10,
        'max_odds': 2.50,
    },
    'S5': {
        'name': 'Under 2.5',
        'odds_col': 'odds_ft_under25',
        'target_col': 'target_under25',
        'min_odds': 1.10,
        'max_odds': 3.50,
    },
    'S6': {
        'name': 'Over 2.5',
        'odds_col': 'odds_ft_over25',
        'target_col': 'target_over25',
        'min_odds': 1.10,
        'max_odds': 3.50,
    },
    'S7': {
        'name': 'Under 9.5 C',
        'odds_col': 'odds_corners_under_95',
        'target_col': 'target_under95c',
        'min_odds': 1.10,
        'max_odds': 3.50,
    },
    'S8': {
        'name': 'Over 9.5 C',
        'odds_col': 'odds_corners_over_95',
        'target_col': 'target_over95c',
        'min_odds': 1.10,
        'max_odds': 3.50,
    },
    'S9': {
        'name': 'BTTS Yes',
        'odds_col': 'odds_btts_yes',
        'target_col': 'target_btts',
        'min_odds': 1.10,
        'max_odds': 3.50,
    },
    'S10': {
        'name': 'BTTS No',
        'odds_col': 'odds_btts_no',
        'target_col': 'target_btts_no',
        'min_odds': 1.10,
        'max_odds': 3.50,
    },
}

# ── DATA LOADING ───────────────────────────────────────────────────────────────

def load_historical_data(sezon: int = None) -> Tuple[pd.DataFrame, List[int]]:
    """Load all complete matches from 2022–2026 data.

    Args:
        sezon: Optional season year filter (e.g. 2025 or 2026)

    Returns:
        Tuple of (DataFrame, allowed_season_ids)
    """

    # Build list of allowed season IDs if sezon filter is specified
    allowed_ids = None
    if sezon is not None:
        allowed_ids = [s["id"] for s in ALL_HISTORICAL if s["year"] == sezon]

    dfs = []

    # If loading 2026 (current season), use data/current/{league}_matches_2026.csv
    if sezon == 2026:
        current_dir = os.path.join(os.path.dirname(__file__), 'data', 'current')
        if not os.path.exists(current_dir):
            print(f"  ✗ Data directory not found: {current_dir}")
            return pd.DataFrame(), allowed_ids if allowed_ids else []

        for league_name in ['allsvenskan', 'eliteserien', 'veikkausliiga']:
            csv_file = os.path.join(current_dir, f'{league_name}_matches_2026.csv')

            if not os.path.exists(csv_file):
                print(f"  ✗ File not found: {csv_file}")
                continue

            try:
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
                # Filter only complete matches
                df = df[df['status'] == 'complete'].copy()
                df['league'] = league_name
                dfs.append(df)
                print(f"  ✓ Loaded {len(df)} matches from {os.path.basename(csv_file)}")
            except Exception as e:
                print(f"  ✗ Error loading {csv_file}: {e}")
    else:
        # Load from historical data (2022-2025)
        league_dirs = {
            'allsvenskan': ALLSVENSKAN_DIR,
            'eliteserien': ELITESERIEN_DIR,
            'veikkausliiga': VEIKKAUSLIIGA_DIR,
        }

        for league_name, league_dir in league_dirs.items():
            csv_files = glob.glob(os.path.join(league_dir, 'advanced_league_matches_*.csv'))

            for csv_file in csv_files:
                # Filter by filename if season is specified
                if allowed_ids:
                    fname = os.path.basename(csv_file)
                    # Check if filename contains one of the allowed season IDs
                    match = any(str(sid) in fname for sid in allowed_ids)
                    if not match:
                        continue

                try:
                    df = pd.read_csv(csv_file, encoding='utf-8-sig')
                    # Filter only complete matches
                    df = df[df['status'] == 'complete'].copy()
                    df['league'] = league_name
                    dfs.append(df)
                    print(f"  ✓ Loaded {len(df)} matches from {os.path.basename(csv_file)}")
                except Exception as e:
                    print(f"  ✗ Error loading {csv_file}: {e}")

    if not dfs:
        print("ERROR: No data loaded!")
        return pd.DataFrame(), allowed_ids if allowed_ids else []

    data = pd.concat(dfs, ignore_index=True)

    if sezon is not None:
        print(f"\n  Filtered by sezon {sezon}: {len(data)} complete matches")
    else:
        print(f"\n  Total: {len(data)} complete matches loaded")

    return data, allowed_ids if allowed_ids else []

# ── TARGET CALCULATION ─────────────────────────────────────────────────────────

def calculate_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate betting targets for each match."""

    df = df.copy()

    # 1: Home win
    df['target_1'] = (df['homeGoalCount'] > df['awayGoalCount']).astype(int)

    # 2: Away win
    df['target_2'] = (df['awayGoalCount'] > df['homeGoalCount']).astype(int)

    # X2: Away win or draw
    df['target_x2'] = (df['awayGoalCount'] >= df['homeGoalCount']).astype(int)

    # 1X: Home win or draw
    df['target_1x'] = (df['homeGoalCount'] >= df['awayGoalCount']).astype(int)

    # Under 2.5 goals
    df['target_under25'] = (df['totalGoalCount'] <= 2).astype(int)

    # Over 2.5 goals
    df['target_over25'] = (df['totalGoalCount'] > 2).astype(int)

    # Under 9.5 corners
    df['target_under95c'] = (df['totalCornerCount'] <= 9).astype(int)

    # Over 9.5 corners
    df['target_over95c'] = (df['totalCornerCount'] > 9).astype(int)

    # BTTS Yes
    df['target_btts'] = ((df['btts'] == True) | (df['btts'] == 1)).astype(int)

    # BTTS No
    df['target_btts_no'] = ((df['btts'] == False) | (df['btts'] == 0)).astype(int)

    return df

# ── BACKTEST CALCULATION ───────────────────────────────────────────────────────

def backtest_strategy(df: pd.DataFrame, strategy_key: str, stake: float) -> Dict:
    """Backtest a single strategy on the dataset."""

    strategy = STRATEGIES[strategy_key]
    odds_col = strategy['odds_col']
    target_col = strategy['target_col']
    min_odds = strategy['min_odds']
    max_odds = strategy['max_odds']

    # Filter: odds in valid range and not null/zero
    valid_rows = (
        (df[odds_col] > 0) &
        (df[odds_col].notna()) &
        (df[odds_col] >= min_odds) &
        (df[odds_col] <= max_odds)
    )

    bets = df[valid_rows].copy()

    if len(bets) == 0:
        return {
            'strategy': strategy_key,
            'name': strategy['name'],
            'n': 0,
            'n_won': 0,
            'win_rate': 0.0,
            'avg_odds': 0.0,
            'total_stake': 0.0,
            'total_profit': 0.0,
            'roi_pct': 0.0,
        }

    # Calculate P&L
    wins = bets[target_col].sum()
    losses = len(bets) - wins

    profits = np.where(
        bets[target_col] == 1,
        (bets[odds_col] - 1) * stake,
        -stake
    )

    total_stake = len(bets) * stake
    total_profit = profits.sum()
    roi = (total_profit / total_stake * 100) if total_stake > 0 else 0.0
    win_rate = (wins / len(bets) * 100) if len(bets) > 0 else 0.0
    avg_odds = bets[odds_col].mean()

    return {
        'strategy': strategy_key,
        'name': strategy['name'],
        'n': len(bets),
        'n_won': int(wins),
        'win_rate': win_rate,
        'avg_odds': avg_odds,
        'total_stake': total_stake,
        'total_profit': total_profit,
        'roi_pct': roi,
    }

# ── LEAGUE-SPECIFIC BACKTEST ───────────────────────────────────────────────────

def backtest_by_league(df: pd.DataFrame, strategy_key: str, stake: float) -> Dict[str, Dict]:
    """Backtest strategy per league."""

    result = {}

    for league in ['allsvenskan', 'eliteserien', 'veikkausliiga']:
        league_df = df[df['league'] == league]

        if len(league_df) == 0:
            result[league] = {
                'league': league,
                'n': 0,
                'n_won': 0,
                'win_rate': 0.0,
                'avg_odds': 0.0,
                'total_stake': 0.0,
                'total_profit': 0.0,
                'roi_pct': 0.0,
            }
        else:
            result[league] = backtest_strategy(league_df, strategy_key, stake)

    return result

# ── ODDS RANGE ANALYSIS ────────────────────────────────────────────────────────

def analyze_odds_ranges(df: pd.DataFrame, strategy_key: str, stake: float) -> List[Dict]:
    """Analyze performance by odds ranges (for S4: Under 9.5 Corners)."""

    strategy = STRATEGIES[strategy_key]
    odds_col = strategy['odds_col']
    target_col = strategy['target_col']

    valid_rows = (
        (df[odds_col] > 0) &
        (df[odds_col].notna())
    )

    bets = df[valid_rows].copy()

    ranges = [
        (1.10, 1.50, '1.10 – 1.50'),
        (1.51, 2.00, '1.51 – 2.00'),
        (2.00, 10.0, '2.00+'),
    ]

    results = []

    for min_r, max_r, label in ranges:
        range_bets = bets[(bets[odds_col] >= min_r) & (bets[odds_col] <= max_r)]

        if len(range_bets) == 0:
            results.append({
                'range': label,
                'n': 0,
                'win_rate': 0.0,
                'roi_pct': 0.0,
            })
            continue

        wins = range_bets[target_col].sum()
        losses = len(range_bets) - wins
        win_rate = (wins / len(range_bets) * 100)

        profits = np.where(
            range_bets[target_col] == 1,
            (range_bets[odds_col] - 1) * stake,
            -stake
        )

        total_stake = len(range_bets) * stake
        total_profit = profits.sum()
        roi = (total_profit / total_stake * 100) if total_stake > 0 else 0.0

        results.append({
            'range': label,
            'n': len(range_bets),
            'win_rate': win_rate,
            'roi_pct': roi,
        })

    return results

# ── OUTPUT FORMATTING ──────────────────────────────────────────────────────────

def format_table_row(cols: List, widths: List) -> str:
    """Format a row for a table."""
    parts = []
    for col, width in zip(cols, widths):
        parts.append(str(col).ljust(width) if isinstance(col, str) else f"{col:>{width}}")
    return ' │ '.join(parts)

def print_results(all_results: Dict, by_league: Dict[str, Dict], odds_analysis: Dict[str, List], total_matches: int, stake: float, sezon: int = None):
    """Print formatted backtest results."""

    print("\n")
    print("╔" + "═" * 120 + "╗")
    title = f"NORDIC BACKTEST — {sezon}" if sezon else "NORDIC BACKTEST — 2022–2026"
    print("║ " + title.center(118) + " ║")
    print("╚" + "═" * 120 + "╝")
    print()
    sezon_label = f"{sezon}" if sezon else "2022–2026"
    print(f"  Sezon: {sezon_label}")
    print(f"  Dane: {total_matches} meczów complete")
    print(f"  Stake: {stake:.0f} PLN flat per zakład")
    print()

    # ── OVERALL RESULTS ────────────────────────────────────────────────────────
    print("  ══ WYNIKI OGÓLNE ═══════════════════════════════════════════════════════════════════════════════════════════════")
    print()

    header = ['Strategia', 'N', 'WR', 'Śr.Kurs', 'ROI', 'Profit']
    widths = [25, 6, 6, 9, 9, 12]

    print("  " + format_table_row(header, widths))
    print("  " + "─" * 128)

    for key in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']:
        r = all_results[key]
        profit_str = f"+{r['total_profit']:.0f}" if r['total_profit'] >= 0 else f"{r['total_profit']:.0f}"
        roi_str = f"+{r['roi_pct']:.1f}%" if r['roi_pct'] >= 0 else f"{r['roi_pct']:.1f}%"

        row = [
            f"{key}: {r['name']}",
            f"{r['n']}",
            f"{r['win_rate']:.0f}%",
            f"{r['avg_odds']:.2f}",
            roi_str,
            profit_str,
        ]
        print("  " + format_table_row(row, widths))

    # ── PER-LEAGUE RESULTS (ALL STRATEGIES) ────────────────────────────────────
    for strategy_key in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']:
        strategy = STRATEGIES[strategy_key]
        league_results = by_league[strategy_key]

        print()
        print(f"  ══ PER LIGA — {strategy_key} ({strategy['name']}) ═══════════════════════════════════════════════════════════════════════════════")
        print()

        header = ['Liga', 'N', 'WR', 'Śr.Kurs', 'ROI', 'Profit']
        print("  " + format_table_row(header, [15, 6, 6, 9, 9, 12]))
        print("  " + "─" * 118)

        for league in ['allsvenskan', 'eliteserien', 'veikkausliiga']:
            r = league_results[league]
            if r['n'] == 0:
                continue

            profit_str = f"+{r['total_profit']:.0f}" if r['total_profit'] >= 0 else f"{r['total_profit']:.0f}"
            roi_str = f"+{r['roi_pct']:.1f}%" if r['roi_pct'] >= 0 else f"{r['roi_pct']:.1f}%"
            league_label = {'allsvenskan': 'Allsvenskan', 'eliteserien': 'Eliteserien', 'veikkausliiga': 'Veikkausliiga'}[league]

            row = [
                league_label,
                f"{r['n']}",
                f"{r['win_rate']:.0f}%",
                f"{r['avg_odds']:.2f}",
                roi_str,
                profit_str,
            ]
            print("  " + format_table_row(row, [15, 6, 6, 9, 9, 12]))

    # ── ODDS RANGE ANALYSIS (S7 ONLY) ──────────────────────────────────────────
    print()
    print(f"  ══ ANALIZA KURSOWA — S7 (Under 9.5 C) ═════════════════════════════════════════════════════════════════════════")
    print()

    header = ['Przedział', 'N', 'WR', 'ROI']
    widths = [15, 6, 6, 9]
    print("  " + format_table_row(header, widths))
    print("  " + "─" * 50)

    for r in odds_analysis['S7']:
        roi_str = f"+{r['roi_pct']:.1f}%" if r['roi_pct'] >= 0 else f"{r['roi_pct']:.1f}%"
        row = [
            r['range'],
            f"{r['n']}",
            f"{r['win_rate']:.0f}%",
            roi_str,
        ]
        print("  " + format_table_row(row, widths))

    print()
    print("╚" + "═" * 120 + "╝")
    print()

# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Backtest Nordic betting strategies 2022–2026')
    parser.add_argument('--liga', default='all', choices=['all', 'allsvenskan', 'eliteserien', 'veikkausliiga'],
                        help='Filter by league (default: all)')
    parser.add_argument('--stake', type=float, default=100.0, help='Flat stake per bet in PLN (default: 100)')
    parser.add_argument('--sezon', type=int, default=None, help='Filter only given season year (e.g. 2025 or 2026 for current season)')

    args = parser.parse_args()

    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║        NORDIC BACKTEST — Loading historical data...          ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    # Load data
    df, allowed_ids = load_historical_data(sezon=args.sezon)

    # Print season confirmation if filtered
    if allowed_ids:
        print(f"\n  Dozwolone season_id dla sezonu {args.sezon}:")
        for s in ALL_HISTORICAL:
            if s["id"] in allowed_ids:
                # Extract league name from season name
                season_name = s['name'].lower()
                if 'allsv' in season_name:
                    league = 'Allsvenskan'
                elif 'elite' in season_name:
                    league = 'Eliteserien'
                elif 'veikk' in season_name:
                    league = 'Veikkausliiga'
                else:
                    league = 'Unknown'
                print(f"    {league}: {s['id']}")
        print()

    if df.empty:
        print("ERROR: Failed to load data. Exiting.")
        return

    # Filter by league if specified
    if args.liga != 'all':
        df = df[df['league'] == args.liga]

    # Calculate targets
    df = calculate_targets(df)

    if df.empty:
        print(f"ERROR: No matches found after filtering. Exiting.")
        return

    total_matches = len(df)
    stake = args.stake

    # Run backtests
    all_results = {}
    by_league = {}
    odds_analysis = {}

    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║                  Running backtest analysis...                 ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    for strategy_key in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']:
        all_results[strategy_key] = backtest_strategy(df, strategy_key, stake)
        by_league[strategy_key] = backtest_by_league(df, strategy_key, stake)
        odds_analysis[strategy_key] = analyze_odds_ranges(df, strategy_key, stake)

        strategy = STRATEGIES[strategy_key]
        print(f"  ✓ {strategy_key}: {strategy['name']:<25} | {all_results[strategy_key]['n']:>4} bets | WR: {all_results[strategy_key]['win_rate']:>5.1f}% | ROI: {all_results[strategy_key]['roi_pct']:>6.1f}%")

    print()

    # Print formatted results
    print_results(all_results, by_league, odds_analysis, total_matches, stake, sezon=args.sezon)

    # Save JSON
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d')
    report_file = os.path.join(REPORTS_DIR, f'backtest_{timestamp}.json')

    report_data = {
        'timestamp': datetime.now().isoformat(),
        'total_matches': total_matches,
        'stake_pln': stake,
        'league_filter': args.liga,
        'all_results': {k: v for k, v in all_results.items()},
        'by_league': by_league,
        'odds_analysis': odds_analysis,
    }

    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Results saved to {report_file}")
    except Exception as e:
        print(f"✗ Error saving results: {e}")

if __name__ == '__main__':
    main()
