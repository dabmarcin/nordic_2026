#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate portfolio CSV files - remove signals with odds outside their brackets
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

from nordic_config import PORTFOLIO_DIR, PORTFOLIO_SIGNALS

def get_odds_range(signal_id):
    """Extract min/max odds from signal definition"""
    if signal_id not in PORTFOLIO_SIGNALS:
        return None, None

    signal = PORTFOLIO_SIGNALS[signal_id]

    # Scorer-based signals have no odds constraint
    if signal.get("model_src") == "scorer":
        return None, None

    # Parse condition string
    condition = signal.get("condition", "")

    # Format: "odds_col >= X and odds_col <= Y" or "odds_col >= X"
    if ">=" in condition and "<=" in condition:
        # Range: X <= odds <= Y
        parts = condition.split("and")
        min_str = parts[0].strip().split(">=")[1].strip()
        max_str = parts[1].strip().split("<=")[1].strip()
        return float(min_str), float(max_str)
    elif ">=" in condition:
        # Minimum only: odds >= X
        min_str = condition.split(">=")[1].strip()
        return float(min_str), None

    return None, None

def validate_signal(signal_id, kurs):
    """Check if odds are within signal's defined range"""
    min_odds, max_odds = get_odds_range(signal_id)

    if min_odds is None and max_odds is None:
        return True  # No constraint

    if min_odds is not None and kurs < min_odds:
        return False
    if max_odds is not None and kurs > max_odds:
        return False

    return True

def clean_portfolio_files():
    """Remove invalid signals from all portfolio files"""

    portfolio_files = sorted(glob.glob(os.path.join(PORTFOLIO_DIR, "portfolio_*.csv")))

    print(f"📋 Validating {len(portfolio_files)} portfolio files...\n")

    total_removed = 0
    files_updated = 0

    for portfolio_path in portfolio_files:
        basename = os.path.basename(portfolio_path)

        try:
            df = pd.read_csv(portfolio_path, encoding='utf-8-sig')
        except Exception as e:
            print(f"❌ Error reading {basename}: {e}")
            continue

        original_count = len(df)
        invalid_rows = []
        valid_df = df.copy()

        # Check each row
        for idx, row in df.iterrows():
            signal_id = row.get('Signal_ID', '')
            kurs = row.get('Kurs', 0)

            # Skip scorer-based and non-rule signals
            if not signal_id or signal_id not in PORTFOLIO_SIGNALS:
                continue

            signal = PORTFOLIO_SIGNALS[signal_id]
            if signal.get("model_src") != "rule":
                continue

            # Validate
            if not validate_signal(signal_id, float(kurs)):
                invalid_rows.append(idx)

        # Remove invalid rows
        if invalid_rows:
            valid_df = df.drop(invalid_rows).reset_index(drop=True)
            removed = len(invalid_rows)
            total_removed += removed
            files_updated += 1

            # Save
            valid_df.to_csv(portfolio_path, index=False, encoding='utf-8-sig')

            print(f"📄 {basename}")
            print(f"   Removed: {removed} invalid signals (had {original_count}, now {len(valid_df)})")

            for idx in invalid_rows:
                row = df.loc[idx]
                sig_id = row['Signal_ID']
                kurs = row['Kurs']
                min_o, max_o = get_odds_range(sig_id)

                if min_o and max_o:
                    print(f"     - {sig_id}: Kurs {kurs} outside {min_o}-{max_o}")
                elif min_o:
                    print(f"     - {sig_id}: Kurs {kurs} below min {min_o}")
                else:
                    print(f"     - {sig_id}: Kurs {kurs}")

    print(f"\n✅ Cleaned {files_updated} portfolio files")
    print(f"📊 Total invalid signals removed: {total_removed}\n")

    # Summary
    print("Signal validation rules:")
    for sig_id, signal in PORTFOLIO_SIGNALS.items():
        if signal.get("model_src") == "rule":
            min_o, max_o = get_odds_range(sig_id)
            if min_o and max_o:
                print(f"  {sig_id}: {min_o} <= odds <= {max_o}")
            elif min_o:
                print(f"  {sig_id}: odds >= {min_o}")
            else:
                print(f"  {sig_id}: no constraint")

if __name__ == "__main__":
    clean_portfolio_files()
