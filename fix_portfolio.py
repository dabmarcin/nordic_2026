#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix Portfolio - Naprawia wyniki w istniejących plikach portfolio
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import glob
import os
from pathlib import Path

# Add script directory to path
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from nordic_config import PORTFOLIO_DIR
from online_settle import repair_portfolio_signals

def main():
    """Naprawia wszystkie pliki portfolio"""
    portfolio_files = sorted(glob.glob(os.path.join(PORTFOLIO_DIR, "portfolio_*.csv")), reverse=True)

    if not portfolio_files:
        print("❌ Brak plików portfolio do naprawienia")
        return

    print(f"🔧 Znaleziono {len(portfolio_files)} plik(i) portfolio")

    total_updated = 0
    for file_path in portfolio_files:
        print(f"\n📄 {os.path.basename(file_path)}...")
        updated, msg = repair_portfolio_signals(file_path)
        print(f"   {msg}")
        total_updated += updated

    print(f"\n✅ Naprawiono łącznie {total_updated} zakładów")

if __name__ == "__main__":
    main()
