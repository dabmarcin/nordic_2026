# -*- coding: utf-8 -*-
"""
Pobiera mecze dla konkretnych dat (29-05, 30-05, 31-05 2026)
i zapisuje w data/daily/ jako today_matches_2026-MM-DD.csv.

Wykorzystuje logikę _fetch_daily_matches z fetch_data.py.
"""
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
import pandas as pd

from nordic_config import (
    DAILY_DIR,
    ALLSVENSKAN_2026_ID, ELITESERIEN_2026_ID, VEIKKAUSLIIGA_2026_ID,
    MLS_2026_ID, CSL_2026_ID,
)
from fetch_data import _fetch_daily_matches


TARGET_DATES = [
    "2026-05-29",
    "2026-05-30",
    "2026-05-31",
]


def main():
    print("=" * 50)
    print("POBIERANIE MECZÓW DLA WYBRANYCH DAT")
    print("=" * 50)

    summary = {}

    for date_str in TARGET_DATES:
        print(f"\n  Pobieranie meczów: {date_str}...")
        matches = _fetch_daily_matches(date_str, date_str)

        fname = f"today_matches_{date_str}.csv"
        csv_path = os.path.join(DAILY_DIR, fname)
        df = pd.DataFrame(matches)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        n_allsv = n_elite = n_veikk = n_mls = n_csl = n_other = 0
        for m in matches:
            cid = m.get("competition_id") or m.get("season_id")
            try:
                cid = int(cid)
            except (TypeError, ValueError):
                cid = None
            if cid == ALLSVENSKAN_2026_ID:
                n_allsv += 1
            elif cid == ELITESERIEN_2026_ID:
                n_elite += 1
            elif cid == VEIKKAUSLIIGA_2026_ID:
                n_veikk += 1
            elif cid == MLS_2026_ID:
                n_mls += 1
            elif cid == CSL_2026_ID:
                n_csl += 1
            else:
                n_other += 1

        summary[date_str] = {
            "total": len(matches),
            "allsvenskan": n_allsv,
            "eliteserien": n_elite,
            "veikkausliiga": n_veikk,
            "mls": n_mls,
            "csl": n_csl,
            "other": n_other,
        }
        print(f"  ✅ {date_str} — {len(matches)} meczów → {csv_path}")

    print()
    print("══════════════════════════════════════════")
    print("PODSUMOWANIE")
    print("──────────────────────────────────────────")
    for date_str, s in summary.items():
        print(f"  {date_str}: {s['total']} meczów łącznie")
        print(f"    Allsvenskan:   {s['allsvenskan']}")
        print(f"    Eliteserien:   {s['eliteserien']}")
        print(f"    Veikkausliiga: {s['veikkausliiga']}")
        print(f"    MLS:           {s['mls']}")
        print(f"    CSL:           {s['csl']}")
        print(f"    Inne:          {s['other']}")
    print("══════════════════════════════════════════")


if __name__ == "__main__":
    main()
