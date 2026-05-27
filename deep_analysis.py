# -*- coding: utf-8 -*-
"""
Deep Analysis — Skanowanie wszystkich meczów z data/current/
w poszukiwaniu zyskownych sygnałów w każdej kombinacji:
  liga × rynek × przedział kursów
"""
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from nordic_config import CURRENT_DIR, DATA_DIR, REPORTS_DIR, PORTFOLIO_SIGNALS


# ── KONFIGURACJA ──────────────────────────────────────────────────────────

STAKE = 100.0
MIN_N = 10
MIN_ROI = 5.0
ROLLING_W = 6


# ── MARKETS ───────────────────────────────────────────────────────────────

MARKETS = {
    # ── WYNIK MECZU ─────────────────────
    "home_win": {
        "odds_col": "odds_ft_1",
        "target": lambda r: 1 if r["homeGoalCount"] > r["awayGoalCount"] else 0,
        "ranges": [
            (1.05, 1.20), (1.20, 1.35), (1.35, 1.50),
            (1.50, 1.65), (1.65, 1.80), (1.80, 2.00),
            (2.00, 2.30), (2.30, 2.70), (2.70, 3.50),
            (3.50, 6.00),
        ],
    },
    "away_win": {
        "odds_col": "odds_ft_2",
        "target": lambda r: 1 if r["awayGoalCount"] > r["homeGoalCount"] else 0,
        "ranges": [
            (1.50, 1.80), (1.80, 2.10), (2.10, 2.50),
            (2.50, 3.00), (3.00, 3.50), (3.50, 4.00),
            (4.00, 5.00), (5.00, 7.00), (7.00, 15.0),
        ],
    },
    "draw": {
        "odds_col": "odds_ft_x",
        "target": lambda r: 1 if r["homeGoalCount"] == r["awayGoalCount"] else 0,
        "ranges": [
            (2.50, 2.80), (2.80, 3.10), (3.10, 3.40),
            (3.40, 3.80), (3.80, 4.20), (4.20, 4.80),
            (4.80, 6.00), (6.00, 12.0),
        ],
    },

    # ── DOUBLE CHANCE ───────────────────
    "dc_1x": {
        "odds_col": "odds_doublechance_1x",
        "target": lambda r: 1 if r["homeGoalCount"] >= r["awayGoalCount"] else 0,
        "ranges": [
            (1.05, 1.15), (1.15, 1.25), (1.25, 1.40),
            (1.40, 1.60), (1.60, 1.90),
        ],
    },
    "dc_x2": {
        "odds_col": "odds_doublechance_x2",
        "target": lambda r: 1 if r["awayGoalCount"] >= r["homeGoalCount"] else 0,
        "ranges": [
            (1.20, 1.40), (1.40, 1.60), (1.60, 1.85),
            (1.85, 2.20), (2.20, 3.00),
        ],
    },

    # ── GOLE ────────────────────────────
    "over25": {
        "odds_col": "odds_ft_over25",
        "target": lambda r: 1 if pd.to_numeric(r.get("totalGoalCount", 0), errors="coerce") > 2.5 else 0,
        "ranges": [
            (1.20, 1.35), (1.35, 1.50), (1.50, 1.65),
            (1.65, 1.80), (1.80, 2.00), (2.00, 2.50),
        ],
    },
    "under25": {
        "odds_col": "odds_ft_under25",
        "target": lambda r: 1 if pd.to_numeric(r.get("totalGoalCount", 0), errors="coerce") <= 2.5 else 0,
        "ranges": [
            (1.40, 1.60), (1.60, 1.80), (1.80, 2.00),
            (2.00, 2.30), (2.30, 2.70), (2.70, 3.50),
        ],
    },
    "over35": {
        "odds_col": "odds_ft_over35",
        "target": lambda r: 1 if pd.to_numeric(r.get("totalGoalCount", 0), errors="coerce") > 3.5 else 0,
        "ranges": [
            (1.50, 1.75), (1.75, 2.00), (2.00, 2.40),
            (2.40, 3.00), (3.00, 4.00),
        ],
    },
    "under35": {
        "odds_col": "odds_ft_under35",
        "target": lambda r: 1 if pd.to_numeric(r.get("totalGoalCount", 0), errors="coerce") <= 3.5 else 0,
        "ranges": [
            (1.20, 1.40), (1.40, 1.60), (1.60, 1.85),
            (1.85, 2.20),
        ],
    },

    # ── BTTS ────────────────────────────
    "btts_yes": {
        "odds_col": "odds_btts_yes",
        "target": lambda r: 1 if str(r.get("btts", "")).lower() in ("true", "1") else 0,
        "ranges": [
            (1.25, 1.40), (1.40, 1.55), (1.55, 1.70),
            (1.70, 1.90), (1.90, 2.20), (2.20, 3.00),
        ],
    },
    "btts_no": {
        "odds_col": "odds_btts_no",
        "target": lambda r: 0 if str(r.get("btts", "")).lower() in ("true", "1") else 1,
        "ranges": [
            (1.50, 1.70), (1.70, 1.90), (1.90, 2.10),
            (2.10, 2.40), (2.40, 2.80), (2.80, 4.00),
        ],
    },

    # ── CORNERS ─────────────────────────
    "over_9_5c": {
        "odds_col": "odds_corners_over_95",
        "target": lambda r: 1 if pd.to_numeric(r.get("totalCornerCount", -1), errors="coerce") > 9.5 else 0,
        "ranges": [
            (1.40, 1.55), (1.55, 1.70), (1.70, 1.85),
            (1.85, 2.00), (2.00, 2.20), (2.20, 3.00),
        ],
    },
    "under_9_5c": {
        "odds_col": "odds_corners_under_95",
        "target": lambda r: 1 if pd.to_numeric(r.get("totalCornerCount", -1), errors="coerce") <= 9.5 else 0,
        "ranges": [
            (1.50, 1.70), (1.70, 1.90), (1.90, 2.10),
            (2.10, 2.40), (2.40, 2.80), (2.80, 4.00),
        ],
    },
    "over_8_5c": {
        "odds_col": "odds_corners_over_85",
        "target": lambda r: 1 if pd.to_numeric(r.get("totalCornerCount", -1), errors="coerce") > 8.5 else 0,
        "ranges": [
            (1.25, 1.40), (1.40, 1.55), (1.55, 1.75),
            (1.75, 2.00), (2.00, 2.50),
        ],
    },
    "over_10_5c": {
        "odds_col": "odds_corners_over_105",
        "target": lambda r: 1 if pd.to_numeric(r.get("totalCornerCount", -1), errors="coerce") > 10.5 else 0,
        "ranges": [
            (1.50, 1.75), (1.75, 2.00), (2.00, 2.30),
            (2.30, 2.80), (2.80, 4.00),
        ],
    },

    # ── PIERWSZA POŁOWA ──────────────────
    "ht_over15": {
        "odds_col": "odds_1st_half_over15",
        "target": lambda r: 1 if (pd.to_numeric(r.get("ht_goals_team_a", 0), errors="coerce") or 0) +
                                  (pd.to_numeric(r.get("ht_goals_team_b", 0), errors="coerce") or 0) > 1.5 else 0,
        "ranges": [
            (1.50, 1.75), (1.75, 2.00), (2.00, 2.30),
            (2.30, 2.80),
        ],
    },
    "ht_under15": {
        "odds_col": "odds_1st_half_under15",
        "target": lambda r: 1 if (pd.to_numeric(r.get("ht_goals_team_a", 0), errors="coerce") or 0) +
                                  (pd.to_numeric(r.get("ht_goals_team_b", 0), errors="coerce") or 0) <= 1.5 else 0,
        "ranges": [
            (1.20, 1.40), (1.40, 1.60), (1.60, 1.85),
            (1.85, 2.20),
        ],
    },
}

LEAGUES = ["allsvenskan", "eliteserien", "veikkausliiga", "mls", "csl"]


# ── FUNKCJE ───────────────────────────────────────────────────────────────

def load_all_matches():
    """Wczytaj wszystkie *_matches_2026.csv z CURRENT_DIR."""
    df_all = pd.DataFrame()

    for league in LEAGUES:
        fname = f"{league}_matches_2026.csv"
        fpath = os.path.join(CURRENT_DIR, fname)
        if not os.path.isfile(fpath):
            continue

        df = pd.read_csv(fpath, encoding="utf-8-sig")
        df["liga"] = league

        # Filtruj tylko complete
        if "status" in df.columns:
            df = df[df["status"].astype(str).str.lower() == "complete"]

        if "date_unix" in df.columns:
            df = df.sort_values("date_unix", ascending=True)

        df_all = pd.concat([df_all, df], ignore_index=True)
        print(f"  ✅ {league}: {len(df)} complete meczów")

    return df_all


def simulate(df_liga, odds_col, target_fn, lo, hi, stake=STAKE):
    """Symuluj stawianie dla danego zakresu kursów."""
    if odds_col not in df_liga.columns:
        return None

    odds = pd.to_numeric(df_liga[odds_col], errors="coerce").replace(0, np.nan)
    mask = (odds >= lo) & (odds < hi) & odds.notna()
    sub = df_liga[mask].copy()
    n = len(sub)

    if n < MIN_N:
        return None

    try:
        sub["wynik"] = sub.apply(target_fn, axis=1)
    except Exception:
        return None

    sub["odds_v"] = odds[mask].values
    sub = sub.sort_values("date_unix", ascending=True) if "date_unix" in sub.columns else sub

    wins = int(sub["wynik"].sum())
    profit = ((sub["odds_v"] - 1) * stake * sub["wynik"] - stake * (1 - sub["wynik"])).sum()
    roi = profit / (n * stake) * 100 if n > 0 else 0

    # Rolling ostatnie ROLLING_W
    last = sub.tail(ROLLING_W)
    if len(last) >= ROLLING_W:
        rp = ((last["odds_v"] - 1) * stake * last["wynik"] - stake * (1 - last["wynik"])).sum()
        roi_r = rp / (ROLLING_W * stake) * 100
    else:
        roi_r = None

    last_results = "".join(["W" if w == 1 else "L" for w in sub.tail(ROLLING_W)["wynik"]])

    return {
        "n": n,
        "wins": wins,
        "wr": wins / n if n > 0 else 0,
        "avg_odds": sub["odds_v"].mean(),
        "roi": round(roi, 1),
        "roi_rolling": round(roi_r, 1) if roi_r is not None else None,
        "profit": round(profit, 0),
        "last": last_results,
        "in_portfolio": False,
    }


def matches_portfolio_signal(liga, market, lo, hi):
    """Sprawdź czy ten sygnał już jest w PORTFOLIO_SIGNALS."""
    for sig_name, sig_cfg in PORTFOLIO_SIGNALS.items():
        # Uproszczony matching — można rozszerzyć
        if liga.lower() in sig_name.lower() and market.lower() in sig_name.lower():
            return True
    return False


# ── GŁÓWNA LOGIKA ─────────────────────────────────────────────────────────

def main():
    global MIN_N, MIN_ROI

    parser = argparse.ArgumentParser(description="Deep Analysis — skaner zyskownych sygnałów")
    parser.add_argument("--min-n", type=int, default=MIN_N, help=f"Min liczba zakładów (default: {MIN_N})")
    parser.add_argument("--min-roi", type=float, default=MIN_ROI, help=f"Min ROI% (default: {MIN_ROI})")
    parser.add_argument("--liga", type=str, help="Filtruj po lidze (allsvenskan, eliteserien, mls, csl, veikkausliiga)")
    parser.add_argument("--output", type=str, default="table", choices=["table", "json"], help="Format output")
    args = parser.parse_args()

    MIN_N = args.min_n
    MIN_ROI = args.min_roi

    print("=" * 70)
    print("DEEP ANALYSIS — Skanowanie zyskownych sygnałów")
    print("=" * 70)
    print()
    print("  Wczytywanie meczów z data/current/...")

    df_all = load_all_matches()
    n_total = len(df_all)
    print(f"  Łącznie: {n_total} meczów\n")

    if df_all.empty:
        print("  ❌ Brak meczów do analizy!")
        return

    # Skanowanie
    results = []

    for liga in LEAGUES:
        if args.liga and args.liga != liga:
            continue

        df_liga = df_all[df_all["liga"] == liga].copy()
        if df_liga.empty:
            continue

        for market_key, market_cfg in MARKETS.items():
            odds_col = market_cfg["odds_col"]
            target_fn = market_cfg["target"]
            ranges = market_cfg["ranges"]

            if odds_col not in df_liga.columns:
                continue

            for lo, hi in ranges:
                r = simulate(df_liga, odds_col, target_fn, lo, hi)
                if r is None:
                    continue
                if r["roi"] < MIN_ROI:
                    continue

                r["liga"] = liga
                r["market"] = market_key
                r["odds_range"] = f"{lo:.2f}-{hi:.2f}"
                r["in_portfolio"] = matches_portfolio_signal(liga, market_key, lo, hi)
                results.append(r)

    if not results:
        print(f"  ❌ Nie znaleziono sygnałów z ROI > {MIN_ROI}%\n")
        return

    # Sortowanie
    results.sort(key=lambda x: x["roi"], reverse=True)

    # Podział na grupy
    new_candidates = [r for r in results if r["roi"] > MIN_ROI and (r["roi_rolling"] is None or r["roi_rolling"] > 0)]
    strong_candidates = [r for r in results if r["roi"] > 20.0]
    trending = [r for r in results if r["roi"] > MIN_ROI and r["roi_rolling"] is not None and r["roi_rolling"] > r["roi"]]
    portfolio_conf = [r for r in results if r["in_portfolio"]]

    # OUTPUT
    if args.output == "json":
        output_json(results, args.min_n, args.min_roi)
    else:
        output_table(results, new_candidates, strong_candidates, trending, portfolio_conf, MIN_N, MIN_ROI)


def output_table(results, new_cands, strong, trending, portfolio, min_n, min_roi):
    """Wydrukuj wyniki w formacie tabelarycznym."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print()
    print("=" * 100)
    print(f"  DEEP ANALYSIS — {now}")
    print(f"  Min N: {min_n} | Min ROI: {min_roi}%")
    print("=" * 100)

    # Nowe kandydaty
    if new_cands:
        print()
        print("  ── NOWE KANDYDATY (ROI > {}%, rolling > 0) ──".format(min_roi))
        print("  " + "-" * 96)
        print(
            "  {:<20} {:<20} {:<12} {:<4} {:<6} {:<8} {:<8} {:<12}".format(
                "Liga", "Market", "Kurs", "N", "WR", "ROI", "Roll6", "Seria"
            )
        )
        print("  " + "-" * 96)
        for r in new_cands[:10]:
            print(
                "  {:<20} {:<20} {:<12} {:<4} {:<7} {:<8} {:<10} {:<12}".format(
                    r["liga"],
                    r["market"],
                    r["odds_range"],
                    r["n"],
                    f"{r['wr']:.0%}",
                    f"{r['roi']:.1f}%",
                    f"{r['roi_rolling']:.1f}%" if r["roi_rolling"] is not None else "—",
                    r["last"],
                )
            )

    # Silne (roi > 20%)
    if strong:
        print()
        print("  ── SILNE (ROI > 20%) ──")
        print("  " + "-" * 96)
        print(
            "  {:<20} {:<20} {:<12} {:<4} {:<6} {:<8} {:<8} {:<12}".format(
                "Liga", "Market", "Kurs", "N", "WR", "ROI", "Roll6", "Seria"
            )
        )
        print("  " + "-" * 96)
        for r in strong[:8]:
            print(
                "  {:<20} {:<20} {:<12} {:<4} {:<7} {:<8} {:<10} {:<12}".format(
                    r["liga"],
                    r["market"],
                    r["odds_range"],
                    r["n"],
                    f"{r['wr']:.0%}",
                    f"{r['roi']:.1f}%",
                    f"{r['roi_rolling']:.1f}%" if r["roi_rolling"] is not None else "—",
                    r["last"],
                )
            )

    # Trend rosnący
    if trending:
        print()
        print("  ── TREND ROSNĄCY (roll > roi) ──")
        print("  " + "-" * 96)
        print(
            "  {:<20} {:<20} {:<12} {:<4} {:<6} {:<8} {:<8} {:<12}".format(
                "Liga", "Market", "Kurs", "N", "WR", "ROI", "Roll6", "Seria"
            )
        )
        print("  " + "-" * 96)
        for r in trending[:8]:
            print(
                "  {:<20} {:<20} {:<12} {:<4} {:<7} {:<8} {:<10} {:<12}".format(
                    r["liga"],
                    r["market"],
                    r["odds_range"],
                    r["n"],
                    f"{r['wr']:.0%}",
                    f"{r['roi']:.1f}%",
                    f"{r['roi_rolling']:.1f}%" if r["roi_rolling"] is not None else "—",
                    r["last"],
                )
            )

    # Portfolio confirmation
    if portfolio:
        print()
        print("  ── POTWIERDZENIE PORTFELA ──")
        print("  " + "-" * 96)
        for r in portfolio:
            status = "✅" if r["roi"] > 10 else "⚠️"
            print(
                f"  {status} {r['liga']:15} {r['market']:20} n={r['n']:3} roi={r['roi']:+.1f}% roll={r['roi_rolling'] or '—':>8}"
            )

    print()
    print("=" * 100)
    print(f"  Łącznie znalezionych: {len(results)} sygnałów > {min_roi}% ROI")
    print("=" * 100)
    print()


def output_json(results, min_n, min_roi):
    """Zapisz wyniki do JSON w data/reports/."""
    now = datetime.now()
    fname = f"deep_analysis_{now.strftime('%Y-%m-%d_%H%M%S')}.json"
    fpath = os.path.join(REPORTS_DIR, fname)

    output = {
        "timestamp": now.isoformat(),
        "params": {"min_n": min_n, "min_roi": min_roi},
        "results": results,
        "top_candidates": [r for r in results if r["roi"] > 20.0][:10],
    }

    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print()
    print(f"  ✅ JSON zapisany: {fpath}")
    print()


if __name__ == "__main__":
    main()
