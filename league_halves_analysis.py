# -*- coding: utf-8 -*-
"""
league_halves_analysis.py
==========================
Porównanie KAŻDEJ ligi z osobna pod kątem rynków, które dają pieniądze,
z podziałem sezonu na DWIE POŁOWY (przed/po przerwie mid-season).

Pomysł (na przykładzie MLS, ale liczone osobno dla każdej ligi):
  1. Wczytaj wszystkie pliki historyczne (sezony) danej ligi z data/historical/{liga}.
  2. Dla każdego sezonu wykryj PRZERWĘ w sezonie: mecze odbywają się ~raz w tygodniu,
     więc odstęp ~3 tygodni między kolejnymi datami = przerwa = punkt podziału.
     (Gdy brak wyraźnej przerwy -> fallback: mediana po liczbie meczów.)
  3. Dla każdego rynku policz ROI (cały rynek, flat stake) OSOBNO w I i II połowie.
     Zestaw, co się zmieniało między połowami i które rynki były +ROI.
  4. Wczytaj aktualny sezon 2026 z data/current/{liga}_matches_2026.csv,
     policz I połowę (zwykle tylko ona jest dostępna — teraz jest przerwa)
     i zestaw z danymi historycznymi, żeby przygotować się na II połowę.

Wynik: tabele w konsoli (per liga, per połowa) + raport JSON do data/reports/.

Użycie:
  python league_halves_analysis.py                 # wszystkie ligi
  python league_halves_analysis.py --liga mls      # jedna liga
  python league_halves_analysis.py --stake 100     # własna stawka (PLN)
  python league_halves_analysis.py --break-days 18 # próg długości przerwy (dni)
  python league_halves_analysis.py --min-n 10      # min. liczba zakładów aby pokazać rynek
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import os
import glob
import json
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from nordic_config import DATA_DIR, HISTORICAL_DIR, CURRENT_DIR
# Single source of truth dla definicji rynków (reuse z monitor.py):
from monitor import MARKETS, MARKET_KEY_TO_TYP

# ── KONFIGURACJA ─────────────────────────────────────────────────────────
LEAGUES = ["allsvenskan", "eliteserien", "veikkausliiga", "mls", "csl"]

STAKE = 100.0          # flat stake (PLN)
BREAK_DAYS = 18.0      # odstęp >= tylu dni między meczami = przerwa w sezonie
MIN_N = 10             # min. liczba zakładów, aby rynek pojawił się w tabeli
SECONDS_DAY = 86400.0

REPORTS_DIR = os.path.join(DATA_DIR, "reports")


# ── ŁADOWANIE DANYCH ─────────────────────────────────────────────────────
def load_historical(liga: str) -> dict:
    """Zwraca {season_id: DataFrame} dla wszystkich plików historycznych ligi."""
    out = {}
    pattern = os.path.join(HISTORICAL_DIR, liga, "advanced_league_matches_*.csv")
    for f in sorted(glob.glob(pattern)):
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
        except Exception as e:
            print(f"  [WARN] nie wczytano {os.path.basename(f)}: {e}")
            continue
        if "status" in df.columns:
            df = df[df["status"].astype(str).str.lower() == "complete"]
        if "date_unix" not in df.columns or len(df) == 0:
            continue
        season = int(df["season"].iloc[0]) if "season" in df.columns else os.path.basename(f)
        out[season] = df.sort_values("date_unix").reset_index(drop=True)
    return out


def load_current(liga: str) -> pd.DataFrame | None:
    """Aktualny sezon 2026 z data/current/."""
    f = os.path.join(CURRENT_DIR, f"{liga}_matches_2026.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f, encoding="utf-8-sig")
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "complete"]
    if "date_unix" not in df.columns or len(df) == 0:
        return None
    return df.sort_values("date_unix").reset_index(drop=True)


# ── WYKRYWANIE PRZERWY / PODZIAŁ SEZONU ───────────────────────────────────
def detect_split(df: pd.DataFrame, break_days: float, allow_median: bool) -> dict:
    """
    Zwraca dict: {split_unix, method, gap_days, h1_idx_end}.
    Mecze z date_unix <= split_unix -> I połowa, reszta -> II połowa.

    method:
      'break'  -> wykryto przerwę >= break_days w środkowej części sezonu
      'median' -> brak przerwy, podział na medianie liczby meczów
      'none'   -> brak podziału (wszystko = I połowa); używane dla bieżącego sezonu
    """
    days = np.sort(df["date_unix"].dropna().unique().astype(float))
    n = len(days)
    if n < 4:
        return {"split_unix": days[-1] if n else 0, "method": "none",
                "gap_days": 0.0, "n_days": n}

    gaps = np.diff(days) / SECONDS_DAY

    # kandydaci na przerwę tylko w środkowej części sezonu (25%-75% dni),
    # żeby nie złapać pauzy na samym początku/końcu
    lo_i, hi_i = int(n * 0.25), int(n * 0.75)
    best_i, best_gap = -1, 0.0
    for i in range(lo_i, max(lo_i + 1, hi_i)):
        if i < len(gaps) and gaps[i] > best_gap:
            best_gap, best_i = gaps[i], i

    if best_i >= 0 and best_gap >= break_days:
        return {"split_unix": float(days[best_i]), "method": "break",
                "gap_days": round(best_gap, 1), "n_days": n}

    if allow_median:
        mid = days[n // 2 - 1]
        return {"split_unix": float(mid), "method": "median",
                "gap_days": round(best_gap, 1), "n_days": n}

    # brak przerwy i brak fallbacku -> wszystko I połowa (bieżący, ucięty sezon)
    return {"split_unix": float(days[-1]), "method": "none",
            "gap_days": round(best_gap, 1), "n_days": n}


def split_halves(df: pd.DataFrame, split: dict) -> tuple:
    h1 = df[df["date_unix"] <= split["split_unix"]].copy()
    h2 = df[df["date_unix"] > split["split_unix"]].copy()
    return h1, h2


# ── SYMULACJA RYNKU (cały rynek, flat stake) ──────────────────────────────
def simulate_market(df: pd.DataFrame, odds_col: str, target_fn, stake: float) -> dict | None:
    """
    Obstawiamy KAŻDY mecz w danym rynku (cały rynek), flat stake.
    Bierzemy tylko mecze z prawidłowym kursem (1.01–51).
    """
    if odds_col not in df.columns or len(df) == 0:
        return None
    odds = pd.to_numeric(df[odds_col], errors="coerce").replace(0, np.nan)
    mask = odds.notna() & (odds >= 1.01) & (odds <= 51.0)
    sub = df[mask].copy()
    n = len(sub)
    if n == 0:
        return None
    try:
        wins = sub.apply(target_fn, axis=1).astype(int)
    except Exception:
        return None
    odds_v = odds[mask]
    profit = ((odds_v - 1) * stake * wins - stake * (1 - wins)).sum()
    w = int(wins.sum())
    return {
        "n": int(n),
        "wins": w,
        "wr": round(w / n * 100, 1) if n else 0.0,
        "avg_odds": round(float(odds_v.mean()), 2),
        "roi": round(profit / (n * stake) * 100, 1) if n else 0.0,
        "profit": round(float(profit), 0),
    }


def _fmt(m: dict | None) -> str:
    if not m or m["n"] == 0:
        return f"{'—':>26}"
    return f"N={m['n']:>3} WR={m['wr']:>4.1f}% ROI={m['roi']:>+6.1f}% ({m['avg_odds']:>4.2f})"


# ── ANALIZA POJEDYNCZEJ LIGI ──────────────────────────────────────────────
def analyze_league(liga: str, stake: float, break_days: float, min_n: int) -> dict:
    print("\n" + "=" * 100)
    print(f"  LIGA: {liga.upper()}")
    print("=" * 100)

    historical = load_historical(liga)
    if not historical:
        print("  Brak danych historycznych.")
        return {"liga": liga, "error": "no historical data"}

    # — wykryj podział dla każdego sezonu i pokaż —
    season_splits = {}
    print("\n  Wykryte podziały sezonów (przerwa ~%.0f dni):" % break_days)
    for season, df in sorted(historical.items()):
        sp = detect_split(df, break_days, allow_median=True)
        season_splits[season] = sp
        sd = datetime.fromtimestamp(sp["split_unix"], tz=timezone.utc).strftime("%Y-%m-%d")
        h1, h2 = split_halves(df, sp)
        print(f"    sezon {season}: {sp['method']:>6} @ {sd}  "
              f"(gap={sp['gap_days']:.0f}d)  ->  I poł={len(h1):>3}  II poł={len(h2):>3}")

    # — agregacja per rynek per połowa (pooled po sezonach) —
    market_rows = []
    for mkey, cfg in MARKETS.items():
        odds_col = cfg["odds_col"]
        tfn = cfg["target"]

        h1_frames, h2_frames = [], []
        per_season = {}  # season -> {"h1": roi, "h2": roi}
        for season, df in historical.items():
            h1, h2 = split_halves(df, season_splits[season])
            h1_frames.append(h1)
            h2_frames.append(h2)
            r1 = simulate_market(h1, odds_col, tfn, stake)
            r2 = simulate_market(h2, odds_col, tfn, stake)
            per_season[season] = {
                "h1_roi": r1["roi"] if r1 else None,
                "h2_roi": r2["roi"] if r2 else None,
            }

        all_h1 = pd.concat(h1_frames, ignore_index=True) if h1_frames else pd.DataFrame()
        all_h2 = pd.concat(h2_frames, ignore_index=True) if h2_frames else pd.DataFrame()
        m1 = simulate_market(all_h1, odds_col, tfn, stake)
        m2 = simulate_market(all_h2, odds_col, tfn, stake)

        if (not m1 or m1["n"] < min_n) and (not m2 or m2["n"] < min_n):
            continue

        # ile sezonów +ROI w każdej połowie
        h2_seasons_pos = sum(1 for v in per_season.values()
                             if v["h2_roi"] is not None and v["h2_roi"] > 0)
        h2_seasons_tot = sum(1 for v in per_season.values() if v["h2_roi"] is not None)
        h1_seasons_pos = sum(1 for v in per_season.values()
                             if v["h1_roi"] is not None and v["h1_roi"] > 0)
        h1_seasons_tot = sum(1 for v in per_season.values() if v["h1_roi"] is not None)

        delta = None
        if m1 and m2:
            delta = round(m2["roi"] - m1["roi"], 1)

        market_rows.append({
            "market": mkey,
            "typ": MARKET_KEY_TO_TYP.get(mkey, mkey),
            "h1": m1,
            "h2": m2,
            "delta_roi": delta,
            "h1_seasons_pos": h1_seasons_pos,
            "h1_seasons_tot": h1_seasons_tot,
            "h2_seasons_pos": h2_seasons_pos,
            "h2_seasons_tot": h2_seasons_tot,
            "per_season": per_season,
        })

    # — bieżący sezon 2026 (zwykle tylko I połowa) —
    cur_df = load_current(liga)
    cur_info = None
    cur_markets = {}
    if cur_df is not None:
        cur_split = detect_split(cur_df, break_days, allow_median=False)
        c_h1, c_h2 = split_halves(cur_df, cur_split)
        first = datetime.fromtimestamp(cur_df["date_unix"].min(), tz=timezone.utc).strftime("%Y-%m-%d")
        last = datetime.fromtimestamp(cur_df["date_unix"].max(), tz=timezone.utc).strftime("%Y-%m-%d")
        cur_info = {
            "n_total": int(len(cur_df)),
            "n_h1": int(len(c_h1)),
            "n_h2": int(len(c_h2)),
            "method": cur_split["method"],
            "first_date": first, "last_date": last,
        }
        for mkey, cfg in MARKETS.items():
            r1 = simulate_market(c_h1, cfg["odds_col"], cfg["target"], stake)
            r2 = simulate_market(c_h2, cfg["odds_col"], cfg["target"], stake)
            cur_markets[mkey] = {"h1": r1, "h2": r2}

    # — wydruk: tabela rynków, sortowana po ROI II połowy (przygotowanie na II poł.) —
    market_rows.sort(key=lambda r: (r["h2"]["roi"] if r["h2"] else -999), reverse=True)

    print("\n  RYNKI — I połowa vs II połowa (pooled po sezonach historycznych), flat stake %.0f PLN" % stake)
    print("  " + "-" * 96)
    print(f"  {'Rynek':<20} {'I POŁOWA':<28} {'II POŁOWA':<28} {'ΔROI':>7}  {'II poł +sez':>10}")
    print("  " + "-" * 96)
    for r in market_rows:
        d = f"{r['delta_roi']:+.1f}" if r["delta_roi"] is not None else "—"
        sp = f"{r['h2_seasons_pos']}/{r['h2_seasons_tot']}"
        print(f"  {r['typ']:<20} {_fmt(r['h1']):<28} {_fmt(r['h2']):<28} {d:>7}  {sp:>10}")

    # — wydruk: bieżący sezon 2026 (I połowa) zestawiony z historią —
    if cur_info:
        print("\n  BIEŻĄCY SEZON 2026: %d meczów (%s → %s), I poł=%d, II poł=%d [%s]"
              % (cur_info["n_total"], cur_info["first_date"], cur_info["last_date"],
                 cur_info["n_h1"], cur_info["n_h2"], cur_info["method"]))
        print("  Zestawienie I połowy 2026 vs historyczne I/II połowy (gdzie szukać pieniędzy w II poł.):")
        print("  " + "-" * 96)
        print(f"  {'Rynek':<20} {'2026 I POŁ (teraz)':<28} {'HIST I POŁ':<22} {'HIST II POŁ':<22}")
        print("  " + "-" * 96)
        for r in market_rows:
            cm = cur_markets.get(r["market"], {})
            c1 = cm.get("h1")
            h1s = f"ROI={r['h1']['roi']:+.1f}%" if r["h1"] else "—"
            h2s = f"ROI={r['h2']['roi']:+.1f}%" if r["h2"] else "—"
            print(f"  {r['typ']:<20} {_fmt(c1):<28} {h1s:<22} {h2s:<22}")

    # — wnioski: rynki konsystentnie +ROI w II połowie —
    winners = [r for r in market_rows
               if r["h2"] and r["h2"]["roi"] > 0
               and r["h2_seasons_tot"] > 0
               and r["h2_seasons_pos"] / r["h2_seasons_tot"] >= 0.6
               and r["h2"]["n"] >= min_n]
    if winners:
        print("\n  💰 KANDYDACI NA II POŁOWĘ (ROI>0 w II poł. + zyskowne w ≥60%% sezonów):")
        for r in winners:
            dtxt = f" | ΔROI(II-I)={r['delta_roi']:+.1f}%" if r["delta_roi"] is not None else ""
            print(f"     • {r['typ']:<20} II poł ROI={r['h2']['roi']:+.1f}% "
                  f"(N={r['h2']['n']}, zysk w {r['h2_seasons_pos']}/{r['h2_seasons_tot']} sez.){dtxt}")
    else:
        print("\n  💰 Brak rynków konsystentnie zyskownych w II połowie.")

    return {
        "liga": liga,
        "stake": stake,
        "break_days": break_days,
        "season_splits": {
            str(s): {
                "method": sp["method"],
                "split_date": datetime.fromtimestamp(sp["split_unix"], tz=timezone.utc).strftime("%Y-%m-%d"),
                "gap_days": sp["gap_days"],
            } for s, sp in season_splits.items()
        },
        "markets": [
            {
                "market": r["market"],
                "typ": r["typ"],
                "h1": r["h1"],
                "h2": r["h2"],
                "delta_roi": r["delta_roi"],
                "h1_seasons_pos": r["h1_seasons_pos"],
                "h1_seasons_tot": r["h1_seasons_tot"],
                "h2_seasons_pos": r["h2_seasons_pos"],
                "h2_seasons_tot": r["h2_seasons_tot"],
                "per_season": r["per_season"],
            } for r in market_rows
        ],
        "current_2026": cur_info,
        "current_markets": cur_markets,
        "h2_candidates": [r["typ"] for r in winners],
    }


# ── MAIN ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Analiza rynków per liga z podziałem na połowy sezonu")
    ap.add_argument("--liga", choices=LEAGUES, help="tylko jedna liga")
    ap.add_argument("--stake", type=float, default=STAKE, help="flat stake w PLN")
    ap.add_argument("--break-days", type=float, default=BREAK_DAYS, help="próg długości przerwy (dni)")
    ap.add_argument("--min-n", type=int, default=MIN_N, help="min. liczba zakładów aby pokazać rynek")
    args = ap.parse_args()

    ligi = [args.liga] if args.liga else LEAGUES
    os.makedirs(REPORTS_DIR, exist_ok=True)

    report = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stake": args.stake,
        "break_days": args.break_days,
        "min_n": args.min_n,
        "leagues": {},
    }
    for liga in ligi:
        report["leagues"][liga] = analyze_league(liga, args.stake, args.break_days, args.min_n)
        out = os.path.join(REPORTS_DIR, f"season_halves_{liga}.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report["leagues"][liga], fh, ensure_ascii=False, indent=2, default=str)
        print(f"\n  → zapisano {out}")

    combined = os.path.join(REPORTS_DIR, "season_halves_all.json")
    with open(combined, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n  → zapisano zbiorczy raport {combined}")


if __name__ == "__main__":
    main()
