#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal Monitor — Real-time portfolio signal health tracking
Monitors rolling ROI, consecutive losses, and identifies new candidates.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from nordic_config import (
    DATA_DIR, PORTFOLIO_DIR, CURRENT_DIR, SCRIPT_DIR,
    PORTFOLIO_SIGNALS, CUSTOM_SIGNALS_FILE,
    ALLSVENSKAN_2026_ID, ELITESERIEN_2026_ID, VEIKKAUSLIIGA_2026_ID,
    MLS_2026_ID, CSL_2026_ID,
)

# ── CONFIGURATION ────────────────────────────────────────────────────────

ROLLING_WINDOW = 6
STAKE = 100.0

WARN_ROI_ROLLING = -5.0
ALARM_ROI_ROLLING = -10.0
WARN_CONSEC_LOSS = 3
ALARM_CONSEC_LOSS = 5
REENABLE_ROI = +5.0
CANDIDATE_MIN_N = 6
CANDIDATE_MIN_ROI = +10.0

STATUS_ACTIVE = "ACTIVE"
STATUS_WARNING = "WARNING"
STATUS_ALARM = "ALARM"
STATUS_DISABLED = "DISABLED"

MONITOR_DIR = os.path.join(DATA_DIR, "monitor")
STATE_FILE = os.path.join(MONITOR_DIR, "signal_state.json")

# ── CUSTOM SIGNALS (user-added z monitora) ───────────────────────────────

def load_custom_signals() -> dict:
    """Załaduj custom signals z JSON. Pusty dict gdy plik nie istnieje."""
    if not os.path.isfile(CUSTOM_SIGNALS_FILE):
        return {}
    try:
        with open(CUSTOM_SIGNALS_FILE, encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save_custom_signals(data: dict):
    """Zapisz custom signals do JSON."""
    os.makedirs(os.path.dirname(CUSTOM_SIGNALS_FILE), exist_ok=True)
    with open(CUSTOM_SIGNALS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_all_signals() -> dict:
    """Built-in PORTFOLIO_SIGNALS + custom signals merged."""
    merged = dict(PORTFOLIO_SIGNALS)
    for sid, cfg in load_custom_signals().items():
        if sid in merged:
            continue
        merged[sid] = {**cfg, 'is_custom': True}
    return merged

# Mapowanie target_key z CANDIDATE_SCAN do target_key w portfolio_scorer CUSTOM_TARGETS
CANDIDATE_TARGET_TO_PORTFOLIO = {
    "home_win":  "home_win",
    "away":      "away_win",
    "draw":      "draw",
    "btts_yes":  "btts_yes",
    "btts_no":   "btts_no",
    "over_9_5c": "over_9_5c",
}

# Polskie display labels dla typu (bukmacherski)
CANDIDATE_TYP_LABEL = {
    "home_win":  "Home Win",
    "away":      "Away Win",
    "draw":      "Draw",
    "btts_yes":  "BTTS Yes",
    "btts_no":   "BTTS No",
    "over_9_5c": "Over 9.5 corners",
}

# Wbudowane signal.typ → semantic target_key (do wykrywania duplikatów)
BUILTIN_TYP_TO_TARGET = {
    "Home Win":          "home_win",
    "Away Win":          "away_win",
    "Draw":              "draw",
    "Under 2.5":         "under25",
    "Over 2.5":          "over25",
    "BTTS Yes":          "btts_yes",
    "BTTS No":           "btts_no",
    "Under 9.5 corners": "under_9_5c",
    "Over 9.5 corners":  "over_9_5c",
}

def _signal_target(signal_cfg: dict) -> str:
    """Semantic target_key sygnału (działa dla built-in i custom)."""
    if signal_cfg.get('is_custom'):
        return signal_cfg.get('target_key', '')
    return BUILTIN_TYP_TO_TARGET.get(signal_cfg.get('typ', ''), '')

def _signal_odds_range(signal_cfg: dict) -> tuple:
    """(lo, hi) z config sygnału. Built-in parsuje condition; custom czyta odds_min/max."""
    if signal_cfg.get('is_custom'):
        try:
            lo = float(signal_cfg.get('odds_min', 0))
        except (TypeError, ValueError):
            lo = 0.0
        hi_raw = signal_cfg.get('odds_max')
        try:
            hi = float(hi_raw) if hi_raw is not None else 999.0
        except (TypeError, ValueError):
            hi = 999.0
        return (lo, hi)
    cond = signal_cfg.get('condition', '')
    if ">=" in cond and "<=" in cond:
        try:
            parts = cond.split("and")
            lo = float(parts[0].strip().split(">=")[1].strip())
            hi = float(parts[1].strip().split("<=")[1].strip())
            return (lo, hi)
        except (IndexError, ValueError):
            return (0.0, 999.0)
    if ">=" in cond:
        try:
            return (float(cond.split(">=")[1].strip()), 999.0)
        except (IndexError, ValueError):
            return (0.0, 999.0)
    return (0.0, 999.0)

def _ranges_overlap(r1: tuple, r2: tuple) -> bool:
    """Czy dwa zakresy odds overlapują (sąsiadowanie != overlap)."""
    a1, b1 = r1
    a2, b2 = r2
    return a1 < b2 and a2 < b1

def is_duplicate_candidate(liga: str, odds_col: str, target_key: str,
                           lo: float, hi: float,
                           all_signals: dict | None = None) -> str | None:
    """
    Czy kandydat duplikuje istniejący sygnał?
    Kryteria: ta sama liga + odds_col + target + overlapping odds range.
    Zwraca signal_id duplikatu lub None.
    """
    target_norm = CANDIDATE_TARGET_TO_PORTFOLIO.get(target_key, target_key)
    sigs = all_signals if all_signals is not None else get_all_signals()
    for sid, cfg in sigs.items():
        if cfg.get('league') != liga:
            continue
        if cfg.get('odds_col') != odds_col:
            continue
        if _signal_target(cfg) != target_norm:
            continue
        if _ranges_overlap((lo, hi), _signal_odds_range(cfg)):
            return sid
    return None

# ── CANDIDATE SCAN CONFIG ────────────────────────────────────────────────

CANDIDATE_SCAN = [
    # (liga, odds_col, target_col, label, lo, hi)
    ("mls", "odds_ft_1", "home_win", "MLS Home 1.60-1.90", 1.60, 1.90),
    ("mls", "odds_ft_1", "home_win", "MLS Home 1.30-1.60", 1.30, 1.60),
    ("mls", "odds_ft_x", "draw", "MLS Draw 4.50+", 4.50, 9.00),
    ("mls", "odds_btts_yes", "btts_yes", "MLS BTTS Yes 1.50-1.65", 1.50, 1.65),
    ("csl", "odds_ft_x", "draw", "CSL Draw 2.80-3.30", 2.80, 3.30),
    ("csl", "odds_ft_2", "away", "CSL Away 4.00-6.00", 4.00, 6.00),
    ("csl", "odds_btts_yes", "btts_yes", "CSL BTTS Yes 1.65-1.85", 1.65, 1.85),
    ("csl", "odds_corners_over_95", "over_9_5c", "CSL Over 9.5C 1.90+", 1.90, 3.00),
    ("allsvenskan", "odds_ft_x", "draw", "Allsv Draw 3.30-4.50", 3.30, 4.50),
    ("allsvenskan", "odds_ft_1", "home_win", "Allsv Home 1.60-1.90", 1.60, 1.90),
    ("eliteserien", "odds_ft_1", "home_win", "Elite Home 1.60-1.90", 1.60, 1.90),
    ("eliteserien", "odds_btts_no", "btts_no", "Elite BTTS No 2.60+", 2.60, 5.00),
    ("veikkausliiga", "odds_ft_1", "home_win", "Veikk Home 2.50+", 2.50, 5.00),
    ("veikkausliiga", "odds_corners_over_95", "over_9_5c", "Veikk Over 9.5C 1.70-1.90", 1.70, 1.90),
]

# ── LOAD DATA ────────────────────────────────────────────────────────────

def load_all_portfolio() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(PORTFOLIO_DIR, "portfolio_*.csv")))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding='utf-8-sig')
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)
    wynik = pd.to_numeric(df_all.get("Wynik", 0), errors="coerce")
    df_s = df_all[wynik.isin([0.0, 1.0])].copy()
    df_s["Wynik_num"] = wynik[wynik.isin([0.0, 1.0])]
    df_s["Kurs_num"] = pd.to_numeric(df_s.get("Kurs", 0), errors="coerce")
    df_s["Profit_calc"] = df_s.apply(
        lambda r: round((r["Kurs_num"] - 1) * STAKE, 2)
        if r["Wynik_num"] == 1
        else -STAKE,
        axis=1,
    )
    df_s = df_s.drop_duplicates(subset=["ID", "Signal_ID"], keep="first")
    df_s = df_s.sort_values("Data").reset_index(drop=True)
    return df_s

# ── METRICS ──────────────────────────────────────────────────────────────

def calc_signal_metrics(df_all, signal_id) -> dict:
    sub = df_all[df_all["Signal_ID"] == signal_id].copy()

    if sub.empty:
        return {
            "n": 0,
            "roi_overall": 0,
            "roi_rolling": None,
            "consecutive_losses": 0,
            "trend": "NO_DATA",
            "last_results": "",
            "profit_total": 0,
            "avg_odds": 0,
            "win_rate": 0,
            "last_date": None,
        }

    n = len(sub)
    profit_total = sub["Profit_calc"].sum()
    roi_overall = (profit_total / (n * STAKE) * 100) if n > 0 else 0

    last_n = sub.tail(ROLLING_WINDOW)
    roi_rolling = None
    if len(last_n) >= ROLLING_WINDOW:
        roi_rolling = last_n["Profit_calc"].sum() / (ROLLING_WINDOW * STAKE) * 100

    consecutive_losses = 0
    for wynik in reversed(sub["Wynik_num"].tolist()):
        if wynik == 0:
            consecutive_losses += 1
        else:
            break

    trend = "NO_DATA"
    if len(sub) >= ROLLING_WINDOW * 2 and roi_rolling is not None:
        prev_n = sub.iloc[-(ROLLING_WINDOW * 2) : -ROLLING_WINDOW]
        roi_prev = prev_n["Profit_calc"].sum() / (ROLLING_WINDOW * STAKE) * 100
        delta = roi_rolling - roi_prev
        if delta > 5:
            trend = "GROWING"
        elif delta < -5:
            trend = "DECLINING"
        else:
            trend = "STABLE"
    elif roi_rolling is not None:
        trend = "STABLE"

    last_results = "".join(["W" if w == 1 else "L" for w in sub.tail(ROLLING_WINDOW)["Wynik_num"]])

    return {
        "n": n,
        "roi_overall": round(roi_overall, 1),
        "roi_rolling": round(roi_rolling, 1) if roi_rolling is not None else None,
        "consecutive_losses": consecutive_losses,
        "trend": trend,
        "last_results": last_results,
        "profit_total": round(profit_total, 0),
        "avg_odds": round(sub["Kurs_num"].mean(), 2),
        "win_rate": round(sub["Wynik_num"].mean() * 100, 1),
        "last_date": str(sub["Data"].iloc[-1]) if n > 0 else None,
    }

# ── STATUS DETERMINATION ────────────────────────────────────────────────

def determine_status(metrics, current_status, user_disabled=False) -> tuple:
    if user_disabled:
        roi_r = metrics.get("roi_rolling")
        if roi_r is not None and roi_r >= REENABLE_ROI and metrics["n"] >= ROLLING_WINDOW:
            return (
                STATUS_DISABLED,
                f"RE-ENABLE CANDIDATE: rolling ROI={roi_r:+.1f}% ({metrics['last_results']})",
            )
        return (STATUS_DISABLED, None)

    roi_r = metrics.get("roi_rolling")
    cl = metrics.get("consecutive_losses", 0)
    n = metrics.get("n", 0)

    if n < ROLLING_WINDOW:
        return (STATUS_ACTIVE, None)

    if (roi_r is not None and roi_r <= ALARM_ROI_ROLLING) or cl >= ALARM_CONSEC_LOSS:
        msg = (
            f"🔴 ALARM — {metrics['last_results']} "
            f"| rolling ROI={roi_r:+.1f}% "
            f"| {cl} straty z rzędu. "
            f"Rozważ WYŁĄCZENIE."
        )
        return (STATUS_ALARM, msg)

    if (roi_r is not None and roi_r <= WARN_ROI_ROLLING) or cl >= WARN_CONSEC_LOSS:
        msg = (
            f"🟡 WARNING — {metrics['last_results']} "
            f"| rolling ROI={roi_r:+.1f}% "
            f"| {cl} straty z rzędu."
        )
        return (STATUS_WARNING, msg)

    return (STATUS_ACTIVE, None)

# ── CANDIDATE SCANNING ───────────────────────────────────────────────────

# ── MARKETS (z deep_analysis.py) ────────────────────────────────────────

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

# ── LEGACY TARGET_FUNC (dla CANDIDATE_SCAN) ─────────────

TARGET_FUNC = {
    "home_win": lambda r: 1 if pd.to_numeric(r.get("homeGoalCount", 0), errors="coerce") > pd.to_numeric(r.get("awayGoalCount", 0), errors="coerce") else 0,
    "away": lambda r: 1 if pd.to_numeric(r.get("awayGoalCount", 0), errors="coerce") > pd.to_numeric(r.get("homeGoalCount", 0), errors="coerce") else 0,
    "draw": lambda r: 1 if pd.to_numeric(r.get("homeGoalCount", 0), errors="coerce") == pd.to_numeric(r.get("awayGoalCount", 0), errors="coerce") else 0,
    "btts_yes": lambda r: 1 if str(r.get("btts", "")).lower() in ("true", "1") else 0,
    "btts_no": lambda r: 0 if str(r.get("btts", "")).lower() in ("true", "1") else 1,
    "over_9_5c": lambda r: 1 if pd.to_numeric(r.get("totalCornerCount", 0), errors="coerce") > 9.5 else 0,
}

# ── MARKET SIMULATE & SCAN ──────────────────────────────────────────────

def simulate(df_liga, odds_col, target_fn, lo, hi, min_n=CANDIDATE_MIN_N, stake=STAKE):
    """Symuluj stawianie dla danego zakresu kursów. Zwraca dict z metrykami lub None."""
    if odds_col not in df_liga.columns:
        return None

    odds = pd.to_numeric(df_liga[odds_col], errors="coerce").replace(0, np.nan)
    mask = (odds >= lo) & (odds < hi) & odds.notna()
    sub = df_liga[mask].copy()
    n = len(sub)

    if n < min_n:
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

    # Rolling ostatnie ROLLING_WINDOW
    last = sub.tail(ROLLING_WINDOW)
    if len(last) >= ROLLING_WINDOW:
        rp = ((last["odds_v"] - 1) * stake * last["wynik"] - stake * (1 - last["wynik"])).sum()
        roi_r = rp / (ROLLING_WINDOW * stake) * 100
    else:
        roi_r = None

    last_results = "".join(["W" if w == 1 else "L" for w in sub.tail(ROLLING_WINDOW)["wynik"]])

    return {
        "n": n,
        "wins": wins,
        "wr": wins / n if n > 0 else 0,
        "avg_odds": sub["odds_v"].mean(),
        "roi": round(roi, 1),
        "roi_rolling": round(roi_r, 1) if roi_r is not None else None,
        "profit": round(profit, 0),
        "last": last_results,
    }


def scan_market(min_n=CANDIDATE_MIN_N, min_roi=CANDIDATE_MIN_ROI) -> list:
    """
    Skanuje wszystkie mecze complete z CURRENT_DIR
    per liga × rynek × przedział kursowy.
    Zwraca listę sygnałów z ROI > min_roi posortowanych po roi_rolling DESC.
    """
    league_files = {
        "allsvenskan": "allsvenskan_matches_2026.csv",
        "eliteserien": "eliteserien_matches_2026.csv",
        "veikkausliiga": "veikkausliiga_matches_2026.csv",
        "mls": "mls_matches_2026.csv",
        "csl": "csl_matches_2026.csv",
    }

    all_sigs = get_all_signals()
    results = []

    for liga, fname in league_files.items():
        path = os.path.join(CURRENT_DIR, fname)
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path, encoding='utf-8-sig')
        except Exception:
            continue

        df = df[df.get("status", "") == "complete"].copy()
        if df.empty:
            continue

        for market_key, market_cfg in MARKETS.items():
            odds_col = market_cfg["odds_col"]
            target_fn = market_cfg["target"]
            if odds_col not in df.columns:
                continue

            for lo, hi in market_cfg["ranges"]:
                r = simulate(df, odds_col, target_fn, lo, hi, min_n=min_n)
                if r is None:
                    continue
                if r["roi"] < min_roi:
                    continue

                dup_id = is_duplicate_candidate(
                    liga, odds_col, market_key, lo, hi, all_sigs)

                label = f"{liga} {market_key} {lo:.2f}-{hi:.2f}"

                results.append({
                    "liga": liga,
                    "market": market_key,
                    "odds_col": odds_col,
                    "odds_range": f"{lo:.2f}-{hi:.2f}",
                    "lo": lo,
                    "hi": hi,
                    "label": label,
                    "n": r["n"],
                    "wr": r["wr"],
                    "roi_overall": r["roi"],
                    "roi_rolling": r["roi_rolling"],
                    "last_results": r["last"],
                    "avg_odds": r["avg_odds"],
                    "in_portfolio": dup_id is not None,
                    "portfolio_signal_id": dup_id,
                    "alert": (
                        f"🔵 KANDYDAT: {label} | "
                        f"rolling={r['roi_rolling']:+.1f}% "
                        f"({r['last']}) | "
                        f"overall={r['roi']:+.1f}% "
                        f"n={r['n']}"
                    ),
                })

    results.sort(key=lambda x: x["roi_rolling"] or -999, reverse=True)
    return results


def compare_portfolio_vs_market(df_portfolio, market_results) -> list:
    """
    Porównuje aktywne sygnały portfelowe z wynikami skanera rynkowego.
    Zwraca listę rekomendacji: KEEP / MODIFY / DISABLE / RE-ENABLE / WATCH.
    """
    all_sigs = get_all_signals()
    state = load_state()
    comparisons = []

    for sig_id, sig_cfg in all_sigs.items():
        user_disabled = state.get(sig_id, {}).get("disabled", False)

        # Znajdź wynik skanera dla tego sygnału
        market_match = None
        for m in market_results:
            if m.get("portfolio_signal_id") == sig_id:
                market_match = m
                break

        metrics = calc_signal_metrics(df_portfolio, sig_id)
        roi_portfolio = metrics.get("roi_rolling")
        roi_market = market_match["roi_rolling"] if market_match else None

        # Rekomendacja
        if user_disabled:
            if roi_market is not None and roi_market > REENABLE_ROI:
                rec = "RE-ENABLE"
                msg = f"Wyłączony sygnał wraca: market rolling={roi_market:+.1f}%"
            else:
                rec = "KEEP_DISABLED"
                msg = "Nadal wyłączony"
        elif (roi_portfolio is not None and roi_portfolio < -10 and
              roi_market is not None and roi_market < 0):
            rec = "DISABLE"
            msg = f"Portfolio rolling={roi_portfolio:+.1f}% AND market rolling={roi_market:+.1f}% — wyłącz"
        elif (roi_portfolio is not None and roi_portfolio < -10 and
              roi_market is not None and roi_market > 5):
            rec = "MODIFY"
            msg = f"Portfolio traci ({roi_portfolio:+.1f}%) ale market zarabia ({roi_market:+.1f}%) — zmień zakres"
        elif (roi_portfolio is not None and roi_portfolio > 0 and
              roi_market is not None and roi_market > 0):
            rec = "KEEP"
            msg = f"Oba dodatnie — trzymaj"
        else:
            rec = "WATCH"
            if roi_portfolio is not None and roi_market is not None:
                msg = f"Obserwuj — portfolio={roi_portfolio:+.1f}% market={roi_market:+.1f}%"
            else:
                msg = "Brak danych skanera"

        comparisons.append({
            "signal_id": sig_id,
            "label": sig_cfg["label"],
            "tier": sig_cfg.get("tier", "B"),
            "disabled": user_disabled,
            "roi_portfolio": roi_portfolio,
            "roi_market": roi_market,
            "recommendation": rec,
            "message": msg,
            "market_match": market_match,
        })

    # Sortuj: DISABLE/MODIFY najpierw
    priority = {
        "DISABLE": 0, "MODIFY": 1,
        "RE-ENABLE": 2, "WATCH": 3,
        "KEEP": 4, "KEEP_DISABLED": 5,
    }
    comparisons.sort(key=lambda x: priority.get(x["recommendation"], 9))
    return comparisons


def scan_candidates(df_portfolio) -> list:
    league_files = {
        "allsvenskan": "allsvenskan_matches_2026.csv",
        "eliteserien": "eliteserien_matches_2026.csv",
        "veikkausliiga": "veikkausliiga_matches_2026.csv",
        "mls": "mls_matches_2026.csv",
        "csl": "csl_matches_2026.csv",
    }

    candidates = []

    for (liga, odds_col, target_key, label, lo, hi) in CANDIDATE_SCAN:
        fname = league_files.get(liga)
        if not fname:
            continue

        path = os.path.join(CURRENT_DIR, fname)
        if not os.path.isfile(path):
            continue

        try:
            df = pd.read_csv(path, encoding='utf-8-sig')
        except Exception:
            continue

        df = df[df.get("status", "") == "complete"].copy()
        if df.empty:
            continue

        odds = pd.to_numeric(df.get(odds_col, 0), errors="coerce").replace(0, float("nan"))
        mask = (odds >= lo) & (odds < hi) & odds.notna()
        df_f = df[mask].copy()
        n = len(df_f)

        if n < CANDIDATE_MIN_N:
            continue

        target_fn = TARGET_FUNC.get(target_key)
        if not target_fn:
            continue

        df_f["wynik"] = df_f.apply(target_fn, axis=1)
        df_f["odds_v"] = odds[mask].values

        last = df_f.tail(ROLLING_WINDOW)
        if len(last) < ROLLING_WINDOW:
            continue

        profit = ((last["odds_v"] - 1) * STAKE * last["wynik"] - STAKE * (1 - last["wynik"])).sum()
        roi_r = profit / (ROLLING_WINDOW * STAKE) * 100

        profit_all = ((df_f["odds_v"] - 1) * STAKE * df_f["wynik"] - STAKE * (1 - df_f["wynik"])).sum()
        roi_all = profit_all / (n * STAKE) * 100

        all_sigs_now = get_all_signals()
        dup_id = is_duplicate_candidate(liga, odds_col, target_key, lo, hi, all_sigs_now)
        already_in_portfolio = dup_id is not None

        if roi_r >= CANDIDATE_MIN_ROI:
            last_results = "".join(["W" if w == 1 else "L" for w in last["wynik"]])
            candidates.append({
                "label": label,
                "liga": liga,
                "odds_range": f"{lo:.2f}-{hi:.2f}",
                "n": n,
                "roi_overall": round(roi_all, 1),
                "roi_rolling": round(roi_r, 1),
                "last_results": last_results,
                "in_portfolio": already_in_portfolio,
                "alert": (
                    f"🔵 KANDYDAT: {label} | "
                    f"rolling={roi_r:+.1f}% "
                    f"({last_results}) | "
                    f"overall={roi_all:+.1f}% n={n}"
                ),
            })

    candidates.sort(key=lambda x: x["roi_rolling"], reverse=True)
    return candidates

# ── STATE FILE ───────────────────────────────────────────────────────────

def load_state() -> dict:
    if os.path.isfile(STATE_FILE):
        try:
            with open(STATE_FILE, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_state(state: dict):
    os.makedirs(MONITOR_DIR, exist_ok=True)
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

# ── MAIN CHECK ───────────────────────────────────────────────────────────

def run_check(debug=False):
    os.makedirs(MONITOR_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    date_str = datetime.now().strftime("%Y-%m-%d")

    df_all = load_all_portfolio()
    state = load_state()
    alerts = []
    report = {
        "timestamp": timestamp,
        "signals": {},
        "candidates": [],
        "alerts": [],
    }

    print(f'\n{"═" * 60}')
    print(f'  SIGNAL MONITOR — {timestamp}')
    print(f'  Rolling window: {ROLLING_WINDOW} zakładów')
    print(f'{"═" * 60}')

    print('\n  ── SYGNAŁY PORTFELOWE ──────────────')
    print(f'  {"Sygnał":<35} {"N":>4} {"ROI":>7} {"Roll6":>7} {"Seria":>6} {"Status"}')
    print(f'  {"-" * 35}─┼─{"─" * 4}─┼─{"─" * 7}─┼─{"─" * 7}─┼─{"─" * 6}─┼─{"─" * 8}')

    all_signals = get_all_signals()
    for sig_id, sig_cfg in all_signals.items():
        metrics = calc_signal_metrics(df_all, sig_id)
        user_disabled = state.get(sig_id, {}).get("disabled", False)
        prev_status = state.get(sig_id, {}).get("status", STATUS_ACTIVE)

        new_status, alert_msg = determine_status(metrics, prev_status, user_disabled)

        if sig_id not in state:
            state[sig_id] = {}
        state[sig_id]["status"] = new_status
        state[sig_id]["last_check"] = timestamp

        icons = {
            STATUS_ACTIVE: "🟢",
            STATUS_WARNING: "🟡",
            STATUS_ALARM: "🔴",
            STATUS_DISABLED: "⚫",
        }
        icon = icons.get(new_status, "❓")

        roi_r_str = (
            f"{metrics['roi_rolling']:+.1f}%"
            if metrics["roi_rolling"] is not None
            else "   —  "
        )
        last_r = metrics.get("last_results", "")

        print(
            f'  {sig_cfg["label"]:<35}'
            f' {metrics["n"]:>4}'
            f' {metrics["roi_overall"]:>+6.1f}%'
            f' {roi_r_str:>7}'
            f' {last_r:>6}'
            f'  {icon} {new_status}'
        )

        if alert_msg:
            alerts.append(f'{sig_cfg["label"]}: {alert_msg}')

        report["signals"][sig_id] = {
            **metrics,
            "status": new_status,
            "label": sig_cfg["label"],
            "tier": sig_cfg.get("tier", "B"),
            "alert": alert_msg,
            "is_custom": bool(sig_cfg.get("is_custom", False)),
        }

    # ── SEKCJA A: PORTFOLIO vs RYNEK ─────────────────────────────────────
    print('\n  ── PORTFOLIO vs RYNEK ──────────────')
    market_results = scan_market(min_n=8, min_roi=5.0)
    comparisons = compare_portfolio_vs_market(df_all, market_results)

    rec_icons = {
        "DISABLE": "🔴",
        "MODIFY": "🟠",
        "RE-ENABLE": "🔵",
        "WATCH": "🟡",
        "KEEP": "🟢",
        "KEEP_DISABLED": "⚫",
    }

    for c in comparisons:
        icon = rec_icons.get(c["recommendation"], "❓")
        p_str = f"{c['roi_portfolio']:+.1f}%" if c["roi_portfolio"] is not None else "    —  "
        m_str = f"{c['roi_market']:+.1f}%" if c["roi_market"] is not None else "   —  "
        print(
            f'  {c["label"]:<35}'
            f' {p_str:>9}'
            f' {m_str:>8}'
            f'  {icon} {c["recommendation"]}'
        )
        if c["recommendation"] in ("DISABLE", "MODIFY", "RE-ENABLE"):
            alerts.append(f'{c["label"]}: {icon} {c["recommendation"]} — {c["message"]}')

    # ── SEKCJA B: NOWE KANDYDATY Z RYNKU ─────────────────────────────────
    print('\n  ── NOWE KANDYDATY Z RYNKU ──────────')
    new_candidates = [
        m for m in market_results
        if not m["in_portfolio"]
        and (m["roi_rolling"] or 0) >= CANDIDATE_MIN_ROI
    ]

    if not new_candidates:
        print('  Brak nowych kandydatów powyżej progu.')
    else:
        print(f'  {"Kandydat":<40} {"N":>4} {"ROI":>7} {"Roll6":>7} {"Seria":>6}')
        for c in new_candidates[:15]:
            print(
                f'  {c["label"]:<40}'
                f' {c["n"]:>4}'
                f' {c["roi_overall"]:>+6.1f}%'
                f' {c["roi_rolling"]:>+6.1f}%'
                f' {c["last_results"]:>6}'
            )
            alerts.append(c["alert"])

    report["market_scan"] = market_results[:50]
    report["comparisons"] = comparisons
    report["new_candidates"] = new_candidates

    # ── LEGACY CANDIDATE SCAN (zachowywane dla kompatybilności) ───────────
    print('\n  ── KANDYDACI (legacy) ──────────────')
    candidates = scan_candidates(df_all)

    if not candidates:
        print('  Brak kandydatów legacy powyżej progu.')
    else:
        print(f'  {"Kandydat":<35} {"N":>4} {"ROI":>7} {"Roll6":>7} {"Seria":>6}')
        for c in candidates:
            in_p = " ✅" if c["in_portfolio"] else ""
            print(
                f'  {c["label"]:<35}'
                f' {c["n"]:>4}'
                f' {c["roi_overall"]:>+6.1f}%'
                f' {c["roi_rolling"]:>+6.1f}%'
                f' {c["last_results"]:>6}'
                f'{in_p}'
            )
            if not c["in_portfolio"]:
                alerts.append(c["alert"])

    report["candidates"] = candidates
    report["alerts"] = alerts

    if alerts:
        print(f'\n  {"═" * 60}')
        print(f'  ⚡ ALERTY ({len(alerts)})')
        print(f'  {"═" * 60}')
        for a in alerts:
            print(f'  {a}')
    else:
        print('\n  ✅ Brak alertów — wszystko OK')

    print(f'\n{"═" * 60}\n')

    save_state(state)
    report_path = os.path.join(MONITOR_DIR, f"monitor_{date_str}.json")
    os.makedirs(MONITOR_DIR, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if debug:
        print(f'  Raport: {report_path}')

    return report

# ── HISTORY ─────────────────────────────────────────────────────────────

def run_history():
    files = sorted(glob.glob(os.path.join(MONITOR_DIR, "monitor_*.json")))
    if not files:
        print('Brak historii monitorów.')
        return

    print(f'\n{"═" * 60}')
    print(f'  HISTORIA ALERTÓW ({len(files)} raportów)')
    print(f'{"═" * 60}')

    for f in files[-10:]:
        try:
            with open(f, encoding='utf-8') as fp:
                d = json.load(fp)
        except Exception:
            continue

        ts = d.get("timestamp", "?")
        alerts = d.get("alerts", [])
        if alerts:
            print(f'\n  [{ts}]')
            for a in alerts:
                print(f'    {a}')

    print()

# ── SIGNAL MANAGEMENT ────────────────────────────────────────────────────

def disable_signal(signal_id):
    state = load_state()
    if signal_id not in get_all_signals():
        print(f'Nieznany sygnał: {signal_id}')
        return
    state.setdefault(signal_id, {})
    state[signal_id]["disabled"] = True
    save_state(state)
    print(f'⚫ Wyłączono: {signal_id}')

def enable_signal(signal_id):
    state = load_state()
    state.setdefault(signal_id, {})
    state[signal_id]["disabled"] = False
    save_state(state)
    print(f'🟢 Włączono: {signal_id}')

# ── CANDIDATE → CUSTOM SIGNAL ────────────────────────────────────────────

def _candidate_signal_id(liga: str, target_key: str, lo: float, hi: float) -> str:
    """Stabilne signal_id dla custom signal z parametrów kandydata."""
    lo_s = f'{int(round(lo * 100)):03d}'
    hi_s = f'{int(round(hi * 100)):03d}'
    return f'custom_{liga}_{target_key}_{lo_s}_{hi_s}'

def add_candidate(label: str) -> str | None:
    """
    Konwertuje kandydata (wg label z CANDIDATE_SCAN) na custom signal w JSON.
    Zwraca nowy signal_id lub None gdy nie znaleziono.
    """
    match = None
    for entry in CANDIDATE_SCAN:
        if entry[3] == label:
            match = entry
            break

    if match is None:
        print(f'Nie znaleziono kandydata o label: "{label}"')
        return None

    liga, odds_col, target_key, lbl, lo, hi = match

    target_portfolio = CANDIDATE_TARGET_TO_PORTFOLIO.get(target_key)
    if target_portfolio is None:
        print(f'Nieobsługiwany target_key: {target_key}')
        return None

    # Defense in depth: odmów dodania jeśli kandydat duplikuje istniejący sygnał
    dup = is_duplicate_candidate(liga, odds_col, target_key, lo, hi)
    if dup is not None:
        print(f'Pominięto — duplikat istniejącego sygnału: {dup}')
        print(f'   (liga={liga}, odds_col={odds_col}, target={target_portfolio}, zakres {lo:.2f}-{hi:.2f})')
        return None

    sig_id = _candidate_signal_id(liga, target_portfolio, lo, hi)

    custom = load_custom_signals()
    if sig_id in custom:
        print(f'Sygnał już istnieje w custom_signals.json: {sig_id}')
        return sig_id

    custom[sig_id] = {
        'league':      liga,
        'model_src':   'rule',
        'typ':         CANDIDATE_TYP_LABEL.get(target_key, target_key),
        'target_key':  target_portfolio,
        'odds_col':    odds_col,
        'odds_min':    float(lo),
        'odds_max':    float(hi),
        'label':       lbl,
        'tier':        'B',
        'added_at':    datetime.now().strftime('%Y-%m-%d %H:%M'),
        'added_from':  'monitor_candidate',
    }

    save_custom_signals(custom)
    print(f'➕ Dodano custom signal: {sig_id}')
    print(f'   Label: {lbl}')
    print(f'   Liga: {liga} | Kurs: {lo:.2f}-{hi:.2f} | Target: {target_portfolio}')
    return sig_id

# ── MAIN ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Signal Monitor — Portfolio health tracking"
    )
    parser.add_argument('--check', action='store_true', help='Run signal check')
    parser.add_argument('--history', action='store_true', help='Show alert history')
    parser.add_argument('--debug', action='store_true', help='Debug output')
    parser.add_argument('--disable', type=str, default=None, help='Disable signal')
    parser.add_argument('--enable', type=str, default=None, help='Enable signal')
    parser.add_argument('--add-candidate', type=str, default=None,
                        help='Dodaj kandydata (label z CANDIDATE_SCAN) do custom_signals.json')

    args = parser.parse_args()

    if args.check:
        run_check(args.debug)
    elif args.history:
        run_history()
    elif args.disable:
        disable_signal(args.disable)
    elif args.enable:
        enable_signal(args.enable)
    elif args.add_candidate:
        add_candidate(args.add_candidate)
    else:
        run_check(args.debug)

if __name__ == '__main__':
    main()
