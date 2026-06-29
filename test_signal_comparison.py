# -*- coding: utf-8 -*-
"""
Test Signal Comparison — Detailed analysis of Portfolio vs Market signals
Extracts all bets for portfolio signal and compares with market signal.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import glob
import os
from datetime import datetime

from nordic_config import PORTFOLIO_DIR, CURRENT_DIR

STAKE = 100.0


def load_portfolio_matches(signal_id: str, liga: str) -> pd.DataFrame:
    """Wczytaj wszystkie mecze portfelowe dla danego sygnału."""
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

    # Filtruj po signal_id (unikalny — Liga w portfolio bywa "CSL"/"Allsvenskan",
    # więc filtr po lidze zerowałby wynik; signal_id jednoznacznie identyfikuje sygnał)
    mask = (df_all.get("Signal_ID", "") == signal_id)
    sub = df_all[mask].copy()

    # Konwertuj kolumny na liczby
    sub["Kurs_num"] = pd.to_numeric(sub.get("Kurs", 0), errors="coerce")
    sub["Wynik_num"] = pd.to_numeric(sub.get("Wynik", 0), errors="coerce")
    sub["Data_dt"] = pd.to_datetime(sub.get("Data", None), errors="coerce")

    return sub.sort_values("Data_dt")


def load_market_matches(liga: str, market: str, lo: float, hi: float) -> pd.DataFrame:
    """Wczytaj mecze rynkowe dla zakresu kursów."""
    league_files = {
        "allsvenskan": "allsvenskan_matches_2026.csv",
        "eliteserien": "eliteserien_matches_2026.csv",
        "veikkausliiga": "veikkausliiga_matches_2026.csv",
        "mls": "mls_matches_2026.csv",
        "csl": "csl_matches_2026.csv",
    }

    fname = league_files.get(liga)
    if not fname:
        return pd.DataFrame()

    path = os.path.join(CURRENT_DIR, fname)
    if not os.path.isfile(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except Exception:
        return pd.DataFrame()

    # Filtruj complete
    df = df[df.get("status", "") == "complete"].copy()
    if df.empty:
        return df

    # Mapuj market → odds_col
    market_to_odds = {
        "draw": "odds_ft_x",
        "home_win": "odds_ft_1",
        "away_win": "odds_ft_2",
        "btts_yes": "odds_btts_yes",
        "btts_no": "odds_btts_no",
        "over_9_5c": "odds_corners_over_95",
        "under_9_5c": "odds_corners_under_95",
    }

    odds_col = market_to_odds.get(market)
    if not odds_col or odds_col not in df.columns:
        return pd.DataFrame()

    # Filtruj po zakresie kursów
    odds = pd.to_numeric(df.get(odds_col), errors="coerce")
    mask = (odds >= lo) & (odds < hi) & odds.notna()
    sub = df[mask].copy()

    # Mapuj wyniki
    if market == "draw":
        sub["wynik"] = (
            (pd.to_numeric(sub.get("homeGoalCount", 0), errors="coerce") ==
             pd.to_numeric(sub.get("awayGoalCount", 0), errors="coerce")).astype(int)
        )
    elif market == "home_win":
        sub["wynik"] = (
            (pd.to_numeric(sub.get("homeGoalCount", 0), errors="coerce") >
             pd.to_numeric(sub.get("awayGoalCount", 0), errors="coerce")).astype(int)
        )
    elif market == "away_win":
        sub["wynik"] = (
            (pd.to_numeric(sub.get("awayGoalCount", 0), errors="coerce") >
             pd.to_numeric(sub.get("homeGoalCount", 0), errors="coerce")).astype(int)
        )
    elif market == "btts_yes":
        sub["wynik"] = (sub.get("btts", "").astype(str).str.lower().isin(["true", "1"])).astype(int)
    elif market == "under_9_5c":
        sub["wynik"] = (
            (pd.to_numeric(sub.get("totalCornerCount", -1), errors="coerce") <= 9.5).astype(int)
        )

    sub["odds_num"] = odds[mask].values
    sub["data_dt"] = pd.to_datetime(sub.get("date_unix"), unit='s', errors="coerce")

    return sub.sort_values("data_dt")


def analyze_signal(signal_id: str, liga: str, portfolio_kurs_range: tuple,
                  market_name: str, market_kurs_range: tuple):
    """Porównaj sygnał portfela z sygnałem rynkowym."""

    print(f"\n{'═' * 100}")
    print(f"ANALIZA SYGNAŁU: {signal_id.upper()}")
    print(f"{'═' * 100}")

    # ── PORTFOLIO ─────────────────────────────────────────────────────────
    print(f"\n  📊 PORTFOLIO — {liga.upper()} kurs {portfolio_kurs_range[0]:.2f}-{portfolio_kurs_range[1]:.2f}")
    print(f"  {'-' * 96}")

    df_portfolio = load_portfolio_matches(signal_id, liga)

    if df_portfolio.empty:
        print(f"  ❌ Brak meczów portfelowych dla {signal_id}")
    else:
        n_port = len(df_portfolio)
        wins_port = int(df_portfolio["Wynik_num"].sum())
        wr_port = wins_port / n_port if n_port > 0 else 0
        avg_odds_port = df_portfolio["Kurs_num"].mean()
        profit_port = (
            ((df_portfolio["Kurs_num"] - 1) * STAKE * df_portfolio["Wynik_num"] -
             STAKE * (1 - df_portfolio["Wynik_num"])).sum()
        )
        roi_port = profit_port / (n_port * STAKE) * 100 if n_port > 0 else 0

        print(f"  Meczów: {n_port} | Wygrane: {wins_port} ({wr_port:.0%}) | Średni kurs: {avg_odds_port:.2f} | ROI: {roi_port:+.1f}%")
        print()
        print(f"  {'Data':<12} {'Mecz':<25} {'Kurs':>6} {'W/L':<3} {'Profit':>8}")
        print(f"  {'-' * 96}")

        for idx, row in df_portfolio.iterrows():
            data = str(row.get("Data", "?"))[:10]
            mecz = str(row.get("Mecz", "?"))[:25]
            kurs = row["Kurs_num"]
            wynik = "W" if row["Wynik_num"] == 1 else "L"
            profit = (kurs - 1) * STAKE if row["Wynik_num"] == 1 else -STAKE
            print(f"  {data:<12} {mecz:<25} {kurs:>6.2f} {wynik:<3} {profit:>+8.0f}")

        print(f"  {'-' * 96}")
        print(f"  Łącznie:     {n_port:>24} {wins_port:>10} Profit: {profit_port:+.0f} ({roi_port:+.1f}%)\n")

    # ── RYNEK ──────────────────────────────────────────────────────────────
    print(f"  🎯 RYNEK — {liga.upper()} {market_name} kurs {market_kurs_range[0]:.2f}-{market_kurs_range[1]:.2f}")
    print(f"  {'-' * 96}")

    df_market = load_market_matches(liga, market_name, market_kurs_range[0], market_kurs_range[1])

    if df_market.empty:
        print(f"  ❌ Brak meczów rynkowych dla zakresu {market_kurs_range}")
    else:
        n_market = len(df_market)
        wins_market = int(df_market["wynik"].sum())
        wr_market = wins_market / n_market if n_market > 0 else 0
        avg_odds_market = df_market["odds_num"].mean()
        profit_market = (
            ((df_market["odds_num"] - 1) * STAKE * df_market["wynik"] -
             STAKE * (1 - df_market["wynik"])).sum()
        )
        roi_market = profit_market / (n_market * STAKE) * 100 if n_market > 0 else 0

        print(f"  Meczów: {n_market} | Wygrane: {wins_market} ({wr_market:.0%}) | Średni kurs: {avg_odds_market:.2f} | ROI: {roi_market:+.1f}%")
        print()
        print(f"  {'Data':<12} {'Mecz':<25} {'Kurs':>6} {'W/L':<3} {'Profit':>8}")
        print(f"  {'-' * 96}")

        for idx, row in df_market.iterrows():
            data = str(row.get("data_dt", "?"))[:10]
            home = str(row.get("home_name", "?"))[:12]
            away = str(row.get("away_name", "?"))[:12]
            mecz = f"{home} vs {away}"[:25]
            kurs = row["odds_num"]
            wynik = "W" if row["wynik"] == 1 else "L"
            profit = (kurs - 1) * STAKE if row["wynik"] == 1 else -STAKE
            print(f"  {data:<12} {mecz:<25} {kurs:>6.2f} {wynik:<3} {profit:>+8.0f}")

        print(f"  {'-' * 96}")
        print(f"  Łącznie:     {n_market:>24} {wins_market:>10} Profit: {profit_market:+.0f} ({roi_market:+.1f}%)\n")

    # ── PODSUMOWANIE ───────────────────────────────────────────────────────
    print(f"  📈 PORÓWNANIE")
    print(f"  {'-' * 96}")

    if not df_portfolio.empty and not df_market.empty:
        diff_roi = roi_market - roi_port
        diff_avg_odds = avg_odds_market - avg_odds_port

        print(f"  ROI Portfolio:      {roi_port:>+7.1f}%")
        print(f"  ROI Rynek:          {roi_market:>+7.1f}%")
        print(f"  Różnica ROI:        {diff_roi:>+7.1f}% {'⬆️ Rynek lepszy' if diff_roi > 0 else '⬇️ Portfolio lepszy'}")
        print()
        print(f"  Średni kurs Port:   {avg_odds_port:>+7.2f}")
        print(f"  Średni kurs Rynek:  {avg_odds_market:>+7.2f}")
        print(f"  Różnica kursów:     {diff_avg_odds:>+7.2f} {'⬆️ Rynek wyższe' if diff_avg_odds > 0 else '⬇️ Portfolio wyższe'}")
        print()

        if diff_roi > 10:
            print(f"  💡 WNIOSEK: Rynek znacznie lepszy — CONSIDER RE-ENABLING lub zmiana zakresu kursów")
        elif diff_roi > 0:
            print(f"  💡 WNIOSEK: Rynek nieco lepszy — obserwuj lub rozważ modyfikację")
        elif diff_roi > -10:
            print(f"  💡 WNIOSEK: Podobne wyniki — KEEP status quo")
        else:
            print(f"  💡 WNIOSEK: Portfolio znacznie lepszy — KEEP portfolio signal")

    print(f"\n{'═' * 100}\n")


def main():
    print(f"\n{'═' * 100}")
    print(f"  TEST SIGNAL COMPARISON — Portfolio vs Market Analysis")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Stake: {STAKE:.0f} PLN | Rolling window: 6 meczów")
    print(f"{'═' * 100}")

    # Analizuj CSL Draw
    analyze_signal(
        signal_id="csl_draw",
        liga="csl",
        portfolio_kurs_range=(3.30, 4.50),
        market_name="draw",
        market_kurs_range=(4.20, 4.80),
    )

    # Analizuj Allsvenskan BTTS Yes
    analyze_signal(
        signal_id="allsv_btts_yes",
        liga="allsvenskan",
        portfolio_kurs_range=(0, 999),  # Portfolio nie ma zakresu (scorer-based)
        market_name="btts_yes",
        market_kurs_range=(1.25, 3.00),
    )

    # Analizuj Eliteserien Under 9.5C
    analyze_signal(
        signal_id="elite_under_corners",
        liga="eliteserien",
        portfolio_kurs_range=(0, 999),  # Scorer-based
        market_name="under_9_5c",
        market_kurs_range=(1.50, 4.00),
    )


if __name__ == "__main__":
    main()
