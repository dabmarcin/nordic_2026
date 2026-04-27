# -*- coding: utf-8 -*-
import sys
import io

# Wrapper stdout UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

import json

from nordic_config import (
    API_KEY, BASE_URL,
    ALL_HISTORICAL, LEAGUE_BY_SEASON_ID,
    ALLSVENSKAN_DIR, ELITESERIEN_DIR, VEIKKAUSLIIGA_DIR,
    ALLSVENSKAN_HISTORICAL, ELITESERIEN_HISTORICAL, VEIKKAUSLIIGA_HISTORICAL,
    ALLSVENSKAN_2026_ID, ELITESERIEN_2026_ID, VEIKKAUSLIIGA_2026_ID,
    ACTIVE_SEASON_IDS, LEAGUE_BY_SEASON_ID,
    DAILY_DIR, REPORTS_DIR,
    CURRENT_DIR,
    TEAMS_ALLSV_HIST, TEAMS_ELITE_HIST, TEAMS_VEIKK_HIST,
)

LEAGUE_DIR_MAP = {
    "allsvenskan":   ALLSVENSKAN_DIR,
    "eliteserien":   ELITESERIEN_DIR,
    "veikkausliiga": VEIKKAUSLIIGA_DIR,
}

TEAMS_HIST_DIR_MAP = {
    "allsvenskan":   TEAMS_ALLSV_HIST,
    "eliteserien":   TEAMS_ELITE_HIST,
    "veikkausliiga": TEAMS_VEIKK_HIST,
}

# season_id → liga
ACTIVE_LEAGUE_NAMES = {
    ALLSVENSKAN_2026_ID:   "allsvenskan",
    ELITESERIEN_2026_ID:   "eliteserien",
    VEIKKAUSLIIGA_2026_ID: "veikkausliiga",
}

TEAMS_STATS_KEYS = [
    "seasonPPG_overall", "seasonPPG_home", "seasonPPG_away",
    "seasonBTTSPercentage_overall", "seasonBTTSPercentage_home", "seasonBTTSPercentage_away",
    "seasonOver25Percentage_overall", "seasonOver25Percentage_home", "seasonOver25Percentage_away",
    "seasonScoredAVG_overall", "seasonScoredAVG_home", "seasonScoredAVG_away",
    "seasonConcededAVG_overall", "seasonConcededAVG_home", "seasonConcededAVG_away",
    "cornersAVG_overall", "cornersAVG_home", "cornersAVG_away",
    "cornersAgainstAVG_overall", "cornersAgainstAVG_home", "cornersAgainstAVG_away",
    "xg_for_avg_overall", "xg_for_avg_home", "xg_for_avg_away",
    "xg_against_avg_overall", "xg_against_avg_home", "xg_against_avg_away",
    "winPercentage_overall", "winPercentage_home", "winPercentage_away",
    "seasonMatchesPlayed_overall", "seasonMatchesPlayed_home", "seasonMatchesPlayed_away",
    "leaguePosition_overall",
    "over95CornersPercentage_overall", "over95CornersPercentage_home", "over95CornersPercentage_away",
    "cardsAVG_overall", "cardsAVG_home", "cardsAVG_away",
    "seasonAVG_overall",
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'Expect': '',
    'Connection': 'close',
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _fetch_with_retry(url: str, params: dict, context: str, max_retries: int = 3):
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)

            if resp.status_code == 429:
                print(f"  ⚠️  HTTP 429 dla {context} — czekam 60s...")
                time.sleep(60)
                continue

            resp.raise_for_status()

            if not resp.content:
                raise ValueError("Pusta odpowiedź")

            text = resp.text.strip()
            if "Connection failed" in text or "SQLSTATE" in text:
                raise ValueError(f"Błąd bazy danych API: {text[:100]}")

            return resp.json()

        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  ⚠️  Błąd (próba {attempt + 1}/{max_retries + 1}): {e} — retry za {wait}s")
                time.sleep(wait)
            else:
                raise
    return None


def _fetch_all_pages(season_id: int, context: str):
    """Pobiera wszystkie strony league-matches dla danego season_id."""
    url = f"{BASE_URL}/league-matches"
    params = {"key": API_KEY, "season_id": season_id, "max_per_page": 3000, "page": 1}

    data = _fetch_with_retry(url, params, context)
    if not data or not data.get("success"):
        raise ValueError(f"API error: {data.get('message', 'brak szczegółów') if data else 'brak odpowiedzi'}")

    matches = list(data.get("data", []))
    pager = data.get("pager", {})
    max_page = int(pager.get("max_page", 1))

    for page in range(2, max_page + 1):
        time.sleep(0.5)
        params["page"] = page
        page_data = _fetch_with_retry(url, params, f"{context} (strona {page})")
        if page_data and page_data.get("success"):
            matches.extend(page_data.get("data", []))

    return matches


def _log_error(msg: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(REPORTS_DIR, f"fetch_errors_{ts}.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


# ── TEAMS HELPERS ─────────────────────────────────────────────────────────────

def _fetch_teams(season_id: int, context: str):
    """Pobiera league-teams dla danego season_id i zwraca spłaszczony DataFrame."""
    url = f"{BASE_URL}/league-teams"
    params = {"key": API_KEY, "season_id": season_id, "include": "stats"}
    data = _fetch_with_retry(url, params, context)
    if not data or not data.get("success"):
        raise ValueError(
            f"API error: {data.get('message', 'brak') if data else 'brak odpowiedzi'}")
    teams = data.get("data", [])
    rows = []
    for t in teams:
        stats = t.get("stats", {})
        if isinstance(stats, str):
            try:
                stats = json.loads(stats)
            except Exception:
                stats = {}
        if not isinstance(stats, dict):
            stats = {}
        add = t.get("additional_info", {}) or {}
        row = {
            "team_id":   t.get("id"),
            "team_name": t.get("cleanName") or t.get("name") or t.get("shortName", ""),
            "season_id": season_id,
        }
        for key in TEAMS_STATS_KEYS:
            val = stats.get(key)
            if val is None:
                val = add.get(key)
            row[key] = val
        rows.append(row)
    return pd.DataFrame(rows)


# ── TRYB --historical ─────────────────────────────────────────────────────────

def run_historical(force: bool = False):
    print("=" * 50)
    print("TRYB: --historical")
    print("=" * 50)

    stats = {
        "allsvenskan":   {"seasons": 0, "matches": 0, "skipped": 0, "errors": 0,
                          "teams_seasons": 0, "teams_skipped": 0},
        "eliteserien":   {"seasons": 0, "matches": 0, "skipped": 0, "errors": 0,
                          "teams_seasons": 0, "teams_skipped": 0},
        "veikkausliiga": {"seasons": 0, "matches": 0, "skipped": 0, "errors": 0,
                          "teams_seasons": 0, "teams_skipped": 0},
    }
    total_skipped = 0
    total_errors = 0
    request_counter = 0

    for season in ALL_HISTORICAL:
        name = season["name"]
        sid = season["id"]
        league = LEAGUE_BY_SEASON_ID.get(sid)
        out_dir = LEAGUE_DIR_MAP.get(league, ALLSVENSKAN_DIR)
        csv_path = os.path.join(out_dir, f"advanced_league_matches_{sid}.csv")

        # Skip jeśli plik istnieje i brak --force
        if os.path.isfile(csv_path) and not force:
            print(f"  ⏭  {name} — już w cache")
            stats[league]["skipped"] += 1
            total_skipped += 1
        else:
            try:
                request_counter += 1
                if request_counter > 1 and (request_counter - 1) % 10 == 0:
                    print(f"  ⏳ Rate limit pause (10 requestów)...")
                    time.sleep(3)
                else:
                    time.sleep(0.5)

                matches = _fetch_all_pages(sid, name)
                df = pd.DataFrame(matches)
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")

                n = len(df)
                stats[league]["seasons"] += 1
                stats[league]["matches"] += n
                print(f"  ✅ {name} (id={sid}) — {n} meczów")

            except Exception as e:
                msg = f"{name} (id={sid}): {e}"
                print(f"  ❌ {name} — błąd: {e}")
                _log_error(msg)
                stats[league]["errors"] += 1
                total_errors += 1

        # Teams per sezon
        teams_dir = TEAMS_HIST_DIR_MAP.get(league, TEAMS_ALLSV_HIST)
        teams_path = os.path.join(teams_dir, f"advanced_league_teams_{sid}.csv")
        if os.path.isfile(teams_path) and not force:
            print(f"  ⏭  {name} teams — już w cache")
            stats[league]["teams_skipped"] += 1
        else:
            try:
                time.sleep(0.5)
                df_teams = _fetch_teams(sid, f"{name} teams")
                df_teams.to_csv(teams_path, index=False, encoding="utf-8-sig")
                stats[league]["teams_seasons"] += 1
                print(f"  ✅ {name} teams (id={sid}) — {len(df_teams)} drużyn")
            except Exception as e:
                print(f"  ❌ {name} teams — błąd: {e}")
                _log_error(f"{name} teams (id={sid}): {e}")

    # Podsumowanie
    total_seasons = sum(v["seasons"] for v in stats.values())
    total_matches = sum(v["matches"] for v in stats.values())
    total_teams_seasons = sum(v["teams_seasons"] for v in stats.values())
    print()
    print("══════════════════════════════════════════")
    print("HISTORYCZNE — PODSUMOWANIE")
    print("──────────────────────────────────────────")
    print(f"  Allsvenskan:   {stats['allsvenskan']['seasons']} sezony │ {stats['allsvenskan']['matches']} meczów")
    print(f"  Eliteserien:   {stats['eliteserien']['seasons']} sezony │ {stats['eliteserien']['matches']} meczów")
    print(f"  Veikkausliiga: {stats['veikkausliiga']['seasons']} sezony │ {stats['veikkausliiga']['matches']} meczów")
    print("──────────────────────────────────────────")
    print(f"  Łącznie:      {total_seasons} sezonów │ {total_matches} meczów")
    print(f"  Teams pobrane: {total_teams_seasons} sezonów")
    print(f"  Pominięto:    {total_skipped} (już w cache)")
    print(f"  Błędy:        {total_errors}")
    print("══════════════════════════════════════════")


# ── TRYB --daily ──────────────────────────────────────────────────────────────

def _fetch_daily_matches(date_str: str, label: str):
    """Pobiera wszystkie mecze z API dla danej daty (z paginacją)."""
    url = f"{BASE_URL}/todays-matches"
    all_matches = []
    page = 1

    while True:
        params = {
            "key": API_KEY,
            "date": date_str,
            "timezone": "Europe/Helsinki",
            "page": page,
        }
        try:
            data = _fetch_with_retry(url, params, f"{label} strona {page}")
        except Exception as e:
            print(f"  ❌ Błąd pobierania {label} (strona {page}): {e}")
            _log_error(f"daily {label} page {page}: {e}")
            break

        if not data or not data.get("success"):
            break

        batch = data.get("data", [])
        all_matches.extend(batch)

        next_page = data.get("pager", {}).get("next_page") or data.get("next_page")
        if not next_page or next_page <= page:
            break
        page = next_page
        time.sleep(0.5)

    return all_matches


def run_daily():
    print("=" * 50)
    print("TRYB: --daily")
    print("=" * 50)

    today = datetime.now()
    dates = {
        "today":    today.strftime("%Y-%m-%d"),
        "tomorrow": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
    }

    summary = {}

    for label, date_str in dates.items():
        print(f"\n  Pobieranie meczów: {date_str}...")
        matches = _fetch_daily_matches(date_str, date_str)

        if label == "today":
            fname = f"today_matches_{date_str}.csv"
        else:
            fname = f"tomorrow_matches_{date_str}.csv"

        csv_path = os.path.join(DAILY_DIR, fname)
        df = pd.DataFrame(matches)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # Zlicz per liga
        n_allsv = 0
        n_elite = 0
        n_veikk = 0
        n_other = 0

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
            else:
                n_other += 1

        summary[label] = {
            "date": date_str,
            "total": len(matches),
            "allsvenskan": n_allsv,
            "eliteserien": n_elite,
            "veikkausliiga": n_veikk,
            "other": n_other,
        }
        print(f"  ✅ {date_str} — {len(matches)} meczów → {csv_path}")

    # Current teams 2026
    print()
    print("  Pobieranie current teams 2026...")
    teams_summary = {}
    for sid, fname in [
        (ALLSVENSKAN_2026_ID,   "allsvenskan_teams_2026.csv"),
        (ELITESERIEN_2026_ID,   "eliteserien_teams_2026.csv"),
        (VEIKKAUSLIIGA_2026_ID, "veikkausliiga_teams_2026.csv"),
    ]:
        league = ACTIVE_LEAGUE_NAMES[sid]
        teams_path = os.path.join(CURRENT_DIR, fname)
        try:
            time.sleep(0.5)
            df_teams = _fetch_teams(sid, f"{league} 2026 current")
            df_teams.to_csv(teams_path, index=False, encoding="utf-8-sig")
            teams_summary[league] = len(df_teams)
            print(f"  ✅ {league.capitalize()} 2026: {len(df_teams)} drużyn")
        except Exception as e:
            print(f"  ❌ {league} teams — błąd: {e}")
            _log_error(f"daily teams {league}: {e}")
            teams_summary[league] = 0

    # Podsumowanie
    print()
    print("══════════════════════════════════════════")
    print("DAILY — PODSUMOWANIE")
    print("──────────────────────────────────────────")
    for label in ("today", "tomorrow"):
        s = summary.get(label, {})
        lbl = "Dziś " if label == "today" else "Jutro"
        print(f"  {lbl} ({s.get('date', '?')}): {s.get('total', 0)} meczów łącznie")
        print(f"    Allsvenskan:   {s.get('allsvenskan', 0)}")
        print(f"    Eliteserien:   {s.get('eliteserien', 0)}")
        print(f"    Veikkausliiga: {s.get('veikkausliiga', 0)}")
        print(f"    Inne:          {s.get('other', 0)}")
    print("──────────────────────────────────────────")
    print("  Current teams zaktualizowano:")
    for lg, n in teams_summary.items():
        print(f"    {lg.capitalize()} 2026: {n} drużyn")
    print("══════════════════════════════════════════")


# ── TRYB --weekly ─────────────────────────────────────────────────────────────

def run_weekly():
    print("=" * 50)
    print("TRYB: --weekly")
    print("=" * 50)

    weekly_summary = {}

    for sid, fname_matches, fname_teams in [
        (ALLSVENSKAN_2026_ID,
         "allsvenskan_matches_2026.csv",
         "allsvenskan_teams_2026.csv"),
        (ELITESERIEN_2026_ID,
         "eliteserien_matches_2026.csv",
         "eliteserien_teams_2026.csv"),
        (VEIKKAUSLIIGA_2026_ID,
         "veikkausliiga_matches_2026.csv",
         "veikkausliiga_teams_2026.csv"),
    ]:
        league = ACTIVE_LEAGUE_NAMES[sid]
        print(f"\n  {league.capitalize()} 2026...")

        # A) Matches 2026
        try:
            time.sleep(0.5)
            matches = _fetch_all_pages(sid, f"{league} 2026")
            complete = [m for m in matches if str(m.get("status", "")).lower() == "complete"]
            df_m = pd.DataFrame(complete)
            m_path = os.path.join(CURRENT_DIR, fname_matches)
            df_m.to_csv(m_path, index=False, encoding="utf-8-sig")
            n_complete = len(complete)
            print(f"  ✅ {league} matches: {n_complete} complete → {m_path}")
        except Exception as e:
            print(f"  ❌ {league} matches — błąd: {e}")
            _log_error(f"weekly matches {league}: {e}")
            n_complete = 0

        # B) Teams 2026
        try:
            time.sleep(0.5)
            df_t = _fetch_teams(sid, f"{league} 2026 teams")
            t_path = os.path.join(CURRENT_DIR, fname_teams)
            df_t.to_csv(t_path, index=False, encoding="utf-8-sig")
            n_teams = len(df_t)
            print(f"  ✅ {league} teams: {n_teams} → {t_path}")
        except Exception as e:
            print(f"  ❌ {league} teams — błąd: {e}")
            _log_error(f"weekly teams {league}: {e}")
            n_teams = 0

        weekly_summary[league] = {"matches": n_complete, "teams": n_teams}

    print()
    print("══════════════════════════════════════════")
    print("WEEKLY — PODSUMOWANIE")
    print("──────────────────────────────────────────")
    for lg, s in weekly_summary.items():
        print(f"  {lg.capitalize():15} {s['matches']:4} complete meczów, {s['teams']} drużyn")
    print("══════════════════════════════════════════")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pobieranie danych dla nordic_2026 (Allsvenskan / Eliteserien / Veikkausliiga)"
    )
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Pobierz dane historyczne (2022–2025) dla wszystkich 3 lig",
    )
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Pobierz mecze na dziś i jutro + current teams 2026",
    )
    parser.add_argument(
        "--weekly",
        action="store_true",
        help="Pobierz current matches + teams 2026 (do retreningu modeli)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Nadpisz istniejące pliki (używaj z --historical)",
    )

    args = parser.parse_args()

    if not args.historical and not args.daily and not args.weekly:
        parser.print_help()
        sys.exit(0)

    if args.historical:
        run_historical(force=args.force)

    if args.daily:
        run_daily()

    if args.weekly:
        run_weekly()


if __name__ == "__main__":
    main()
