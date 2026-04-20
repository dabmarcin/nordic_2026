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

from nordic_config import (
    API_KEY, BASE_URL,
    ALL_HISTORICAL, LEAGUE_BY_SEASON_ID,
    ALLSVENSKAN_DIR, ELITESERIEN_DIR, VEIKKAUSLIIGA_DIR,
    ALLSVENSKAN_HISTORICAL, ELITESERIEN_HISTORICAL, VEIKKAUSLIIGA_HISTORICAL,
    ALLSVENSKAN_2026_ID, ELITESERIEN_2026_ID, VEIKKAUSLIIGA_2026_ID,
    DAILY_DIR, REPORTS_DIR,
)

LEAGUE_DIR_MAP = {
    "allsvenskan":   ALLSVENSKAN_DIR,
    "eliteserien":   ELITESERIEN_DIR,
    "veikkausliiga": VEIKKAUSLIIGA_DIR,
}

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


# ── TRYB --historical ─────────────────────────────────────────────────────────

def run_historical(force: bool = False):
    print("=" * 50)
    print("TRYB: --historical")
    print("=" * 50)

    stats = {
        "allsvenskan":   {"seasons": 0, "matches": 0, "skipped": 0, "errors": 0},
        "eliteserien":   {"seasons": 0, "matches": 0, "skipped": 0, "errors": 0},
        "veikkausliiga": {"seasons": 0, "matches": 0, "skipped": 0, "errors": 0},
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
            continue

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

    # Podsumowanie
    total_seasons = sum(v["seasons"] for v in stats.values())
    total_matches = sum(v["matches"] for v in stats.values())
    print()
    print("══════════════════════════════════════════")
    print("HISTORYCZNE — PODSUMOWANIE")
    print("──────────────────────────────────────────")
    print(f"  Allsvenskan:   {stats['allsvenskan']['seasons']} sezony │ {stats['allsvenskan']['matches']} meczów")
    print(f"  Eliteserien:   {stats['eliteserien']['seasons']} sezony │ {stats['eliteserien']['matches']} meczów")
    print(f"  Veikkausliiga: {stats['veikkausliiga']['seasons']} sezony │ {stats['veikkausliiga']['matches']} meczów")
    print("──────────────────────────────────────────")
    print(f"  Łącznie:      {total_seasons} sezonów │ {total_matches} meczów")
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
        help="Pobierz mecze na dziś i jutro",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Nadpisz istniejące pliki (używaj z --historical)",
    )

    args = parser.parse_args()

    if not args.historical and not args.daily:
        parser.print_help()
        sys.exit(0)

    if args.historical:
        run_historical(force=args.force)

    if args.daily:
        run_daily()


if __name__ == "__main__":
    main()
