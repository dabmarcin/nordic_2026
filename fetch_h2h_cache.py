import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from nordic_config import API_KEY, BASE_URL, H2H_CACHE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LIMIT_PER_HOUR = 1600
PAUSE_SECONDS = 3900
DELAY_BETWEEN = 0.5
ENDPOINT = f"{BASE_URL}/match"


def collect_match_ids():
    patterns = [
        os.path.join(os.path.dirname(__file__), "data", "historical", "allsvenskan",   "advanced_league_matches_*.csv"),
        os.path.join(os.path.dirname(__file__), "data", "historical", "eliteserien",   "advanced_league_matches_*.csv"),
        os.path.join(os.path.dirname(__file__), "data", "historical", "veikkausliiga", "advanced_league_matches_*.csv"),
    ]
    files = sorted(f for pattern in patterns for f in glob.glob(pattern))
    ids = set()

    for path in files:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", usecols=["id"])
            col = pd.to_numeric(df["id"], errors="coerce").dropna().astype(int)
            ids.update(col.tolist())
        except Exception as exc:
            print(f"[WARN] Nie udało się wczytać {os.path.basename(path)}: {exc}")

    return files, ids


def existing_cached_ids():
    cached = set()
    for path in glob.glob(os.path.join(H2H_CACHE, "match_*.json")):
        name = os.path.basename(path)
        try:
            match_id = int(name.replace("match_", "").replace(".json", ""))
            cached.add(match_id)
        except Exception:
            continue
    return cached


def estimate_duration(seconds_total):
    hours = int(seconds_total // 3600)
    minutes = int((seconds_total % 3600) // 60)
    return hours, minutes


def pause_with_countdown(seconds):
    resume_at = datetime.now() + timedelta(seconds=seconds)
    print(f"⏸ Limit {LIMIT_PER_HOUR} osiągnięty. Pauza do {resume_at.strftime('%H:%M:%S')}...")
    minutes_left = int(seconds // 60)
    while minutes_left > 0:
        print(f"⏳ Wznowienie za {minutes_left} minut...")
        time.sleep(60)
        minutes_left -= 1
    remaining = int(seconds % 60)
    if remaining > 0:
        time.sleep(remaining)
    print("▶ Wznawianie pobierania...")


def fetch_one_match(match_id):
    url = f"{ENDPOINT}?key={API_KEY}&match_id={match_id}"
    headers = {"User-Agent": "Mozilla/5.0"}

    for attempt in range(3):
        try:
            response = requests.get(url, timeout=30, headers=headers)
            status = response.status_code
            if status == 429:
                return {"status": "rate_limited", "payload": None}
            if status != 200:
                return {"status": "http_error", "http_status": status, "payload": None}

            payload = response.json()
            if payload.get("success"):
                return {"status": "ok", "payload": payload}
            return {"status": "success_false", "payload": payload}
        except Exception as exc:
            if attempt == 2:
                return {"status": "exception", "error": str(exc), "payload": None}
            backoff = 2 ** attempt
            time.sleep(backoff)

    return {"status": "exception", "error": "unknown", "payload": None}


def main(dry_run=False):
    os.makedirs(H2H_CACHE, exist_ok=True)

    files, all_ids = collect_match_ids()
    cached_ids = existing_cached_ids()
    to_download = sorted(all_ids - cached_ids)

    total = len(all_ids)
    already = len(cached_ids & all_ids)
    remaining = len(to_download)
    est_seconds = (remaining / LIMIT_PER_HOUR) * 3600 if remaining > 0 else 0
    est_h, est_m = estimate_duration(est_seconds)

    print("════════════════════════════════════════")
    print("FETCH H2H CACHE — Nordic 2026")
    print("════════════════════════════════════════")
    print(f"Pliki historical:     {len(files)}")
    print(f"Unikalne match_id:    {total}")
    print(f"Już w cache:          {already}")
    print(f"Do pobrania:          {remaining}")
    print(f"Szacowany czas:       ~{est_h} godzin {est_m} minut")
    print("════════════════════════════════════════")

    print(f"Znaleziono {total} unikalnych match_id z {len(files)} plików")
    print(f"Do pobrania: {remaining} | Już w cache: {already} | Łącznie: {total}")
    print(f"Szacowany czas (przy 1600/h): ~{est_h} godzin")

    if dry_run:
        print("[DRY-RUN] Bez pobierania danych.")
        return

    requests_this_hour = 0
    downloaded = 0
    skipped = already
    errors = 0
    start_time = time.time()

    idx = 0
    while idx < len(to_download):
        match_id = to_download[idx]

        if requests_this_hour >= LIMIT_PER_HOUR:
            pause_with_countdown(PAUSE_SECONDS)
            requests_this_hour = 0

        result = fetch_one_match(match_id)
        status = result.get("status")

        if status == "rate_limited":
            print("[429] Rate limit — natychmiastowa pauza 65 min")
            pause_with_countdown(PAUSE_SECONDS)
            requests_this_hour = 0
            continue

        if status == "ok":
            out_path = os.path.join(H2H_CACHE, f"match_{match_id}.json")
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result["payload"], f, ensure_ascii=False, indent=2)
                downloaded += 1
            except Exception as exc:
                errors += 1
                print(f"[ERR] match_{match_id}: zapis JSON nieudany ({exc})")
            requests_this_hour += 1

        elif status == "success_false":
            print(f"[WARN] match_{match_id}: success=false — pomijam")
            requests_this_hour += 1
            errors += 1

        elif status == "http_error":
            print(f"[ERR] match_{match_id}: HTTP {result.get('http_status')} — pomijam")
            requests_this_hour += 1
            errors += 1

        else:
            print(f"[ERR] match_{match_id}: {result.get('error', 'unknown')} — pomijam")
            errors += 1

        processed = idx + 1
        if processed % 50 == 0:
            elapsed = int(time.time() - start_time)
            em = elapsed // 60
            es = elapsed % 60
            print(
                f"[{processed}/{len(to_download)}] match_{match_id} ✓ | "
                f"req/h={requests_this_hour} | elapsed={em}m{es:02d}s"
            )

        idx += 1
        time.sleep(DELAY_BETWEEN)

    total_cache_now = len(existing_cached_ids())
    elapsed_total = int(time.time() - start_time)
    h_total = elapsed_total // 3600
    m_total = (elapsed_total % 3600) // 60

    print("════════════════════════════════════════")
    print("GOTOWE")
    print(f"Pobrano:       {downloaded} plików")
    print(f"Pominięto:     {skipped} (już były w cache)")
    print(f"Błędy:         {errors}")
    print(f"Cache łącznie: {total_cache_now} plików")
    print(f"Czas całkowity: {h_total}h {m_total}m")
    print("════════════════════════════════════════")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pobiera cache H2H dla meczów historical")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tylko statystyki bez pobierania danych",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
