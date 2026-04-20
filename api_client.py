import requests
from nordic_config import API_KEY
import time
import random
from functools import lru_cache

BASE_URL = "https://api.football-data-api.com/"

def _get_safe_headers():
    """
    Zwraca nagłówki HTTP bezpieczne dla API (bez błędu 417).
    Fix dla problemu 'Expectation Failed' z niektórymi ligami.
    """
    return {
        'User-Agent': 'Mozilla/5.0',
        'Expect': '',  # Wyłącz nagłówek Expect (powoduje błąd 417)
        'Connection': 'close'  # Unikaj keep-alive problems
    }

def _handle_response_with_retry(url, params, headers, context="API call", max_retries=3):
    """
    Helper function to handle API responses with retry logic and better error reporting.
    """
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)

            if not response.content:
                print(f"API returned empty response for {context} (attempt {attempt + 1})")
                if attempt < max_retries:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                return None

            response_text = response.text.strip()
            if "Connection failed" in response_text or "SQLSTATE" in response_text:
                print(f"API database connection error for {context} (attempt {attempt + 1}): {response_text}")
                if attempt < max_retries:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                return None

            try:
                response.raise_for_status()
                data = response.json()
                return data

            except ValueError as json_error:
                print(f"JSON parsing error for {context} (attempt {attempt + 1}): {json_error}")
                print(f"Response status code: {response.status_code}")
                print(f"Response content (first 200 chars): {response_text[:200]}")
                if attempt < max_retries:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                return None

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error for {context} (attempt {attempt + 1}): {e}")

            if '417' in str(e):
                print(f"   → Błąd 417 Expectation Failed - próbuję z innymi nagłówkami...")
                headers_fixed = headers.copy()
                headers_fixed['Expect'] = ''
                headers_fixed['Connection'] = 'close'

                try:
                    response = requests.get(url, params=params, headers=headers_fixed, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    print(f"   ✅ Sukces po fiksie nagłówków!")
                    return data
                except Exception as fix_error:
                    print(f"   ❌ Fix nie pomógł: {fix_error}")

            if attempt < max_retries:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                continue
            return None

        except requests.exceptions.RequestException as e:
            print(f"Request error for {context} (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                continue
            return None

    print(f"❌ All retry attempts failed for {context}")
    return None

def _handle_response(response, context="API call"):
    try:
        response.raise_for_status()

        if not response.content:
            print(f"API returned empty response for {context}")
            return None

        try:
            data = response.json()
        except ValueError as json_error:
            print(f"JSON parsing error for {context}: {json_error}")
            print(f"Response status code: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            print(f"Response content (first 500 chars): {response.text[:500]}")
            return None

        return data

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error for {context}: {e}")
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text[:500]}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error for {context}: {e}")
        return None

def get_todays_matches(use_retry=True):
    endpoint = "todays-matches"
    params = {'key': API_KEY}
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f"{BASE_URL}{endpoint}"

    if use_retry:
        data = _handle_response_with_retry(url, params, headers, "today's matches")
    else:
        try:
            response = requests.get(url, params=params, headers=headers)
            data = _handle_response(response, "today's matches")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while connecting to the API: {e}")
            return None

    if data and data.get("success"):
        return data.get("data", [])
    else:
        if data:
            print(f"API returned an error: {data.get('message', 'Unknown error')}")
        return None

def get_match_details(match_id):
    try:
        match_id = int(float(str(match_id).strip()))
    except (TypeError, ValueError):
        pass
    endpoint = "match"
    params = {'key': API_KEY, 'match_id': match_id}
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers)
        data = _handle_response(response, f"match details for ID {match_id}")

        if data and data.get("success"):
            return data
        else:
            if data:
                print(f"API returned an error for match {match_id}: {data.get('message', 'Unknown error')}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching match details: {e}")
        return None

def get_league_season_data(season_id):
    endpoint = "league-season"
    params = {'key': API_KEY, 'season_id': season_id}
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers)
        data = _handle_response(response, f"league season data for ID {season_id}")

        if data and data.get("success"):
            return data
        else:
            if data:
                print(f"API returned an error for season {season_id}: {data.get('message', 'Unknown error')}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching league season data: {e}")
        return None

def get_league_teams(season_id, include_stats=True, use_retry=True):
    endpoint = "league-teams"
    params = {'key': API_KEY, 'season_id': season_id}

    if include_stats:
        params['include'] = 'stats'

    headers = _get_safe_headers()
    url = f"{BASE_URL}{endpoint}"

    if use_retry:
        data = _handle_response_with_retry(url, params, headers, f"league teams for season {season_id}")
    else:
        try:
            response = requests.get(url, params=params, headers=headers)
            data = _handle_response(response, f"league teams for season {season_id}")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching league teams: {e}")
            return None

    if data and data.get("success"):
        return data
    else:
        if data:
            print(f"API returned an error for league teams (season {season_id}): {data.get('message', 'Unknown error')}")
        return None

def get_league_matches(season_id, use_retry=True):
    endpoint = "league-matches"
    params = {'key': API_KEY, 'season_id': season_id}
    headers = _get_safe_headers()
    url = f"{BASE_URL}{endpoint}"

    if use_retry:
        data = _handle_response_with_retry(url, params, headers, f"league matches for season {season_id}")
    else:
        try:
            response = requests.get(url, params=params, headers=headers)
            data = _handle_response(response, f"league matches for season {season_id}")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching league matches: {e}")
            return None

    if data and data.get("success"):
        return data
    else:
        if data:
            print(f"API returned an error for league matches (season {season_id}): {data.get('message', 'Unknown error')}")
        return None

@lru_cache(maxsize=256)
def get_matches_by_date(date_str: str, timezone: str = "Europe/Helsinki", page: int = 1):
    endpoint = "todays-matches"
    params = {
        'key': API_KEY,
        'date': date_str,
        'timezone': timezone,
        'page': page
    }
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f"{BASE_URL}{endpoint}"

    try:
        response = requests.get(url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json() or {}
        matches = data.get("data", []) or []

        if data.get("next_page", None):
            next_page_matches = get_matches_by_date(date_str, timezone, page=data["next_page"])
            matches.extend(next_page_matches)

        return matches

    except requests.exceptions.RequestException as e:
        print(f"❌ Błąd API dla daty {date_str}: {e}")
        return []
    except Exception as e:
        print(f"❌ Nieoczekiwany błąd dla daty {date_str}: {e}")
        return []

def test_api_connection():
    print("🔍 Testing API connection...")
    print(f"🔗 Base URL: {BASE_URL}")
    print(f"🔑 API Key (first 10 chars): {API_KEY[:10] if API_KEY else 'No API key'}...")

    endpoint = "league-list"
    params = {'key': API_KEY}
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f"{BASE_URL}{endpoint}"

    print(f"🌐 Testing URL: {url}")

    try:
        response = requests.get(url, params=params, headers=headers)
        print(f"📊 Response Status: {response.status_code}")
        print(f"📊 Response Length: {len(response.content)} bytes")

        if response.content:
            print(f"📊 Response Content (first 200 chars):")
            print(f"   {response.text[:200]}")

        try:
            data = response.json()
            print(f"✅ JSON parsing successful")
            if isinstance(data, dict) and 'success' in data:
                print(f"📊 API Success: {data.get('success')}")
        except ValueError as e:
            print(f"❌ JSON parsing failed: {e}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

    print("🔍 API connection test complete")

if __name__ == "__main__":
    test_api_connection()
