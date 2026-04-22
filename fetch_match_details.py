import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import time
import argparse
import datetime

import requests
import pandas as pd

from nordic_config import (
    API_KEY, BASE_URL,
    DAILY_DIR,
    ACTIVE_SEASON_IDS,
    ALLSVENSKAN_2026_ID, ELITESERIEN_2026_ID, VEIKKAUSLIIGA_2026_ID,
    MATCH_DETAILS_DIR,
)

ENDPOINT = f"{BASE_URL}/match"
DELAY = 0.5

BOOKMAKER_PRIORITY = ['Pinnacle', 'bet365', '1xbet', '10Bet', 'WilliamHill', 'Unibet', '888Sport']

LEAGUE_NAME_MAP = {
    ALLSVENSKAN_2026_ID:   "allsvenskan",
    ELITESERIEN_2026_ID:   "eliteserien",
    VEIKKAUSLIIGA_2026_ID: "veikkausliiga",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Pobiera szczegóły meczów nordic z FootyStats /match API")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--today',    action='store_true', help="Mecze na dziś")
    group.add_argument('--tomorrow', action='store_true', help="Mecze na jutro")
    group.add_argument('--date',     type=str, metavar='YYYY-MM-DD', help="Konkretna data")
    parser.add_argument('--force',   action='store_true', help="Pobierz ponownie nawet z cache")
    return parser.parse_args()


def load_match_ids(args):
    today = datetime.date.today()
    if args.today:
        date_str = today.strftime('%Y-%m-%d')
        prefix = 'today_matches'
    elif args.tomorrow:
        date_str = (today + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        prefix = 'tomorrow_matches'
    else:
        date_str = args.date
        prefix = 'today_matches'

    path = os.path.join(DAILY_DIR, f'{prefix}_{date_str}.csv')
    if not os.path.isfile(path):
        print(f'[ERROR] Brak pliku: {path}')
        return [], date_str, {}

    df = pd.read_csv(path, encoding='utf-8-sig')
    df['competition_id'] = pd.to_numeric(df['competition_id'], errors='coerce')
    df_nordic = df[df['competition_id'].isin(ACTIVE_SEASON_IDS)].copy()

    matches = []
    per_league = {ALLSVENSKAN_2026_ID: 0, ELITESERIEN_2026_ID: 0, VEIKKAUSLIIGA_2026_ID: 0}
    for _, row in df_nordic.iterrows():
        mid = int(row['id'])
        cid = int(row['competition_id'])
        per_league[cid] = per_league.get(cid, 0) + 1
        matches.append({
            'match_id':       mid,
            'competition_id': cid,
            'home':           str(row.get('home_name', mid)),
            'away':           str(row.get('away_name', mid)),
        })

    return matches, date_str, per_league


def fetch_match(match_id):
    params = {'key': API_KEY, 'match_id': match_id}
    headers = {'User-Agent': 'Mozilla/5.0'}
    for attempt in range(3):
        try:
            resp = requests.get(ENDPOINT, params=params, headers=headers, timeout=30)
            if resp.status_code == 429:
                print(f'    [429] Rate limit — czekam 60s...')
                time.sleep(60)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f'    [WARN] Próba {attempt + 1}/3: {e} — czekam {wait}s')
            time.sleep(wait)
    return None


def extract_pinnacle_odds(data):
    pin = {
        'home':              None,
        'draw':              None,
        'away':              None,
        'over25':            None,
        'btts_yes':          None,
        'corners_over_95':   None,
        'corners_under_95':  None,
    }
    oc = data.get('odds_comparison', {})
    if not oc or not isinstance(oc, dict):
        return pin

    def _get(val):
        try:
            f = float(val)
            return f if f > 0 else None
        except (TypeError, ValueError):
            return None

    def _best(outcome_dict):
        if not isinstance(outcome_dict, dict):
            return None
        for bk in BOOKMAKER_PRIORITY:
            val = _get(outcome_dict.get(bk, 0))
            if val is not None:
                return val
        return None

    # FT Result
    ft = oc.get('FT Result', {})
    pin['home'] = _best(ft.get('Home', {}))
    pin['draw'] = _best(ft.get('Draw', {}))
    pin['away'] = _best(ft.get('Away', {}))

    # Goals Over/Under 2.5
    goals  = oc.get('Goals Over/Under', {})
    over25 = goals.get('Over 2.5', {})
    pin['over25'] = _best(over25)

    # Both Teams To Score
    btts_oc  = oc.get('Both Teams To Score', {})
    btts_yes = btts_oc.get('Yes', {})
    pin['btts_yes'] = _best(btts_yes)

    # Corners Over/Under 9.5
    corners_oc = oc.get('Corners', {})
    over95  = corners_oc.get('Over 9.5', {})
    under95 = corners_oc.get('Under 9.5', {})
    pin['corners_over_95']  = _best(over95)
    pin['corners_under_95'] = _best(under95)

    return pin


def _get_best_odds(bookmakers: dict) -> float | None:
    """Zwraca kurs Pinnacle jeśli dostępny, fallback — max z dostępnych."""
    if not isinstance(bookmakers, dict) or not bookmakers:
        return None
    if 'Pinnacle' in bookmakers:
        try:
            v = float(bookmakers['Pinnacle'])
            if v > 1.0:
                return v
        except (TypeError, ValueError):
            pass
    vals = []
    for v in bookmakers.values():
        try:
            f = float(v)
            if f > 1.0:
                vals.append(f)
        except (TypeError, ValueError):
            pass
    return max(vals) if vals else None


def _resolve_odds(tip: dict, odds_summary: dict) -> float | None:
    """Uzupełnia kurs dla typu gdy GPT go nie podał."""
    kierunek = tip.get('kierunek', '')
    linia    = tip.get('linia')
    rynek    = tip.get('rynek', '').lower()

    direct = {
        'home_win': 'home',  'away_win': 'away', 'draw': 'draw',
        'btts_yes': 'btts_yes', 'btts_no': 'btts_no',
        '1x': '1x', 'x2': 'x2', '12': '12',
    }
    if kierunek in direct:
        return odds_summary.get(direct[kierunek])

    if linia:
        line_str = str(linia).replace('.', '')
        if 'corner' in rynek:
            if kierunek == 'over':
                return odds_summary.get(f'corners_over_{line_str}')
            if kierunek == 'under':
                return odds_summary.get(f'corners_under_{line_str}')
        else:
            if kierunek == 'over':
                return odds_summary.get(f'over{line_str}')
            if kierunek == 'under':
                return odds_summary.get(f'under{line_str}')
    return None


def extract_gpt_tips_claude(gpt_pl: str, gpt_en: str,
                             odds_comparison: dict) -> list:
    """Zwraca listę polecanych typów zakładowych wyciągniętych z tekstu GPT."""
    tekst = gpt_pl.strip() if gpt_pl and gpt_pl.strip() else gpt_en.strip()
    if not tekst or len(tekst) < 50:
        return []

    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    if not ANTHROPIC_API_KEY:
        return []

    # Buduj skrócone odds_summary dla Claude
    oc = odds_comparison if isinstance(odds_comparison, dict) else {}
    odds_summary: dict = {}
    try:
        ft = oc.get('FT Result', {})
        odds_summary['home'] = _get_best_odds(ft.get('Home', {}))
        odds_summary['draw'] = _get_best_odds(ft.get('Draw', {}))
        odds_summary['away'] = _get_best_odds(ft.get('Away', {}))
        dc = oc.get('Double Chance', {})
        odds_summary['1x'] = _get_best_odds(dc.get('Home/Draw', {}))
        odds_summary['x2'] = _get_best_odds(dc.get('Draw/Away', {}))
        odds_summary['12'] = _get_best_odds(dc.get('Home/Away', {}))
        goals = oc.get('Goals Over/Under', {})
        for line in ['1.5', '2.5', '3.5', '4.5']:
            k = line.replace('.', '')
            odds_summary[f'over{k}']  = _get_best_odds(goals.get(f'Over {line}', {}))
            odds_summary[f'under{k}'] = _get_best_odds(goals.get(f'Under {line}', {}))
        btts = oc.get('Both Teams To Score', {})
        odds_summary['btts_yes'] = _get_best_odds(btts.get('Yes', {}))
        odds_summary['btts_no']  = _get_best_odds(btts.get('No', {}))
        corners = oc.get('Corners', {})
        for line in ['7.5', '8.5', '9.5', '10.5', '11.5']:
            k = line.replace('.', '')
            odds_summary[f'corners_over_{k}']  = _get_best_odds(corners.get(f'Over {line}', {}))
            odds_summary[f'corners_under_{k}'] = _get_best_odds(corners.get(f'Under {line}', {}))
    except Exception:
        pass

    SYSTEM_PROMPT = (
        'Jesteś ekspertem analizy zakładów sportowych. Przeczytaj analizę meczu '
        'i wyciągnij TYLKO polecane typy zakładowe '
        '(nie ryzykowne, nie "bezpieczne alternatywy" — '
        'tylko główne rekomendacje oznaczone jako polecane, główne lub "najbardziej wartościowe").\n\n'
        'Zwróć TYLKO tablicę JSON bez żadnego tekstu przed ani po. Nie używaj markdown.\n\n'
        'Dla każdego polecanego typu zwróć obiekt:\n'
        '{\n'
        '  "typ": "czytelna nazwa np. KuPS wygra / BTTS Yes / Over 2.5 / Corners Over 9.5",\n'
        '  "rynek": "1X2 / BTTS / Over25 / Under25 / Over15 / Corners / DC",\n'
        '  "kierunek": "home_win / away_win / draw / btts_yes / btts_no / over / under / 1x / x2 / 12",\n'
        '  "linia": null lub liczba np. 2.5 dla Over/Under goli, 9.5 dla corners,\n'
        '  "kurs": liczba z tekstu lub null jeśli brak,\n'
        '  "pewnosc": "wysoka / srednia / niska",\n'
        '  "uzasadnienie": "max 100 znaków"\n'
        '}\n\n'
        'Jeśli kurs nie jest podany w tekście → null (zostanie uzupełniony z rynku).\n'
        'Jeśli nie ma żadnych polecanych typów → zwróć [].\n'
        'Zwróć maksymalnie 5 typów.'
    )

    user_msg = (
        f'Analiza meczu:\n\n{tekst[:4000]}\n\n'
        f'Dostępne kursy rynkowe:\n'
        f'{json.dumps(odds_summary, ensure_ascii=False)}'
    )

    try:
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key':         ANTHROPIC_API_KEY,
                'anthropic-version': '2023-06-01',
                'content-type':      'application/json',
            },
            json={
                'model':    'claude-sonnet-4-20250514',
                'max_tokens': 800,
                'system':   SYSTEM_PROMPT,
                'messages': [{'role': 'user', 'content': user_msg}],
            },
            timeout=30,
        )
        time.sleep(1)

        if response.status_code == 200:
            text = response.json()['content'][0]['text']
            text = text.replace('```json', '').replace('```', '').strip()
            tips = json.loads(text)
            if not isinstance(tips, list):
                return []
            for tip in tips:
                if tip.get('kurs') is None:
                    tip['kurs'] = _resolve_odds(tip, odds_summary)
            return tips

        if response.status_code == 529:
            print('  [WARN] Anthropic overloaded — pominięto GPT tips')
            return []

        return []

    except Exception as e:
        print(f'  [WARN] Claude tips error: {e}')
        return []


def build_gpt_data(match_id, data, competition_id):
    liga = LEAGUE_NAME_MAP.get(competition_id, 'allsvenskan')

    oc           = data.get('odds_comparison', {})
    pinnacle     = extract_pinnacle_odds(data)
    gpt_int      = data.get('gpt_int', {}) if isinstance(data.get('gpt_int'), dict) else {}
    gpt_en       = data.get('gpt_en') or gpt_int.get('en') or ''
    gpt_pl       = gpt_int.get('pl') or ''
    gpt_se       = gpt_int.get('se') or ''
    gpt_no       = gpt_int.get('no') or ''
    gpt_tips     = extract_gpt_tips_claude(gpt_pl, gpt_en, oc)

    # xG
    xg_home = data.get('team_a_xg_prematch') or data.get('xg_home')
    xg_away = data.get('team_b_xg_prematch') or data.get('xg_away')
    xg_total = None
    if xg_home is not None and xg_away is not None:
        try:
            xg_total = round(float(xg_home) + float(xg_away), 2)
        except Exception:
            pass

    # H2H
    h2h_raw = data.get('h2h_stats') or data.get('h2h') or {}
    if isinstance(h2h_raw, dict):
        h2h_stats = {
            'total_matches': h2h_raw.get('total_matches', 0),
            'btts_pct':      h2h_raw.get('btts_pct', 0),
            'over25_pct':    h2h_raw.get('over25_pct', 0),
            'home_win_pct':  h2h_raw.get('home_win_pct', 0),
            'avg_goals':     h2h_raw.get('avg_goals', 0),
        }
    else:
        h2h_stats = {'total_matches': 0, 'btts_pct': 0, 'over25_pct': 0,
                     'home_win_pct': 0, 'avg_goals': 0}

    # Potencjały
    potencjaly = {
        'btts_potential':        data.get('btts_potential', 50),
        'o25_potential':         data.get('o25_potential', 50),
        'corners_o95_potential': data.get('corners_o95_potential') or data.get('corners_potential', 50),
        'avg_potential':         data.get('avg_potential', 2.5),
    }

    # Nazwy drużyn
    home_name = ''
    away_name = ''
    if isinstance(data.get('homeTeam'), dict):
        home_name = data['homeTeam'].get('name', '')
    if not home_name:
        home_name = data.get('home_name', '')
    if isinstance(data.get('awayTeam'), dict):
        away_name = data['awayTeam'].get('name', '')
    if not away_name:
        away_name = data.get('away_name', '')

    # Data meczu
    date_val = data.get('date') or data.get('date_string', '')
    if not date_val and data.get('date_unix'):
        try:
            date_val = datetime.datetime.fromtimestamp(
                int(data['date_unix'])
            ).strftime('%Y-%m-%d')
        except Exception:
            date_val = ''

    return {
        'match_id':       match_id,
        'home':           home_name,
        'away':           away_name,
        'date':           str(date_val),
        'competition_id': competition_id,
        'liga':           liga,
        'stadium':        data.get('stadium', '') or data.get('venue', ''),
        'gpt_pl':         gpt_pl,
        'gpt_en':         gpt_en,
        'gpt_se':         gpt_se,
        'gpt_no':         gpt_no,
        'gpt_tips':       gpt_tips,
        'pinnacle_odds':  pinnacle,
        'potencjaly':     potencjaly,
        'h2h_stats':      h2h_stats,
        'xg_prematch': {
            'home':  xg_home,
            'away':  xg_away,
            'total': xg_total,
        },
        'pobrano': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
    }


def main():
    args = parse_args()
    matches, date_str, per_league = load_match_ids(args)

    n_total = len(matches)
    n_allsv = per_league.get(ALLSVENSKAN_2026_ID, 0)
    n_elite = per_league.get(ELITESERIEN_2026_ID, 0)
    n_veikk = per_league.get(VEIKKAUSLIIGA_2026_ID, 0)

    print('══════════════════════════════════════════════════')
    print('FETCH MATCH DETAILS — Nordic 2026')
    print('══════════════════════════════════════════════════')
    print(f'Data: {date_str}')
    print(f'Meczów nordic: {n_total}')
    print(f'  Allsvenskan:   {n_allsv}')
    print(f'  Eliteserien:   {n_elite}')
    print(f'  Veikkausliiga: {n_veikk}')
    print()

    if not matches:
        print('Brak meczów do pobrania.')
        return

    n_fetched   = 0
    n_refreshed = 0
    n_skipped   = 0
    n_errors    = 0
    n_gpt       = 0

    for match in matches:
        mid   = match['match_id']
        cid   = match['competition_id']
        home  = match['home']
        away  = match['away']
        label = f'{home} vs {away}'

        detail_path = os.path.join(MATCH_DETAILS_DIR, f'match_{mid}.json')
        gpt_path    = os.path.join(MATCH_DETAILS_DIR, f'gpt_{mid}.json')

        was_incomplete = False

        # Skip logic
        if os.path.isfile(detail_path) and not args.force:
            try:
                with open(detail_path, encoding='utf-8') as f:
                    cached = json.load(f)
                status = (
                    cached.get('data', {}).get('status', '')
                    or cached.get('status', '')
                )
                if str(status).lower() != 'incomplete':
                    print(f'  ⏭  match_{mid} → już w cache (complete)')
                    n_skipped += 1
                    time.sleep(DELAY)
                    continue
                else:
                    was_incomplete = True
            except Exception:
                pass

        data_raw = fetch_match(mid)

        if data_raw is None:
            print(f'  ❌ match_{mid} ({label}) → błąd: nie udało się pobrać')
            n_errors += 1
            time.sleep(DELAY)
            continue

        with open(detail_path, 'w', encoding='utf-8') as f:
            json.dump(data_raw, f, ensure_ascii=False, indent=2, default=str)

        data = data_raw.get('data', data_raw)

        try:
            gpt_data = build_gpt_data(mid, data, cid)
            with open(gpt_path, 'w', encoding='utf-8') as f:
                json.dump(gpt_data, f, ensure_ascii=False, indent=2, default=str)
            n_gpt += 1
        except Exception as e:
            print(f'  [WARN] match_{mid}: błąd zapisu GPT: {e}')

        if was_incomplete:
            print(f'  🔄 match_{mid} ({label}) [odświeżono]')
            n_refreshed += 1
        else:
            print(f'  ✅ match_{mid} ({label}) → zapisano')
            n_fetched += 1

        time.sleep(DELAY)

    print()
    print('══════════════════════════════════════════════════')
    print('PODSUMOWANIE')
    print(f'  Pobrano:    {n_fetched}')
    print(f'  Odświeżono: {n_refreshed}')
    print(f'  Pominięto:  {n_skipped} (cache)')
    print(f'  Błędy:      {n_errors}')
    print(f'  GPT pliki:  {n_gpt} zapisanych')
    print('══════════════════════════════════════════════════')


if __name__ == '__main__':
    main()
