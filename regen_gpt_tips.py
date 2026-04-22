import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import os
import glob
import time

import requests
from dotenv import load_dotenv
load_dotenv()

from nordic_config import MATCH_DETAILS_DIR, ALLSVENSKAN_2026_ID


def _get_best_odds(bookmakers):
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


def _resolve_odds(tip, odds_summary):
    k     = tip.get('kierunek', '')
    linia = tip.get('linia')
    rynek = tip.get('rynek', '').lower()
    direct = {
        'home_win': 'home', 'away_win': 'away', 'draw': 'draw',
        'btts_yes': 'btts_yes', 'btts_no': 'btts_no',
        '1x': '1x', 'x2': 'x2', '12': '12',
    }
    if k in direct:
        return odds_summary.get(direct[k])
    if linia:
        ls = str(linia).replace('.', '')
        if 'corner' in rynek:
            return odds_summary.get(f'corners_over_{ls}') if k == 'over' else odds_summary.get(f'corners_under_{ls}')
        return odds_summary.get(f'over{ls}') if k == 'over' else odds_summary.get(f'under{ls}')
    return None


def extract_gpt_tips(gpt_pl, gpt_en, oc):
    tekst = (gpt_pl or '').strip() or (gpt_en or '').strip()
    if len(tekst) < 50:
        return []
    key = os.getenv('ANTHROPIC_API_KEY')
    if not key:
        return []

    odds_summary = {}
    try:
        ft = oc.get('FT Result', {})
        odds_summary.update({
            'home': _get_best_odds(ft.get('Home', {})),
            'draw': _get_best_odds(ft.get('Draw', {})),
            'away': _get_best_odds(ft.get('Away', {})),
        })
        dc = oc.get('Double Chance', {})
        odds_summary.update({
            '1x': _get_best_odds(dc.get('Home/Draw', {})),
            'x2': _get_best_odds(dc.get('Draw/Away', {})),
            '12': _get_best_odds(dc.get('Home/Away', {})),
        })
        goals = oc.get('Goals Over/Under', {})
        for ln in ['1.5', '2.5', '3.5', '4.5']:
            k = ln.replace('.', '')
            odds_summary[f'over{k}']  = _get_best_odds(goals.get(f'Over {ln}', {}))
            odds_summary[f'under{k}'] = _get_best_odds(goals.get(f'Under {ln}', {}))
        btts = oc.get('Both Teams To Score', {})
        odds_summary['btts_yes'] = _get_best_odds(btts.get('Yes', {}))
        odds_summary['btts_no']  = _get_best_odds(btts.get('No', {}))
        corners = oc.get('Corners', {})
        for ln in ['7.5', '8.5', '9.5', '10.5', '11.5']:
            k = ln.replace('.', '')
            odds_summary[f'corners_over_{k}']  = _get_best_odds(corners.get(f'Over {ln}', {}))
            odds_summary[f'corners_under_{k}'] = _get_best_odds(corners.get(f'Under {ln}', {}))
    except Exception:
        pass

    SYSTEM = (
        'Jesteś ekspertem analizy zakładów sportowych. Przeczytaj analizę meczu '
        'i wyciągnij TYLKO polecane typy zakładowe (główne rekomendacje).\n'
        'Zwróć TYLKO tablicę JSON bez żadnego tekstu. Nie używaj markdown.\n'
        'Każdy typ:\n'
        '{"typ":"...","rynek":"1X2/BTTS/Over25/Under25/Over15/Corners/DC",'
        '"kierunek":"home_win/away_win/draw/btts_yes/btts_no/over/under/1x/x2/12",'
        '"linia":null_lub_liczba,"kurs":null_lub_liczba,"pewnosc":"wysoka/srednia/niska",'
        '"uzasadnienie":"max 100 znaków"}\n'
        'kurs=null jeśli brak w tekście. Brak typów → []. Max 5 typów.'
    )

    try:
        resp = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'x-api-key':         key,
                'anthropic-version': '2023-06-01',
                'content-type':      'application/json',
            },
            json={
                'model':    'claude-sonnet-4-20250514',
                'max_tokens': 800,
                'system':   SYSTEM,
                'messages': [{'role': 'user', 'content':
                    f'Analiza meczu:\n\n{tekst[:4000]}\n\n'
                    f'Kursy:\n{json.dumps(odds_summary, ensure_ascii=False)}'}],
            },
            timeout=30,
        )
        time.sleep(1)
        if resp.status_code == 200:
            txt = resp.json()['content'][0]['text'].replace('```json', '').replace('```', '').strip()
            tips = json.loads(txt)
            if isinstance(tips, list):
                for t in tips:
                    if t.get('kurs') is None:
                        t['kurs'] = _resolve_odds(t, odds_summary)
                return tips
        elif resp.status_code == 529:
            print('  [WARN] Anthropic overloaded')
    except Exception as e:
        print(f'  [WARN] {e}')
    return []


def main():
    files = sorted(glob.glob(os.path.join(MATCH_DETAILS_DIR, 'match_*.json')))
    print(f'Regeneruję gpt_tips dla {len(files)} meczów...')
    print()

    for path in files:
        mid = int(os.path.basename(path).replace('match_', '').replace('.json', ''))
        gpt_path = os.path.join(MATCH_DETAILS_DIR, f'gpt_{mid}.json')
        if not os.path.isfile(gpt_path):
            print(f'  Brak gpt_{mid}.json — pomijam')
            continue

        with open(path, encoding='utf-8') as f:
            data_raw = json.load(f)
        data = data_raw.get('data', data_raw)

        with open(gpt_path, encoding='utf-8') as f:
            old_gpt = json.load(f)

        home = old_gpt.get('home', '?')
        away = old_gpt.get('away', '?')
        print(f'  {home} vs {away} (match_{mid})...', flush=True)

        oc  = data.get('odds_comparison', {})
        gi  = data.get('gpt_int', {}) if isinstance(data.get('gpt_int'), dict) else {}
        gpl = gi.get('pl', '') or ''
        gen = data.get('gpt_en') or gi.get('en', '') or ''

        tips = extract_gpt_tips(gpl, gen, oc)

        old_gpt.pop('rekomendacja', None)
        old_gpt['gpt_tips'] = tips

        with open(gpt_path, 'w', encoding='utf-8') as f:
            json.dump(old_gpt, f, ensure_ascii=False, indent=2, default=str)
        print(f'    -> {len(tips)} tips')

    print()
    print('Gotowe.')


if __name__ == '__main__':
    main()
