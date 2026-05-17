# ADD_LEAGUE_INSTRUCTIONS.md
# Instrukcja dodawania nowej ligi do projektu Nordic 2026

Gdy użytkownik mówi "dodaj ligę XYZ, sezon historyczny ID=XXXX,
aktywny 2026 ID=YYYY" — wykonaj WSZYSTKIE kroki po kolei.
Nie pomijaj żadnego kroku. Przed każdym krokiem sprawdź
aktualną zawartość pliku który modyfikujesz.

---

## INFORMACJE WEJŚCIOWE od użytkownika

Użytkownik poda:
- `LEAGUE_KEY`    — klucz wewnętrzny (małe litery, bez spacji, np. "bundesliga")
- `LEAGUE_NAME`   — pełna nazwa (np. "Germany Bundesliga")
- `LEAGUE_SHORT`  — skrót do wyświetlania (np. "BUND")
- `TIMEZONE`      — strefa czasowa (np. "Europe/Berlin")
- `HISTORICAL_IDS` — lista {year: id} dla sezonów treningowych (np. 2023, 2024, 2025)
- `ACTIVE_2026_ID` — ID aktywnego sezonu 2026
- `PORTFOLIO_SIGNALS` — opcjonalna lista sygnałów portfelowych (może być pusta)
  Format: [{signal_id, typ, condition, odds_col, label, tier}]
- `P_EMPIRICAL_OVER25` — empiryczna p dla Over 2.5 (domyślnie None = nie odwracaj)
- `P_EMPIRICAL_CORNERS` — empiryczna p dla Corners Over (domyślnie None = nie odwracaj)

---

## KROK 1 — nordic_config.py

Otwórz C:\Projects\nordic_2026\nordic_config.py

### 1a. Dodaj sezony historyczne

```python
{LEAGUE_KEY.upper()}_HISTORICAL = [
    {"name": "{LEAGUE_NAME} 2023", "id": XXXX, "year": 2023},
    {"name": "{LEAGUE_NAME} 2024", "id": XXXX, "year": 2024},
    {"name": "{LEAGUE_NAME} 2025", "id": XXXX, "year": 2025},
]
```

### 1b. Dodaj aktywny sezon

```python
{LEAGUE_KEY.upper()}_2026_ID = {ACTIVE_2026_ID}
```

### 1c. Rozszerz ALL_HISTORICAL

```python
ALL_HISTORICAL = (
    ...istniejące...
    + {LEAGUE_KEY.upper()}_HISTORICAL
)
```

### 1d. Rozszerz ACTIVE_SEASON_IDS

```python
ACTIVE_SEASON_IDS = [
    ...istniejące...,
    {LEAGUE_KEY.upper()}_2026_ID,
]
```

### 1e. Rozszerz LEAGUE_BY_SEASON_ID

```python
LEAGUE_BY_SEASON_ID = (
    ...istniejące...
    | {s["id"]: "{LEAGUE_KEY}" for s in {LEAGUE_KEY.upper()}_HISTORICAL}
    | {{LEAGUE_KEY.upper()}_2026_ID: "{LEAGUE_KEY}"}
)
```

### 1f. Rozszerz LEAGUE_NAMES

```python
LEAGUE_NAMES = {
    ...istniejące...,
    "{LEAGUE_KEY}": "{LEAGUE_NAME}",
}
```

### 1g. Dodaj ścieżki

```python
{LEAGUE_KEY.upper()}_DIR        = os.path.join(HISTORICAL_DIR, "{LEAGUE_KEY}")
TEAMS_{LEAGUE_KEY.upper()}_HIST = os.path.join(TEAMS_HIST_DIR, "{LEAGUE_KEY}")
{LEAGUE_KEY.upper()}_SCORER_DIR = os.path.join(TELEMETRY, "{LEAGUE_NAME} scorer")
MODELS_{LEAGUE_KEY.upper()}     = os.path.join(MODELS_DIR, "{LEAGUE_KEY}")
```

### 1h. Dodaj do makedirs

Dodaj nowe ścieżki do pętli makedirs na końcu pliku.

### 1i. Dodaj sygnały portfelowe (jeśli podano)

Dodaj do PORTFOLIO_SIGNALS:
```python
"{signal_id}": {
    "league":    "{LEAGUE_KEY}",
    "model_src": "rule",
    "typ":       "{typ}",
    "condition": "{condition}",
    "odds_col":  "{odds_col}",
    "label":     "{label}",
    "tier":      "{tier}",
},
```

### Weryfikacja kroku 1

```bash
python -c "
from nordic_config import (
    {LEAGUE_KEY.upper()}_HISTORICAL,
    {LEAGUE_KEY.upper()}_2026_ID,
    {LEAGUE_KEY.upper()}_DIR,
    MODELS_{LEAGUE_KEY.upper()},
)
print('Config OK')
for s in {LEAGUE_KEY.upper()}_HISTORICAL:
    print(f'  {s[\"name\"]} id={s[\"id\"]}')
print(f'Active 2026: {{LEAGUE_KEY.upper()}_2026_ID}')
"
```

---

## KROK 2 — fetch_data.py

Otwórz C:\Projects\nordic_2026\fetch_data.py

### 2a. Dodaj importy z nordic_config

```python
from nordic_config import (
    ...istniejące...,
    {LEAGUE_KEY.upper()}_HISTORICAL,
    {LEAGUE_KEY.upper()}_DIR,
    TEAMS_{LEAGUE_KEY.upper()}_HIST,
    {LEAGUE_KEY.upper()}_2026_ID,
    {LEAGUE_KEY.upper()}_SCORER_DIR,
)
```

### 2b. Rozszerz LEAGUE_DIR_MAP

```python
LEAGUE_DIR_MAP = {
    ...istniejące...,
    "{LEAGUE_KEY}": {LEAGUE_KEY.upper()}_DIR,
}
```

### 2c. Rozszerz TEAMS_HIST_DIR_MAP

```python
TEAMS_HIST_DIR_MAP = {
    ...istniejące...,
    "{LEAGUE_KEY}": TEAMS_{LEAGUE_KEY.upper()}_HIST,
}
```

### 2d. Rozszerz tryb --daily

Dodaj pobieranie teams dla nowej ligi:
```python
# W sekcji pobierania current teams
("{LEAGUE_KEY}", {LEAGUE_KEY.upper()}_2026_ID,
 f"CURRENT_DIR/{LEAGUE_KEY}_teams_2026.csv"),
```

### 2e. Rozszerz tryb --weekly

Dodaj pobieranie matches i teams dla nowej ligi:
```python
("{LEAGUE_KEY}", {LEAGUE_KEY.upper()}_2026_ID,
 f"CURRENT_DIR/{LEAGUE_KEY}_matches_2026.csv",
 f"CURRENT_DIR/{LEAGUE_KEY}_teams_2026.csv"),
```

### Weryfikacja kroku 2

```bash
python fetch_data.py --help
python fetch_data.py --historical --force
```

Sprawdź czy nowa liga pojawia się w logach.

---

## KROK 3 — Pobierz dane historyczne

```bash
python fetch_data.py --historical
python fetch_data.py --weekly
python fetch_data.py --daily
```

Oczekiwany output:
```
✅ {LEAGUE_NAME} 2023 (id=XXXX) — NNN meczów
✅ {LEAGUE_NAME} 2024 (id=XXXX) — NNN meczów
✅ {LEAGUE_NAME} 2025 (id=XXXX) — NNN meczów
```

---

## KROK 4 — build_dataset.py

Otwórz C:\Projects\nordic_2026\build_dataset.py

### 4a. Dodaj importy

```python
from nordic_config import (
    ...istniejące...,
    {LEAGUE_KEY.upper()}_DIR,
    TEAMS_{LEAGUE_KEY.upper()}_HIST,
    {LEAGUE_KEY.upper()}_2026_ID,
)
```

### 4b. Dodaj do LEAGUE_CONFIG

```python
LEAGUE_CONFIG = {
    ...istniejące...,
    "{LEAGUE_KEY}": {
        "hist_dirs":  [{LEAGUE_KEY.upper()}_DIR],
        "teams_dirs": [TEAMS_{LEAGUE_KEY.upper()}_HIST],
        "current_matches": [
            ("{LEAGUE_KEY}", {LEAGUE_KEY.upper()}_2026_ID),
        ],
        "current_teams": [
            "{LEAGUE_KEY}_teams_2026.csv",
        ],
        "output": os.path.join(DATA_DIR,
                  "{LEAGUE_KEY}_training_dataset.csv"),
        "league_flag": "is_{LEAGUE_KEY}",
    },
}
```

### 4c. Rozszerz choices w argparse

```python
parser.add_argument('--league',
    choices=['nordic','mls','csl','{LEAGUE_KEY}','all'],
    ...)
```

### Weryfikacja i budowa datasetu

```bash
python build_dataset.py --league {LEAGUE_KEY} --debug
```

Sprawdź output:
- Liczba meczów (historyczne + current)
- Teams coverage (powinno być > 85%)
- Wypełnienie potencjałów

---

## KROK 5 — train_models.py

Otwórz C:\Projects\nordic_2026\train_models.py

### 5a. Dodaj importy

```python
from nordic_config import (
    ...istniejące...,
    MODELS_{LEAGUE_KEY.upper()},
)
```

### 5b. Dodaj LEAGUE_SETS

```python
LEAGUE_SETS_{LEAGUE_KEY.upper()} = {
    "{LEAGUE_KEY}": ("{LEAGUE_KEY}", MODELS_{LEAGUE_KEY.upper()}),
}
```

### 5c. Dodaj do DATASET_PATHS

```python
DATASET_PATHS = {
    ...istniejące...,
    "{LEAGUE_KEY}": os.path.join(DATA_DIR,
                    "{LEAGUE_KEY}_training_dataset.csv"),
}
```

### 5d. Rozszerz choices w argparse

```python
parser.add_argument('--league',
    choices=['nordic','mls','csl','{LEAGUE_KEY}','all'],
    ...)
```

### 5e. Rozszerz --all

Gdy args.league == 'all' — dodaj '{LEAGUE_KEY}' do listy.

### Trening modeli

```bash
python train_models.py --league {LEAGUE_KEY} --debug
```

Sprawdź AUC_cv dla wszystkich 5 modeli.
Porównaj z Nordic/MLS/CSL — powinny być na podobnym poziomie.

Opcjonalnie sprawdź regularyzację:
```bash
python train_models.py --league {LEAGUE_KEY} --compare
```

---

## KROK 6 — Utwórz scorer

Skopiuj mls_scorer.py jako {LEAGUE_KEY}_scorer.py

### 6a. Zmień konfigurację ligi

```python
ACTIVE_LEAGUES = {{LEAGUE_KEY}: {LEAGUE_KEY.upper()}_2026_ID}

LEAGUE_DIRS = {
    '{LEAGUE_KEY}': (
        {LEAGUE_KEY.upper()}_DIR,
        MODELS_{LEAGUE_KEY.upper()},
        {LEAGUE_KEY.upper()}_SCORER_DIR,
    ),
}
```

### 6b. Zmień timezone

```python
TZ = '{TIMEZONE}'  # np. 'Europe/Berlin'
```

### 6c. Zmień nagłówek konsolowy

```python
# Zamień 'MLS' na '{LEAGUE_SHORT}'
```

### 6d. Zmień zapis CSV

```python
# Zapis: {LEAGUE_KEY}_scorer_{date}.csv
# w {LEAGUE_KEY.upper()}_SCORER_DIR
```

### 6e. Zmień teams lookup

```python
# Wczytuje: CURRENT_DIR/{LEAGUE_KEY}_teams_2026.csv
```

### 6f. Ustaw empiryczne prawdopodobieństwa

Jeśli użytkownik podał P_EMPIRICAL_CORNERS (ligi ofensywne):
```python
P_CORNERS_OVER = {P_EMPIRICAL_CORNERS}
# Logika: gdy model mówi Under → graj Over z tą p
```

Jeśli użytkownik podał P_EMPIRICAL_OVER25:
```python
P_OVER25_FORCED = {P_EMPIRICAL_OVER25}
# Logika: gdy model mówi Under 2.5 → graj Over 2.5 z tą p
```

Jeśli żadne nie podano → zostaw domyślną logikę Nordic
(model mówi Under → graj Under).

### 6g. Zaktualizuj importy

```python
from nordic_config import (
    {LEAGUE_KEY.upper()}_DIR,
    MODELS_{LEAGUE_KEY.upper()},
    {LEAGUE_KEY.upper()}_SCORER_DIR,
    {LEAGUE_KEY.upper()}_2026_ID,
    DAILY_DIR, CURRENT_DIR,
    BANKROLL, KELLY_FRAC, MAX_STAKE,
)
```

### Weryfikacja scorera

```bash
python {LEAGUE_KEY}_scorer.py today --debug
python {LEAGUE_KEY}_scorer.py tomorrow --debug
```

### Backfill historyczny

```bash
python {LEAGUE_KEY}_scorer.py --backfill
```

Sprawdź ile predykcji i jaki % rozliczonych.

---

## KROK 7 — online_settle.py

Otwórz C:\Projects\nordic_2026\online_settle.py

### 7a. Dodaj mapowanie w _build_league_to_season_map()

```python
if hasattr(_nc, "{LEAGUE_KEY.upper()}_2026_ID"):
    sid = int(_nc.{LEAGUE_KEY.upper()}_2026_ID)
    for key in (
        "{LEAGUE_KEY}",
        "{LEAGUE_KEY.upper()}",
        "{LEAGUE_NAME}",
        "{LEAGUE_NAME} 2026",
    ):
        m[key] = sid
```

### 7b. Dodaj substring matching w _get_season_id_for_league()

```python
if _nc and hasattr(_nc, "{LEAGUE_KEY.upper()}_2026_ID") and \
   "{LEAGUE_KEY}" in league_lower:
    return int(_nc.{LEAGUE_KEY.upper()}_2026_ID)
```

---

## KROK 8 — nordic_app.py

Otwórz C:\Projects\nordic_2026\nordic_app.py

### 8a. Dodaj importy

```python
from nordic_config import (
    ...istniejące...,
    {LEAGUE_KEY.upper()}_SCORER_DIR,
    {LEAGUE_KEY.upper()}_2026_ID,
)
```

### 8b. Rozszerz get_all_scorer_files()

```python
def get_all_scorer_files():
    ...
    for league, d in [
        ...istniejące...,
        ("{LEAGUE_KEY}", {LEAGUE_KEY.upper()}_SCORER_DIR),
    ]:
```

### 8c. Rozszerz TAB Mecze

Dodaj do league_options i liga_map:
```python
"{LEAGUE_NAME}": {LEAGUE_KEY.upper()}_SCORER_DIR,
```

### 8d. Rozszerz TAB Wyniki

Dodaj do selectbox ligi:
```python
"{LEAGUE_KEY}",
```

Dodaj do format_func:
```python
"{LEAGUE_KEY}": "🏳️ {LEAGUE_NAME}",
```

Dodaj do get_scorer_files():
```python
"{LEAGUE_KEY}": {LEAGUE_KEY.upper()}_SCORER_DIR,
```

### 8e. Rozszerz TAB Statystyki

Dodaj flagę w liga_icons:
```python
"{LEAGUE_KEY}": "🏳️",
```

### 8f. Rozszerz TAB Drużyny

Dodaj do load_league_data():
```python
"{LEAGUE_KEY}": "{LEAGUE_KEY}",
```

### 8g. Rozszerz TAB Ustawienia

Dodaj przycisk generowania predykcji:
```python
if st.button("▶ {LEAGUE_SHORT} — Generuj predykcje"):
    rc, out, err = run_script(
        "{LEAGUE_KEY}_scorer.py", [day_score])
    ...
```

---

## KROK 9 — portfolio_scorer.py (jeśli sygnały podano)

Jeśli użytkownik podał PORTFOLIO_SIGNALS dla nowej ligi:

Otwórz C:\Projects\nordic_2026\portfolio_scorer.py

### 9a. Dodaj obsługę nowych sygnałów rule-based

W funkcji generate_rule_signals() dodaj bloki:

```python
if signal_id == "{signal_id}":
    if liga == "{LEAGUE_KEY}":
        kurs = safe_float(row.get("{odds_col}"))
        # Sprawdź warunek z PORTFOLIO_SIGNALS
        if {condition_as_python_expression}:
            → dodaj zakład
```

### 9b. Dodaj do backfill

W sekcji backfill dodaj wczytywanie:
```python
("{LEAGUE_KEY}", {LEAGUE_KEY.upper()}_2026_ID,
 f"CURRENT_DIR/{LEAGUE_KEY}_matches_2026.csv"),
```

### 9c. Uruchom backfill portfolio

```bash
python portfolio_scorer.py --backfill
```

---

## KROK 10 — Weryfikacja końcowa

```bash
# 1. Sprawdź config
python -c "
from nordic_config import ALL_HISTORICAL, ACTIVE_SEASON_IDS
print('Historyczne:', len(ALL_HISTORICAL))
print('Aktywne:', ACTIVE_SEASON_IDS)
"

# 2. Sprawdź dataset
python build_dataset.py --league {LEAGUE_KEY} --debug

# 3. Sprawdź modele
python train_models.py --league {LEAGUE_KEY} --debug

# 4. Sprawdź scorer
python {LEAGUE_KEY}_scorer.py today --debug
python {LEAGUE_KEY}_scorer.py --backfill

# 5. Sprawdź online_settle
python -c "
from online_settle import _get_season_id_for_league
print(_get_season_id_for_league('{LEAGUE_KEY}'))
print(_get_season_id_for_league('{LEAGUE_NAME}'))
"

# 6. Uruchom aplikację
streamlit run nordic_app.py
```

---

## PRZYKŁAD UŻYCIA

Użytkownik mówi:
> Dodaj ligę Bundesliga. Sezon 2023 ID=9186,
> 2024 ID=11217, 2025 ID=14153, aktywny 2026 ID=16789.
> Timezone: Europe/Berlin. Brak sygnałów portfolio.

Claude Code wykonuje kroki 1-10 z:
- LEAGUE_KEY = "bundesliga"
- LEAGUE_NAME = "Germany Bundesliga"
- LEAGUE_SHORT = "BUND"
- TIMEZONE = "Europe/Berlin"
- HISTORICAL_IDS = {2023: 9186, 2024: 11217, 2025: 14153}
- ACTIVE_2026_ID = 16789
- PORTFOLIO_SIGNALS = [] (brak)
- P_EMPIRICAL_OVER25 = None
- P_EMPIRICAL_CORNERS = None

---

## UWAGI WAŻNE

1. **Nigdy nie usuwaj** istniejącej logiki Nordic, MLS, CSL
   — tylko DODAWAJ nowe bloki

2. **Zawsze sprawdź** wypełnienie potencjałów
   (btts_potential, o25_potential) w build_dataset
   — jeśli < 50% → zostaną automatycznie pominięte

3. **Empiryczne p** podajesz tylko gdy wiesz z backtestów
   że liga jest ofensywna (Over 2.5 > 58%) lub
   ma dużo cornersów (Over 9.5C > 55%)
   — dla defensywnych lig zostawiasz domyślną logikę

4. **Backfill scorer** wymaga danych z current matches
   — najpierw uruchom: python fetch_data.py --weekly

5. **Kolejność kroków jest obowiązkowa**
   — scorer wymaga wytrenowanych modeli
   — build_dataset wymaga pobranych danych
   — train_models wymaga buildu datasetu
