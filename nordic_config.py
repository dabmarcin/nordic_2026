import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FOOTYSTATS_API_KEY")
if not API_KEY:
    raise ValueError("Brak FOOTYSTATS_API_KEY w .env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
BASE_URL = "https://api.football-data-api.com"

# ── SEZONY TRENINGOWE (historyczne, 2022–2025) ────────
# Pobierane jednorazowo --historical
# NIE mieszamy z 2026 (data leakage prevention)

ALLSVENSKAN_HISTORICAL = [
    {"name": "Allsvenskan 2022", "id": 7064,  "year": 2022},
    {"name": "Allsvenskan 2023", "id": 8737,  "year": 2023},
    {"name": "Allsvenskan 2024", "id": 10969, "year": 2024},
    {"name": "Allsvenskan 2025", "id": 16263, "year": 2025},
]

ELITESERIEN_HISTORICAL = [
    {"name": "Eliteserien 2022", "id": 7048,  "year": 2022},
    {"name": "Eliteserien 2023", "id": 8739,  "year": 2023},
    {"name": "Eliteserien 2024", "id": 10976, "year": 2024},
    {"name": "Eliteserien 2025", "id": 16260, "year": 2025},
]

VEIKKAUSLIIGA_HISTORICAL = [
    {"name": "Veikkausliiga 2022", "id": 7120,  "year": 2022},
    {"name": "Veikkausliiga 2023", "id": 8935,  "year": 2023},
    {"name": "Veikkausliiga 2024", "id": 11120, "year": 2024},
    {"name": "Veikkausliiga 2025", "id": 14089, "year": 2025},
]

MLS_HISTORICAL = [
    {"name": "MLS 2023", "id": 8777,  "year": 2023},
    {"name": "MLS 2024", "id": 10977, "year": 2024},
    {"name": "MLS 2025", "id": 13973, "year": 2025},
]

CSL_HISTORICAL = [
    {"name": "CSL 2023", "id": 9186,  "year": 2023},
    {"name": "CSL 2024", "id": 11217, "year": 2024},
    {"name": "CSL 2025", "id": 14153, "year": 2025},
]

# ── AKTYWNE SEZONY 2026 ───────────────────────────────
# Pobierane codziennie --daily
ALLSVENSKAN_2026_ID   = 16576
ELITESERIEN_2026_ID   = 16558
VEIKKAUSLIIGA_2026_ID = 16572
MLS_2026_ID           = 16504
CSL_2026_ID           = 16789

# ── AGREGATY ─────────────────────────────────────────
# Note: ALL_HISTORICAL includes 2022-2025; 2026 is added for backtest compatibility
ALLSVENSKAN_2026_HISTORICAL = [
    {"name": "Allsvenskan 2026", "id": ALLSVENSKAN_2026_ID, "year": 2026},
]

ELITESERIEN_2026_HISTORICAL = [
    {"name": "Eliteserien 2026", "id": ELITESERIEN_2026_ID, "year": 2026},
]

VEIKKAUSLIIGA_2026_HISTORICAL = [
    {"name": "Veikkausliiga 2026", "id": VEIKKAUSLIIGA_2026_ID, "year": 2026},
]

MLS_2026_HISTORICAL = [
    {"name": "MLS 2026", "id": MLS_2026_ID, "year": 2026},
]

CSL_2026_HISTORICAL = [
    {"name": "CSL 2026", "id": CSL_2026_ID, "year": 2026},
]

ALL_HISTORICAL = (
    ALLSVENSKAN_HISTORICAL +
    ELITESERIEN_HISTORICAL +
    VEIKKAUSLIIGA_HISTORICAL +
    MLS_HISTORICAL +
    CSL_HISTORICAL +
    ALLSVENSKAN_2026_HISTORICAL +
    ELITESERIEN_2026_HISTORICAL +
    VEIKKAUSLIIGA_2026_HISTORICAL +
    MLS_2026_HISTORICAL +
    CSL_2026_HISTORICAL
)

ACTIVE_SEASON_IDS = [
    ALLSVENSKAN_2026_ID,
    ELITESERIEN_2026_ID,
    VEIKKAUSLIIGA_2026_ID,
    MLS_2026_ID,
    CSL_2026_ID,
]

# Mapowania pomocnicze
SEASON_ID_TO_NAME = {s["id"]: s["name"] for s in ALL_HISTORICAL}

LEAGUE_BY_SEASON_ID = (
    {s["id"]: "allsvenskan"   for s in ALLSVENSKAN_HISTORICAL}  |
    {s["id"]: "eliteserien"   for s in ELITESERIEN_HISTORICAL}  |
    {s["id"]: "veikkausliiga" for s in VEIKKAUSLIIGA_HISTORICAL} |
    {s["id"]: "mls"           for s in MLS_HISTORICAL}           |
    {s["id"]: "csl"           for s in CSL_HISTORICAL}           |
    {ALLSVENSKAN_2026_ID:   "allsvenskan"}   |
    {ELITESERIEN_2026_ID:   "eliteserien"}   |
    {VEIKKAUSLIIGA_2026_ID: "veikkausliiga"} |
    {MLS_2026_ID:           "mls"}           |
    {CSL_2026_ID:           "csl"}
)

LEAGUE_NAMES = {
    "allsvenskan":   "Sweden Allsvenskan",
    "eliteserien":   "Norway Eliteserien",
    "veikkausliiga": "Finland Veikkausliiga",
    "mls":           "USA MLS",
    "csl":           "China Super League",
}

# ── PORTFOLIO SYGNAŁY INWESTYCYJNE ──────────────────────
PORTFOLIO_SIGNALS = {
    "csl_draw": {
        "league":    "csl",
        "model_src": "rule",
        "typ":       "Draw",
        "condition": "odds_ft_x >= 3.80 and odds_ft_x <= 4.50",
        "odds_col":  "odds_ft_x",
        "label":     "CSL Draw 3.80-4.50",
        "tier":      "A",
    },
    "mls_away_win_hi": {
        "league":    "mls",
        "model_src": "rule",
        "typ":       "Away Win",
        "condition": "odds_ft_2 >= 3.80 and odds_ft_2 <= 5.00",
        "odds_col":  "odds_ft_2",
        "label":     "MLS Away Win 3.80-5.00",
        "tier":      "A",
    },
    "csl_under_corners": {
        "league":    "csl",
        "model_src": "rule",
        "typ":       "Under 9.5 corners",
        "condition": "odds_corners_under_95 >= 2.20",
        "odds_col":  "odds_corners_under_95",
        "label":     "CSL Under 9.5C kurs 2.20+",
        "tier":      "A",
    },
    "mls_over_corners": {
        "league":    "mls",
        "model_src": "rule",
        "typ":       "Over 9.5 corners",
        "condition": "odds_corners_over_95 >= 2.00",
        "odds_col":  "odds_corners_over_95",
        "label":     "MLS Over 9.5C kurs 2.00+",
        "tier":      "B",
    },
    "elite_under_corners": {
        "league":    "eliteserien",
        "model_src": "scorer",
        "typ":       "Under 9.5 corners",
        "scorer_filter": {
            "Model_type": "liga",
            "Typ":        "Under 9.5 corners",
        },
        "label":     "Eliteserien ML Under 9.5C",
        "tier":      "B",
    },
    "allsv_btts_yes": {
        "league":    "allsvenskan",
        "model_src": "scorer",
        "typ":       "BTTS Yes",
        "scorer_filter": {
            "Model_type": "gpt_pred",
            "Typ":        "BTTS Yes",
        },
        "label":     "Allsvenskan GPT BTTS Yes",
        "tier":      "B",
    },
}

PORTFOLIO_FLAT_STAKE = 100.0

# ── PORTALE DO PARSOWANIA ARTYKUŁÓW ──────────────────
ARTICLE_SOURCES = {
    "allsvenskan": [
        {
            "name": "bettingstugan",
            "base_url": "https://bettingstugan.se",
            "language": "sv",
            "description": "Artykuły per omgång z typami",
        },
        {
            "name": "speltips",
            "base_url": "https://speltips.se",
            "language": "sv",
            "description": "Codzienne artykuły per mecz",
        },
    ],
    "eliteserien": [
        {
            "name": "oddspodden",
            "base_url": "https://www.oddspodden.com",
            "language": "no",
            "description": "Artykuły per mecz z oddstips",
        },
        {
            "name": "statsbet",
            "base_url": "https://statsbet.no",
            "language": "no",
            "description": "Analizy per mecz EN/NO",
        },
    ],
    "veikkausliiga": [
        {
            "name": "vedonlyonti",
            "base_url": "https://vedonlyonti.com",
            "language": "fi",
            "description": "Codzienne vihjeet per mecz",
        },
        {
            "name": "ristikaksi",
            "base_url": "https://www.ristikaksi.com",
            "language": "fi",
            "description": "Głębokie ennakko per mecz",
        },
    ],
}

# ── ŚCIEŻKI ───────────────────────────────────────────
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(SCRIPT_DIR, "data")
HISTORICAL_DIR    = os.path.join(DATA_DIR, "historical")
ALLSVENSKAN_DIR   = os.path.join(HISTORICAL_DIR, "allsvenskan")
ELITESERIEN_DIR   = os.path.join(HISTORICAL_DIR, "eliteserien")
VEIKKAUSLIIGA_DIR = os.path.join(HISTORICAL_DIR, "veikkausliiga")
MLS_DIR           = os.path.join(HISTORICAL_DIR, "mls")
CSL_DIR           = os.path.join(HISTORICAL_DIR, "csl")
DAILY_DIR         = os.path.join(DATA_DIR, "daily")
H2H_CACHE         = os.path.join(DATA_DIR, "h2h_cache")
ARTICLES_DIR      = os.path.join(DATA_DIR, "articles")
ARTICLES_ALLSV    = os.path.join(ARTICLES_DIR, "allsvenskan")
ARTICLES_ELITE    = os.path.join(ARTICLES_DIR, "eliteserien")
ARTICLES_VEIKK    = os.path.join(ARTICLES_DIR, "veikkausliiga")
TELEMETRY         = os.path.join(DATA_DIR, "telemetry")
ALLSV_SCORER_DIR  = os.path.join(TELEMETRY, "Allsvenskan scorer")
ELITE_SCORER_DIR  = os.path.join(TELEMETRY, "Eliteserien scorer")
VEIKK_SCORER_DIR  = os.path.join(TELEMETRY, "Veikkausliiga scorer")
MLS_SCORER_DIR    = os.path.join(TELEMETRY, "MLS scorer")
CSL_SCORER_DIR    = os.path.join(TELEMETRY, "CSL scorer")
REPORTS_DIR       = os.path.join(DATA_DIR, "reports")
PORTFOLIO_DIR     = os.path.join(DATA_DIR, "portfolio")
MATCH_DETAILS_DIR = os.path.join(DATA_DIR, "match_details")
MODELS_DIR        = os.path.join(SCRIPT_DIR, "ml_models")
MODELS_ALLSV      = os.path.join(MODELS_DIR, "allsvenskan")
MODELS_ELITE      = os.path.join(MODELS_DIR, "eliteserien")
MODELS_VEIKK      = os.path.join(MODELS_DIR, "veikkausliiga")
MODELS_MLS        = os.path.join(MODELS_DIR, "mls")
MODELS_CSL        = os.path.join(MODELS_DIR, "csl")
MODELS_COMBINED   = os.path.join(MODELS_DIR, "nordic_combined")
CURRENT_DIR       = os.path.join(DATA_DIR, "current")
TEAMS_HIST_DIR    = os.path.join(DATA_DIR, "teams_historical")
TEAMS_ALLSV_HIST  = os.path.join(TEAMS_HIST_DIR, "allsvenskan")
TEAMS_ELITE_HIST  = os.path.join(TEAMS_HIST_DIR, "eliteserien")
TEAMS_VEIKK_HIST  = os.path.join(TEAMS_HIST_DIR, "veikkausliiga")
TEAMS_MLS_HIST    = os.path.join(TEAMS_HIST_DIR, "mls")
TEAMS_CSL_HIST    = os.path.join(TEAMS_HIST_DIR, "csl")

# Utwórz wszystkie katalogi jeśli nie istnieją
for _dir in [
    ALLSVENSKAN_DIR, ELITESERIEN_DIR, VEIKKAUSLIIGA_DIR,
    MLS_DIR, CSL_DIR,
    DAILY_DIR, H2H_CACHE,
    ARTICLES_ALLSV, ARTICLES_ELITE, ARTICLES_VEIKK,
    ALLSV_SCORER_DIR, ELITE_SCORER_DIR, VEIKK_SCORER_DIR,
    MLS_SCORER_DIR, CSL_SCORER_DIR,
    REPORTS_DIR, MATCH_DETAILS_DIR, PORTFOLIO_DIR,
    MODELS_ALLSV, MODELS_ELITE, MODELS_VEIKK, MODELS_MLS, MODELS_CSL, MODELS_COMBINED,
    CURRENT_DIR,
    TEAMS_ALLSV_HIST, TEAMS_ELITE_HIST, TEAMS_VEIKK_HIST,
    TEAMS_MLS_HIST, TEAMS_CSL_HIST,
]:
    os.makedirs(_dir, exist_ok=True)
