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

# ── AKTYWNE SEZONY 2026 ───────────────────────────────
# Pobierane codziennie --daily
ALLSVENSKAN_2026_ID   = 16576
ELITESERIEN_2026_ID   = 16558
VEIKKAUSLIIGA_2026_ID = 16572

# ── AGREGATY ─────────────────────────────────────────
ALL_HISTORICAL = (
    ALLSVENSKAN_HISTORICAL +
    ELITESERIEN_HISTORICAL +
    VEIKKAUSLIIGA_HISTORICAL
)

ACTIVE_SEASON_IDS = [
    ALLSVENSKAN_2026_ID,
    ELITESERIEN_2026_ID,
    VEIKKAUSLIIGA_2026_ID,
]

# Mapowania pomocnicze
SEASON_ID_TO_NAME = {s["id"]: s["name"] for s in ALL_HISTORICAL}

LEAGUE_BY_SEASON_ID = (
    {s["id"]: "allsvenskan"   for s in ALLSVENSKAN_HISTORICAL}  |
    {s["id"]: "eliteserien"   for s in ELITESERIEN_HISTORICAL}  |
    {s["id"]: "veikkausliiga" for s in VEIKKAUSLIIGA_HISTORICAL} |
    {ALLSVENSKAN_2026_ID:   "allsvenskan"}  |
    {ELITESERIEN_2026_ID:   "eliteserien"}  |
    {VEIKKAUSLIIGA_2026_ID: "veikkausliiga"}
)

LEAGUE_NAMES = {
    "allsvenskan":   "Sweden Allsvenskan",
    "eliteserien":   "Norway Eliteserien",
    "veikkausliiga": "Finland Veikkausliiga",
}

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
REPORTS_DIR       = os.path.join(DATA_DIR, "reports")
MATCH_DETAILS_DIR = os.path.join(DATA_DIR, "match_details")
MODELS_DIR        = os.path.join(SCRIPT_DIR, "ml_models")
MODELS_ALLSV      = os.path.join(MODELS_DIR, "allsvenskan")
MODELS_ELITE      = os.path.join(MODELS_DIR, "eliteserien")
MODELS_VEIKK      = os.path.join(MODELS_DIR, "veikkausliiga")
MODELS_COMBINED   = os.path.join(MODELS_DIR, "nordic_combined")
CURRENT_DIR       = os.path.join(DATA_DIR, "current")
TEAMS_HIST_DIR    = os.path.join(DATA_DIR, "teams_historical")
TEAMS_ALLSV_HIST  = os.path.join(TEAMS_HIST_DIR, "allsvenskan")
TEAMS_ELITE_HIST  = os.path.join(TEAMS_HIST_DIR, "eliteserien")
TEAMS_VEIKK_HIST  = os.path.join(TEAMS_HIST_DIR, "veikkausliiga")

# Utwórz wszystkie katalogi jeśli nie istnieją
for _dir in [
    ALLSVENSKAN_DIR, ELITESERIEN_DIR, VEIKKAUSLIIGA_DIR,
    DAILY_DIR, H2H_CACHE,
    ARTICLES_ALLSV, ARTICLES_ELITE, ARTICLES_VEIKK,
    ALLSV_SCORER_DIR, ELITE_SCORER_DIR, VEIKK_SCORER_DIR,
    REPORTS_DIR, MATCH_DETAILS_DIR,
    MODELS_ALLSV, MODELS_ELITE, MODELS_VEIKK, MODELS_COMBINED,
    CURRENT_DIR,
    TEAMS_ALLSV_HIST, TEAMS_ELITE_HIST, TEAMS_VEIKK_HIST,
]:
    os.makedirs(_dir, exist_ok=True)
