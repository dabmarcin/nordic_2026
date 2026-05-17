# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Nordic 2026** is an ML-based betting prediction system for three Nordic football leagues:
- **Allsvenskan** (Sweden) - Season 2026 ID: 16576
- **Eliteserien** (Norway) - Season 2026 ID: 16558
- **Veikkausliiga** (Finland) - Season 2026 ID: 16572

The system fetches match data, builds ML models for predicting match outcomes, and scores predictions against bookmaker odds. A Streamlit dashboard displays predictions and performance analytics.

## Architecture

### Data Pipeline

```
API (football-data-api.com)
    ↓
fetch_data.py (--daily or --historical)
    ↓
data/daily/ and data/historical/{allsvenskan,eliteserien,veikkausliiga}
    ↓
build_dataset.py (feature engineering, time-series window)
    ↓
data/current/
    ↓
train_models.py (HistGradientBoosting + calibration)
    ↓
ml_models/{allsvenskan,eliteserien,veikkausliiga}
    ↓
nordic_app.py (Streamlit dashboard) → nordic_scorer.py (score predictions)
    ↓
data/telemetry/{league}_scorer/ (daily CSV logs)
```

### Key Modules

- **nordic_config.py** — Central configuration hub:
  - API credentials (FOOTYSTATS_API_KEY from .env)
  - Directory structure (data/, ml_models/, etc.)
  - League IDs and metadata
  - Article sources for web scraping
  - Automatically creates all required directories

- **api_client.py** — Football-data-api.com wrapper:
  - Retry logic with exponential backoff
  - Safe headers (Expect: '' fix for 417 errors)
  - Functions: `get_league_matches()`, `get_fixtures()`, `get_team_stats()`, `get_h2h_cache()`

- **fetch_data.py** — Data acquisition:
  - `python fetch_data.py --daily` — Fetch live 2026 season data
  - `python fetch_data.py --historical` — Fetch 2022–2025 archived data
  - Stores match data (fixtures, results, stats) by league and season

- **build_dataset.py** — Feature engineering:
  - Aggregates historical data into rolling windows (last 5/10 matches)
  - Computes team stats: PPG (points per game), BTTS%, Over 2.5%, corners, xG
  - Handles missing features (threshold: ≥50% non-zero)
  - Outputs: `data/current/{league}_train_data.csv`

- **train_models.py** — ML training:
  - Models: HistGradientBoostingClassifier + CalibratedClassifierCV
  - Targets: `result_home`, `result_away`, `btts`, `over25`, `corners_under`
  - CV: TimeSeriesSplit (respects time order)
  - Metrics: ROC-AUC, Brier Score
  - Saves: `ml_models/{league}/{target}_model.pkl`, `{target}_calibrator.pkl`
  - Feature sets: TEAM_STATS_HOME, TEAM_STATS_AWAY, TEAM_STATS_DIFF, etc.

- **nordic_scorer.py** — Prediction & scoring:
  - `python nordic_scorer.py --daily` — Score current day's predictions
  - FILTERS dict: min_score, min_odds, max_odds per prediction type
  - BANKROLL=1000 PLN, KELLY_FRAC=0.25, MAX_STAKE=50 PLN
  - Outputs CSV to `data/telemetry/{league}_scorer/`

- **nordic_app.py** — Streamlit dashboard:
  - `streamlit run nordic_app.py`
  - 6 tabs: Mecze (Matches), Artykuły (Articles), Wyniki (Results), Drużyny (Teams), Statystyki (Statistics), Ustawienia (Settings)
  - **Statistics tab:**
    - Overall KPI summary: Zakłady, Wygrane, Win Rate, ROI, Profit
    - Per-league breakdown with charts
    - ML vs GPT model comparison
    - Per-bet-type analysis
    - Per-league-per-type matrix with ROI/Profit charts
    - Odds bracket analysis (1.01–1.30, 1.31–1.50, etc.)
    - GPT accuracy per bet type
    - Match result distribution (goals, BTTS, corners)
    - Equity curve with drawdown tracking
    - **All Bets filter section:** Liga/Model/Typ multiselects with filtered summary KPI above data table (Zakłady, Wygrane, Win Rate, Śr. kurs, ROI, Profit)
  - Charts: performance over time, prediction accuracy
  - H2H history lookup for team matchups
  - JSON export of all statistics to `data/reports/`

- **online_settle.py** — Bet settlement:
  - Maps league abbreviations → season IDs
  - Fetches final match results, updates scorer CSV

- **process_existing_predictions.py** — Batch processing:
  - Reprocesses historical predictions
  - Useful for backfill or model recalibration

- **nordic_backtest.py** — Backtesting strategies on historical (2022–2025) and current (2026) data:
  - `python nordic_backtest.py` — Test all 10 strategies on full dataset (2022–2026)
  - `python nordic_backtest.py --liga allsvenskan` — Filter by league
  - `python nordic_backtest.py --stake 50` — Custom flat stake (PLN)
  - `python nordic_backtest.py --sezon 2025` — Filter by season year (loads from `data/historical/`)
  - `python nordic_backtest.py --sezon 2026` — Backtest current season (loads from `data/current/{league}_matches_2026.csv`, only completed matches)
  - **10 strategies:**
    - S1: 1 (Home Win) — odds_ft_1
    - S2: 2 (Away Win) — odds_ft_2
    - S3: X2 (Draw or Away) — odds_doublechance_x2
    - S4: 1X (Home or Draw) — odds_doublechance_1x
    - S5: Under 2.5 — odds_ft_under25
    - S6: Over 2.5 — odds_ft_over25
    - S7: Under 9.5 Corners — odds_corners_under_95
    - S8: Over 9.5 Corners — odds_corners_over_95
    - S9: BTTS Yes — odds_btts_yes
    - S10: BTTS No — odds_btts_no
  - **Outputs:** Console table + JSON report to `data/reports/backtest_YYYY-MM-DD.json`
  - **Per-league breakdown:** All 10 strategies (S1–S10) shown with Liga, N, WR, Śr.Kurs, ROI, Profit per league
  - **Metrics:** Win rate, ROI%, average odds, profit per league
  - **Season filtering:** `--sezon` loads only CSV files matching season IDs (e.g., 2025 loads from historical; 2026 loads from daily)

## Common Commands

### Setup
```bash
# Install dependencies (if requirements.txt exists) or manually:
pip install pandas numpy scikit-learn joblib streamlit altair requests python-dotenv

# Create .env in project root:
# FOOTYSTATS_API_KEY=your_api_key
# ANTHROPIC_API_KEY=your_anthropic_key (if needed)
```

### Daily Workflow
```bash
# 1. Fetch today's matches and stats
python fetch_data.py --daily

# 2. Rebuild features from current data
python build_dataset.py

# 3. Retrain models
python train_models.py

# 4. Score predictions
python nordic_scorer.py --daily

# 5. Settle results (run after matches finish)
python online_settle.py

# 6. View dashboard
streamlit run nordic_app.py

# 7. Backtest strategies on historical and current season data
python nordic_backtest.py                                    # All data (2022–2026)
python nordic_backtest.py --liga allsvenskan --stake 50     # Filter by league
python nordic_backtest.py --sezon 2025                       # Filter by season year (historical)
python nordic_backtest.py --sezon 2026                       # Current season results (from data/daily/)
```

### Historical Data (one-time setup)
```bash
python fetch_data.py --historical
python build_dataset.py
python train_models.py
```

## Data Structures

### Scorer CSV Schema
Each league's `data/telemetry/{league}_scorer/{league}_scorer_YYYY-MM-DD.csv`:
- **ID** — Match ID
- **Data** — Date (YYYY-MM-DD)
- **Mecz** — Match (e.g., "Hammarby vs Halmstad")
- **Liga** — League name
- **Model** — Which model/combo generated prediction
- **Typ** — Bet type (e.g., "Over 2.5", "BTTS Yes", "1", "Under 9.5 corners")
- **Score[%]** — Model confidence (0–100)
- **P_model** — Probability from calibrated model
- **Kurs** — Bookmaker odds
- **EV[%]** — Expected value %
- **Stake_PLN** — Kelly-fraction stake
- **Rezultat** — Final match result
- **Corners**, **Kartki** — Actual corners and yellow cards
- **Profit_PLN** — P&L if prediction was staked

### Team Stats Snapshot
Historical team stats aggregated by rolling windows (last 5/10 matches):
- `home_team_ppg_home` — Points per game at home
- `home_team_btts_pct_home` — BTTS frequency at home
- `home_team_corners_avg_home` — Average corners conceded at home
- Similar for away teams

## Key Configuration & Filters

From `nordic_scorer.py`:
```python
FILTERS = {
    'result_home':   {'min_score': 62.0, 'min_odds': 1.40, 'max_odds': 5.0},
    'result_away':   {'min_score': 62.0, 'min_odds': 1.40, 'max_odds': 5.0},
    'btts':          {'min_score': 62.0, 'min_odds': 1.20, 'max_odds': 5.0},
    'over25':        {'min_score': 62.0, 'min_odds': 1.20, 'max_odds': 5.0},
    'corners_under': {'min_score': 60.0, 'min_odds': 1.30, 'max_odds': 5.0},
}

BANKROLL = 1000.0  # Initial bank (PLN)
KELLY_FRAC = 0.25  # Kelly fraction (conservative 25% bet sizing)
MAX_STAKE = 50.0   # Max per bet (PLN)
```

Adjust these in `nordic_scorer.py` to fine-tune stake sizing and filtering.

## Model Features

**Result prediction (Home/Away Win):**
- Team stats (PPG, win%, xG)
- Implied probabilities from odds
- Last 5 match aggregate stats
- League dummy variables

**BTTS & Over 2.5:**
- Team goal-scoring metrics
- Goal conceded rates
- BTTS frequency in rolling windows

**Corners Under 9.5:**
- Average corners conceded/created
- Odds-based features
- Historical corner distribution

Models use **time-series cross-validation** to prevent data leakage.

## Important Notes

### UTF-8 Encoding
Many scripts set UTF-8 stdout explicitly (Polish characters):
```python
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

### API Stability
- `api_client.py` includes retry logic and safe headers
- Avoid rapid polling; use caching where possible
- H2H matches cached in `data/h2h_cache/`

### Data Leakage Prevention
- Historical data (2022–2025) stored separately from 2026 season
- `build_dataset.py` respects season boundaries
- TimeSeriesSplit ensures no future data leaks into training

### Time Zone
- Dates in CSVs and logs are assumed UTC/game-local
- Verify match kickoff times match your betting platform

## Directory Structure Summary
```
nordic_2026/
├── nordic_config.py          # Central config & paths
├── api_client.py             # API wrapper
├── fetch_data.py             # Data ingestion
├── build_dataset.py          # Feature engineering
├── train_models.py           # Model training
├── nordic_scorer.py          # Prediction & scoring
├── nordic_app.py             # Streamlit dashboard
├── online_settle.py          # Bet settlement
├── process_existing_predictions.py  # Batch reprocessing
├── nordic_backtest.py        # Backtest strategies on 2022–2026 data
├── data/
│   ├── daily/                # Current season matches (2026)
│   ├── historical/           # 2022–2025 data by league
│   ├── telemetry/            # Scorer CSV logs
│   ├── current/              # Feature-engineered datasets
│   ├── h2h_cache/            # Cached head-to-head data
│   ├── articles/             # Web-scraped articles (future)
│   ├── match_details/        # Match stats from API
│   ├── reports/              # Analysis & metrics
│   └── teams_historical/     # Team stats snapshots
└── ml_models/
    ├── allsvenskan/          # Models for Sweden
    ├── eliteserien/          # Models for Norway
    ├── veikkausliiga/        # Models for Finland
    └── nordic_combined/      # Cross-league models (if used)
```

## Maintenance & Updating Documentation

**After code changes:**
1. If altering league IDs or API endpoints → update `nordic_config.py` section in CLAUDE.md
2. If modifying filters or stake logic → update **Key Configuration & Filters** section
3. If adding new features → document in **Model Features**
4. If changing the data pipeline flow → update **Architecture > Data Pipeline**
5. If adding new scripts → list in **Common Commands** and **Key Modules**
6. If modifying backtest strategies or metrics → update **Architecture > Key Modules > nordic_backtest.py**

Keep this file as the source of truth for project structure and workflows.

The **Git Hook** (configured in `.claude/settings.json`) monitors changes to core modules (train_models, api_client, fetch_data, build_dataset, nordic_config, nordic_scorer) and reminds you to update CLAUDE.md when these change.
