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
  - **Prediction logic:**
    - Over 2.5: Generated if model confidence > 50%
    - Under 2.5: Generated if Over 2.5 model confidence < 50% (inverted: score = 100% - Over2.5_score, p recalculated for Kelly)
    - Inverted score used for filtering and Kelly stake calculation, ensuring proper confidence weighting
  - BANKROLL=1000 PLN, KELLY_FRAC=0.25, MAX_STAKE=50 PLN
  - Outputs CSV to `data/telemetry/{league}_scorer/`
  - Supports both Over 2.5 and Under 2.5 predictions with appropriate odds for all leagues

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

- **portfolio_scorer.py** — Investment portfolio signal generation:
  - `python portfolio_scorer.py --daily` — Generate portfolio signals from rule-based and scorer-based sources
  - `python portfolio_scorer.py --backfill` — Reprocess historical portfolio data
  - Integrates: Allsvenskan BTTS Yes (gpt_pred model), Eliteserien Under 9.5 Corners (liga model)
  - Outputs portfolio CSVs to `data/portfolio/portfolio_YYYY-MM-DD.csv`
  - **CRITICAL:** Validates all rule-based signals against odds ranges defined in `PORTFOLIO_SIGNALS`
  - Uses `validate_odds(signal_id, kurs)` function to ensure odds match signal criteria
  - **SOURCE OF TRUTH:** Portfolio CSVs are used by `invest_app.py` for all statistics and performance tracking

- **sync_portfolio_with_scorers.py** — Data sync utility:
  - `python sync_portfolio_with_scorers.py` — Add missing Nordic signals to existing portfolio CSVs
  - Scans scorers for Allsvenskan BTTS Yes and Eliteserien Under 9.5C signals
  - Merges any missing signals into their corresponding portfolio files

- **create_missing_portfolio_files.py** — Portfolio file creation utility:
  - `python create_missing_portfolio_files.py` — Create portfolio CSVs for dates with scorer data but no portfolio file
  - Ensures complete coverage of all Nordic signal dates

- **validate_portfolio_odds.py** — Portfolio quality assurance:
  - `python validate_portfolio_odds.py` — Scan all portfolio files for signals with invalid odds
  - Validates rule-based signals against `PORTFOLIO_SIGNALS` odds ranges
  - Automatically removes and reports all violations
  - **Must run after cleaning or before major analysis**

- **invest_app.py** — Streamlit portfolio investment dashboard:
  - `streamlit run invest_app.py --logger.level=error`
  - Port: 8501
  - **Source of data:** Reads from portfolio CSV files only (not scorers)
  - KPI cards: ROI, Profit, Zakłady, Win Rate, Zainwestowano
  - Visualizations: Equity curve with drawdown, per-signal analysis, daily profit chart
  - Detailed signal table with filtering and CSV export
  - **3 tabs:** 📊 Dashboard, 🔔 Monitor, 🔍 Analiza Lig
  - **Analiza Lig tab:** per-league market scan across all seasons (reads `data/monitor/league_analysis.json`)
    - League selector + season profile (goals, O2.5, BTTS, corners, H/D/A)
    - Full markets table with overall ROI + per-season ROI columns (2022→2026)
    - "💰 Konsystentnie zyskowne" cards: markets profitable across most seasons, not yet in portfolio
    - "🔄 Przelicz analizę lig" button runs `monitor.py --leagues`

- **monitor.py** — Signal health monitor + league market analysis:
  - `python monitor.py --check` — Rolling ROI / consecutive-loss tracking, portfolio-vs-market scan, candidate detection (writes `data/monitor/monitor_YYYY-MM-DD.json`)
  - `python monitor.py --leagues [--min-n N]` — **Per-league deep scan** (wzór: `tournament_analysis.py` z mundial_2026): each league examined separately across ALL markets × ALL seasons (2022–2026), WITHOUT odds-range splitting. For every market computes combined ROI/WR plus ROI per season, flagging long-term consistent winners (`seasons_profitable ≥ 60%` and combined ROI > 0). Loads 2022–2025 from `data/historical/{league}/`, 2026 from `data/current/`. Writes `data/monitor/league_analysis.json` (source for invest_app "Analiza Lig" tab)
  - `python monitor.py --disable/--enable <signal_id>` — Toggle signals
  - `python monitor.py --add-candidate "<label>"` — Promote candidate to custom signal
  - `python monitor.py --history` — Show alert history

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
python mls_scorer.py --daily
python csl_scorer.py --daily

# 5. Sync portfolio with scorer data (ensures invest_app has all Nordic signals)
python create_missing_portfolio_files.py
python sync_portfolio_with_scorers.py

# 6. Generate portfolio investment signals
python portfolio_scorer.py --daily

# 7. Settle results (run after matches finish)
python online_settle.py

# 8. View dashboard
streamlit run nordic_app.py

# 9. View portfolio dashboard
streamlit run invest_app.py

# 10. Backtest strategies on historical and current season data (optional)
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

### Nordic Scorer (`nordic_scorer.py`):
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

### Portfolio Signals (`nordic_config.py` → `PORTFOLIO_SIGNALS`):
**CRITICAL VALIDATION RULES** — All rule-based portfolio signals must respect these odds ranges:

```python
PORTFOLIO_SIGNALS = {
    "csl_draw": {              # odds_ft_x must be 3.80-4.50
        "condition": "odds_ft_x >= 3.80 and odds_ft_x <= 4.50",
    },
    "mls_away_win_hi": {       # odds_ft_2 must be 3.80-5.00
        "condition": "odds_ft_2 >= 3.80 and odds_ft_2 <= 5.00",
    },
    "csl_under_corners": {     # odds_corners_under_95 must be >= 2.20
        "condition": "odds_corners_under_95 >= 2.20",
    },
    "mls_over_corners": {      # odds_corners_over_95 must be >= 2.00
        "condition": "odds_corners_over_95 >= 2.00",
    },
    # Scorer-based signals (no odds constraint):
    "elite_under_corners": {...},      # Eliteserien Under 9.5 Corners
    "allsv_btts_yes": {...},           # Allsvenskan BTTS Yes
}
```

**Portfolio Scorer Validation:** The `portfolio_scorer.py` script automatically validates all rule-based signals:
- Uses `validate_odds(signal_id, kurs)` function
- Skips any signal where odds fall outside the defined range
- Scorer-based signals (elite_under_corners, allsv_btts_yes) have no restrictions
- **Check this by running:** `python validate_portfolio_odds.py` to find and remove any violating signals

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

### Portfolio Data Consistency
- **Portfolio CSV files are the source of truth** for `invest_app.py` performance metrics
- Nordic signals (Allsvenskan BTTS Yes, Eliteserien Under 9.5 Corners) are generated by scorers but must be synced to portfolio
- **Sync workflow:**
  1. `nordic_scorer.py --daily` generates Allsvenskan signals in `data/telemetry/Allsvenskan scorer/`
  2. `sync_portfolio_with_scorers.py` adds missing signals to existing portfolio CSVs
  3. `create_missing_portfolio_files.py` creates portfolio CSVs for dates that have scorer data but no portfolio file
  4. `portfolio_scorer.py --daily` generates additional rule-based and other signals
- **Verification:** Run this to check consistency:
  ```bash
  python << 'EOF'
  import pandas as pd, glob, os
  scorer_ids = [int(id) for f in glob.glob(r"data/telemetry/Allsvenskan scorer/*.csv") 
                for id in pd.read_csv(f, encoding='utf-8-sig')
                [(pd.read_csv(f, encoding='utf-8-sig')['Model_type']=='gpt_pred') & 
                 (pd.read_csv(f, encoding='utf-8-sig')['Typ']=='BTTS Yes')]['ID']]
  portfolio_ids = [int(id) for f in glob.glob(r"data/portfolio/*.csv") 
                   for id in pd.read_csv(f, encoding='utf-8-sig')
                   [(pd.read_csv(f, encoding='utf-8-sig')['Liga']=='Allsvenskan') & 
                    (pd.read_csv(f, encoding='utf-8-sig')['Signal_ID']=='allsv_btts_yes')]['ID']]
  print(f"Match: {set(scorer_ids)==set(portfolio_ids)}")
  EOF
  ```

## Directory Structure Summary
```
nordic_2026/
├── nordic_config.py          # Central config & paths
├── api_client.py             # API wrapper
├── fetch_data.py             # Data ingestion
├── build_dataset.py          # Feature engineering
├── train_models.py           # Model training
├── nordic_scorer.py          # Prediction & scoring
├── mls_scorer.py             # MLS predictions
├── csl_scorer.py             # CSL predictions
├── nordic_app.py             # Betting dashboard (Streamlit)
├── invest_app.py             # Portfolio investment dashboard (Streamlit)
├── portfolio_scorer.py       # Portfolio signal generation
├── online_settle.py          # Bet settlement
├── sync_portfolio_with_scorers.py  # Sync Nordic signals to portfolio
├── create_missing_portfolio_files.py  # Create missing portfolio CSVs
├── process_existing_predictions.py  # Batch reprocessing
├── nordic_backtest.py        # Backtest strategies on 2022–2026 data
├── data/
│   ├── daily/                # Current season matches (2026)
│   ├── historical/           # 2022–2025 data by league
│   ├── telemetry/            # Scorer CSV logs by league
│   ├── portfolio/            # Portfolio CSV files (invest_app source of truth)
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
    ├── mls/                  # Models for USA
    ├── csl/                  # Models for China
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
