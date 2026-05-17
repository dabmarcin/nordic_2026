# Portfolio-Scorer Data Sync

## The Problem

The portfolio CSV files are the source of truth for the investment dashboard (`invest_app.py`). However, Nordic signals (Allsvenskan BTTS Yes and Eliteserien Under 9.5 Corners) were being generated in the scorer files but not always reflected in the portfolio CSVs, causing data discrepancies.

### Root Causes

1. **Missing portfolio files for some dates**: If `portfolio_scorer.py` never ran for a particular date, no portfolio CSV was created for that date, even if scorer files existed.

2. **Incomplete backfill**: The original `update_portfolio_with_nordic.py` was a one-time backfill that only ran once. Any new scorer signals generated after that script ran were never added to portfolio.

## Solution

Three complementary scripts have been created:

### 1. `sync_portfolio_with_scorers.py`
**Purpose**: Add missing Nordic signals to existing portfolio files

**Usage**:
```bash
python sync_portfolio_with_scorers.py
```

**What it does**:
- Scans all Allsvenskan and Eliteserien scorer files
- For each existing portfolio CSV, adds any missing Nordic signals from scorers
- Removes duplicates and re-sorts

### 2. `create_missing_portfolio_files.py`
**Purpose**: Create portfolio CSVs for dates that have scorer data but no portfolio file

**Usage**:
```bash
python create_missing_portfolio_files.py
```

**What it does**:
- Identifies dates with Nordic signals in scorers that don't have a portfolio CSV
- Creates a new portfolio CSV for each missing date with those signals

### 3. `backfill_missing_nordic_signals.py`
**Purpose**: (Deprecated/Legacy) First attempt at syncing; replaced by sync scripts above

## Recommended Workflow

### After running scorer files:

```bash
# Step 1: Create any missing portfolio files for those dates
python create_missing_portfolio_files.py

# Step 2: Sync existing portfolio files with any new scorer signals
python sync_portfolio_with_scorers.py

# Step 3: Clear cache in invest_app
# Click "🗑️ Wyczyść cache" button in Settings tab, then refresh browser
```

### Or run both steps automatically:

```bash
python create_missing_portfolio_files.py && python sync_portfolio_with_scorers.py
```

## Data Verification

To verify that scorer data matches portfolio data:

```bash
python << 'EOF'
import os, pandas as pd, glob

# Check Allsvenskan BTTS Yes
scorer_dir = r"C:\Projects\nordic_2026\data\telemetry\Allsvenskan scorer"
portfolio_dir = r"C:\Projects\nordic_2026\data\portfolio"

scorer_ids = []
for f in glob.glob(os.path.join(scorer_dir, "*.csv")):
    df = pd.read_csv(f, encoding='utf-8-sig')
    scorer_ids.extend(df[(df['Model_type']=='gpt_pred') & (df['Typ']=='BTTS Yes')]['ID'].tolist())

portfolio_ids = []
for f in glob.glob(os.path.join(portfolio_dir, "*.csv")):
    df = pd.read_csv(f, encoding='utf-8-sig')
    portfolio_ids.extend(df[(df['Liga']=='Allsvenskan') & (df['Signal_ID']=='allsv_btts_yes')]['ID'].tolist())

print(f"Allsvenskan BTTS Yes: Scorers={len(set(scorer_ids))}, Portfolio={len(set(portfolio_ids))}, Match={'YES' if set(scorer_ids)==set(portfolio_ids) else 'NO'}")
EOF
```

## Integration with Daily Workflow

To make this automatic, add to your daily `.bat` script after running scorers:

```batch
@REM Sync portfolio with scorer data
python sync_portfolio_with_scorers.py
python create_missing_portfolio_files.py
```

Or integrate into the Streamlit Settings tab as a button.

## Future Improvements

1. **Automatic sync in portfolio_scorer.py**: Modify portfolio_scorer.py to automatically pull and sync Nordic signals from scorers during its daily run, rather than requiring manual sync.

2. **Continuous monitoring**: Add a scheduled job (cron, Windows Task Scheduler) to periodically verify scorer/portfolio data consistency.

3. **API-based settlement**: Once portfolio data matches scorers perfectly, the `invest_app.py` settling button could automatically sync with latest results.
