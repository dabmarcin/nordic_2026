# Portfolio Signals - Odds Validation & Quality Assurance

## The Problem

Portfolio signals were being generated with odds (Kurs) that violated their signal type's defined range. Examples:

```
Signal: csl_draw (required 3.80-4.50)        → Kurs: 6.0    ❌
Signal: mls_away_win_hi (required 3.80-5.00) → Kurs: 2.19   ❌
```

This resulted in **502 invalid signals** across all portfolio files being added to investment calculations, distorting ROI and performance metrics.

## Root Cause

The portfolio files were created without proper validation against `PORTFOLIO_SIGNALS` odds ranges defined in `nordic_config.py`.

## Solution

### 1. Cleaned All Existing Portfolio Files

**Script:** `validate_portfolio_odds.py`

Removed 502 invalid signals from 41 portfolio files:
- **csl_draw**: 3.80 <= odds <= 4.50
- **mls_away_win_hi**: 3.80 <= odds <= 5.00
- **csl_under_corners**: odds >= 2.20
- **mls_over_corners**: odds >= 2.00

**To re-run validation:**
```bash
python validate_portfolio_odds.py
```

### 2. Fixed Code to Use Dynamic Validation

**File:** `portfolio_scorer.py`

Refactored `generate_rule_signals()` to:
- Use `validate_odds(signal_id, kurs)` function
- Dynamically read odds ranges from `PORTFOLIO_SIGNALS` configuration
- Skip any signal that doesn't meet the criteria
- **No hardcoded odds checks anymore**

**Key change:**
```python
def validate_odds(signal_id, kurs):
    """Sprawdź czy odds spełniają warunki sygnału"""
    signal = PORTFOLIO_SIGNALS[signal_id]
    condition = signal.get('condition', '')
    # Parse "X <= odds <= Y" or "odds >= X" and validate
    # Returns True if valid, False otherwise
```

### 3. Documented Validation Rules

**File:** `CLAUDE.md`

Added comprehensive documentation of all signal odds ranges and validation requirements.

## Portfolio Signal Definitions

From `nordic_config.py`:

| Signal ID | League | Type | Odds Range | Model Source |
|-----------|--------|------|-----------|--------------|
| csl_draw | CSL | Draw | 3.80-4.50 | rule |
| mls_away_win_hi | MLS | Away Win | 3.80-5.00 | rule |
| csl_under_corners | CSL | Under 9.5C | >= 2.20 | rule |
| mls_over_corners | MLS | Over 9.5C | >= 2.00 | rule |
| elite_under_corners | Eliteserien | Under 9.5C | *no limit* | scorer |
| allsv_btts_yes | Allsvenskan | BTTS Yes | *no limit* | scorer |

**Note:** Scorer-based signals (elite_under_corners, allsv_btts_yes) have no odds restrictions as they come directly from ML models.

## Prevention & QA Workflow

### Before Adding New Signals

1. **Check `PORTFOLIO_SIGNALS`** in `nordic_config.py`
   - Verify signal ID exists
   - Confirm odds range is correct

2. **Run `portfolio_scorer.py --daily`**
   - Automatically validates all rule-based signals
   - Only adds signals that pass validation

3. **Verify with `validate_portfolio_odds.py`**
   - Run weekly to catch any invalid additions
   ```bash
   python validate_portfolio_odds.py
   ```

### Daily Workflow

```bash
# Generate new signals
python portfolio_scorer.py --daily

# Create missing portfolio files for new dates
python create_missing_portfolio_files.py

# Sync portfolio with scorer data
python sync_portfolio_with_scorers.py

# Optional: Validate entire portfolio
python validate_portfolio_odds.py
```

## For Developers

### To Add a New Portfolio Signal

1. **Define in `nordic_config.py`:**
```python
PORTFOLIO_SIGNALS = {
    ...
    "new_signal_id": {
        "league":    "league_code",        # allsvenskan, eliteserien, mls, csl, etc.
        "model_src": "rule",               # "rule" or "scorer"
        "typ":       "Signal Type",        # e.g., "Draw", "BTTS Yes"
        "condition": "odds_col >= X and odds_col <= Y",  # or just "odds_col >= X"
        "odds_col":  "odds_column_name",   # e.g., "odds_ft_x", "odds_corners_under_95"
        "label":     "Display Label X-Y",  # shown in reports
        "tier":      "A",                  # or "B"
    },
}
```

2. **Code automatically handles validation**
   - No need to modify `portfolio_scorer.py`
   - `validate_odds()` reads from config
   - New signal is validated on next run

3. **Test with:**
```python
from portfolio_scorer import validate_odds
assert validate_odds("new_signal_id", 4.2) == True   # Test valid odds
assert validate_odds("new_signal_id", 6.0) == False  # Test invalid odds
```

## Metrics Summary

| Metric | Before | After |
|--------|--------|-------|
| Invalid signals (total) | 502 | 0 |
| Portfolio files affected | 41 | 41 (cleaned) |
| Validation automated | No | Yes |
| Code maintainability | Low | High |

## Files Modified

- ✅ `portfolio_scorer.py` — Added `validate_odds()` function, refactored signal generation
- ✅ `validate_portfolio_odds.py` — New validation script
- ✅ `CLAUDE.md` — Updated documentation
- ✅ All 46 portfolio CSV files — Cleaned, 502 invalid signals removed

## Next Steps

1. ✅ Run invest_app and verify KPIs are now correct (cleared cache)
2. ✅ Monitor daily portfolio_scorer runs for any validation rejections
3. Optional: Set up scheduled validation with `python validate_portfolio_odds.py` (weekly)
