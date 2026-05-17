#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Invest App - Professional Portfolio Investment Dashboard
Displays portfolio performance analytics with equity curves, per-signal analysis, and detailed metrics.
"""

import sys
import io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except (AttributeError, ValueError):
    pass

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
import glob
import os
from pathlib import Path

# Add script directory to path
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from nordic_config import (
    PORTFOLIO_DIR, PORTFOLIO_SIGNALS,
    ALLSV_SCORER_DIR, ELITE_SCORER_DIR, VEIKK_SCORER_DIR,
    MLS_SCORER_DIR, CSL_SCORER_DIR,
)

# Streamlit page config
st.set_page_config(
    page_title="Nordic 2026 - Portfolio",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
:root {
    --primary-color: #1f77b4;
    --success-color: #2ca02c;
    --danger-color: #d62728;
    --warning-color: #ff7f0e;
    --neutral-color: #7f7f7f;
}

body {
    background-color: #f8f9fa;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.kpi-card {
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid #1f77b4;
    margin-bottom: 10px;
}

.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: #1f77b4;
    margin: 10px 0;
}

.kpi-label {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #666;
    margin-bottom: 5px;
}

.kpi-trend {
    font-size: 12px;
    margin-top: 5px;
}

.trend-up {
    color: #2ca02c;
}

.trend-down {
    color: #d62728;
}

.signal-card {
    background: white;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-top: 3px solid #1f77b4;
    margin-bottom: 10px;
}

.signal-label {
    font-weight: 600;
    font-size: 14px;
    color: #333;
    margin-bottom: 12px;
}

.signal-metric {
    display: inline-block;
    margin-right: 20px;
    font-size: 12px;
    color: #666;
}

.signal-metric-value {
    font-weight: 700;
    color: #1f77b4;
    display: block;
    font-size: 16px;
}

.tier-a {
    border-left: 4px solid #2ca02c;
    background: #f0fdf4;
}

.tier-b {
    border-left: 4px solid #ff7f0e;
    background: #fff7ed;
}

.chart-container {
    background: white;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.metric-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}

.divider {
    border-top: 1px solid #e0e0e0;
    margin: 20px 0;
}

</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_portfolio():
    """Load portfolio CSV + scorer data to match nordic_app statistics."""
    # Load portfolio CSV
    portfolio_dfs = []
    for file in glob.glob(os.path.join(PORTFOLIO_DIR, "portfolio_*.csv")):
        try:
            df = pd.read_csv(file, encoding='utf-8-sig')
            portfolio_dfs.append(df)
        except Exception:
            pass

    # Load scorer data (allsvenskan + eliteserien)
    scorer_dfs = []

    # Allsvenskan BTTS Yes (gpt_pred)
    for file in glob.glob(os.path.join(ALLSV_SCORER_DIR, "allsvenskan_scorer_*.csv")):
        try:
            df = pd.read_csv(file, encoding='utf-8-sig')
            df_btts = df[(df.get('Model_type') == 'gpt_pred') & (df.get('Typ') == 'BTTS Yes')].copy()
            if not df_btts.empty:
                df_btts['Liga'] = 'Allsvenskan'
                df_btts['Signal_ID'] = 'allsv_btts_yes'
                df_btts['Signal_Label'] = 'Allsvenskan GPT BTTS Yes'
                df_btts['Tier'] = 'B'
                df_btts['Source'] = 'scorer'
                df_btts['Stake_PLN'] = 100.0
                scorer_dfs.append(df_btts[['ID', 'Data', 'Godzina', 'Mecz', 'Liga', 'Signal_ID', 'Signal_Label',
                                           'Tier', 'Source', 'Typ', 'Kurs', 'Stake_PLN', 'Wynik', 'Profit_PLN']])
        except Exception:
            pass

    # Eliteserien Under 9.5 Corners (liga)
    for file in glob.glob(os.path.join(ELITE_SCORER_DIR, "eliteserien_scorer_*.csv")):
        try:
            df = pd.read_csv(file, encoding='utf-8-sig')
            df_under = df[(df.get('Model_type') == 'liga') & (df.get('Typ') == 'Under 9.5 corners')].copy()
            if not df_under.empty:
                df_under['Liga'] = 'Eliteserien'
                df_under['Signal_ID'] = 'elite_under_corners'
                df_under['Signal_Label'] = 'Eliteserien ML Under 9.5C'
                df_under['Tier'] = 'B'
                df_under['Source'] = 'scorer'
                df_under['Stake_PLN'] = 100.0
                scorer_dfs.append(df_under[['ID', 'Data', 'Godzina', 'Mecz', 'Liga', 'Signal_ID', 'Signal_Label',
                                            'Tier', 'Source', 'Typ', 'Kurs', 'Stake_PLN', 'Wynik', 'Profit_PLN']])
        except Exception:
            pass

    # Combine all data
    all_dfs = portfolio_dfs + scorer_dfs
    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    # Ensure required columns exist
    required_cols = ['ID', 'Data', 'Godzina', 'Mecz', 'Liga', 'Signal_ID', 'Signal_Label',
                     'Tier', 'Source', 'Typ', 'Kurs', 'Stake_PLN', 'Wynik', 'Profit_PLN']
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Convert data types
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df['Kurs'] = pd.to_numeric(df['Kurs'], errors='coerce')
    df['Stake_PLN'] = pd.to_numeric(df['Stake_PLN'], errors='coerce')
    df['Wynik'] = pd.to_numeric(df['Wynik'], errors='coerce')
    df['Profit_PLN'] = pd.to_numeric(df['Profit_PLN'], errors='coerce')

    # Deduplication - keep first occurrence
    df = df.drop_duplicates(subset=['ID', 'Signal_ID'], keep='first')

    # Sort by date
    df = df.sort_values('Data', ascending=False)

    return df

def format_currency(value):
    """Format value as PLN currency."""
    if pd.isna(value):
        return "—"
    return f"{value:,.2f} PLN" if value != 0 else "0.00 PLN"

def format_percent(value):
    """Format value as percentage."""
    if pd.isna(value):
        return "—"
    return f"{value:.1f}%"

def calculate_equity_curve(df):
    """Calculate cumulative profit over time."""
    # Filter only settled bets
    df_settled = df[df['Wynik'].isin([0.0, 1.0])].copy()
    if df_settled.empty:
        return pd.DataFrame()

    df_settled = df_settled.sort_values('Data')
    df_settled['Cumulative_Profit'] = df_settled['Profit_PLN'].cumsum()
    df_settled['Running_Max'] = df_settled['Cumulative_Profit'].cummax()
    df_settled['Drawdown'] = df_settled['Cumulative_Profit'] - df_settled['Running_Max']

    return df_settled

def calculate_daily_profit(df):
    """Calculate daily profit aggregation."""
    df_settled = df[df['Wynik'].isin([0.0, 1.0])].copy()
    if df_settled.empty:
        return pd.DataFrame()

    daily = df_settled.groupby('Data').agg({
        'Profit_PLN': 'sum',
        'Wynik': 'sum',
        'ID': 'count'
    }).rename(columns={'ID': 'N'})
    daily['Win_Count'] = daily['Wynik'].astype(int)
    daily['Win_Rate'] = (daily['Win_Count'] / daily['N'] * 100).round(1)
    daily = daily.reset_index()
    daily['Color'] = daily['Profit_PLN'].apply(lambda x: 'Wygrana' if x > 0 else ('Strata' if x < 0 else 'Remis'))

    return daily

def get_signal_metrics(df, signal_id):
    """Calculate metrics for a specific signal."""
    df_signal = df[df['Signal_ID'] == signal_id].copy()
    df_settled = df_signal[df_signal['Wynik'].isin([0.0, 1.0])]

    n = len(df_settled)
    if n == 0:
        return {
            'n': 0,
            'wins': 0,
            'win_rate': 0,
            'avg_odds': 0,
            'total_staked': 0,
            'total_profit': 0,
            'roi': 0,
            'tier': df_signal['Tier'].iloc[0] if not df_signal.empty else 'N/A'
        }

    wins = (df_settled['Wynik'] == 1).sum()
    avg_odds = df_settled['Kurs'].mean()
    total_staked = df_settled['Stake_PLN'].sum()
    total_profit = df_settled['Profit_PLN'].sum()
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

    return {
        'n': n,
        'wins': wins,
        'win_rate': wins / n * 100,
        'avg_odds': avg_odds,
        'total_staked': total_staked,
        'total_profit': total_profit,
        'roi': roi,
        'tier': df_signal['Tier'].iloc[0] if not df_signal.empty else 'N/A'
    }

# Load data
df_portfolio = load_portfolio()

if df_portfolio.empty:
    st.error("❌ Brak danych w portfolio. Uruchom portfolio_scorer.py aby wygenerować sygnały.")
    st.stop()

# Sidebar filters
st.sidebar.markdown("### 🎯 Filtry")

# Bankroll and stake info
bankroll = st.sidebar.number_input("💰 Bankroll (PLN)", value=1000.0, min_value=10.0)
user_stake = st.sidebar.number_input("💵 Stawka na zakład (PLN)", value=100.0, min_value=1.0, step=10.0)
avg_stake = df_portfolio['Stake_PLN'].mean()

# Tier filter
tier_filter = st.sidebar.multiselect(
    "📊 Tier",
    options=['A', 'B'],
    default=['A', 'B'],
    key='tier_filter'
)

# Signal filter
available_signals = sorted(df_portfolio['Signal_ID'].unique())
signal_filter = st.sidebar.multiselect(
    "🎲 Sygnały",
    options=available_signals,
    default=available_signals,
    key='signal_filter'
)

# Date filter by months
available_months = sorted(df_portfolio['Data'].dt.strftime('%Y-%m').unique())
month_filter = st.sidebar.multiselect(
    "📅 Miesiące",
    options=available_months,
    default=available_months,
    key='month_filter'
)

# Apply filters
df_filtered = df_portfolio[
    (df_portfolio['Tier'].isin(tier_filter)) &
    (df_portfolio['Signal_ID'].isin(signal_filter)) &
    (df_portfolio['Data'].dt.strftime('%Y-%m').isin(month_filter))
].copy()

df_settled = df_filtered[df_filtered['Wynik'].isin([0.0, 1.0])].copy()

# Apply stake multiplier to adjust all profits dynamically
multiplier = user_stake / 100.0
if not df_settled.empty:
    df_settled['Profit_PLN'] = df_settled['Profit_PLN'] * multiplier
    df_settled['Stake_PLN'] = df_settled['Stake_PLN'] * multiplier

# Main dashboard
st.title("💼 Portfolio Investment Dashboard")
st.markdown("Professional investment signal tracking and performance analytics")

# KPI Cards Row 1
st.markdown("### 📈 Kluczowe Wskaźniki")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    roi_total = 0
    if not df_settled.empty and df_settled['Stake_PLN'].sum() > 0:
        roi_total = (df_settled['Profit_PLN'].sum() / df_settled['Stake_PLN'].sum() * 100)

    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">ROI</div>
        <div class="kpi-value" style="color: {'#2ca02c' if roi_total >= 0 else '#d62728'}">
            {roi_total:+.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_profit = df_settled['Profit_PLN'].sum() if not df_settled.empty else 0
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Profit</div>
        <div class="kpi-value" style="color: {'#2ca02c' if total_profit >= 0 else '#d62728'}">
            {total_profit:+,.0f}
        </div>
        <div style="font-size: 11px; color: #999;">PLN</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_odds_total = df_settled['Kurs'].mean() if not df_settled.empty else 0
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Śr. Kurs</div>
        <div class="kpi-value">{avg_odds_total:.2f}</div>
        <div style="font-size: 11px; color: #999;">Wszystkie</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    n_bets = len(df_settled)
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Zakłady</div>
        <div class="kpi-value">{n_bets}</div>
        <div style="font-size: 11px; color: #999;">Rozliczone</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    win_rate = 0
    if not df_settled.empty:
        win_rate = (df_settled['Wynik'] == 1).sum() / len(df_settled) * 100
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Win Rate</div>
        <div class="kpi-value">{win_rate:.1f}%</div>
        <div style="font-size: 11px; color: #999;">Wygrane / Razem</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    total_staked = df_settled['Stake_PLN'].sum() if not df_settled.empty else 0
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Zainwestowano</div>
        <div class="kpi-value">{total_staked:,.0f}</div>
        <div style="font-size: 11px; color: #999;">PLN</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Equity Curve
st.markdown("### 📊 Krzywa Kapitału")
df_equity = calculate_equity_curve(df_settled)

if not df_equity.empty:
    # Create equity curve chart
    equity_chart = alt.Chart(df_equity).mark_area(
        line=True,
        opacity=0.3,
        color='#1f77b4'
    ).encode(
        x=alt.X('Data:T', title='Data'),
        y=alt.Y('Cumulative_Profit:Q', title='Zysk/Strata (PLN)'),
        tooltip=['Data:T', 'Cumulative_Profit:Q', 'Drawdown:Q']
    ).properties(
        height=300
    )

    st.altair_chart(equity_chart, use_container_width=True)

    # Drawdown info
    max_drawdown = df_equity['Drawdown'].min()
    st.info(f"📉 Maksymalne obniżenie (Drawdown): **{max_drawdown:.2f} PLN**")
else:
    st.warning("Brak danych do wyświetlenia krzywej kapitału.")

st.markdown("---")

# Per-Signal Analysis
st.markdown("### 🎯 Analiza Sygnałów")

signal_cols = st.columns(2)
for idx, signal_id in enumerate(sorted(signal_filter)):
    metrics = get_signal_metrics(df_filtered, signal_id)
    signal_label = df_filtered[df_filtered['Signal_ID'] == signal_id]['Signal_Label'].iloc[0] if not df_filtered[df_filtered['Signal_ID'] == signal_id].empty else signal_id

    with signal_cols[idx % 2]:
        tier_class = "tier-a" if metrics['tier'] == 'A' else "tier-b"
        st.markdown(f"""
        <div class="signal-card {tier_class}">
            <div class="signal-label">🎲 {signal_label}</div>
            <div style="font-size: 11px; color: #999; margin-bottom: 10px;">Tier {metrics['tier']}</div>
            <div>
                <span class="signal-metric">
                    <span class="signal-metric-value">{metrics['win_rate']:.1f}%</span>
                    Win Rate
                </span>
                <span class="signal-metric">
                    <span class="signal-metric-value">{metrics['avg_odds']:.2f}</span>
                    Śr. Kurs
                </span>
                <span class="signal-metric">
                    <span class="signal-metric-value">{metrics['roi']:+.1f}%</span>
                    ROI
                </span>
                <span class="signal-metric">
                    <span class="signal-metric-value">{metrics['total_profit']:+,.0f}</span>
                    Profit (PLN)
                </span>
                <span class="signal-metric">
                    <span class="signal-metric-value">{metrics['n']}</span>
                    Bets
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Per-Signal Equity Curve
st.markdown("### 📈 Krzywa Kapitału Wg Sygnału")

for signal_id in sorted(signal_filter):
    df_signal_settled = df_settled[df_settled['Signal_ID'] == signal_id].copy()
    if df_signal_settled.empty:
        continue

    df_signal_settled = df_signal_settled.sort_values('Data')
    df_signal_settled['Cumulative_Profit'] = df_signal_settled['Profit_PLN'].cumsum()

    signal_label = df_filtered[df_filtered['Signal_ID'] == signal_id]['Signal_Label'].iloc[0]

    chart = alt.Chart(df_signal_settled).mark_line(point=True, color='#1f77b4').encode(
        x=alt.X('Data:T', title='Data'),
        y=alt.Y('Cumulative_Profit:Q', title='Zysk/Strata (PLN)'),
        tooltip=['Data:T', 'Cumulative_Profit:Q', 'Mecz:N']
    ).properties(
        height=200,
        title=f"📊 {signal_label}"
    )

    st.altair_chart(chart, use_container_width=True)

st.markdown("---")

# Daily Profit Chart
st.markdown("### 💵 Dzienny Zysk/Strata")
df_daily = calculate_daily_profit(df_settled)

if not df_daily.empty:
    daily_chart = alt.Chart(df_daily).mark_bar().encode(
        x=alt.X('Data:T', title='Data'),
        y=alt.Y('Profit_PLN:Q', title='Zysk/Strata (PLN)'),
        color=alt.Color('Color:N', scale=alt.Scale(
            domain=['Wygrana', 'Strata', 'Remis'],
            range=['#2ca02c', '#d62728', '#7f7f7f']
        )),
        tooltip=['Data:T', 'Profit_PLN:Q', 'N:Q', 'Win_Rate:Q']
    ).properties(
        height=250
    )

    st.altair_chart(daily_chart, use_container_width=True)
else:
    st.warning("Brak danych do wyświetlenia wykresu dziennych zysków.")

st.markdown("---")

# Detailed Table with CSV Export
st.markdown("### 📋 Szczegółowa Tabela")

with st.expander("Pełna historia zakładów", expanded=False):
    if not df_filtered.empty:
        # Display columns
        display_cols = ['Data', 'Godzina', 'Liga', 'Mecz', 'Signal_Label', 'Typ', 'Tier',
                       'Kurs', 'Stake_PLN', 'Wynik', 'Profit_PLN']

        df_display = df_filtered[display_cols].copy()
        df_display = df_display.sort_values('Data', ascending=False)

        # Format for display
        df_display['Kurs'] = df_display['Kurs'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        df_display['Stake_PLN'] = df_display['Stake_PLN'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
        df_display['Wynik'] = df_display['Wynik'].apply(lambda x: "✓" if x == 1 else ("✗" if x == 0 else "—"))
        df_display['Profit_PLN'] = df_display['Profit_PLN'].apply(
            lambda x: f"<span style='color:green'>+{x:.0f}</span>" if x > 0 else (
                f"<span style='color:red'>{x:.0f}</span>" if x < 0 else "—"
            ) if pd.notna(x) else "—"
        )

        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # CSV export
        csv = df_filtered[display_cols].to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="⬇️ Pobierz CSV",
            data=csv,
            file_name=f"portfolio_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("Brak danych do wyświetlenia.")

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; color: #999; font-size: 12px; margin-top: 40px;">
    <p>Nordic 2026 Portfolio Dashboard | Dane zaktualizowane: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
