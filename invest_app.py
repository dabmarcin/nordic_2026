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
import json
import os
from pathlib import Path
import subprocess
import time

# Add script directory to path
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

from nordic_config import (
    PORTFOLIO_DIR, PORTFOLIO_SIGNALS, CUSTOM_SIGNALS_FILE,
    ALLSV_SCORER_DIR, ELITE_SCORER_DIR, VEIKK_SCORER_DIR,
    MLS_SCORER_DIR, CSL_SCORER_DIR,
    DATA_DIR,
)

MONITOR_DIR = os.path.join(DATA_DIR, "monitor")

def _load_custom_signals_local() -> dict:
    """Wczytaj custom signals z JSON dla UI."""
    if not os.path.isfile(CUSTOM_SIGNALS_FILE):
        return {}
    try:
        with open(CUSTOM_SIGNALS_FILE, encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _all_signals_local() -> dict:
    """Built-in + custom signals (dla list w UI)."""
    merged = dict(PORTFOLIO_SIGNALS)
    for sid, cfg in _load_custom_signals_local().items():
        if sid not in merged:
            merged[sid] = {**cfg, 'is_custom': True}
    return merged

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

@st.cache_data(ttl=60)
def load_portfolio():
    """Load portfolio CSV only (scorer data already synced via sync_portfolio_with_scorers.py)."""
    # Load portfolio CSV
    portfolio_dfs = []
    for file in glob.glob(os.path.join(PORTFOLIO_DIR, "portfolio_*.csv")):
        try:
            df = pd.read_csv(file, encoding='utf-8-sig')
            portfolio_dfs.append(df)
        except Exception:
            pass

    if not portfolio_dfs:
        return pd.DataFrame()

    df = pd.concat(portfolio_dfs, ignore_index=True)

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

# Tab selector
tab_dashboard, tab_monitor = st.tabs(["📊 Dashboard", "🔔 Monitor"])

# ── DASHBOARD TAB ──────────────────────────────────────────────────────

with tab_dashboard:

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

# Signal filter — show ALL signals from PORTFOLIO_SIGNALS
available_signals = sorted(PORTFOLIO_SIGNALS.keys())
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
    signal_label = PORTFOLIO_SIGNALS.get(signal_id, {}).get('label', signal_id)

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
        st.info(f"ℹ️ **{PORTFOLIO_SIGNALS.get(signal_id, {}).get('label', signal_id)}** — Brak danych")
        continue

    df_signal_settled = df_signal_settled.sort_values('Data')
    df_signal_settled['Cumulative_Profit'] = df_signal_settled['Profit_PLN'].cumsum()

    signal_label = PORTFOLIO_SIGNALS.get(signal_id, {}).get('label', signal_id)

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

# Daily Predictions Preview
st.markdown("### 📋 Podgląd Predykcji Dnia")

# Get available dates from portfolio files
available_dates = sorted(
    [f.replace(os.path.join(PORTFOLIO_DIR, 'portfolio_'), '').replace('.csv', '')
     for f in glob.glob(os.path.join(PORTFOLIO_DIR, 'portfolio_*.csv'))],
    reverse=True
)

if available_dates:
    selected_date = st.selectbox(
        "📅 Wybierz datę",
        options=available_dates,
        index=0,
        key='date_selector'
    )

    # Load portfolio for selected date
    portfolio_file = os.path.join(PORTFOLIO_DIR, f'portfolio_{selected_date}.csv')
    try:
        df_day = pd.read_csv(portfolio_file, encoding='utf-8-sig')

        # Ensure required columns exist
        required_cols = ['Mecz', 'Liga', 'Signal_Label', 'Typ', 'Kurs', 'Rezultat', 'Corners', 'Wynik']
        for col in required_cols:
            if col not in df_day.columns:
                df_day[col] = None

        # Convert data types
        df_day['Kurs'] = pd.to_numeric(df_day['Kurs'], errors='coerce')
        df_day['Corners'] = pd.to_numeric(df_day['Corners'], errors='coerce')
        df_day['Wynik'] = pd.to_numeric(df_day['Wynik'], errors='coerce')

        # Sort by match time if available
        if 'Godzina' in df_day.columns:
            df_day = df_day.sort_values('Godzina', ascending=True)

        if not df_day.empty:
            # Display columns
            display_cols = ['Mecz', 'Liga', 'Signal_Label', 'Typ', 'Kurs', 'Rezultat', 'Corners', 'Wynik']
            df_display = df_day[display_cols].copy()

            # Format for display
            df_display['Kurs'] = df_display['Kurs'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
            df_display['Corners'] = df_display['Corners'].apply(lambda x: f"{int(x)}" if pd.notna(x) else "—")
            df_display['Wynik'] = df_display['Wynik'].apply(lambda x: "✅ Wygrana" if x == 1 else ("❌ Przegrana" if x == 0 else "⏳ Oczekuje"))
            df_display['Rezultat'] = df_display['Rezultat'].apply(lambda x: str(x) if pd.notna(x) else "—")

            st.dataframe(df_display, use_container_width=True, hide_index=True)

            # Summary stats for selected day
            st.markdown("#### 📊 Statystyka Dnia")
            col1, col2, col3, col4, col5 = st.columns(5)

            df_settled_day = df_day[df_day['Wynik'].isin([0.0, 1.0])]

            with col1:
                st.metric("Predykcji", len(df_day))

            with col2:
                settled_count = len(df_settled_day)
                st.metric("Rozliczonych", settled_count)

            with col3:
                if len(df_settled_day) > 0:
                    win_count = (df_settled_day['Wynik'] == 1).sum()
                    win_rate = (win_count / len(df_settled_day) * 100)
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                else:
                    st.metric("Win Rate", "—")

            with col4:
                avg_odds = df_day['Kurs'].mean()
                if pd.notna(avg_odds):
                    st.metric("Śr. Kurs", f"{avg_odds:.2f}")
                else:
                    st.metric("Śr. Kurs", "—")

            with col5:
                if 'Profit_PLN' in df_day.columns:
                    total_profit = pd.to_numeric(df_day['Profit_PLN'], errors='coerce').sum()
                    st.metric("Zysk/Strata", f"{total_profit:+.0f} PLN",
                             delta_color="inverse" if total_profit > 0 else "normal")
                else:
                    st.metric("Zysk/Strata", "—")

            # CSV export
            csv = df_day[display_cols].to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="⬇️ Pobierz CSV",
                data=csv,
                file_name=f"portfolio_predictions_{selected_date}.csv",
                mime="text/csv"
            )
        else:
            st.info(f"Brak danych dla dnia {selected_date}")

    except Exception as e:
        st.error(f"❌ Błąd ładowania danych: {str(e)}")
    else:
        st.warning("Brak dostępnych dat w portfolio.")

# ── MONITOR TAB ────────────────────────────────────────────────────────

with tab_monitor:
    st.markdown("### 🔔 Signal Monitor")

    # Load latest monitor report
    monitor_files = sorted(glob.glob(os.path.join(MONITOR_DIR, "monitor_*.json")))

    if not monitor_files:
        st.info("📋 Brak raportów monitora. Uruchom: `python monitor.py --check`")
    else:
        with open(monitor_files[-1], encoding='utf-8') as f:
            report = json.load(f)

        st.caption(f"Ostatnia aktualizacja: **{report.get('timestamp', '?')}** | Rolling window: **6 zakładów**")

        # ── Run monitor button ──────────────────────────────────
        if st.button("🔄 Uruchom Monitor Teraz", key="run_monitor_btn"):
            try:
                subprocess.run([
                    "python", "monitor.py", "--check"
                ], cwd=str(SCRIPT_DIR), capture_output=True, timeout=30)
                st.success("✅ Monitor zaktualizowany!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Błąd uruchomienia: {str(e)}")

        st.markdown("---")

        # ── Status grid ────────────────────────────────────────
        st.markdown("#### 📊 Status Sygnałów")

        signals = report.get("signals", {})
        cols = st.columns(3)
        colors = {
            "ACTIVE": ("#10b981", "🟢"),
            "WARNING": ("#f59e0b", "🟡"),
            "ALARM": ("#ef4444", "🔴"),
            "DISABLED": ("#6b7280", "⚫"),
        }

        for i, (sig_id, sig_data) in enumerate(
            sorted(signals.items(),
                   key=lambda x: x[1].get("roi_rolling") or -999,
                   reverse=True)
        ):
            status = sig_data.get("status", "ACTIVE")
            color, icon = colors.get(status, ("#6b7280", "❓"))
            roi_r = sig_data.get("roi_rolling")
            roi_str = f"{roi_r:+.1f}%" if roi_r is not None else "—"
            last_r = sig_data.get("last_results", "")

            with cols[i % 3]:
                st.markdown(f"""
                <div style="background:#1f2937;border:1px solid {color};
                     border-left:4px solid {color};border-radius:8px;
                     padding:12px;margin:4px 0">
                  <div style="font-size:11px;color:#9ca3af;
                       text-transform:uppercase;letter-spacing:.08em">
                    {sig_data.get('tier','?')} · {icon} {status}</div>
                  <div style="font-size:13px;font-weight:700;margin:4px 0;color:#ffffff">
                    {sig_data.get('label','?')}</div>
                  <div style="display:flex;justify-content:space-between;font-size:12px;color:#ffffff">
                    <span>Roll6: <b style="color:{color}">{roi_str}</b></span>
                    <span>Overall: <b style="color:#ffffff">{sig_data.get('roi_overall',0):+.1f}%</b></span>
                  </div>
                  <div style="font-size:11px;color:#9ca3af;margin-top:4px;
                       letter-spacing:.1em">{last_r}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Alerts ─────────────────────────────────────────────
        alerts = report.get("alerts", [])
        if alerts:
            st.markdown("#### ⚡ Aktywne Alerty")
            for a in alerts:
                if "ALARM" in a or "🔴" in a:
                    st.error(a)
                elif "WARNING" in a or "🟡" in a:
                    st.warning(a)
                elif "RE-ENABLE" in a:
                    st.success(a)
                else:
                    st.info(a)
        else:
            st.success("✅ Brak alertów — wszystko OK")

        st.markdown("---")

        # ── Candidates ──────────────────────────────────────────
        candidates = report.get("candidates", [])
        if candidates:
            st.markdown("#### 🔵 Kandydaci — Nowe Sygnały")

            cand_new = [c for c in candidates if not c.get("in_portfolio")]
            cand_existing = [c for c in candidates if c.get("in_portfolio")]

            if cand_new:
                st.markdown("**Nowe (nie w portfelu) — kliknij ➕ aby dodać na stałe:**")
                hdr = st.columns([3, 1, 1, 1, 1, 1, 1])
                hdr[0].markdown("**Sygnał**")
                hdr[1].markdown("**Liga**")
                hdr[2].markdown("**Kurs**")
                hdr[3].markdown("**N**")
                hdr[4].markdown("**ROI Roll6**")
                hdr[5].markdown("**Seria**")
                hdr[6].markdown("**Akcja**")

                for idx, c in enumerate(cand_new):
                    cols = st.columns([3, 1, 1, 1, 1, 1, 1])
                    cols[0].text(c.get("label", ""))
                    cols[1].text(c.get("liga", ""))
                    cols[2].text(c.get("odds_range", ""))
                    cols[3].text(str(c.get("n", 0)))
                    roi_r = c.get("roi_rolling", 0)
                    cols[4].markdown(f"<span style='color:#10b981'>+{roi_r:.1f}%</span>" if roi_r >= 0 else f"<span style='color:#ef4444'>{roi_r:.1f}%</span>", unsafe_allow_html=True)
                    cols[5].text(c.get("last_results", ""))
                    if cols[6].button("➕ Dodaj", key=f"add_cand_{idx}_{c.get('label','')}"):
                        with st.spinner(f"Dodawanie '{c.get('label')}' do portfela..."):
                            try:
                                # 1. Add candidate to custom_signals.json via monitor
                                subprocess.run([
                                    sys.executable, "monitor.py", "--add-candidate", c.get("label", "")
                                ], cwd=str(SCRIPT_DIR), capture_output=True, timeout=15, check=False)

                                # 2. Backfill portfolio with new custom signal
                                subprocess.run([
                                    sys.executable, "portfolio_scorer.py", "--backfill"
                                ], cwd=str(SCRIPT_DIR), capture_output=True, timeout=90, check=False)

                                # 3. Refresh monitor metrics
                                subprocess.run([
                                    sys.executable, "monitor.py", "--check"
                                ], cwd=str(SCRIPT_DIR), capture_output=True, timeout=30, check=False)

                                st.success(f"✓ Dodano: {c.get('label')}\n\nSygnał trafił do custom_signals.json. Portfel przebudowany.")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Błąd: {str(e)}")

            if cand_existing:
                st.markdown("**Już w portfelu (potwierdzenie):**")
                df_ce = pd.DataFrame(cand_existing)[[
                    "label", "roi_rolling", "last_results"
                ]]
                df_ce.columns = ["Sygnał", "ROI Roll6", "Seria"]
                st.dataframe(df_ce, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Signal management ───────────────────────────────────
        st.markdown("#### ⚙️ Zarządzaj Sygnałami")

        col_d, col_e = st.columns(2)

        _ui_all_signals = _all_signals_local()
        _ui_signal_keys = sorted(
            _ui_all_signals.keys(),
            key=lambda k: (not _ui_all_signals[k].get('is_custom', False), k)
        )

        def _fmt_sig(sid: str) -> str:
            cfg = _ui_all_signals.get(sid, {})
            prefix = '★ ' if cfg.get('is_custom') else ''
            return f"{prefix}{sid}"

        with col_d:
            sig_to_disable = st.selectbox(
                "Wyłącz sygnał",
                _ui_signal_keys,
                key="mon_disable_sel",
                format_func=_fmt_sig,
            )
            if st.button("⚫ Wyłącz", key="mon_disable_btn"):
                with st.spinner("Wyłączanie sygnału i odświeżanie portfela..."):
                    try:
                        # 1. Disable signal in state
                        subprocess.run([
                            sys.executable, "monitor.py", "--disable", sig_to_disable
                        ], cwd=str(SCRIPT_DIR), capture_output=True, timeout=10, check=False)

                        # 2. Backfill portfolio to remove unsettled bets for disabled signal
                        subprocess.run([
                            sys.executable, "portfolio_scorer.py", "--backfill"
                        ], cwd=str(SCRIPT_DIR), capture_output=True, timeout=60, check=False)

                        # 3. Update monitor status
                        subprocess.run([
                            sys.executable, "monitor.py", "--check"
                        ], cwd=str(SCRIPT_DIR), capture_output=True, timeout=30, check=False)

                        st.success(f"✓ Wyłączono: {sig_to_disable}\n\nPortfel przebudowany — niezapisane zakłady usunięte.")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Błąd: {str(e)}")

        with col_e:
            sig_to_enable = st.selectbox(
                "Włącz sygnał",
                _ui_signal_keys,
                key="mon_enable_sel",
                format_func=_fmt_sig,
            )
            if st.button("🟢 Włącz", key="mon_enable_btn"):
                with st.spinner("Włączanie sygnału i odświeżanie portfela..."):
                    try:
                        # 1. Enable signal in state
                        subprocess.run([
                            sys.executable, "monitor.py", "--enable", sig_to_enable
                        ], cwd=str(SCRIPT_DIR), capture_output=True, timeout=10, check=False)

                        # 2. Backfill portfolio to regenerate bets for re-enabled signal
                        subprocess.run([
                            sys.executable, "portfolio_scorer.py", "--backfill"
                        ], cwd=str(SCRIPT_DIR), capture_output=True, timeout=60, check=False)

                        # 3. Update monitor status
                        subprocess.run([
                            sys.executable, "monitor.py", "--check"
                        ], cwd=str(SCRIPT_DIR), capture_output=True, timeout=30, check=False)

                        st.success(f"✓ Włączono: {sig_to_enable}\n\nPortfel przebudowany — nowe zakłady dodane z historii.")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Błąd: {str(e)}")

        st.markdown("---")

        # ── History ────────────────────────────────────────────
        with st.expander("📋 Historia Alertów"):
            for f in sorted(glob.glob(os.path.join(MONITOR_DIR, "monitor_*.json")))[-10:]:
                try:
                    with open(f, encoding='utf-8') as fp:
                        hist = json.load(fp)
                except Exception:
                    continue

                ts = hist.get("timestamp", "?")
                ha = hist.get("alerts", [])
                if ha:
                    st.markdown(f"**{ts}**")
                    for a in ha:
                        st.text(f"  {a}")

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; color: #999; font-size: 12px; margin-top: 40px;">
    <p>Nordic 2026 Portfolio Dashboard | Dane zaktualizowane: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
