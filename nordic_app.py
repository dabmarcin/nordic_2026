import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from nordic_config import (
    SCRIPT_DIR,
    H2H_CACHE,
    DAILY_DIR,
    ALLSV_SCORER_DIR, ELITE_SCORER_DIR, VEIKK_SCORER_DIR,
    MATCH_DETAILS_DIR,
    CURRENT_DIR,
    ALLSVENSKAN_2026_ID, ELITESERIEN_2026_ID, VEIKKAUSLIIGA_2026_ID,
)

st.set_page_config(
    page_title="Nordic 2026 — Betting Dashboard",
    page_icon="⚽",
    layout="wide",
)

# ── FUNKCJE POMOCNICZE ────────────────────────────────────────────────────────

def load_csv(path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()


def get_scorer_files(league: str) -> list:
    dirs = {
        "allsvenskan":   ALLSV_SCORER_DIR,
        "eliteserien":   ELITE_SCORER_DIR,
        "veikkausliiga": VEIKK_SCORER_DIR,
    }
    d = dirs.get(league, ALLSV_SCORER_DIR)
    files = glob.glob(os.path.join(d, "*scorer*.csv"))
    return sorted(files, reverse=True)


def get_all_scorer_files() -> pd.DataFrame:
    dfs = []
    for league, d in [
        ("allsvenskan",   ALLSV_SCORER_DIR),
        ("eliteserien",   ELITE_SCORER_DIR),
        ("veikkausliiga", VEIKK_SCORER_DIR),
    ]:
        for f in glob.glob(os.path.join(d, "*scorer*.csv")):
            df = load_csv(f)
            if not df.empty:
                if "league" not in df.columns:
                    df["league"] = league
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def extract_date(path: str) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def run_script(script: str, args: list = None):
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, script)] + (args or [])
    result = subprocess.run(
        cmd,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=SCRIPT_DIR,
    )
    return result.returncode, result.stdout, result.stderr


def calc_roi(df: pd.DataFrame) -> dict:
    n = len(df)
    wynik = pd.to_numeric(df.get("Wynik", pd.Series(dtype=float)), errors="coerce")
    kurs  = pd.to_numeric(df.get("Kurs",  pd.Series(dtype=float)), errors="coerce")
    stake = pd.to_numeric(df.get("Stake_PLN", pd.Series(dtype=float)), errors="coerce").fillna(0)
    n_won = int(wynik.fillna(0).sum())
    win_rate = n_won / n if n else 0.0
    total_stake = stake.sum()
    profit = pd.to_numeric(df.get("Profit_PLN", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    roi_pct = (profit / total_stake * 100) if total_stake else 0.0
    avg_kurs = kurs.mean() if not kurs.empty else 0.0
    return {
        "n": n,
        "n_won": n_won,
        "win_rate": win_rate,
        "total_stake": total_stake,
        "total_profit": profit,
        "roi_pct": roi_pct,
        "avg_kurs": avg_kurs,
    }


def settle_match(df: pd.DataFrame, match_id, home_goals: int, away_goals: int,
                 total_corners: int, total_cards: int) -> pd.DataFrame:
    mask = df["ID"] == match_id
    for idx in df[mask].index:
        typ = str(df.at[idx, "Typ"]).lower() if "Typ" in df.columns else ""
        stake = pd.to_numeric(df.at[idx, "Stake_PLN"], errors="coerce") or 0
        kurs  = pd.to_numeric(df.at[idx, "Kurs"],     errors="coerce") or 1

        wynik = None
        if "home win" in typ or "home" in typ and "win" in typ:
            wynik = 1 if home_goals > away_goals else 0
        elif "away win" in typ or "away" in typ and "win" in typ:
            wynik = 1 if away_goals > home_goals else 0
        elif "btts yes" in typ or typ == "btts":
            wynik = 1 if home_goals > 0 and away_goals > 0 else 0
        elif "btts no" in typ:
            wynik = 0 if home_goals > 0 and away_goals > 0 else 1
        elif "over 2.5" in typ or "over2.5" in typ:
            wynik = 1 if (home_goals + away_goals) > 2 else 0
        elif "under 2.5" in typ or "under2.5" in typ:
            wynik = 0 if (home_goals + away_goals) > 2 else 1
        elif "over corners" in typ or "over corner" in typ:
            wynik = 1 if total_corners > 9.5 else 0
        elif "under corners" in typ or "under corner" in typ:
            wynik = 0 if total_corners > 9.5 else 1

        if wynik is not None:
            df.at[idx, "Wynik"] = wynik
            df.at[idx, "Rezultat"] = f"{home_goals}:{away_goals}"
            df.at[idx, "Corners"] = total_corners
            df.at[idx, "Kartki"] = total_cards
            profit = stake * (kurs - 1) if wynik == 1 else -stake
            df.at[idx, "Profit_PLN"] = round(profit, 2)
    return df


def normalize_typ(typ: str, mecz: str) -> str:
    if not isinstance(typ, str) or not typ.strip():
        return typ

    t = typ.strip()
    t_lower = t.lower()

    if " + " in t:
        return "Parlay"

    if t_lower in ("over corners", "corners over"):
        return "Corners Over 9.5"

    m = re.match(r"corners?\s+(over|under)\s+([\d.]+)", t_lower)
    if m:
        direction = m.group(1).capitalize()
        line = m.group(2)
        return f"Corners {direction} {line}"

    m = re.match(
        r"(over|under)\s+([\d.]+)"
        r"(?:\s+(?:goli|goals|bramek|gola|bramki|mål|maalia))?$",
        t_lower)
    if m:
        direction = m.group(1).capitalize()
        line = m.group(2)
        return f"{direction} {line}"

    if t_lower in ("btts yes", "btts tak", "btts – tak", "btts - tak", "btts yes + over 2.5"):
        return "BTTS Yes"
    if t_lower in ("btts no", "btts nie", "btts – nie", "btts - nie"):
        return "BTTS No"
    if t_lower.startswith("btts"):
        return "BTTS Yes"

    if re.search(r"double\s*chance\s*1x", t_lower):
        return "1X"
    if re.search(r"double\s*chance\s*x2", t_lower):
        return "X2"
    if re.search(r"double\s*chance\s*12", t_lower):
        return "12"

    teams = re.split(r"\s+(?:vs|-)?\s+", mecz, maxsplit=1, flags=re.IGNORECASE)
    home = teams[0].strip().lower() if teams else ""
    away = teams[1].strip().lower() if len(teams) > 1 else ""

    dc_m = re.match(r"double\s*chance\s+(.+?)(?:\s+win[/\s]draw)?$", t_lower)
    if dc_m:
        team_mention = dc_m.group(1).strip()
        if home and home in team_mention:
            return "1X"
        if away and away in team_mention:
            return "X2"
        return "1X"

    win_draw_m = re.search(
        r"wygra\s+lub\s+remis|win\s+or\s+draw|win/draw"
        r"|vinner\s+eller\s+uavgjort|vinner\s+eller\s+oavgjort",
        t_lower)
    if win_draw_m:
        if home and home in t_lower:
            return "1X"
        if away and away in t_lower:
            return "X2"
        return "1X"

    win_m = re.search(
        r"wygra|wins?(?!\s+or)|vinner(?!\s+eller)|vann|siegt|voittaa",
        t_lower)
    if win_m:
        if home and home in t_lower:
            return "1"
        if away and away in t_lower:
            return "2"
        pre_win = t_lower[:win_m.start()].strip()
        if home and home in pre_win:
            return "1"
        if away and away in pre_win:
            return "2"
        return "1"

    if t_lower in ("remis", "draw", "x", "oavgjort"):
        return "X"

    if t_lower in ("1", "x", "2", "1x", "x2", "12"):
        return t_lower.upper()

    return t


def apply_flat_stake(df: pd.DataFrame, stake: float) -> pd.DataFrame:
    df = df.copy()
    wynik = pd.to_numeric(df["Wynik"], errors="coerce")
    kurs  = pd.to_numeric(df["Kurs"],  errors="coerce")
    df["Stake_PLN"] = stake
    df["Profit_PLN"] = float("nan")
    df.loc[wynik == 1, "Profit_PLN"] = (kurs[wynik == 1] - 1) * stake
    df.loc[wynik == 0, "Profit_PLN"] = -stake
    return df


def calc_stats_table(df: pd.DataFrame, group_col: str, stake: float) -> pd.DataFrame:
    rows = []
    for val in df[group_col].dropna().unique():
        sub = df[df[group_col] == val].copy()
        sub = apply_flat_stake(sub, stake)
        wynik  = pd.to_numeric(sub["Wynik"],      errors="coerce")
        kurs   = pd.to_numeric(sub["Kurs"],        errors="coerce")
        profit = pd.to_numeric(sub["Profit_PLN"],  errors="coerce")
        n      = len(sub)
        n_won  = int(wynik.fillna(0).sum())
        wr     = n_won / n if n else 0.0
        tot_stake  = n * stake
        tot_profit = profit.fillna(0).sum()
        roi        = (tot_profit / tot_stake * 100) if tot_stake else 0.0
        avg_k      = kurs.mean() if not kurs.empty else 0.0
        rows.append({
            group_col:     val,
            "N":           n,
            "Wygrane":     n_won,
            "Win Rate":    f"{wr:.0%}",
            "Śr. kurs":    f"{avg_k:.2f}",
            "ROI":         f"{roi:+.1f}%",
            "Profit":      f"{tot_profit:+.1f} PLN",
            "_roi_raw":    roi,
            "_profit_raw": tot_profit,
        })
    result = pd.DataFrame(rows)
    if not result.empty and "_roi_raw" in result.columns:
        result = result.sort_values("_roi_raw", ascending=False)
        result = result.drop(columns=["_roi_raw", "_profit_raw"])
    return result


def safe_float(val, default=0.0) -> float:
    try:
        return float(val) if pd.notna(val) else default
    except (TypeError, ValueError):
        return default


def load_league_data(league: str) -> tuple:
    key = {"allsvenskan": "allsvenskan",
           "eliteserien": "eliteserien",
           "veikkausliiga": "veikkausliiga"}.get(league, "allsvenskan")
    teams_path   = os.path.join(CURRENT_DIR, f"{key}_teams_2026.csv")
    matches_path = os.path.join(CURRENT_DIR, f"{key}_matches_2026.csv")
    df_teams   = pd.DataFrame()
    df_matches = pd.DataFrame()
    if os.path.isfile(teams_path):
        df_teams = pd.read_csv(teams_path, encoding='utf-8-sig')
    if os.path.isfile(matches_path):
        df_matches = pd.read_csv(matches_path, encoding='utf-8-sig')
        df_matches = df_matches[df_matches['status'] == 'complete'].copy()

    if not df_teams.empty and not df_matches.empty:
        hid = df_matches['homeID'].astype(str)
        aid = df_matches['awayID'].astype(str)
        hg  = pd.to_numeric(df_matches['homeGoalCount'], errors='coerce').fillna(0)
        ag  = pd.to_numeric(df_matches['awayGoalCount'], errors='coerce').fillna(0)
        home_df = pd.DataFrame({'tid': hid,
                                'w': (hg > ag).astype(int),
                                'd': (hg == ag).astype(int),
                                'l': (hg < ag).astype(int),
                                'gd': hg - ag})
        away_df = pd.DataFrame({'tid': aid,
                                'w': (ag > hg).astype(int),
                                'd': (ag == hg).astype(int),
                                'l': (ag < hg).astype(int),
                                'gd': ag - hg})
        combined = pd.concat([home_df, away_df]).groupby('tid').sum().reset_index()
        combined.columns = ['tid', 'seasonWinsNum_overall',
                            'seasonDrawsNum_overall',
                            'seasonLossesNum_overall',
                            'seasonGoalDifference_overall']
        df_teams['_tid_str'] = df_teams['team_id'].astype(str)
        df_teams = df_teams.merge(combined,
                                  left_on='_tid_str', right_on='tid',
                                  how='left').drop(columns=['_tid_str', 'tid'],
                                                   errors='ignore')
        for col in ['seasonWinsNum_overall', 'seasonDrawsNum_overall',
                    'seasonLossesNum_overall', 'seasonGoalDifference_overall']:
            df_teams[col] = df_teams[col].fillna(0).astype(int)

    if not df_teams.empty and 'cornersTotalAVG_overall' not in df_teams.columns:
        c_for = pd.to_numeric(df_teams.get('cornersAVG_overall', 0),
                              errors='coerce').fillna(0)
        c_ag  = pd.to_numeric(df_teams.get('cornersAgainstAVG_overall', 0),
                              errors='coerce').fillna(0)
        df_teams['cornersTotalAVG_overall'] = c_for + c_ag

    return df_teams, df_matches


def get_team_row(df_teams: pd.DataFrame, team_id) -> dict:
    row = df_teams[df_teams['team_id'].astype(str) == str(team_id)]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def trio_bar_html(label, val_all, val_home, val_away,
                  max_val=100, suffix="", pct=False) -> str:
    def fmt(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{v:.0f}%" if pct else f"{v:.2f}{suffix}".rstrip('0').rstrip('.')

    def bar(v, css_class):
        pct_w = 0
        if v is not None and not (isinstance(v, float) and pd.isna(v)) and max_val:
            pct_w = min(100, (v / max_val) * 100)
        return (f'<div class="bar-bg">'
                f'<div class="bar-fill {css_class}" '
                f'style="width:{pct_w:.1f}%"></div></div>')

    rows = [("ALL", val_all, "bar-overall"),
            ("H",   val_home, "bar-home"),
            ("A",   val_away, "bar-away")]
    bars_html = ""
    for tag, val, css in rows:
        bars_html += (f'<div class="trio-row">'
                      f'<span class="trio-tag">{tag}</span>'
                      f'{bar(val, css)}'
                      f'<span class="trio-val">{fmt(val)}</span>'
                      f'</div>')
    return (f'<div class="trio-stat">'
            f'<span class="trio-label">{label}</span>'
            f'<div class="trio-bars">{bars_html}</div>'
            f'</div>')


def form_badges_html(form_str: str) -> str:
    if not form_str or not isinstance(form_str, str):
        return '<span style="color:var(--muted)">brak danych</span>'
    badges = ""
    for ch in form_str.upper():
        if ch in "WDL":
            badges += (f'<div class="form-badge form-{ch}">{ch}</div>')
    return f'<div class="form-run">{badges}</div>'


# ── ZAKŁADKI ──────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "📅 Mecze",
    "📰 Artykuły",
    "🏆 Wyniki",
    "🏟 Drużyny",
    "📊 Statystyki",
    "⚙️ Ustawienia",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MECZE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Predykcje — Nordic 2026")

    col1, col2 = st.columns([2, 3])
    league_sel = col1.selectbox(
        "Liga",
        ["Wszystkie", "Allsvenskan", "Eliteserien", "Veikkausliiga"],
        key="mecze_liga",
    )
    day_sel = col2.radio(
        "Dzień",
        ["today", "tomorrow"],
        horizontal=True,
        format_func=lambda x: "Dziś" if x == "today" else "Jutro",
        key="mecze_dzien",
    )

    date_str = (
        datetime.now().strftime("%Y-%m-%d")
        if day_sel == "today"
        else (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    )

    liga_map = {
        "Allsvenskan":   ALLSV_SCORER_DIR,
        "Eliteserien":   ELITE_SCORER_DIR,
        "Veikkausliiga": VEIKK_SCORER_DIR,
    }
    ligi_do_pokazania = (
        list(liga_map.keys()) if league_sel == "Wszystkie" else [league_sel]
    )

    dfs = []
    for liga in ligi_do_pokazania:
        for f in glob.glob(os.path.join(liga_map[liga], f"*scorer*{date_str}*.csv")):
            df = load_csv(f)
            if not df.empty:
                df["_liga"] = liga
                dfs.append(df)

    if not dfs:
        st.info(
            f"Brak predykcji na {date_str}. "
            "Przejdź do Ustawień i uruchom scorer."
        )
    else:
        df_show = pd.concat(dfs, ignore_index=True)

        col_f1, col_f2, col_f3 = st.columns(3)
        model_type_sel = col_f1.multiselect(
            "Typ modelu", ["combined", "gpt_pred"], default=["combined", "gpt_pred"]
        )
        min_ev    = col_f2.slider("Min EV%",    -20, 50, -5)
        min_score = col_f3.slider("Min Score%",  50, 80, 60)

        if "Model_type" in df_show.columns:
            df_show = df_show[df_show["Model_type"].isin(model_type_sel)]
        if "EV[%]" in df_show.columns:
            ev_num = (
                df_show["EV[%]"].astype(str)
                .str.replace("%", "").str.replace("+", "")
                .pipe(pd.to_numeric, errors="coerce")
            )
            is_gpt = df_show.get("Model_type", pd.Series()).astype(str) == "gpt_pred"
            df_show = df_show[is_gpt | (ev_num >= min_ev)]
        if "Score[%]" in df_show.columns:
            score_num = pd.to_numeric(df_show["Score[%]"], errors="coerce")
            is_gpt    = df_show.get("Model_type", pd.Series()).astype(str) == "gpt_pred"
            df_show   = df_show[is_gpt | (score_num >= min_score)]

        cols_show = [
            c for c in [
                "Godzina", "Mecz", "_liga", "Model", "Model_type",
                "Typ", "Score[%]", "Kurs", "Pinnacle_odds", "EV[%]", "Stake_PLN",
            ]
            if c in df_show.columns
        ]
        st.dataframe(df_show[cols_show], use_container_width=True, hide_index=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Typów łącznie", len(df_show))
        if "Stake_PLN" in df_show.columns:
            total_stake = pd.to_numeric(df_show["Stake_PLN"], errors="coerce").sum()
            m2.metric("Łączna stawka", f"{total_stake:.1f} PLN")
        if "_liga" in df_show.columns:
            m3.metric("Lig", df_show["_liga"].nunique())
        if "EV[%]" in df_show.columns:
            ev_vals = (
                df_show["EV[%]"].astype(str)
                .str.replace("%", "").str.replace("+", "")
                .pipe(pd.to_numeric, errors="coerce")
            )
            m4.metric("Śr. EV", f"{ev_vals.mean():+.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ARTYKUŁY (GPT FootyStats)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Analiza meczów — FootyStats GPT")

    col1, col2 = st.columns(2)
    art_league = col1.selectbox(
        "Liga",
        ["allsvenskan", "eliteserien", "veikkausliiga"],
        format_func=lambda x: x.capitalize(),
        key="art_league",
    )
    art_day = col2.radio(
        "Dzień", ["today", "tomorrow"],
        format_func=lambda x: "Dziś" if x == "today" else "Jutro",
        horizontal=True,
        key="art_day",
    )

    art_date_str = (
        datetime.now().strftime("%Y-%m-%d")
        if art_day == "today"
        else (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    )

    daily_file = os.path.join(
        DAILY_DIR,
        f"{'today' if art_day == 'today' else 'tomorrow'}_matches_{art_date_str}.csv",
    )

    league_id_map = {
        "allsvenskan":   ALLSVENSKAN_2026_ID,
        "eliteserien":   ELITESERIEN_2026_ID,
        "veikkausliiga": VEIKKAUSLIIGA_2026_ID,
    }

    nordic_matches = []
    if os.path.isfile(daily_file):
        df_day = load_csv(daily_file)
        if not df_day.empty:
            lid = league_id_map[art_league]
            df_liga = df_day[
                pd.to_numeric(df_day["competition_id"], errors="coerce") == lid
            ]
            for _, row in df_liga.iterrows():
                mid = int(row["id"])
                gpt_path = os.path.join(MATCH_DETAILS_DIR, f"gpt_{mid}.json")
                nordic_matches.append({
                    "match_id": mid,
                    "home":     row.get("home_name", "?"),
                    "away":     row.get("away_name", "?"),
                    "gpt_path": gpt_path,
                    "has_gpt":  os.path.isfile(gpt_path),
                })

    if not nordic_matches:
        st.info(
            f"Brak meczów {art_league} na {art_date_str}. "
            "Pobierz dane dzienne w Ustawieniach."
        )
    else:
        if st.button("🔄 Pobierz/odśwież analizy GPT", type="primary"):
            with st.spinner("Pobieranie..."):
                day_arg = "--today" if art_day == "today" else "--tomorrow"
                rc, out, err = run_script("fetch_match_details.py", [day_arg])
            if rc == 0:
                st.success("Pobrano!")
                st.rerun()
            else:
                st.error("Błąd")
                st.code(err[:300])

        for match in nordic_matches:
            mid   = match["match_id"]
            label = (
                f"{'✅' if match['has_gpt'] else '⏳'} "
                f"{match['home']} vs {match['away']}"
            )

            with st.expander(label):
                if not match["has_gpt"]:
                    st.info("Brak analizy. Kliknij 'Pobierz/odśwież'.")
                    continue

                with open(match["gpt_path"], encoding="utf-8") as f:
                    gpt = json.load(f)

                col_a, col_b = st.columns(2)

                with col_a:
                    st.subheader("📊 Analiza FootyStats")

                    pot = gpt.get("potencjaly", {})
                    if pot:
                        pc1, pc2, pc3, pc4 = st.columns(4)
                        pc1.metric("BTTS%",      f"{pot.get('btts_potential', 0)}%")
                        pc2.metric("O2.5%",      f"{pot.get('o25_potential', 0)}%")
                        pc3.metric("Corn O9.5%", f"{pot.get('corners_o95_potential', 0)}%")
                        pc4.metric("Avg Pot",     pot.get("avg_potential", "—"))

                    xg = gpt.get("xg_prematch", {})
                    if xg:
                        st.write(
                            f"**xG:** "
                            f"{gpt.get('home', '?')} "
                            f"{xg.get('home', '?')} — "
                            f"{xg.get('away', '?')} "
                            f"{gpt.get('away', '?')} "
                            f"(łącznie {xg.get('total', '?')})"
                        )

                    h2h = gpt.get("h2h_stats", {})
                    if h2h and h2h.get("total_matches", 0) > 0:
                        st.write(
                            f"**H2H** ({h2h.get('total_matches')} meczów): "
                            f"BTTS {h2h.get('btts_pct')}% | "
                            f"Over2.5 {h2h.get('over25_pct')}% | "
                            f"Avg goli {h2h.get('avg_goals')}"
                        )

                    pin = gpt.get("pinnacle_odds", {})
                    if any(v for v in pin.values() if v):
                        st.divider()
                        st.write("**Pinnacle odds:**")
                        pc1, pc2, pc3 = st.columns(3)
                        pc1.metric("1",    pin.get("home", "—"))
                        pc2.metric("X",    pin.get("draw", "—"))
                        pc3.metric("2",    pin.get("away", "—"))
                        pc1b, pc2b, pc3b = st.columns(3)
                        pc1b.metric("O2.5",   pin.get("over25", "—"))
                        pc2b.metric("BTTS",   pin.get("btts_yes", "—"))
                        pc3b.metric("C O9.5", pin.get("corners_over_95", "—"))

                    gpt_tips = gpt.get("gpt_tips", [])
                    if gpt_tips:
                        st.divider()
                        st.markdown("**💡 GPT — polecane typy:**")
                        for tip in gpt_tips:
                            kurs_str = f"@ **{tip['kurs']:.2f}**" if tip.get("kurs") else ""
                            pewnosc  = tip.get("pewnosc", "")
                            badge    = "🟢" if pewnosc == "wysoka" else ("🟡" if pewnosc == "srednia" else "🔴")
                            uzas     = tip.get("uzasadnienie", "")
                            st.write(f"{badge} {tip.get('typ', '?')} {kurs_str}")
                            if uzas:
                                st.caption(f"  {uzas}")

                    st.divider()
                    with st.expander("📝 Pełna analiza (PL)"):
                        st.write(gpt.get("gpt_pl", "—"))

                with col_b:
                    st.subheader("🤖 Predykcja ML")

                    scorer_dir = {
                        "allsvenskan":   ALLSV_SCORER_DIR,
                        "eliteserien":   ELITE_SCORER_DIR,
                        "veikkausliiga": VEIKK_SCORER_DIR,
                    }.get(art_league, ALLSV_SCORER_DIR)

                    scorer_files_art = glob.glob(
                        os.path.join(scorer_dir, f"*scorer*{art_date_str}*.csv")
                    )

                    if scorer_files_art:
                        df_ml = load_csv(scorer_files_art[0])
                        if not df_ml.empty and "ID" in df_ml.columns:
                            df_match = df_ml[
                                pd.to_numeric(df_ml["ID"], errors="coerce") == mid
                            ]
                            # Pokaż tylko combined (ML), gpt_pred jest osobno w col_a
                            df_ml_only = df_match[
                                df_match.get("Model_type", pd.Series()).astype(str) == "combined"
                            ] if "Model_type" in df_match.columns else df_match

                            if not df_ml_only.empty:
                                cols_ml = [
                                    c for c in [
                                        "Model", "Typ", "Score[%]",
                                        "Kurs", "Pinnacle_odds", "EV[%]", "Stake_PLN",
                                    ]
                                    if c in df_ml_only.columns
                                ]
                                st.dataframe(
                                    df_ml_only[cols_ml],
                                    use_container_width=True,
                                    hide_index=True,
                                )

                                # Zgodność: porównaj kierunki gpt_tips z ML
                                gpt_tips_match = gpt.get("gpt_tips", [])
                                gpt_kierunki   = {t.get("kierunek", "") for t in gpt_tips_match}
                                ml_typy = (
                                    df_ml_only["Typ"].str.lower().tolist()
                                    if "Typ" in df_ml_only.columns else []
                                )
                                zgodnosc = (
                                    ("home_win" in gpt_kierunki and any("home" in t for t in ml_typy))
                                    or ("away_win" in gpt_kierunki and any("away" in t for t in ml_typy))
                                    or ("over" in gpt_kierunki and any("over" in t for t in ml_typy))
                                    or ("btts_yes" in gpt_kierunki and any("btts" in t for t in ml_typy))
                                )
                                if zgodnosc:
                                    st.success("✅ Zgodność GPT vs ML")
                                elif gpt_kierunki - {""}:
                                    st.warning("⚠️ Rozbieżność GPT vs ML")
                            else:
                                st.info("Brak predykcji ML (combined) dla tego meczu")
                    else:
                        st.info("Brak pliku scorera. Uruchom predykcje w Ustawieniach.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WYNIKI
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Rozliczenie wyników")

    col_l, col_f = st.columns([1, 2])
    res_league = col_l.selectbox(
        "Liga",
        ["allsvenskan", "eliteserien", "veikkausliiga"],
        format_func=lambda x: x.capitalize(),
        key="res_league",
    )
    scorer_files = get_scorer_files(res_league)

    if not scorer_files:
        st.info("Brak plików scorera.")
    else:
        selected = col_f.selectbox(
            "Plik scorera", scorer_files, format_func=extract_date
        )
        df = load_csv(selected)

        if df.empty:
            st.warning("Pusty plik.")
        else:
            try:
                from online_settle import settle_daily_scorer as _settle_api
                _has_api = True
            except ImportError:
                _has_api = False

            if _has_api:
                if st.button("Pobierz z API i rozlicz", key="api_settle"):
                    with st.spinner("Pobieram wyniki z API..."):
                        n, msg = _settle_api(selected)
                    if n > 0:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.info(msg)

            if "Wynik" in df.columns:
                mask_pending = df["Wynik"].isna() | (
                    df["Wynik"].astype(str).str.strip().isin(["", "nan"])
                )
            else:
                mask_pending = pd.Series([True] * len(df))

            pending = df[mask_pending]
            settled = df[~mask_pending]

            if not pending.empty:
                st.subheader(f"⏳ Oczekujące ({len(pending)})")
                for match_id in pending["ID"].dropna().unique():
                    sub = pending[pending["ID"] == match_id]
                    mecz_name = (
                        sub["Mecz"].iloc[0] if "Mecz" in sub.columns else str(match_id)
                    )
                    godz = sub["Godzina"].iloc[0] if "Godzina" in sub.columns else ""

                    with st.expander(
                        f"{godz} — {mecz_name} ({len(sub)} typów)"
                    ):
                        cols_p = [
                            c for c in [
                                "Model", "Model_type", "Typ",
                                "Score[%]", "Kurs", "Stake_PLN",
                            ]
                            if c in sub.columns
                        ]
                        st.dataframe(sub[cols_p], use_container_width=True, hide_index=True)

                        c1, c2, c3, c4 = st.columns(4)
                        hg      = c1.number_input("Gole home",     0, 20, 0, key=f"hg_{match_id}")
                        ag      = c2.number_input("Gole away",     0, 20, 0, key=f"ag_{match_id}")
                        corners = c3.number_input("Corners",       0, 30, 0, key=f"c_{match_id}")
                        cards   = c4.number_input("Kartki żółte",  0, 20, 0, key=f"k_{match_id}")

                        if st.button("✅ Rozlicz", key=f"settle_{match_id}"):
                            df = settle_match(df, match_id, hg, ag, corners, cards)
                            df.to_csv(selected, index=False, encoding="utf-8-sig")
                            st.success("Rozliczono!")
                            st.rerun()

            if not settled.empty:
                st.subheader(f"✅ Rozliczone ({len(settled)})")
                cols_s = [
                    c for c in [
                        "Godzina", "Mecz", "Model", "Model_type", "Typ",
                        "Kurs", "Stake_PLN", "Wynik", "Rezultat", "Corners", "Kartki", "Profit_PLN",
                    ]
                    if c in settled.columns
                ]
                st.dataframe(settled[cols_s], use_container_width=True, hide_index=True)

                if "Wynik" in settled.columns:
                    w    = settled["Wynik"]
                    n_won  = int(pd.to_numeric(w, errors="coerce").fillna(0).sum())
                    n_lost = len(settled) - n_won
                    profit = pd.to_numeric(
                        settled.get("Profit_PLN", pd.Series()), errors="coerce"
                    ).sum()
                    avg_kurs = pd.to_numeric(
                        settled.get("Kurs", pd.Series()), errors="coerce"
                    ).mean()
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Wyniki", f"✅{n_won} ❌{n_lost}")
                    m2.metric("Profit", f"{profit:+.2f} PLN")
                    if len(settled) > 0:
                        m3.metric("Win Rate", f"{n_won/len(settled):.0%}")
                    if pd.notna(avg_kurs):
                        m4.metric("Śr. kurs", f"{avg_kurs:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DRUŻYNY
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("""
<style>
:root {
  --bg:       #0d1117;
  --surface:  #161b22;
  --surface2: #21262d;
  --border:   #30363d;
  --accent:   #00d4aa;
  --accent2:  #58a6ff;
  --text:     #e6edf3;
  --muted:    #8b949e;
  --green:    #3fb950;
  --red:      #f85149;
  --yellow:   #d29922;
}
.league-table { background:var(--surface); border-radius:12px;
  border:1px solid var(--border); overflow:hidden;
  font-family:'DM Mono',monospace; }
.league-table-header { background:var(--surface2); padding:12px 16px;
  font-size:11px; font-weight:700; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted);
  border-bottom:1px solid var(--border); }
.lt-row { display:grid;
  grid-template-columns:28px 1fr 36px 36px 36px 36px 52px 52px 44px;
  gap:0; padding:9px 16px; border-bottom:1px solid var(--border);
  font-size:13px; color:var(--text); }
.lt-row:hover { background:var(--surface2); }
.lt-row.selected { background:rgba(0,212,170,.08);
  border-left:3px solid var(--accent); }
.lt-row.zone-cl  { border-left:3px solid var(--accent2); }
.lt-row.zone-rel { border-left:3px solid var(--red); }
.lt-pos { color:var(--muted); font-size:12px; }
.lt-name { font-weight:600; }
.lt-num  { text-align:center; color:var(--muted); }
.lt-ppg  { text-align:center; font-weight:700; color:var(--accent); }
.lt-gd.pos { color:var(--green); }
.lt-gd.neg { color:var(--red); }
.stat-card { background:var(--surface); border:1px solid var(--border);
  border-radius:10px; padding:16px; margin-bottom:12px; }
.stat-card-title { font-size:10px; font-weight:700; letter-spacing:.1em;
  text-transform:uppercase; color:var(--muted); margin-bottom:12px;
  padding-bottom:8px; border-bottom:1px solid var(--border); }
.trio-stat { display:grid; grid-template-columns:160px 1fr;
  align-items:center; gap:8px; margin-bottom:10px; }
.trio-label { font-size:12px; color:var(--muted); }
.trio-bars  { display:flex; flex-direction:column; gap:3px; }
.trio-row   { display:flex; align-items:center; gap:6px; font-size:11px; }
.trio-tag   { width:24px; text-align:center; font-size:9px; font-weight:700;
  letter-spacing:.05em; color:var(--muted); flex-shrink:0; }
.bar-bg     { flex:1; height:6px; background:var(--surface2);
  border-radius:3px; overflow:hidden; }
.bar-fill   { height:100%; border-radius:3px; transition:width .4s ease; }
.bar-overall { background:var(--accent); }
.bar-home    { background:var(--accent2); }
.bar-away    { background:var(--yellow); }
.trio-val   { width:36px; text-align:right; font-size:12px;
  font-weight:600; color:var(--text); }
.big-metrics { display:grid; grid-template-columns:repeat(4,1fr);
  gap:8px; margin-bottom:12px; }
.big-metric  { background:var(--surface2); border-radius:8px;
  padding:12px; text-align:center; }
.big-metric-val   { font-size:22px; font-weight:800; color:var(--accent);
  line-height:1; }
.big-metric-label { font-size:10px; color:var(--muted); margin-top:4px;
  text-transform:uppercase; letter-spacing:.08em; }
.form-run   { display:flex; gap:4px; flex-wrap:wrap; }
.form-badge { width:22px; height:22px; border-radius:4px; display:flex;
  align-items:center; justify-content:center; font-size:11px;
  font-weight:800; }
.form-W { background:var(--green); color:#fff; }
.form-D { background:var(--yellow); color:#fff; }
.form-L { background:var(--red); color:#fff; }
.rank-row { display:grid; grid-template-columns:24px 1fr 120px 56px;
  gap:8px; align-items:center; padding:7px 0;
  border-bottom:1px solid var(--border); font-size:13px; }
.rank-pos { color:var(--muted); font-size:11px; text-align:center; }
.rank-name { font-weight:600; }
.rank-bar-wrap { background:var(--surface2); height:8px;
  border-radius:4px; overflow:hidden; }
.rank-bar  { height:100%; border-radius:4px; }
.rank-val  { text-align:right; font-weight:700; color:var(--accent); }
.compare-row { display:grid; grid-template-columns:1fr 80px 1fr;
  gap:8px; align-items:center; padding:8px 0;
  border-bottom:1px solid var(--border); font-size:13px; }
.compare-val-a   { text-align:right; font-weight:700; }
.compare-label   { text-align:center; font-size:10px; color:var(--muted);
  text-transform:uppercase; letter-spacing:.07em; }
.compare-val-b   { text-align:left; font-weight:700; }
.compare-winner-a { color:var(--green); }
.compare-winner-b { color:var(--red); }
.compare-tie      { color:var(--muted); }
.team-header { display:flex; align-items:center; gap:16px; padding:20px;
  background:linear-gradient(135deg,var(--surface) 0%,var(--surface2) 100%);
  border:1px solid var(--border); border-radius:12px; margin-bottom:16px; }
.team-name-big  { font-size:26px; font-weight:900; color:var(--text);
  letter-spacing:-.02em; }
.team-meta      { font-size:12px; color:var(--muted); margin-top:4px; }
.team-pos-badge { background:var(--accent); color:#000; font-weight:800;
  font-size:18px; width:44px; height:44px; border-radius:8px;
  display:flex; align-items:center; justify-content:center;
  flex-shrink:0; }
</style>
""", unsafe_allow_html=True)

    # ── Wybór ligi ──────────────────────────────────────────────────────────
    _col_sv, _col_no, _col_fi = st.columns(3)
    _cur_liga = st.session_state.get("team_league", "allsvenskan")
    with _col_sv:
        if st.button("🇸🇪 Allsvenskan", use_container_width=True,
                     type="primary" if _cur_liga == "allsvenskan" else "secondary",
                     key="tl_sv"):
            st.session_state["team_league"] = "allsvenskan"
            st.rerun()
    with _col_no:
        if st.button("🇳🇴 Eliteserien", use_container_width=True,
                     type="primary" if _cur_liga == "eliteserien" else "secondary",
                     key="tl_no"):
            st.session_state["team_league"] = "eliteserien"
            st.rerun()
    with _col_fi:
        if st.button("🇫🇮 Veikkausliiga", use_container_width=True,
                     type="primary" if _cur_liga == "veikkausliiga" else "secondary",
                     key="tl_fi"):
            st.session_state["team_league"] = "veikkausliiga"
            st.rerun()

    liga = st.session_state.get("team_league", "allsvenskan")
    _df_teams, _df_matches = load_league_data(liga)

    if _df_teams.empty:
        st.warning("Brak danych. Pobierz w Ustawieniach: Pobierz dane dzienne lub Tygodniowy retren.")
        st.stop()

    # ── Layout dwukolumnowy ──────────────────────────────────────────────────
    _col_left, _col_right = st.columns([1, 2.4], gap="medium")

    # ════════════════════════════════════════════════════════════════════════
    # LEWA — tabela ligowa
    # ════════════════════════════════════════════════════════════════════════
    with _col_left:
        _liga_names = {
            "allsvenskan":   "🇸🇪 Allsvenskan 2026",
            "eliteserien":   "🇳🇴 Eliteserien 2026",
            "veikkausliiga": "🇫🇮 Veikkausliiga 2026",
        }
        st.markdown(
            f'<div class="league-table-header">{_liga_names[liga]}</div>',
            unsafe_allow_html=True)

        _pos_col = 'leaguePosition_overall'
        if _pos_col in _df_teams.columns:
            _df_sorted = _df_teams.sort_values(_pos_col).reset_index(drop=True)
        else:
            _df_sorted = _df_teams.reset_index(drop=True)

        st.markdown("""
<div class="league-table">
<div class="lt-row" style="color:var(--muted);font-size:10px;font-weight:700;
letter-spacing:.08em;text-transform:uppercase;border-bottom:2px solid var(--border);
background:var(--surface2)">
  <span>#</span><span>Drużyna</span>
  <span class="lt-num">M</span><span class="lt-num">W</span>
  <span class="lt-num">D</span><span class="lt-num">L</span>
  <span class="lt-num">GD</span>
  <span class="lt-ppg">PPG</span><span class="lt-num">Pkt</span>
</div>""", unsafe_allow_html=True)

        _selected_team = st.session_state.get("selected_team_id", None)
        _n_teams = len(_df_sorted)
        for _, _row in _df_sorted.iterrows():
            _pos  = int(safe_float(_row.get(_pos_col, 0)))
            _name = _row.get('team_name', _row.get('name', '?'))
            _tid  = str(_row.get('team_id', _row.get('id', '')))
            _mp   = int(safe_float(_row.get('seasonMatchesPlayed_overall', 0)))
            _w    = int(safe_float(_row.get('seasonWinsNum_overall', 0)))
            _d    = int(safe_float(_row.get('seasonDrawsNum_overall', 0)))
            _l    = int(safe_float(_row.get('seasonLossesNum_overall', 0)))
            _gd   = int(safe_float(_row.get('seasonGoalDifference_overall', 0)))
            _ppg  = safe_float(_row.get('seasonPPG_overall', 0))
            _pts  = _w * 3 + _d
            _gd_str = f"+{_gd}" if _gd > 0 else str(_gd)
            _gd_cls = "pos" if _gd > 0 else ("neg" if _gd < 0 else "")
            _zone   = ""
            if _pos <= 2:
                _zone = "zone-cl"
            elif _pos >= _n_teams - 1:
                _zone = "zone-rel"
            _sel_cls = "selected" if _tid == str(_selected_team) else ""
            st.markdown(f"""
<div class="lt-row {_zone} {_sel_cls}">
  <span class="lt-pos">{_pos}</span>
  <span class="lt-name">{_name}</span>
  <span class="lt-num">{_mp}</span>
  <span class="lt-num">{_w}</span>
  <span class="lt-num">{_d}</span>
  <span class="lt-num">{_l}</span>
  <span class="lt-num lt-gd {_gd_cls}">{_gd_str}</span>
  <span class="lt-ppg">{_ppg:.2f}</span>
  <span class="lt-num">{_pts}</span>
</div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""
<div style="font-size:10px;color:var(--muted);margin-top:8px;line-height:1.8">
<span style="color:var(--accent2)">■</span> Europejskie puchary&nbsp;&nbsp;
<span style="color:var(--red)">■</span> Strefa spadkowa
</div>""", unsafe_allow_html=True)

        st.divider()
        _name_col = 'team_name' if 'team_name' in _df_sorted.columns else 'name'
        _id_col   = 'team_id'   if 'team_id'   in _df_sorted.columns else 'id'
        _team_names = _df_sorted[_name_col].tolist()
        _team_ids   = _df_sorted[_id_col].tolist()

        _sel_name = st.selectbox("Wybierz drużynę →", _team_names,
                                 key="team_select_left")
        _sel_idx = _team_names.index(_sel_name)
        st.session_state["selected_team_id"] = _team_ids[_sel_idx]

    # ════════════════════════════════════════════════════════════════════════
    # PRAWA — widok interaktywny
    # ════════════════════════════════════════════════════════════════════════
    with _col_right:
        _view_mode = st.radio(
            "Widok",
            ["📋 Profil drużyny", "⚔️ Porównanie drużyn", "🏆 Rankingi"],
            horizontal=True,
            key="team_view_mode",
            label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)

        # ── PROFIL DRUŻYNY ──────────────────────────────────────────────────
        if _view_mode == "📋 Profil drużyny":
            _td = get_team_row(_df_teams,
                               st.session_state.get("selected_team_id",
                                                    _team_ids[0]))
            if not _td:
                st.info("Wybierz drużynę z listy.")
            else:
                _t_name = _td.get('team_name', _td.get('name', '?'))
                _t_pos  = int(safe_float(_td.get(_pos_col, 0)))
                _t_mp   = int(safe_float(_td.get('seasonMatchesPlayed_overall', 0)))
                _ppg    = safe_float(_td.get('seasonPPG_overall'))
                _scored = safe_float(_td.get('seasonScoredAVG_overall'))
                _conc   = safe_float(_td.get('seasonConcededAVG_overall'))
                _xg_for = safe_float(_td.get('xg_for_avg_overall'))
                _xg_ag  = safe_float(_td.get('xg_against_avg_overall'))
                _btts   = safe_float(_td.get('seasonBTTSPercentage_overall'))
                _o25    = safe_float(_td.get('seasonOver25Percentage_overall'))
                _c_all  = safe_float(_td.get('cornersAVG_overall'))
                _c_ag   = safe_float(_td.get('cornersAgainstAVG_overall'))
                _ct     = safe_float(_td.get('cornersTotalAVG_overall',
                                             _c_all + _c_ag))
                _ppg_h  = safe_float(_td.get('seasonPPG_home'))
                _ppg_a  = safe_float(_td.get('seasonPPG_away'))
                _sc_h   = safe_float(_td.get('seasonScoredAVG_home'))
                _sc_a   = safe_float(_td.get('seasonScoredAVG_away'))
                _cn_h   = safe_float(_td.get('seasonConcededAVG_home'))
                _cn_a   = safe_float(_td.get('seasonConcededAVG_away'))
                _o25_h  = safe_float(_td.get('seasonOver25Percentage_home'))
                _o25_a  = safe_float(_td.get('seasonOver25Percentage_away'))
                _bt_h   = safe_float(_td.get('seasonBTTSPercentage_home'))
                _bt_a   = safe_float(_td.get('seasonBTTSPercentage_away'))
                _c_h    = safe_float(_td.get('cornersAVG_home'))
                _c_a2   = safe_float(_td.get('cornersAVG_away'))
                _o95    = safe_float(_td.get('over95CornersPercentage_overall'))
                _o95_h  = safe_float(_td.get('over95CornersPercentage_home'))
                _o95_a  = safe_float(_td.get('over95CornersPercentage_away'))
                _xgf_h  = safe_float(_td.get('xg_for_avg_home'))
                _xgf_a  = safe_float(_td.get('xg_for_avg_away'))
                _xga_h  = safe_float(_td.get('xg_against_avg_home'))
                _xga_a  = safe_float(_td.get('xg_against_avg_away'))
                _k_all  = safe_float(_td.get('cardsAVG_overall'))
                _k_h    = safe_float(_td.get('cardsAVG_home'))
                _k_a    = safe_float(_td.get('cardsAVG_away'))
                _win_pct = safe_float(_td.get('winPercentage_overall'))

                st.markdown(f"""
<div class="team-header">
  <div class="team-pos-badge">{_t_pos}</div>
  <div>
    <div class="team-name-big">{_t_name}</div>
    <div class="team-meta">{_t_mp} meczów · Win% {_win_pct:.0f}%</div>
  </div>
</div>""", unsafe_allow_html=True)

                st.markdown(f"""
<div class="big-metrics">
  <div class="big-metric">
    <div class="big-metric-val">{_ppg:.2f}</div>
    <div class="big-metric-label">PPG</div>
  </div>
  <div class="big-metric">
    <div class="big-metric-val" style="color:var(--green)">{_scored:.2f}</div>
    <div class="big-metric-label">Śr. gole</div>
  </div>
  <div class="big-metric">
    <div class="big-metric-val" style="color:var(--red)">{_conc:.2f}</div>
    <div class="big-metric-label">Śr. stracone</div>
  </div>
  <div class="big-metric">
    <div class="big-metric-val">{_xg_for:.2f}</div>
    <div class="big-metric-label">xG For</div>
  </div>
</div>""", unsafe_allow_html=True)

                _sc1, _sc2 = st.columns(2)
                with _sc1:
                    _html_l = (
                        '<div class="stat-card">'
                        '<div class="stat-card-title">⚽ Gole i wyniki</div>'
                        + trio_bar_html("PPG", _ppg, _ppg_h, _ppg_a, max_val=3.0)
                        + trio_bar_html("Strzelone / mecz", _scored, _sc_h, _sc_a, max_val=3.5)
                        + trio_bar_html("Stracone / mecz", _conc, _cn_h, _cn_a, max_val=3.5)
                        + f'<div class="trio-stat"><span class="trio-label">Win %</span>'
                          f'<span style="font-weight:700;color:var(--accent)">{_win_pct:.0f}%</span></div>'
                        + '</div>'
                        '<div class="stat-card">'
                        '<div class="stat-card-title">📊 Over / BTTS</div>'
                        + trio_bar_html("Over 2.5 %", _o25, _o25_h, _o25_a, max_val=100, pct=True)
                        + trio_bar_html("BTTS Yes %", _btts, _bt_h, _bt_a, max_val=100, pct=True)
                        + '</div>'
                    )
                    st.markdown(_html_l, unsafe_allow_html=True)

                with _sc2:
                    _html_r = (
                        '<div class="stat-card">'
                        '<div class="stat-card-title">🚩 Corners</div>'
                        + trio_bar_html("Corners zdobyte / mecz", _c_all, _c_h, _c_a2, max_val=10)
                        + trio_bar_html("Corners oddane / mecz", _c_ag, None, None, max_val=10)
                        + f'<div class="trio-stat"><span class="trio-label">Łącznie w meczu (avg)</span>'
                          f'<span style="font-weight:700;color:var(--accent)">{_ct:.1f}</span></div>'
                        + trio_bar_html("Over 9.5 %", _o95, _o95_h, _o95_a, max_val=100, pct=True)
                        + '</div>'
                        '<div class="stat-card">'
                        '<div class="stat-card-title">📐 xG</div>'
                        + trio_bar_html("xG tworzone", _xg_for, _xgf_h, _xgf_a, max_val=2.5)
                        + trio_bar_html("xG dopuszczane", _xg_ag, _xga_h, _xga_a, max_val=2.5)
                        + '</div>'
                        '<div class="stat-card">'
                        '<div class="stat-card-title">🟨 Kartki</div>'
                        + trio_bar_html("Kartki drużyny / mecz", _k_all, _k_h, _k_a, max_val=5)
                        + f'<div class="trio-stat"><span class="trio-label">Łącznie w meczu (est.)</span>'
                          f'<span style="font-weight:700">{_k_all * 2:.1f}</span></div>'
                        + '</div>'
                    )
                    st.markdown(_html_r, unsafe_allow_html=True)

        # ── PORÓWNANIE DRUŻYN ───────────────────────────────────────────────
        elif _view_mode == "⚔️ Porównanie drużyn":
            _all_names = _df_teams[_name_col].tolist()
            _all_ids   = _df_teams[_id_col].tolist()
            _cc1, _cc2 = st.columns(2)
            with _cc1:
                _name_a = st.selectbox("Drużyna A", _all_names, index=0,
                                       key="cmp_team_a")
            with _cc2:
                _name_b = st.selectbox("Drużyna B", _all_names,
                                       index=min(1, len(_all_names) - 1),
                                       key="cmp_team_b")
            _td_a = get_team_row(_df_teams, _all_ids[_all_names.index(_name_a)])
            _td_b = get_team_row(_df_teams, _all_ids[_all_names.index(_name_b)])
            _ppg_a2 = safe_float(_td_a.get('seasonPPG_overall'))
            _ppg_b2 = safe_float(_td_b.get('seasonPPG_overall'))
            _pos_a2 = int(safe_float(_td_a.get(_pos_col, 0)))
            _pos_b2 = int(safe_float(_td_b.get(_pos_col, 0)))

            st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 60px 1fr;gap:8px;
align-items:center;background:var(--surface);border:1px solid var(--border);
border-radius:12px;padding:20px;margin-bottom:16px">
  <div style="text-align:right">
    <div style="font-size:22px;font-weight:900">{_name_a}</div>
    <div style="color:var(--muted);font-size:12px">#{_pos_a2} · {_ppg_a2:.2f} PPG</div>
  </div>
  <div style="text-align:center;font-size:14px;font-weight:700;color:var(--muted)">VS</div>
  <div>
    <div style="font-size:22px;font-weight:900">{_name_b}</div>
    <div style="color:var(--muted);font-size:12px">#{_pos_b2} · {_ppg_b2:.2f} PPG</div>
  </div>
</div>""", unsafe_allow_html=True)

            _cmp_cats = st.multiselect(
                "Kategorie",
                ["⚽ Gole", "🔄 BTTS", "🚩 Corners", "📐 xG", "🟨 Kartki"],
                default=["⚽ Gole", "🔄 BTTS", "🚩 Corners", "📐 xG"],
                key="cmp_cats")

            def _cmp_row(label, key_a, td_a, td_b, key_b=None,
                         higher=True, pct=False):
                key_b = key_b or key_a
                va = safe_float(td_a.get(key_a))
                vb = safe_float(td_b.get(key_b))
                fma = f"{va:.0f}%" if pct else f"{va:.2f}"
                fmb = f"{vb:.0f}%" if pct else f"{vb:.2f}"
                if higher:
                    cls_a = "compare-winner-a" if va > vb else ("compare-winner-b" if va < vb else "compare-tie")
                    cls_b = "compare-winner-a" if vb > va else ("compare-winner-b" if vb < va else "compare-tie")
                else:
                    cls_a = "compare-winner-a" if va < vb else ("compare-winner-b" if va > vb else "compare-tie")
                    cls_b = "compare-winner-a" if vb < va else ("compare-winner-b" if vb > va else "compare-tie")
                return (f'<div class="compare-row">'
                        f'<span class="compare-val-a {cls_a}">{fma}</span>'
                        f'<span class="compare-label">{label}</span>'
                        f'<span class="compare-val-b {cls_b}">{fmb}</span>'
                        f'</div>')

            _html_cmp = '<div class="stat-card">'
            _html_cmp += (
                '<div class="compare-row" style="font-size:10px;font-weight:700;'
                'color:var(--muted);letter-spacing:.08em;text-transform:uppercase">'
                f'<span style="text-align:right">{_name_a}</span>'
                f'<span class="compare-label">Stat</span>'
                f'<span>{_name_b}</span></div>')

            if "⚽ Gole" in _cmp_cats:
                _html_cmp += '<div class="stat-card-title" style="margin-top:12px">⚽ Gole</div>'
                for _lbl, _key in [
                    ("PPG overall", 'seasonPPG_overall'),
                    ("PPG dom", 'seasonPPG_home'),
                    ("PPG wyjazd", 'seasonPPG_away'),
                    ("Strzelone/mecz", 'seasonScoredAVG_overall'),
                    ("Over 2.5 %", 'seasonOver25Percentage_overall'),
                ]:
                    _html_cmp += _cmp_row(_lbl, _key, _td_a, _td_b,
                                          pct=("%" in _lbl))
                _html_cmp += _cmp_row("Stracone/mecz", 'seasonConcededAVG_overall',
                                      _td_a, _td_b, higher=False)

            if "🔄 BTTS" in _cmp_cats:
                _html_cmp += '<div class="stat-card-title" style="margin-top:12px">🔄 BTTS</div>'
                for _lbl, _key in [
                    ("BTTS Yes %", 'seasonBTTSPercentage_overall'),
                    ("BTTS dom %", 'seasonBTTSPercentage_home'),
                    ("BTTS wyjazd %", 'seasonBTTSPercentage_away'),
                ]:
                    _html_cmp += _cmp_row(_lbl, _key, _td_a, _td_b, pct=True)

            if "🚩 Corners" in _cmp_cats:
                _html_cmp += '<div class="stat-card-title" style="margin-top:12px">🚩 Corners</div>'
                for _lbl, _key in [
                    ("Corners zdobyte avg", 'cornersAVG_overall'),
                    ("Corners dom", 'cornersAVG_home'),
                    ("Corners wyjazd", 'cornersAVG_away'),
                    ("Łącznie avg", 'cornersTotalAVG_overall'),
                    ("Over 9.5 %", 'over95CornersPercentage_overall'),
                ]:
                    _html_cmp += _cmp_row(_lbl, _key, _td_a, _td_b,
                                          pct=("%" in _lbl))

            if "📐 xG" in _cmp_cats:
                _html_cmp += '<div class="stat-card-title" style="margin-top:12px">📐 xG</div>'
                for _lbl, _key in [
                    ("xG tworzone", 'xg_for_avg_overall'),
                    ("xG dom", 'xg_for_avg_home'),
                    ("xG wyjazd", 'xg_for_avg_away'),
                ]:
                    _html_cmp += _cmp_row(_lbl, _key, _td_a, _td_b)
                _html_cmp += _cmp_row("xG dopuszczane", 'xg_against_avg_overall',
                                      _td_a, _td_b, higher=False)

            if "🟨 Kartki" in _cmp_cats:
                _html_cmp += '<div class="stat-card-title" style="margin-top:12px">🟨 Kartki</div>'
                for _lbl, _key in [
                    ("Kartki dom", 'cardsAVG_home'),
                    ("Kartki wyjazd", 'cardsAVG_away'),
                ]:
                    _html_cmp += _cmp_row(_lbl, _key, _td_a, _td_b, higher=False)
                _html_cmp += _cmp_row("Kartki avg", 'cardsAVG_overall',
                                      _td_a, _td_b, higher=False)

            _html_cmp += '</div>'
            st.markdown(_html_cmp, unsafe_allow_html=True)

        # ── RANKINGI ────────────────────────────────────────────────────────
        elif _view_mode == "🏆 Rankingi":
            _rank_stat = st.selectbox(
                "Rankinguj wg statystyki",
                ["PPG overall", "PPG dom", "PPG wyjazd",
                 "Gole strzelone/mecz",
                 "Gole stracone/mecz (mniej = lepiej)",
                 "Bilans goli",
                 "Over 2.5 %", "BTTS Yes %",
                 "Corners zdobyte avg", "Corners łącznie avg",
                 "Over 9.5 Corners %",
                 "xG tworzone",
                 "xG dopuszczane (mniej = lepiej)",
                 "Kartki avg (mniej = lepiej)",
                 "Win %"],
                key="rank_stat_sel")

            _stat_map = {
                "PPG overall":                        ('seasonPPG_overall', True),
                "PPG dom":                            ('seasonPPG_home', True),
                "PPG wyjazd":                         ('seasonPPG_away', True),
                "Gole strzelone/mecz":                ('seasonScoredAVG_overall', True),
                "Gole stracone/mecz (mniej = lepiej)":('seasonConcededAVG_overall', False),
                "Bilans goli":                        ('seasonGoalDifference_overall', True),
                "Over 2.5 %":                         ('seasonOver25Percentage_overall', True),
                "BTTS Yes %":                         ('seasonBTTSPercentage_overall', True),
                "Corners zdobyte avg":                ('cornersAVG_overall', True),
                "Corners łącznie avg":                ('cornersTotalAVG_overall', True),
                "Over 9.5 Corners %":                 ('over95CornersPercentage_overall', True),
                "xG tworzone":                        ('xg_for_avg_overall', True),
                "xG dopuszczane (mniej = lepiej)":    ('xg_against_avg_overall', False),
                "Kartki avg (mniej = lepiej)":        ('cardsAVG_overall', False),
                "Win %":                              ('winPercentage_overall', True),
            }

            _col_key, _higher = _stat_map[_rank_stat]
            _df_rank = _df_teams[[_name_col, _col_key]].copy()
            _df_rank[_col_key] = pd.to_numeric(_df_rank[_col_key], errors='coerce')
            _df_rank = (_df_rank.dropna()
                        .sort_values(_col_key, ascending=not _higher)
                        .reset_index(drop=True))
            _max_val = _df_rank[_col_key].max() or 1
            _bar_color = "var(--accent)" if _higher else "var(--red)"
            _is_pct = "%" in _rank_stat

            _html_rank = (f'<div class="stat-card">'
                          f'<div class="stat-card-title">🏆 Ranking: {_rank_stat}</div>')
            for _i, _rrow in _df_rank.iterrows():
                _rval  = _rrow[_col_key]
                _rpctw = (_rval / _max_val) * 100
                _medal = {0: "🥇", 1: "🥈", 2: "🥉"}.get(_i, _i + 1)
                _rfmt  = f"{_rval:.0f}%" if _is_pct else f"{_rval:.2f}"
                _html_rank += f"""
<div class="rank-row">
  <span class="rank-pos">{_medal}</span>
  <span class="rank-name">{_rrow[_name_col]}</span>
  <div class="rank-bar-wrap">
    <div class="rank-bar" style="width:{_rpctw:.1f}%;background:{_bar_color}"></div>
  </div>
  <span class="rank-val">{_rfmt}</span>
</div>"""
            _html_rank += '</div>'
            st.markdown(_html_rank, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — STATYSTYKI
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("📊 Statystyki i ROI")

    # ── Wczytaj dane ─────────────────────────────
    df_all = get_all_scorer_files()

    if df_all.empty or "Wynik" not in df_all.columns:
        st.info("Brak rozliczonych zakładów. "
                "Rozlicz wyniki w zakładce Wyniki.")
        st.stop()

    df_settled = df_all[
        pd.to_numeric(df_all["Wynik"], errors="coerce").isin([0, 1])
    ].copy()

    if df_settled.empty:
        st.info("Brak rozliczonych zakładów.")
        st.stop()

    # ── KONFIGURACJA STAWKI ───────────────────────
    st.subheader("⚙️ Konfiguracja stawki")
    cfg1, cfg2, cfg3 = st.columns(3)

    bankroll = cfg1.number_input(
        "Bankroll (PLN)",
        min_value=100.0,
        max_value=1_000_000.0,
        value=1000.0,
        step=100.0,
        key="stats_bankroll")

    stake = cfg2.number_input(
        "Stawka (PLN)",
        min_value=1.0,
        max_value=bankroll,
        value=min(100.0, bankroll),
        step=10.0,
        key="stats_stake")

    pct = (stake / bankroll * 100) if bankroll else 0
    cfg3.metric("% bankrollu", f"{pct:.1f}%")

    st.caption(
        f"Wszystkie zakłady rozliczane flat "
        f"{stake:.0f} PLN (nadpisuje oryginalne stawki)")

    st.divider()

    # ── Normalizuj Typ ────────────────────────────
    if "Typ" in df_settled.columns and "Mecz" in df_settled.columns:
        df_settled["Typ_norm"] = df_settled.apply(
            lambda r: normalize_typ(str(r["Typ"]), str(r.get("Mecz", ""))),
            axis=1)
    else:
        df_settled["Typ_norm"] = df_settled.get("Typ", "")

    df_flat = apply_flat_stake(df_settled, stake)

    # ── BLOK 1 — KPI OGÓLNE ──────────────────────
    wynik_all  = pd.to_numeric(df_flat["Wynik"],      errors="coerce")
    kurs_all   = pd.to_numeric(df_flat["Kurs"],       errors="coerce")
    profit_all = pd.to_numeric(df_flat["Profit_PLN"], errors="coerce")
    n_all      = len(df_flat)
    n_won_all  = int(wynik_all.fillna(0).sum())
    wr_all     = n_won_all / n_all if n_all else 0
    tot_stake  = n_all * stake
    tot_profit = profit_all.fillna(0).sum()
    roi_all    = (tot_profit / tot_stake * 100) if tot_stake else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Zakłady",  n_all)
    k2.metric("Wygrane",  n_won_all)
    k3.metric("Win Rate", f"{wr_all:.1%}")
    k4.metric("ROI",      f"{roi_all:+.1f}%")
    k5.metric("Profit",   f"{tot_profit:+.1f} PLN")

    st.divider()

    # ── BLOK 2 — PER LIGA ────────────────────────
    st.subheader("🏆 Per liga")

    liga_col = next(
        (c for c in ["league", "_liga", "Liga"] if c in df_flat.columns), None)

    if liga_col:
        df_liga_stats = calc_stats_table(df_flat, liga_col, stake)
        df_liga_stats = df_liga_stats.rename(columns={liga_col: "Liga"})
        st.dataframe(df_liga_stats, use_container_width=True, hide_index=True)
    else:
        st.info("Brak kolumny liga.")

    st.divider()

    # ── BLOK 3 — ML vs GPT ───────────────────────
    st.subheader("🤖 ML (combined) vs GPT FootyStats")

    if "Model_type" in df_flat.columns:
        df_mt = calc_stats_table(df_flat, "Model_type", stake)
        df_mt = df_mt.rename(columns={"Model_type": "Model"})
        st.dataframe(df_mt, use_container_width=True, hide_index=True)

        df_mt_raw = []
        for mt in df_flat["Model_type"].dropna().unique():
            sub = apply_flat_stake(df_flat[df_flat["Model_type"] == mt], stake)
            profit = pd.to_numeric(sub["Profit_PLN"], errors="coerce").fillna(0).sum()
            ts = len(sub) * stake
            roi = (profit / ts * 100) if ts else 0
            df_mt_raw.append({"Model": mt, "ROI": roi})
        if df_mt_raw:
            chart_mt = alt.Chart(
                pd.DataFrame(df_mt_raw)
            ).mark_bar().encode(
                x=alt.X("Model:N"),
                y=alt.Y("ROI:Q", title="ROI %"),
                color=alt.condition(
                    alt.datum.ROI >= 0,
                    alt.value("#2ecc71"),
                    alt.value("#e74c3c")),
                tooltip=["Model", "ROI"]
            ).properties(height=180)
            st.altair_chart(chart_mt, use_container_width=True)
    else:
        st.info("Brak kolumny Model_type.")

    st.divider()

    # ── BLOK 4 — PER TYP (znormalizowany) ────────
    st.subheader("🎯 Per typ zakładu (znormalizowany)")

    df_typ_stats = calc_stats_table(df_flat, "Typ_norm", stake)
    df_typ_stats = df_typ_stats.rename(columns={"Typ_norm": "Typ"})
    st.dataframe(df_typ_stats, use_container_width=True, hide_index=True)

    st.divider()

    # ── BLOK 5 — PER PRZEDZIAŁ KURSOWY ───────────
    st.subheader("💰 Per przedział kursowy")

    kurs_num = pd.to_numeric(df_flat["Kurs"], errors="coerce")
    bins   = [1.0, 1.30, 1.50, 1.75, 2.00, 99.0]
    labels = ["1.01–1.30", "1.31–1.50", "1.51–1.75", "1.76–2.00", "2.00+"]
    df_flat["_kurs_bin"] = pd.cut(kurs_num, bins=bins, labels=labels, right=True)

    df_bin_rows = []
    for label in labels:
        sub = apply_flat_stake(df_flat[df_flat["_kurs_bin"] == label], stake)
        if sub.empty:
            continue
        wynik  = pd.to_numeric(sub["Wynik"],      errors="coerce")
        profit = pd.to_numeric(sub["Profit_PLN"], errors="coerce")
        n   = len(sub)
        nw  = int(wynik.fillna(0).sum())
        wr  = nw / n if n else 0
        ts  = n * stake
        tp  = profit.fillna(0).sum()
        roi = (tp / ts * 100) if ts else 0
        avg_k = pd.to_numeric(sub["Kurs"], errors="coerce").mean()
        df_bin_rows.append({
            "Przedział": label,
            "N":         n,
            "Wygrane":   nw,
            "Win Rate":  f"{wr:.0%}",
            "Śr. kurs":  f"{avg_k:.2f}",
            "ROI":       f"{roi:+.1f}%",
            "Profit":    f"{tp:+.1f} PLN",
            "_roi":      roi,
        })

    if df_bin_rows:
        df_bins = pd.DataFrame(df_bin_rows)
        st.dataframe(
            df_bins.drop(columns=["_roi"]),
            use_container_width=True,
            hide_index=True)

        chart_bin = alt.Chart(df_bins).mark_bar(
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3
        ).encode(
            x=alt.X("Przedział:N", sort=labels, title="Przedział kursowy"),
            y=alt.Y("_roi:Q", title="ROI %"),
            color=alt.condition(
                alt.datum._roi >= 0,
                alt.value("#2ecc71"),
                alt.value("#e74c3c")),
            tooltip=["Przedział", "N", "_roi"]
        ).properties(height=200)
        st.altair_chart(chart_bin, use_container_width=True)

    st.divider()

    # ── BLOK 6 — TRAFNOŚĆ GPT PER TYP ────────────
    st.subheader("📰 Trafność GPT FootyStats per typ")

    if "Model_type" in df_flat.columns:
        df_gpt = apply_flat_stake(
            df_flat[df_flat["Model_type"] == "gpt_pred"].copy(), stake)

        if df_gpt.empty:
            st.info("Brak zakładów GPT.")
        else:
            df_gpt_rows = []
            for typ in df_gpt["Typ_norm"].dropna().unique():
                sub    = df_gpt[df_gpt["Typ_norm"] == typ]
                wynik  = pd.to_numeric(sub["Wynik"],      errors="coerce")
                kurs   = pd.to_numeric(sub["Kurs"],       errors="coerce")
                profit = pd.to_numeric(sub["Profit_PLN"], errors="coerce")
                n   = len(sub)
                nw  = int(wynik.fillna(0).sum())
                acc = nw / n if n else 0
                ts  = n * stake
                tp  = profit.fillna(0).sum()
                roi = (tp / ts * 100) if ts else 0
                avg_k = kurs.mean()
                df_gpt_rows.append({
                    "Typ":      typ,
                    "N":        n,
                    "Trafień":  nw,
                    "Accuracy": f"{acc:.0%}",
                    "Śr. kurs": f"{avg_k:.2f}",
                    "ROI":      f"{roi:+.1f}%",
                    "_roi":     roi,
                })

            df_gpt_tab = (pd.DataFrame(df_gpt_rows)
                .sort_values("_roi", ascending=False)
                .drop(columns=["_roi"]))
            st.dataframe(df_gpt_tab, use_container_width=True, hide_index=True)
    else:
        st.info("Brak kolumny Model_type.")

    st.divider()

    # ── BLOK 7 — ROZKŁAD MECZOWY ──────────────────
    st.subheader("⚽ Rozkład wyników meczowych")

    def parse_rezultat(r):
        try:
            parts = str(r).split(":")
            return int(parts[0]), int(parts[1])
        except Exception:
            return None, None

    if "Rezultat" in df_settled.columns:
        df_res = df_settled.copy()
        df_res[["_hg", "_ag"]] = df_res["Rezultat"].apply(
            lambda r: pd.Series(parse_rezultat(r)))
        df_res["_total_goals"] = (
            df_res["_hg"].fillna(0) + df_res["_ag"].fillna(0))
        df_res["_btts_fact"] = (
            (df_res["_hg"] > 0) & (df_res["_ag"] > 0)).astype(int)
        df_res["_over25_fact"] = (df_res["_total_goals"] > 2.5).astype(int)
        corners_num = pd.to_numeric(
            df_res.get("Corners", pd.Series(dtype=float)), errors="coerce")
        df_res["_over95c_fact"] = (corners_num > 9.5).astype(int)

        df_matches = df_res.drop_duplicates(subset=["ID"])

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Meczów",       len(df_matches))
        mc2.metric("Śr. gole",     f"{df_matches['_total_goals'].mean():.2f}")
        mc3.metric("BTTS %",       f"{df_matches['_btts_fact'].mean():.0%}")
        mc4.metric("Over 2.5 %",   f"{df_matches['_over25_fact'].mean():.0%}")
        mc5.metric("Corners O9.5 %", f"{df_matches['_over95c_fact'].mean():.0%}")

        corners_mean = corners_num.mean()
        kartki_mean  = pd.to_numeric(
            df_res.get("Kartki", pd.Series(dtype=float)), errors="coerce").mean()
        if pd.notna(corners_mean) or pd.notna(kartki_mean):
            mc6, mc7 = st.columns(2)
            if pd.notna(corners_mean):
                mc6.metric("Śr. corners", f"{corners_mean:.1f}")
            if pd.notna(kartki_mean):
                mc7.metric("Śr. kartki",  f"{kartki_mean:.1f}")

        if liga_col and liga_col in df_res.columns:
            st.write("**Per liga:**")
            liga_res_rows = []
            for liga in df_res[liga_col].dropna().unique():
                sub = df_res[df_res[liga_col] == liga].drop_duplicates(subset=["ID"])
                if sub.empty:
                    continue
                liga_res_rows.append({
                    "Liga":          liga,
                    "Meczów":        len(sub),
                    "Śr. gole":      f"{sub['_total_goals'].mean():.2f}",
                    "BTTS %":        f"{sub['_btts_fact'].mean():.0%}",
                    "Over 2.5 %":    f"{sub['_over25_fact'].mean():.0%}",
                    "O9.5 Corners %": f"{sub['_over95c_fact'].mean():.0%}",
                })
            if liga_res_rows:
                st.dataframe(
                    pd.DataFrame(liga_res_rows),
                    use_container_width=True,
                    hide_index=True)
    else:
        st.info("Brak kolumny Rezultat — "
                "rozlicz wyniki aby zobaczyć statystyki.")

    st.divider()

    # ── BLOK 8 — EQUITY CURVE ────────────────────
    st.subheader("📈 Equity curve i dzienny profit")

    if "Data" in df_flat.columns:
        df_eq = df_flat.copy()
        df_eq["Profit_PLN"] = pd.to_numeric(
            df_eq["Profit_PLN"], errors="coerce").fillna(0)
        df_eq = df_eq.sort_values("Data")
        df_eq["cum_profit"]   = df_eq["Profit_PLN"].cumsum()
        df_eq["bankroll_val"] = bankroll + df_eq["cum_profit"]

        df_eq["peak"]     = df_eq["bankroll_val"].cummax()
        df_eq["drawdown"] = df_eq["bankroll_val"] - df_eq["peak"]
        max_dd = df_eq["drawdown"].min()

        dd1, dd2 = st.columns(2)
        dd1.metric("Końcowy bankroll",
                   f"{bankroll + tot_profit:.1f} PLN",
                   delta=f"{tot_profit:+.1f} PLN")
        dd2.metric("Max drawdown", f"{max_dd:.1f} PLN")

        chart_eq = alt.Chart(df_eq).mark_line(
            point=True, color="#3498db"
        ).encode(
            x=alt.X("Data:T", title="Data"),
            y=alt.Y("bankroll_val:Q", title="Bankroll (PLN)"),
            tooltip=["Data", "bankroll_val", "cum_profit"]
        ).properties(height=250, title="Equity Curve")
        st.altair_chart(chart_eq, use_container_width=True)

        df_day = (df_eq.groupby("Data")["Profit_PLN"].sum().reset_index())
        df_day.columns = ["Data", "Profit"]
        bar_day = alt.Chart(df_day).mark_bar(
            cornerRadiusTopLeft=3,
            cornerRadiusTopRight=3
        ).encode(
            x=alt.X("Data:T", title="Data"),
            y=alt.Y("Profit:Q", title="Profit dzienny (PLN)"),
            color=alt.condition(
                alt.datum.Profit >= 0,
                alt.value("#2ecc71"),
                alt.value("#e74c3c")),
            tooltip=["Data", "Profit"]
        ).properties(height=180, title="Dzienny Profit")
        st.altair_chart(bar_day, use_container_width=True)
    else:
        st.info("Brak kolumny Data.")

    st.divider()

    # ── PEŁNA TABELA ──────────────────────────────
    st.subheader("📋 Wszystkie zakłady")

    f1, f2, f3 = st.columns(3)
    filter_liga = f1.multiselect(
        "Liga",
        df_flat[liga_col].dropna().unique().tolist() if liga_col else [],
        default=[],
        key="stats_filter_liga")
    filter_mt = f2.multiselect(
        "Model",
        df_flat["Model_type"].dropna().unique().tolist()
        if "Model_type" in df_flat.columns else [],
        default=[],
        key="stats_filter_mt")
    filter_typ = f3.multiselect(
        "Typ (znorm.)",
        df_flat["Typ_norm"].dropna().unique().tolist()
        if "Typ_norm" in df_flat.columns else [],
        default=[],
        key="stats_filter_typ")

    df_table = df_flat.copy()
    if filter_liga and liga_col:
        df_table = df_table[df_table[liga_col].isin(filter_liga)]
    if filter_mt and "Model_type" in df_table.columns:
        df_table = df_table[df_table["Model_type"].isin(filter_mt)]
    if filter_typ and "Typ_norm" in df_table.columns:
        df_table = df_table[df_table["Typ_norm"].isin(filter_typ)]

    cols_table = [c for c in [
        "Data", "Godzina", "Mecz",
        liga_col if liga_col else "Liga",
        "Model_type", "Typ", "Typ_norm",
        "Kurs", "Wynik", "Profit_PLN",
    ] if c and c in df_table.columns]

    st.dataframe(df_table[cols_table], use_container_width=True, hide_index=True)

    # ── EKSPORT JSON ──────────────────────────────
    _REPORTS_DIR = os.path.join(SCRIPT_DIR, "data", "reports")
    os.makedirs(_REPORTS_DIR, exist_ok=True)

    _lv = locals()

    def _recs(name, cols=None):
        df = _lv.get(name)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        if cols:
            df = df[[c for c in cols if c in df.columns]]
        return json.loads(df.to_json(
            orient="records", force_ascii=False, date_format="iso"))

    def _matches_summary():
        dm = _lv.get("df_matches")
        if not isinstance(dm, pd.DataFrame) or dm.empty:
            return {}
        return {
            "meczów":     int(len(dm)),
            "śr_gole":    round(float(dm["_total_goals"].mean()), 2),
            "btts_pct":   round(float(dm["_btts_fact"].mean()), 4),
            "over25_pct": round(float(dm["_over25_fact"].mean()), 4),
            "over95c_pct": round(float(dm["_over95c_fact"].mean()), 4),
        }

    _fname = f"nordic_stats_{datetime.now().strftime('%Y%m%d')}.json"
    _fpath = os.path.join(_REPORTS_DIR, _fname)

    _report = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "bankroll": bankroll,
            "stake":    stake,
        },
        "kpi": {
            "zakłady":    n_all,
            "wygrane":    n_won_all,
            "win_rate":   round(wr_all, 4),
            "roi_pct":    round(roi_all, 2),
            "profit_pln": round(tot_profit, 2),
        },
        "per_liga":     _recs("df_liga_stats"),
        "per_model":    _recs("df_mt"),
        "per_typ":      _recs("df_typ_stats"),
        "per_kurs_bin": _recs("df_bins", cols=[
            "Przedział", "N", "Wygrane", "Win Rate",
            "Śr. kurs", "ROI", "Profit"]),
        "gpt_per_typ":  _recs("df_gpt_tab"),
        "rozkład_meczowy": _matches_summary(),
        "equity_curve": _recs("df_eq", cols=[
            "Data", "Profit_PLN", "cum_profit",
            "bankroll_val"]),
        "tabela": _recs("df_table", cols=cols_table),
    }

    _json_str   = json.dumps(_report, ensure_ascii=False, indent=2, default=str)
    _json_bytes = _json_str.encode("utf-8")

    with open(_fpath, "w", encoding="utf-8") as _f:
        _f.write(_json_str)

    st.caption(f"Raport zapisany: `{_fpath}`")

    st.download_button(
        "⬇️ Pobierz JSON",
        data=_json_bytes,
        file_name=_fname,
        mime="application/json",
        key="stats_dl_json")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — USTAWIENIA
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("Ustawienia i skrypty")

    # ── Dane ─────────────────────────────────────────────────────────────────
    st.subheader("📥 Dane")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.caption("Pobiera mecze na dziś i jutro + szczegóły meczów GPT")
        if st.button("🔄 Pobierz dane dzienne"):
            with st.spinner("Pobieranie listy meczów..."):
                rc, out, err = run_script("fetch_data.py", ["--daily"])
            if rc == 0:
                st.success("Lista meczów pobrana!")
                with st.expander("Log fetch_data"):
                    st.code(out[-2000:])
                with st.spinner("Pobieranie szczegółów i analiz GPT (dziś)..."):
                    rc2, out2, err2 = run_script("fetch_match_details.py", ["--today"])
                with st.spinner("Pobieranie szczegółów i analiz GPT (jutro)..."):
                    rc3, out3, err3 = run_script("fetch_match_details.py", ["--tomorrow"])
                if rc2 == 0 and rc3 == 0:
                    st.success("Szczegóły meczów pobrane!")
                else:
                    st.warning("Szczegóły: częściowy błąd — sprawdź logi")
                with st.expander("Log fetch_match_details (dziś)"):
                    st.code(out2[-1000:])
                with st.expander("Log fetch_match_details (jutro)"):
                    st.code(out3[-1000:])
            else:
                st.error("Błąd pobierania listy meczów")
                st.code(err[:300])
                with st.expander("Log"):
                    st.code(out[-2000:])

    with col_d2:
        st.caption("Pobiera historyczne sezony")
        if st.button("📚 Pobierz dane historyczne"):
            with st.spinner("Pobieranie (~5 min)..."):
                rc, out, err = run_script("fetch_data.py", ["--historical"])
            if rc == 0:
                st.success("Gotowe!")
            else:
                st.error("Błąd")
            with st.expander("Log"):
                st.code(out[-2000:])

    st.divider()

    # ── Scorer ────────────────────────────────────────────────────────────────
    st.subheader("⚽ Predykcje")

    col_s1, col_s2 = st.columns([1, 2])
    day_score = col_s1.radio(
        "Dzień",
        ["today", "tomorrow"],
        format_func=lambda x: "Dziś" if x == "today" else "Jutro",
        horizontal=True,
        key="ustawienia_day",
    )
    league_score = col_s2.multiselect(
        "Liga (puste = wszystkie)",
        ["allsvenskan", "eliteserien", "veikkausliiga"],
        default=[],
    )

    if st.button("▶ Generuj predykcje", type="primary"):
        args = [day_score, "--debug"]
        if league_score:
            for l in league_score:
                args += ["--league", l]
        with st.spinner(f"Generuję predykcje ({day_score})..."):
            rc, out, err = run_script("nordic_scorer.py", args)
        if rc == 0:
            st.success("Predykcje gotowe!")
            st.rerun()
        else:
            st.error("Błąd scorera")
            st.code(err[-500:])
        with st.expander("Log scorera"):
            important = [
                l for l in out.split("\n")
                if any(k in l for k in ["Łącznie", "Meczów", "Zapisano", "DIAG", "Odrzucone", "══"])
            ]
            st.code("\n".join(important[-50:]))

    st.divider()

    # ── Modele ────────────────────────────────────────────────────────────────
    st.subheader("🧠 Modele ML")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        if st.button("🔨 Rebuild dataset"):
            with st.spinner("Buduję dataset..."):
                rc, out, err = run_script("build_dataset.py", ["--debug"])
            if rc == 0:
                st.success("Dataset gotowy!")
            else:
                st.error("Błąd")
            with st.expander("Log"):
                st.code(out[-2000:])

    with col_m2:
        if st.button("🧠 Trenuj modele"):
            with st.spinner("Trening (~3 min)..."):
                rc, out, err = run_script("train_models.py", ["--debug"])
            if rc == 0:
                st.success("Modele wytrenowane!")
                lines = [
                    l for l in out.split("\n")
                    if "│" in l
                    and "Model" not in l
                    and "─" not in l
                    and "═" not in l
                ]
                if lines:
                    st.code("\n".join(lines))
            else:
                st.error("Błąd treningu")
            with st.expander("Log"):
                st.code(out[-3000:])

    st.divider()

    # ── Tygodniowy retren ─────────────────────────────────────────────────────
    st.subheader("🔁 Tygodniowy retren")
    st.caption(
        "Krok 1: fetch_data --weekly (aktualne mecze + statsy drużyn 2026) → "
        "Krok 2: build_dataset --debug → "
        "Krok 3: train_models --debug"
    )

    if st.button("🚀 Uruchom pełny retren (3 kroki)", type="primary"):
        retren_ok = True

        st.write("**Krok 1/3** — pobieranie danych tygodniowych...")
        with st.spinner("fetch_data.py --weekly..."):
            rc1, out1, err1 = run_script("fetch_data.py", ["--weekly"])
        if rc1 == 0:
            st.success("Krok 1 OK — dane pobrane")
        else:
            st.error("Krok 1 BŁĄD — fetch_data --weekly")
            st.code(err1[:500])
            retren_ok = False
        with st.expander("Log Krok 1"):
            st.code(out1[-2000:])

        if retren_ok:
            st.write("**Krok 2/3** — budowanie datasetu...")
            with st.spinner("build_dataset.py --debug..."):
                rc2, out2, err2 = run_script("build_dataset.py", ["--debug"])
            if rc2 == 0:
                st.success("Krok 2 OK — dataset gotowy")
            else:
                st.error("Krok 2 BŁĄD — build_dataset")
                st.code(err2[:500])
                retren_ok = False
            with st.expander("Log Krok 2"):
                st.code(out2[-2000:])

        if retren_ok:
            st.write("**Krok 3/3** — trening modeli...")
            with st.spinner("train_models.py --debug (~3 min)..."):
                rc3, out3, err3 = run_script("train_models.py", ["--debug"])
            if rc3 == 0:
                st.success("Krok 3 OK — modele wytrenowane!")
                lines = [
                    l for l in out3.split("\n")
                    if "│" in l and "Model" not in l and "─" not in l and "═" not in l
                ]
                if lines:
                    st.code("\n".join(lines))
            else:
                st.error("Krok 3 BŁĄD — train_models")
                st.code(err3[:500])
            with st.expander("Log Krok 3"):
                st.code(out3[-3000:])

        if retren_ok:
            st.balloons()
            st.success("Pełny retren zakończony pomyślnie!")

    st.divider()

    # ── H2H Cache ─────────────────────────────────────────────────────────────
    st.subheader("🔗 H2H Cache")
    h2h_files = glob.glob(os.path.join(H2H_CACHE, "match_*.json"))
    st.caption(f"Pliki w cache: {len(h2h_files)} / 2598")

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        if st.button("🔍 Sprawdź cache (dry-run)"):
            rc, out, err = run_script("fetch_h2h_cache.py", ["--dry-run"])
            st.code(out[-1000:])

    with col_h2:
        if st.button("🔄 Uzupełnij H2H Cache"):
            with st.spinner("Pobieranie (~1.6h)..."):
                rc, out, err = run_script("fetch_h2h_cache.py")
            if rc == 0:
                st.success("Cache uzupełniony!")
            else:
                st.error("Błąd")
            with st.expander("Log"):
                st.code(out[-2000:])

    st.divider()

    # ── Analizy meczów GPT ────────────────────────────────────────────────────
    st.subheader("📊 Analizy meczów GPT")
    st.caption(
        "Pobiera szczegółowe dane + analizy GPT "
        "dla meczów nordic z FootyStats /match API"
    )

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        if st.button("📥 Pobierz analizy — Dziś"):
            with st.spinner("Pobieranie..."):
                rc, out, err = run_script("fetch_match_details.py", ["--today"])
            if rc == 0:
                st.success("Gotowe!")
            else:
                st.error("Błąd")
            with st.expander("Log"):
                st.code(out[-1000:])

    with col_g2:
        if st.button("📥 Pobierz analizy — Jutro"):
            with st.spinner("Pobieranie..."):
                rc, out, err = run_script("fetch_match_details.py", ["--tomorrow"])
            if rc == 0:
                st.success("Gotowe!")
            else:
                st.error("Błąd")
            with st.expander("Log"):
                st.code(out[-1000:])

    n_details = len(glob.glob(os.path.join(MATCH_DETAILS_DIR, "match_*.json")))
    n_gpt_files = len(glob.glob(os.path.join(MATCH_DETAILS_DIR, "gpt_*.json")))
    st.caption(
        f"Cache: {n_details} szczegółów meczu | {n_gpt_files} analiz GPT"
    )
