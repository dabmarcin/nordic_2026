import glob
import json
import os
import re
import subprocess
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



# ── ZAKŁADKI ──────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "📅 Mecze",
    "📰 Artykuły",
    "🏆 Wyniki",
    "📊 Statystyki",
    "💼 Investor",
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
            df_show = df_show[ev_num >= min_ev]
        if "Score[%]" in df_show.columns:
            # Score filtr pomija wiersze gpt_pred (mają puste Score)
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
                        "Kurs", "Stake_PLN", "Wynik", "Profit_PLN",
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
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Wyniki", f"✅{n_won} ❌{n_lost}")
                    m2.metric("Profit", f"{profit:+.2f} PLN")
                    if len(settled) > 0:
                        m3.metric("Win Rate", f"{n_won/len(settled):.0%}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — STATYSTYKI
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Statystyki i ROI")

    df_all = get_all_scorer_files()

    if df_all.empty or "Wynik" not in df_all.columns:
        st.info("Brak rozliczonych zakładów.")
    else:
        df_settled = df_all[
            pd.to_numeric(df_all["Wynik"], errors="coerce").isin([0, 1])
        ].copy()

        if df_settled.empty:
            st.info("Brak rozliczonych zakładów.")
        else:
            roi_data = calc_roi(df_settled)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Zakłady",  roi_data.get("n", 0))
            k2.metric("Win Rate", f"{roi_data.get('win_rate', 0):.1%}")
            k3.metric("ROI",      f"{roi_data.get('roi_pct', 0):+.1f}%")
            k4.metric("Profit",   f"{roi_data.get('total_profit', 0):+.1f} PLN")

            st.subheader("ROI per liga")
            liga_col = "league" if "league" in df_settled.columns else (
                "_liga" if "_liga" in df_settled.columns else None
            )
            if liga_col:
                liga_stats = []
                for liga in df_settled[liga_col].unique():
                    sub = df_settled[df_settled[liga_col] == liga]
                    r = calc_roi(sub)
                    r["Liga"] = liga
                    liga_stats.append(r)
                if liga_stats:
                    df_liga = pd.DataFrame(liga_stats)
                    st.dataframe(
                        df_liga[["Liga", "n", "win_rate", "avg_kurs", "roi_pct", "total_profit"]],
                        use_container_width=True,
                        hide_index=True,
                    )

            st.subheader("ROI per model")
            if "Model" in df_settled.columns:
                model_stats = []
                for model in df_settled["Model"].unique():
                    sub = df_settled[df_settled["Model"] == model]
                    r = calc_roi(sub)
                    r["Model"] = model
                    model_stats.append(r)
                if model_stats:
                    df_model = pd.DataFrame(model_stats)
                    st.dataframe(
                        df_model[["Model", "n", "win_rate", "avg_kurs", "roi_pct", "total_profit"]],
                        use_container_width=True,
                        hide_index=True,
                    )

            st.subheader("Liga vs Combined")
            if "Model_type" in df_settled.columns:
                type_stats = []
                for mt in ["liga", "combined"]:
                    sub = df_settled[df_settled["Model_type"] == mt]
                    if not sub.empty:
                        r = calc_roi(sub)
                        r["Model_type"] = mt
                        type_stats.append(r)
                if type_stats:
                    df_type = pd.DataFrame(type_stats)
                    st.dataframe(
                        df_type[["Model_type", "n", "win_rate", "roi_pct", "total_profit"]],
                        use_container_width=True,
                        hide_index=True,
                    )

            st.subheader("Wszystkie zakłady")
            cols_all = [
                c for c in [
                    "Data", "Godzina", "Mecz", "league", "Model", "Model_type",
                    "Typ", "Score[%]", "Kurs", "Stake_PLN", "Wynik", "Profit_PLN",
                ]
                if c in df_settled.columns
            ]
            st.dataframe(df_settled[cols_all], use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — INVESTOR
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("Investor Center")

    df_all = get_all_scorer_files()
    df_settled = pd.DataFrame()
    if not df_all.empty and "Wynik" in df_all.columns:
        df_settled = df_all[
            pd.to_numeric(df_all["Wynik"], errors="coerce").isin([0, 1])
        ].copy()

    if df_settled.empty:
        st.info("Brak rozliczonych zakładów.")
    else:
        inv_bankroll = st.number_input(
            "Bankroll (PLN)", 100.0, 100000.0, 1000.0, 100.0
        )

        roi_data = calc_roi(df_settled)
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("Zakłady",  roi_data.get("n", 0))
        r1c2.metric("Win Rate", f"{roi_data.get('win_rate', 0):.1%}")
        r1c3.metric("ROI",      f"{roi_data.get('roi_pct', 0):+.1f}%")
        r1c4.metric("Profit",   f"{roi_data.get('total_profit', 0):+.1f} PLN")

        if "Profit_PLN" in df_settled.columns and "Data" in df_settled.columns:
            st.subheader("📈 Equity Curve")
            df_eq = df_settled.copy()
            df_eq["Profit_PLN"] = pd.to_numeric(
                df_eq["Profit_PLN"], errors="coerce"
            ).fillna(0)
            df_eq = df_eq.sort_values("Data")
            df_eq["cum_profit"] = df_eq["Profit_PLN"].cumsum()
            df_eq["bankroll"]   = inv_bankroll + df_eq["cum_profit"]

            chart = (
                alt.Chart(df_eq)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Data:T", title="Data"),
                    y=alt.Y("bankroll:Q", title="Bankroll (PLN)"),
                    tooltip=["Data", "bankroll", "cum_profit"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)

        if "Data" in df_settled.columns and "Profit_PLN" in df_settled.columns:
            st.subheader("📊 Dzienny Profit")
            df_day = (
                df_settled.groupby("Data")["Profit_PLN"]
                .apply(lambda x: pd.to_numeric(x, errors="coerce").sum())
                .reset_index()
            )
            df_day.columns = ["Data", "Profit"]
            bar = (
                alt.Chart(df_day)
                .mark_bar()
                .encode(
                    x="Data:T",
                    y="Profit:Q",
                    color=alt.condition(
                        alt.datum.Profit >= 0,
                        alt.value("#2ecc71"),
                        alt.value("#e74c3c"),
                    ),
                    tooltip=["Data", "Profit"],
                )
                .properties(height=200)
            )
            st.altair_chart(bar, use_container_width=True)

        st.subheader("🔍 Liga vs Combined")
        if "Model_type" in df_settled.columns:
            for mt in ["liga", "combined"]:
                sub = df_settled[df_settled["Model_type"] == mt]
                if not sub.empty:
                    r = calc_roi(sub)
                    st.write(
                        f"**{mt.upper()}** — "
                        f"n={r['n']} | "
                        f"WR={r['win_rate']:.1%} | "
                        f"ROI={r['roi_pct']:+.1f}% | "
                        f"Profit={r['total_profit']:+.1f} PLN"
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — USTAWIENIA
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("Ustawienia i skrypty")

    # ── Dane ─────────────────────────────────────────────────────────────────
    st.subheader("📥 Dane")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.caption("Pobiera mecze na dziś i jutro")
        if st.button("🔄 Pobierz dane dzienne"):
            with st.spinner("Pobieranie..."):
                rc, out, err = run_script("fetch_data.py", ["--daily"])
            if rc == 0:
                st.success("Dane pobrane!")
            else:
                st.error("Błąd")
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
            args += ["--league", ",".join(league_score)]
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
