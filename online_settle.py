# -*- coding: utf-8 -*-
"""
online_settle.py - Rozliczanie wynikow zakladow online
Pobiera dane z API football-data-api.com (league-matches), uzupelnia pliki scorera.
season_id pobierane z nordic_config.
"""

import os
import re as _re
from typing import Dict, Any, Optional, Tuple

import pandas as pd

try:
    from api_client import get_league_matches
except ImportError:
    get_league_matches = None

try:
    import nordic_config as _nc
except ImportError:
    _nc = None


# Mapowanie skrotow i pelnych nazw lig uzywanych w kolumnie Liga CSV -> season_id 2026
def _build_league_to_season_map() -> Dict[str, int]:
    if _nc is None:
        return {}
    m: Dict[str, int] = {}

    # Historyczne sezony (pelne nazwy, np. "Allsvenskan 2025")
    if hasattr(_nc, "ALL_HISTORICAL"):
        for s in _nc.ALL_HISTORICAL:
            m[str(s["name"]).strip()] = int(s["id"])

    # Aktualne sezony 2026 - nadpisuja historyczne dla tych samych kluczy bazowych
    if hasattr(_nc, "ALLSVENSKAN_2026_ID"):
        sid = int(_nc.ALLSVENSKAN_2026_ID)
        for key in ("allsvenskan", "allsv", "Sweden Allsvenskan", "Sweden Allsvenskan 2026"):
            m[key] = sid
    if hasattr(_nc, "ELITESERIEN_2026_ID"):
        sid = int(_nc.ELITESERIEN_2026_ID)
        for key in ("eliteserien", "elite", "Norway Eliteserien", "Norway Eliteserien 2026"):
            m[key] = sid
    if hasattr(_nc, "VEIKKAUSLIIGA_2026_ID"):
        sid = int(_nc.VEIKKAUSLIIGA_2026_ID)
        for key in ("veikkausliiga", "veikk", "Finland Veikkausliiga", "Finland Veikkausliiga 2026"):
            m[key] = sid

    return m


def _get_season_id_for_league(league_name: str) -> Optional[int]:
    if _nc is None:
        return None
    league_name = str(league_name).strip()
    m = _build_league_to_season_map()

    # Bezposrednie trafienie (case-sensitive)
    if league_name in m:
        return m[league_name]

    # Case-insensitive
    league_lower = league_name.lower()
    for key, sid in m.items():
        if key.lower() == league_lower:
            return sid

    # Substring - preferuj 2026 (sprawdz najpierw aktywne sezony)
    if _nc and hasattr(_nc, "VEIKKAUSLIIGA_2026_ID") and "veikk" in league_lower:
        return int(_nc.VEIKKAUSLIIGA_2026_ID)
    if _nc and hasattr(_nc, "ALLSVENSKAN_2026_ID") and ("allsv" in league_lower or "allsvenskan" in league_lower):
        return int(_nc.ALLSVENSKAN_2026_ID)
    if _nc and hasattr(_nc, "ELITESERIEN_2026_ID") and ("elite" in league_lower or "eliteserien" in league_lower):
        return int(_nc.ELITESERIEN_2026_ID)

    return None


def _fetch_matches_by_id(df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """Dla kazdej unikalnej ligi w df pobiera mecze z API, zwraca dict: match_id -> match_data."""
    if not get_league_matches:
        return {}
    result: Dict[int, Dict[str, Any]] = {}
    leagues = df["Liga"].dropna().unique().tolist() if "Liga" in df.columns else []
    for liga in leagues:
        season_id = _get_season_id_for_league(liga)
        if season_id is None:
            continue
        data = get_league_matches(season_id)
        if not data or not data.get("success"):
            continue
        matches = data.get("data", [])
        if not isinstance(matches, list):
            continue
        for m in matches:
            mid = m.get("id")
            if mid is not None:
                result[int(mid)] = m
    return result


def _normalize_id(val: Any) -> Optional[int]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return None


def _extract_line_from_typ(typ: str) -> Optional[float]:
    if not typ:
        return None
    m = _re.search(r"(\d+\.?\d*)", str(typ).strip())
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass
    return None


def _is_pending(wynik_val: Any, rezultat_val: Any = None) -> bool:
    """Sprawdza czy zaklad jest nierozliczony lub blednie rozliczony (Rezultat=0:0)."""
    w = str(wynik_val).strip().lower() if wynik_val is not None else ""
    if wynik_val is None or (isinstance(wynik_val, float) and pd.isna(wynik_val)):
        return True
    if w in ("", "nan", "oczekujacy"):
        return True
    # Wynik=0 + Rezultat=0:0 to blad domyslny, nie prawdziwe rozliczenie
    if w in ("0", "0.0") and rezultat_val is not None:
        r = str(rezultat_val).strip()
        if r == "0:0":
            return True
    return False


# ─── SETTLE BTTS MODEL ───────────────────────────────────────────────────────

def settle_btts_model(csv_path: str) -> Tuple[int, str]:
    if not os.path.isfile(csv_path):
        return 0, f"Plik nie istnieje: {csv_path}"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "ID" not in df.columns or "Liga" not in df.columns:
        return 0, "Brak wymaganych kolumn (ID, Liga)."
    if "Wynik" not in df.columns:
        df["Wynik"] = ""

    mask_pending = df.apply(
        lambda r: _is_pending(r["Wynik"], r.get("Rezultat")), axis=1
    )
    if not mask_pending.any():
        return 0, "Wszystkie zaklady sa juz rozliczone."

    matches_by_id = _fetch_matches_by_id(df)
    updated = 0
    for idx in df[mask_pending].index:
        mid = _normalize_id(df.at[idx, "ID"])
        if mid is None or mid not in matches_by_id:
            continue
        m = matches_by_id[mid]
        if str(m.get("status", "")).lower() != "complete":
            continue
        home = m.get("homeGoalCount")
        away = m.get("awayGoalCount")
        if home is None or away is None:
            continue
        win = 1 if (int(home) > 0 and int(away) > 0) else 0
        df.at[idx, "Rezultat"] = f"{int(home)}:{int(away)}"
        df.at[idx, "Wynik"] = win
        updated += 1

    if updated > 0:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return updated, f"Rozliczono {updated} zakladow."


# ─── SETTLE OVER 2.5 MODEL ──────────────────────────────────────────────────

def settle_over25_model(csv_path: str) -> Tuple[int, str]:
    if not os.path.isfile(csv_path):
        return 0, f"Plik nie istnieje: {csv_path}"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "ID" not in df.columns or "Liga" not in df.columns:
        return 0, "Brak wymaganych kolumn (ID, Liga)."
    if "Wynik" not in df.columns:
        df["Wynik"] = ""

    mask_pending = df.apply(
        lambda r: _is_pending(r["Wynik"], r.get("Rezultat")), axis=1
    )
    if not mask_pending.any():
        return 0, "Wszystkie zaklady sa juz rozliczone."

    matches_by_id = _fetch_matches_by_id(df)
    updated = 0
    for idx in df[mask_pending].index:
        mid = _normalize_id(df.at[idx, "ID"])
        if mid is None or mid not in matches_by_id:
            continue
        m = matches_by_id[mid]
        if str(m.get("status", "")).lower() != "complete":
            continue
        home = m.get("homeGoalCount")
        away = m.get("awayGoalCount")
        total_goals = m.get("totalGoalCount")
        if home is None or away is None:
            continue
        tg = int(total_goals) if total_goals is not None else int(home) + int(away)
        win = 1 if tg > 2 else 0
        df.at[idx, "Rezultat"] = f"{int(home)}:{int(away)}"
        df.at[idx, "Wynik"] = win
        updated += 1

    if updated > 0:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return updated, f"Rozliczono {updated} zakladow."


# ─── SETTLE RESULT1 MODEL ───────────────────────────────────────────────────

def settle_result1_model(csv_path: str) -> Tuple[int, str]:
    if not os.path.isfile(csv_path):
        return 0, f"Plik nie istnieje: {csv_path}"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "ID" not in df.columns or "Liga" not in df.columns:
        return 0, "Brak wymaganych kolumn (ID, Liga)."
    if "Wynik" not in df.columns:
        df["Wynik"] = ""

    mask_pending = df.apply(
        lambda r: _is_pending(r["Wynik"], r.get("Rezultat")), axis=1
    )
    if not mask_pending.any():
        return 0, "Wszystkie zaklady sa juz rozliczone."

    matches_by_id = _fetch_matches_by_id(df)
    updated = 0
    for idx in df[mask_pending].index:
        mid = _normalize_id(df.at[idx, "ID"])
        if mid is None or mid not in matches_by_id:
            continue
        m = matches_by_id[mid]
        if str(m.get("status", "")).lower() != "complete":
            continue
        home = m.get("homeGoalCount")
        away = m.get("awayGoalCount")
        if home is None or away is None:
            continue
        typ_str = str(df.at[idx, "Typ"]).strip() if "Typ" in df.columns else "Home Win"
        win = 1 if (int(away) > int(home) if typ_str == "Away Win" else int(home) > int(away)) else 0
        df.at[idx, "Rezultat"] = f"{int(home)}:{int(away)}"
        df.at[idx, "Wynik"] = win
        updated += 1

    if updated > 0:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return updated, f"Rozliczono {updated} zakladow."


# ─── SETTLE CARDS SCORER ────────────────────────────────────────────────────

def settle_cards_scorer(csv_path: str) -> Tuple[int, str]:
    if not os.path.isfile(csv_path):
        return 0, f"Plik nie istnieje: {csv_path}"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "ID" not in df.columns or "Liga" not in df.columns:
        return 0, "Brak wymaganych kolumn (ID, Liga)."
    if "Wynik" not in df.columns:
        df["Wynik"] = ""

    mask_pending = df.apply(
        lambda r: _is_pending(r["Wynik"], r.get("Rezultat")), axis=1
    )
    if not mask_pending.any():
        return 0, "Wszystkie zaklady sa juz rozliczone."

    matches_by_id = _fetch_matches_by_id(df)
    updated = 0
    for idx in df[mask_pending].index:
        mid = _normalize_id(df.at[idx, "ID"])
        if mid is None or mid not in matches_by_id:
            continue
        m = matches_by_id[mid]
        if str(m.get("status", "")).lower() != "complete":
            continue
        y_a = m.get("team_a_yellow_cards", m.get("teamAYellowCards"))
        y_b = m.get("team_b_yellow_cards", m.get("teamBYellowCards"))
        r_a = m.get("team_a_red_cards",    m.get("teamARedCards"))
        r_b = m.get("team_b_red_cards",    m.get("teamBRedCards"))
        try:
            ya = int(y_a) if y_a is not None and pd.notna(y_a) else 0
            yb = int(y_b) if y_b is not None and pd.notna(y_b) else 0
            ra = int(r_a) if r_a is not None and pd.notna(r_a) else 0
            rb = int(r_b) if r_b is not None and pd.notna(r_b) else 0
        except (TypeError, ValueError):
            continue
        total_cards = ya + yb + (ra + rb) * 2
        typ_str = str(df.at[idx, "Typ"]).strip() if "Typ" in df.columns else ""
        line = _extract_line_from_typ(typ_str)
        if line is None:
            continue
        win = 1 if total_cards > line else 0
        if "Cards_SUM" in df.columns:
            df.at[idx, "Cards_SUM"] = total_cards
        df.at[idx, "Wynik"] = win
        updated += 1

    if updated > 0:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return updated, f"Rozliczono {updated} zakladow."


# ─── SETTLE CORNERS SCORER ──────────────────────────────────────────────────

def settle_corners_scorer(csv_path: str) -> Tuple[int, str]:
    if not os.path.isfile(csv_path):
        return 0, f"Plik nie istnieje: {csv_path}"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "ID" not in df.columns or "Liga" not in df.columns:
        return 0, "Brak wymaganych kolumn (ID, Liga)."
    if "Linia" not in df.columns:
        return 0, "Brak kolumny Linia."
    if "Wynik" not in df.columns:
        df["Wynik"] = ""

    mask_pending = df.apply(
        lambda r: _is_pending(r["Wynik"], r.get("Rezultat")), axis=1
    )
    if not mask_pending.any():
        return 0, "Wszystkie zaklady sa juz rozliczone."

    matches_by_id = _fetch_matches_by_id(df)
    updated = 0
    for idx in df[mask_pending].index:
        mid = _normalize_id(df.at[idx, "ID"])
        if mid is None or mid not in matches_by_id:
            continue
        m = matches_by_id[mid]
        if str(m.get("status", "")).lower() != "complete":
            continue
        total_corners = m.get("totalCornerCount")
        try:
            tc = int(total_corners) if total_corners is not None and pd.notna(total_corners) else None
        except (TypeError, ValueError):
            tc = None
        if tc is None:
            continue
        line = _extract_line_from_typ(str(df.at[idx, "Linia"]).strip())
        if line is None:
            continue
        win = 1 if tc > line else 0
        if "Wynik_Suma" in df.columns:
            df.at[idx, "Wynik_Suma"] = tc
        df.at[idx, "Wynik"] = win
        updated += 1

    if updated > 0:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return updated, f"Rozliczono {updated} zakladow."


# ─── SETTLE PORTFOLIO MODEL ─────────────────────────────────────────────────

def settle_portfolio_model(csv_path: str) -> Tuple[int, str]:
    if not os.path.isfile(csv_path):
        return 0, f"Plik nie istnieje: {csv_path}"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "ID" not in df.columns or "Liga" not in df.columns:
        return 0, "Brak wymaganych kolumn (ID, Liga)."
    wynik_col = "Wynik" if "Wynik" in df.columns else "wynik"
    if wynik_col not in df.columns:
        return 0, "Brak kolumny Wynik."

    rezultat_col = next((c for c in df.columns if c.upper() == "REZULTAT"), None)
    mask_pending = df.apply(
        lambda r: _is_pending(r[wynik_col], r.get(rezultat_col) if rezultat_col else None),
        axis=1,
    )
    if not mask_pending.any():
        return 0, "Wszystkie zaklady sa juz rozliczone."

    matches_by_id = _fetch_matches_by_id(df)
    updated = 0
    for idx in df[mask_pending].index:
        mid = _normalize_id(df.at[idx, "ID"])
        if mid is None or mid not in matches_by_id:
            continue
        m = matches_by_id[mid]
        if str(m.get("status", "")).lower() != "complete":
            continue
        home = m.get("homeGoalCount")
        away = m.get("awayGoalCount")
        total_corners = m.get("totalCornerCount")
        total_goals = m.get("totalGoalCount")
        if home is None or away is None:
            continue
        home_int, away_int = int(home), int(away)
        try:
            corners_val = int(total_corners) if total_corners is not None else None
        except (TypeError, ValueError):
            corners_val = None

        if rezultat_col:
            df.at[idx, rezultat_col] = f"{home_int}:{away_int}"
        corners_col = next((c for c in df.columns if c.upper() == "CORNERS"), None)
        if corners_col:
            df.at[idx, corners_col] = corners_val if corners_val is not None else ""

        model_str = str(df.at[idx, "Model"]).strip().upper() if "Model" in df.columns else ""
        typ_str   = str(df.at[idx, "Typ"]).strip()           if "Typ"   in df.columns else ""
        typ_upper = typ_str.upper()

        win = 0
        if "CORNERS" in model_str:
            line = _extract_line_from_typ(typ_str)
            if line is not None and corners_val is not None:
                if "UNDER" in typ_upper or "INVERSE" in model_str:
                    win = 1 if corners_val < line else 0
                else:
                    win = 1 if corners_val > line else 0
        elif model_str == "SMART":
            if "OVER" in typ_upper:
                tg = home_int + away_int if total_goals is None else int(total_goals)
                win = 1 if tg > 2 else 0
            elif "BTTS" in typ_upper and ("YES" in typ_upper or "TAK" in typ_upper):
                win = 1 if (home_int > 0 and away_int > 0) else 0
            elif typ_str == "2":
                win = 1 if away_int > home_int else 0
            elif typ_str == "1":
                win = 1 if home_int > away_int else 0
            elif typ_str.upper() in ("X", "DRAW"):
                win = 1 if home_int == away_int else 0

        df.at[idx, wynik_col] = win
        updated += 1

    if updated > 0:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return updated, f"Rozliczono {updated} zakladow."


# ─── SETTLE FILTERS SCORER ──────────────────────────────────────────────────

def settle_filters_scorer(csv_path: str) -> Tuple[int, str]:
    if not os.path.isfile(csv_path):
        return 0, f"Plik nie istnieje: {csv_path}"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "ID" not in df.columns or "Liga" not in df.columns:
        return 0, "Brak wymaganych kolumn (ID, Liga)."
    if "Wynik" not in df.columns:
        df["Wynik"] = ""

    mask_pending = df.apply(
        lambda r: _is_pending(r["Wynik"], r.get("Rezultat")), axis=1
    )
    if not mask_pending.any():
        return 0, "Wszystkie zaklady sa juz rozliczone."

    matches_by_id = _fetch_matches_by_id(df)
    updated = 0
    for idx in df[mask_pending].index:
        mid = _normalize_id(df.at[idx, "ID"])
        if mid is None or mid not in matches_by_id:
            continue
        m = matches_by_id[mid]
        if str(m.get("status", "")).lower() != "complete":
            continue
        home = m.get("homeGoalCount")
        away = m.get("awayGoalCount")
        total_goals   = m.get("totalGoalCount")
        total_corners = m.get("totalCornerCount")
        if home is None or away is None:
            continue
        home_int, away_int = int(home), int(away)
        typ_str   = str(df.at[idx, "Typ"]).strip() if "Typ" in df.columns else ""
        typ_upper = typ_str.upper()

        if "BTTS YES" in typ_upper or typ_upper == "BTTS":
            win = 1 if (home_int > 0 and away_int > 0) else 0
        elif "OVER 2.5" in typ_upper:
            tg = int(total_goals) if total_goals is not None else home_int + away_int
            win = 1 if tg > 2 else 0
        elif "HOME WIN" in typ_upper:
            win = 1 if home_int > away_int else 0
        elif "AWAY WIN" in typ_upper:
            win = 1 if away_int > home_int else 0
        elif "CORNERS OVER" in typ_upper:
            line = _extract_line_from_typ(typ_str)
            if line is None or total_corners is None:
                continue
            win = 1 if int(total_corners) > line else 0
        else:
            continue

        df.at[idx, "Wynik"] = win
        if "Rezultat" in df.columns:
            df.at[idx, "Rezultat"] = f"{home_int}:{away_int}"
        if "Corners" in df.columns and total_corners is not None:
            df.at[idx, "Corners"] = int(total_corners)
        updated += 1

    if updated > 0:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return updated, f"Rozliczono {updated} zakladow."


# ─── SETTLE INVERSE SCORER ──────────────────────────────────────────────────

def settle_inverse_scorer(csv_path: str) -> Tuple[int, str]:
    if not os.path.isfile(csv_path):
        return 0, f"Plik nie istnieje: {csv_path}"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "ID" not in df.columns:
        return 0, "Brak wymaganej kolumny ID."
    if "Liga" not in df.columns:
        return 0, "Brak kolumny Liga."
    if "Wynik" not in df.columns:
        df["Wynik"] = ""

    mask_pending = df.apply(
        lambda r: _is_pending(r["Wynik"], r.get("Rezultat")), axis=1
    )
    if not mask_pending.any():
        return 0, "Wszystkie zaklady sa juz rozliczone."

    matches_by_id = _fetch_matches_by_id(df)
    updated = 0
    for idx in df[mask_pending].index:
        mid = _normalize_id(df.at[idx, "ID"])
        if mid is None or mid not in matches_by_id:
            continue
        m = matches_by_id[mid]
        if str(m.get("status", "")).lower() != "complete":
            continue
        typ = str(df.at[idx, "Typ_Inverse"] if "Typ_Inverse" in df.columns else "").strip()

        if "Under" in typ:
            total_corners = m.get("totalCornerCount")
            try:
                tc = int(total_corners)
            except (TypeError, ValueError):
                continue
            match_line = _re.search(r"(\d+\.?\d*)", typ)
            if not match_line:
                continue
            linia = float(match_line.group(1))
            df.at[idx, "Wynik"] = 1 if tc < linia else 0
            if "Wynik_Suma" in df.columns:
                df.at[idx, "Wynik_Suma"] = tc
            if "Rezultat" in df.columns:
                home = m.get("homeGoalCount")
                away = m.get("awayGoalCount")
                if home is not None and away is not None:
                    df.at[idx, "Rezultat"] = f"{int(home)}:{int(away)}"
            updated += 1

        elif "BTTS No" in typ:
            home = m.get("homeGoalCount")
            away = m.get("awayGoalCount")
            if home is None or away is None:
                continue
            df.at[idx, "Wynik"] = 1 if (int(home) == 0 or int(away) == 0) else 0
            if "Rezultat" in df.columns:
                df.at[idx, "Rezultat"] = f"{int(home)}:{int(away)}"
            updated += 1

    if updated > 0:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return updated, f"Rozliczono {updated} zakladow."


# ─── SETTLE DAILY SCORER ────────────────────────────────────────────────────

def settle_daily_scorer(csv_path: str) -> Tuple[int, str]:
    """
    Rozlicza plik [league]_scorer_*.csv (Daily Scorer / GPT FootyStats).
    Typy: Corners Over X.5, BTTS Yes, Over 2.5, Home Win, Away Win, [Druzyna] wygra.
    """
    if not os.path.isfile(csv_path):
        return 0, f"Plik nie istnieje: {csv_path}"

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "Liga" not in df.columns:
        return 0, "Brak wymaganych kolumn (Liga)."
    if "ID" not in df.columns:
        return 0, "Brak kolumny ID."

    if "Godzina" in df.columns:
        df = df[df["Godzina"].astype(str).str.match(r"^\d{1,2}:\d{2}$", na=False)].copy()

    if "Wynik" not in df.columns:
        df["Wynik"] = ""

    rezultat_col = "Rezultat" if "Rezultat" in df.columns else None
    mask_pending = df.apply(
        lambda r: _is_pending(r["Wynik"], r.get(rezultat_col) if rezultat_col else None),
        axis=1,
    )
    if not mask_pending.any():
        return 0, "Wszystkie zaklady sa juz rozliczone."

    matches_by_id = _fetch_matches_by_id(df)
    if not matches_by_id:
        return 0, "Brak danych z API (sprawdz klucz API i polaczenie)."

    updated = 0
    for idx in df[mask_pending].index:
        mid = _normalize_id(df.at[idx, "ID"])
        if mid is None or mid not in matches_by_id:
            continue
        m = matches_by_id[mid]
        if str(m.get("status", "")).lower() != "complete":
            continue

        home = m.get("homeGoalCount")
        away = m.get("awayGoalCount")
        total_goals   = m.get("totalGoalCount")
        total_corners = m.get("totalCornerCount")
        if home is None or away is None:
            continue

        home_int, away_int = int(home), int(away)
        typ_str   = str(df.at[idx, "Typ"]).strip() if "Typ" in df.columns else ""
        typ_upper = typ_str.upper()
        mecz_str  = str(df.at[idx, "Mecz"]).strip() if "Mecz" in df.columns else ""
        parts      = mecz_str.split(" vs ", 1) if " vs " in mecz_str else []
        home_team  = parts[0].strip().upper() if parts else ""
        away_team  = parts[1].strip().upper() if len(parts) > 1 else ""

        win: Optional[int] = None
        tg = int(total_goals) if total_goals is not None else home_int + away_int
        tc = int(total_corners) if total_corners is not None else None

        if "CORNERS" in typ_upper and "OVER" in typ_upper:
            line = _extract_line_from_typ(typ_str) or 9.5
            if tc is not None:
                win = 1 if tc > line else 0
        elif "CORNERS" in typ_upper and "UNDER" in typ_upper:
            line = _extract_line_from_typ(typ_str) or 9.5
            if tc is not None:
                win = 1 if tc < line else 0
        elif "BTTS" in typ_upper and ("YES" in typ_upper or "TAK" in typ_upper):
            win = 1 if (home_int > 0 and away_int > 0) else 0
        elif "BTTS" in typ_upper and "NO" in typ_upper:
            win = 0 if (home_int > 0 and away_int > 0) else 1
        elif "OVER" in typ_upper and "2.5" in typ_str:
            win = 1 if tg > 2 else 0
        elif "UNDER" in typ_upper and "2.5" in typ_str:
            win = 1 if tg < 3 else 0
        elif "DOUBLE CHANCE" in typ_upper:
            if "1X" in typ_upper:
                win = 1 if home_int >= away_int else 0
            elif "X2" in typ_upper:
                win = 1 if away_int >= home_int else 0
            elif "12" in typ_upper:
                win = 1 if home_int != away_int else 0
            elif "WIN OR DRAW" in typ_upper or "WYGRA LUB REMIS" in typ_upper:
                # "[Druzyna] Win or Draw" — sprawdz czy druzyna to home czy away
                if home_team and home_team in typ_upper:
                    win = 1 if home_int >= away_int else 0
                elif away_team and away_team in typ_upper:
                    win = 1 if away_int >= home_int else 0
        elif "AWAY WIN" in typ_upper or typ_str == "2":
            win = 1 if away_int > home_int else 0
        elif "HOME WIN" in typ_upper or typ_str == "1" or "RESULT 1" in typ_upper:
            win = 1 if home_int > away_int else 0
        elif "WYGRA" in typ_upper:
            if home_team and home_team in typ_upper:
                win = 1 if home_int > away_int else 0
            elif away_team and away_team in typ_upper:
                win = 1 if away_int > home_int else 0

        if win is None:
            continue

        df.at[idx, "Wynik"] = win
        if rezultat_col:
            df.at[idx, rezultat_col] = f"{home_int}:{away_int}"
        corners_col = next((c for c in df.columns if c.upper() == "CORNERS"), None)
        if corners_col and total_corners is not None:
            df.at[idx, corners_col] = int(total_corners)
        if "Kartki" in df.columns:
            y_a = m.get("team_a_yellow_cards", m.get("teamAYellowCards", 0)) or 0
            y_b = m.get("team_b_yellow_cards", m.get("teamBYellowCards", 0)) or 0
            r_a = m.get("team_a_red_cards",    m.get("teamARedCards",    0)) or 0
            r_b = m.get("team_b_red_cards",     m.get("teamBRedCards",    0)) or 0
            df.at[idx, "Kartki"] = int(y_a) + int(y_b) + int(r_a) + int(r_b)
        if "Kurs" in df.columns and "Profit_PLN" in df.columns:
            stake = 100.0
            kurs  = pd.to_numeric(df.at[idx, "Kurs"], errors="coerce") or 1
            df.at[idx, "Profit_PLN"] = round(stake * (kurs - 1) if win == 1 else -stake, 2)
        updated += 1

    if updated > 0:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return updated, f"Rozliczono {updated} zakladow."
