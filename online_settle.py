# -*- coding: utf-8 -*-
"""
online_settle.py – Rozliczanie wyników zakładów online
Pobiera dane z API football-data-api.com (league-matches), uzupełnia pliki corners_model i btts_model.
season_id pobierane z nordic_config (LEAGUE_BY_SEASON_ID).
"""

import os
from typing import Dict, Any, Optional, Tuple

import pandas as pd

try:
    from api_client import get_league_matches
except ImportError:
    get_league_matches = None

try:
    import nordic_config as config
except ImportError:
    config = None


def _build_league_to_season_map() -> Dict[str, int]:
    """Buduje mapowanie nazwa ligi -> season_id z nordic_config.
    Dla każdej ligi wybiera najnowszy dostępny sezon (najwyższy rok)."""
    m = {}
    if config and hasattr(config, "ALL_HISTORICAL"):
        for s in config.ALL_HISTORICAL:
            liga = s["name"]
            # Usuń rok z nazwy, żeby mapować po nazwie ligi bez roku
            # np. "Allsvenskan 2025" -> season_id najnowszego
            m[liga] = s["id"]
        # Dodaj mapowania po nazwie ligi (bez roku) -> najnowszy sezon
        for league_key, league_full in config.LEAGUE_NAMES.items():
            seasons = [s for s in config.ALL_HISTORICAL
                       if config.LEAGUE_BY_SEASON_ID.get(s["id"]) == league_key]
            if seasons:
                latest = max(seasons, key=lambda x: x["year"])
                m[league_full] = latest["id"]
    if config and hasattr(config, "ACTIVE_SEASON_IDS"):
        if hasattr(config, "ALLSVENSKAN_2026_ID"):
            m["Sweden Allsvenskan 2026"] = config.ALLSVENSKAN_2026_ID
        if hasattr(config, "ELITESERIEN_2026_ID"):
            m["Norway Eliteserien 2026"] = config.ELITESERIEN_2026_ID
        if hasattr(config, "VEIKKAUSLIIGA_2026_ID"):
            m["Finland Veikkausliiga 2026"] = config.VEIKKAUSLIIGA_2026_ID
    return m


def _get_season_id_for_league(league_name: str) -> Optional[int]:
    """Pobiera season_id dla ligi z nordic_config (najnowszy sezon)."""
    if not config:
        return None
    league_name = str(league_name).strip()
    m = _build_league_to_season_map()
    if league_name in m:
        return m[league_name]
    for liga, sid in m.items():
        if league_name.lower() in liga.lower() or liga.lower() in league_name.lower():
            return sid
    return None


def _fetch_matches_by_id(df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """
    Dla każdej unikalnej ligi w df pobiera mecze z API, zwraca dict: match_id -> match_data.
    """
    if not get_league_matches:
        return {}
    result = {}
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
    """Normalizuje ID meczu do int."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return None


def settle_btts_model(csv_path: str) -> Tuple[int, str]:
    """
    Rozlicza plik btts_model_*.csv.
    - Rezultat: np. "2:1" (wynik bramkowy)
    - Wynik: 1 jeśli BTTS Yes (obie drużyny strzeliły), 0 w przeciwnym razie
    """
    if not os.path.isfile(csv_path):
        return 0, f"Plik nie istnieje: {csv_path}"

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "ID" not in df.columns or "Liga" not in df.columns:
        return 0, "Brak wymaganych kolumn (ID, Liga)."

    if "Wynik" not in df.columns:
        df["Wynik"] = ""

    mask_pending = (
        df["Wynik"].astype(str).str.strip().str.lower().isin(["", "nan", "oczekujacy"])
        | df["Wynik"].isna()
    )
    if not mask_pending.any():
        return 0, "Wszystkie zakłady są już rozliczone."

    matches_by_id = _fetch_matches_by_id(df)
    updated = 0
    for idx in df[mask_pending].index:
        mid = _normalize_id(df.at[idx, "ID"])
        if mid is None or mid not in matches_by_id:
            continue
        m = matches_by_id[mid]
        status = str(m.get("status", "")).lower()
        if status != "complete":
            continue

        home = m.get("homeGoalCount")
        away = m.get("awayGoalCount")
        if home is None or away is None:
            continue

        btts_yes = int(home) > 0 and int(away) > 0
        win = 1 if btts_yes else 0

        df.at[idx, "Rezultat"] = f"{int(home)}:{int(away)}"
        df.at[idx, "Wynik"] = win
        updated += 1

    if updated > 0:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return updated, f"Rozliczono {updated} zakładów."


def settle_over25_model(csv_path: str) -> Tuple[int, str]:
    """
    Rozlicza plik over25_model_*.csv.
    - Rezultat: np. "2:1" (wynik bramkowy)
    - Wynik: 1 jeśli Over 2.5 (totalGoalCount > 2), 0 w przeciwnym razie
    """
    if not os.path.isfile(csv_path):
        return 0, f"Plik nie istnieje: {csv_path}"

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "ID" not in df.columns or "Liga" not in df.columns:
        return 0, "Brak wymaganych kolumn (ID, Liga)."

    mask_pending = (df["Wynik"].astype(str).str.strip().str.lower() == "oczekujacy")
    if not mask_pending.any():
        return 0, "Wszystkie zakłady są już rozliczone."

    matches_by_id = _fetch_matches_by_id(df)
    updated = 0
    for idx in df[mask_pending].index:
        mid = _normalize_id(df.at[idx, "ID"])
        if mid is None or mid not in matches_by_id:
            continue
        m = matches_by_id[mid]
        status = str(m.get("status", "")).lower()
        if status != "complete":
            continue

        home = m.get("homeGoalCount")
        away = m.get("awayGoalCount")
        total_goals = m.get("totalGoalCount")
        if home is None or away is None:
            continue

        over25 = total_goals is not None and int(total_goals) > 2
        win = 1 if over25 else 0

        df.at[idx, "Rezultat"] = f"{int(home)}:{int(away)}"
        df.at[idx, "Wynik"] = win
        updated += 1

    if updated > 0:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return updated, f"Rozliczono {updated} zakładów."


def settle_result1_model(csv_path: str) -> Tuple[int, str]:
    """
    Rozlicza plik result1_model_*.csv.
    - Rezultat: np. "2:1" (wynik bramkowy)
    - Wynik: 1 jeśli Result 1 (gospodarz wygrywa, homeGoalCount > awayGoalCount), 0 w przeciwnym razie
    """
    if not os.path.isfile(csv_path):
        return 0, f"Plik nie istnieje: {csv_path}"

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "ID" not in df.columns or "Liga" not in df.columns:
        return 0, "Brak wymaganych kolumn (ID, Liga)."

    mask_pending = (df["Wynik"].astype(str).str.strip().str.lower() == "oczekujacy")
    if not mask_pending.any():
        return 0, "Wszystkie zakłady są już rozliczone."

    matches_by_id = _fetch_matches_by_id(df)
    updated = 0
    for idx in df[mask_pending].index:
        mid = _normalize_id(df.at[idx, "ID"])
        if mid is None or mid not in matches_by_id:
            continue
        m = matches_by_id[mid]
        status = str(m.get("status", "")).lower()
        if status != "complete":
            continue

        home = m.get("homeGoalCount")
        away = m.get("awayGoalCount")
        if home is None or away is None:
            continue

        typ_str = str(df.at[idx, "Typ"]).strip() if "Typ" in df.columns else "Home Win"
        if typ_str == "Away Win":
            result_win = int(away) > int(home)
        else:
            result_win = int(home) > int(away)
        win = 1 if result_win else 0

        df.at[idx, "Rezultat"] = f"{int(home)}:{int(away)}"
        df.at[idx, "Wynik"] = win
        updated += 1

    if updated > 0:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return updated, f"Rozliczono {updated} zakładów."


def settle_cards_scorer(csv_path: str) -> Tuple[int, str]:
    """
    Rozlicza plik cards_scorer_*.csv.
    - Cards_SUM: liczba kartek (yellow_a + yellow_b + (red_a + red_b)*2)
    - Wynik: 1 jeśli Cards_SUM > linia z Typ (np. O 3.5K -> 3.5), 0 w przeciwnym razie
    """
    if not os.path.isfile(csv_path):
        return 0, f"Plik nie istnieje: {csv_path}"

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "ID" not in df.columns or "Liga" not in df.columns:
        return 0, "Brak wymaganych kolumn (ID, Liga)."

    mask_pending = (
        df["Wynik"].astype(str).str.strip().str.lower().isin(["", "nan"])
        | df["Wynik"].isna()
    )
    if not mask_pending.any():
        return 0, "Wszystkie zakłady są już rozliczone."

    matches_by_id = _fetch_matches_by_id(df)
    updated = 0
    for idx in df[mask_pending].index:
        mid = _normalize_id(df.at[idx, "ID"])
        if mid is None or mid not in matches_by_id:
            continue
        m = matches_by_id[mid]
        status = str(m.get("status", "")).lower()
        if status != "complete":
            continue

        y_a = m.get("team_a_yellow_cards", m.get("teamAYellowCards"))
        y_b = m.get("team_b_yellow_cards", m.get("teamBYellowCards"))
        r_a = m.get("team_a_red_cards", m.get("teamARedCards"))
        r_b = m.get("team_b_red_cards", m.get("teamBRedCards"))
        try:
            ya = int(y_a) if y_a is not None and pd.notna(y_a) and int(y_a) >= 0 else 0
            yb = int(y_b) if y_b is not None and pd.notna(y_b) and int(y_b) >= 0 else 0
            ra = int(r_a) if r_a is not None and pd.notna(r_a) and int(r_a) >= 0 else 0
            rb = int(r_b) if r_b is not None and pd.notna(r_b) and int(r_b) >= 0 else 0
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
    return updated, f"Rozliczono {updated} zakładów."


def settle_corners_scorer(csv_path: str) -> Tuple[int, str]:
    """
    Rozlicza plik corners_scorer_*.csv.
    - Wynik_Suma: liczba rzutów rożnych (totalCornerCount)
    - Wynik: 1 jeśli totalCornerCount > linia z Linia (np. O 9.5 -> 9.5), 0 w przeciwnym razie
    """
    if not os.path.isfile(csv_path):
        return 0, f"Plik nie istnieje: {csv_path}"

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "ID" not in df.columns or "Liga" not in df.columns:
        return 0, "Brak wymaganych kolumn (ID, Liga)."
    if "Linia" not in df.columns:
        return 0, "Brak kolumny Linia."

    if "Wynik" not in df.columns:
        df["Wynik"] = ""

    mask_pending = (
        df["Wynik"].astype(str).str.strip().str.lower().isin(["", "nan"])
        | df["Wynik"].isna()
    )
    if not mask_pending.any():
        return 0, "Wszystkie zakłady są już rozliczone."

    matches_by_id = _fetch_matches_by_id(df)
    updated = 0
    for idx in df[mask_pending].index:
        mid = _normalize_id(df.at[idx, "ID"])
        if mid is None or mid not in matches_by_id:
            continue
        m = matches_by_id[mid]
        status = str(m.get("status", "")).lower()
        if status != "complete":
            continue

        total_corners = m.get("totalCornerCount")
        try:
            tc = int(total_corners) if total_corners is not None and pd.notna(total_corners) and int(total_corners) >= 0 else None
        except (TypeError, ValueError):
            tc = None
        if tc is None:
            continue

        linia_str = str(df.at[idx, "Linia"]).strip()
        line = _extract_line_from_typ(linia_str)
        if line is None:
            continue
        win = 1 if tc > line else 0

        if "Wynik_Suma" in df.columns:
            df.at[idx, "Wynik_Suma"] = tc
        df.at[idx, "Wynik"] = win
        updated += 1

    if updated > 0:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return updated, f"Rozliczono {updated} zakładów."


def _extract_line_from_typ(typ: str) -> Optional[float]:
    """Wyciąga linię z Typ (np. 'Over 9.5' -> 9.5)."""
    if not typ:
        return None
    import re
    m = re.search(r"(\d+\.?\d*)", str(typ).strip())
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass
    return None


def settle_portfolio_model(csv_path: str) -> Tuple[int, str]:
    """
    Rozlicza FINAL_PORTFOLIO_*.csv jak Corners Model: Liga->nordic_config->API get_league_matches.
    Rezultat: wynik meczu (np. "2:1"). Corners: liczba rożnych.
    Wynik: Corners – Corners > linia z Typ; SMART – Over/BTTS Yes/1/2/X według Rezultat.
    """
    if not os.path.isfile(csv_path):
        return 0, f"Plik nie istnieje: {csv_path}"

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "ID" not in df.columns or "Liga" not in df.columns:
        return 0, "Brak wymaganych kolumn (ID, Liga)."

    wynik_col = "Wynik" if "Wynik" in df.columns else "wynik"
    if wynik_col not in df.columns:
        return 0, "Brak kolumny Wynik."

    mask_pending = (
        df[wynik_col].astype(str).str.strip().str.lower().isin(["", "oczekujacy", "nan"])
        | df[wynik_col].isna()
    )
    if not mask_pending.any():
        return 0, "Wszystkie zakłady są już rozliczone."

    matches_by_id = _fetch_matches_by_id(df)
    updated = 0

    for idx in df[mask_pending].index:
        mid = _normalize_id(df.at[idx, "ID"])
        if mid is None or mid not in matches_by_id:
            continue
        m = matches_by_id[mid]
        status = str(m.get("status", "")).lower()
        if status != "complete":
            continue

        home = m.get("homeGoalCount")
        away = m.get("awayGoalCount")
        total_corners = m.get("totalCornerCount")
        total_goals = m.get("totalGoalCount")
        if home is None or away is None:
            continue

        home_int = int(home)
        away_int = int(away)
        rezultat_str = f"{home_int}:{away_int}"
        try:
            corners_val = int(total_corners) if total_corners is not None else None
        except (TypeError, ValueError):
            corners_val = None

        model_str = str(df.at[idx, "Model"]).strip().upper() if "Model" in df.columns else ""
        typ_str = str(df.at[idx, "Typ"]).strip() if "Typ" in df.columns else ""
        typ_upper = typ_str.upper()

        rezultat_col = "REZULTAT" if "REZULTAT" in df.columns else "Rezultat"
        if rezultat_col in df.columns:
            df.at[idx, rezultat_col] = rezultat_str
        corners_col = next((c for c in df.columns if c.upper() == "CORNERS"), None)
        if corners_col:
            df.at[idx, corners_col] = corners_val if corners_val is not None else ""

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
                total_g = home_int + away_int if total_goals is None else int(total_goals)
                win = 1 if total_g > 2 else 0
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
    return updated, f"Rozliczono {updated} zakładów."


def settle_filters_scorer(csv_path: str) -> Tuple[int, str]:
    """
    Rozlicza plik filters_scorer_*.csv.
    Typy: BTTS Yes, Over 2.5, Home Win, Away Win, Corners Over 9.5.
    """
    if not os.path.isfile(csv_path):
        return 0, f"Plik nie istnieje: {csv_path}"

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "ID" not in df.columns or "Liga" not in df.columns:
        return 0, "Brak wymaganych kolumn (ID, Liga)."

    if "Wynik" not in df.columns:
        df["Wynik"] = ""

    mask_pending = (
        df["Wynik"].astype(str).str.strip().str.lower().isin(["", "nan", "oczekujacy"])
        | df["Wynik"].isna()
    )
    if not mask_pending.any():
        return 0, "Wszystkie zakłady są już rozliczone."

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
        total_corners = m.get("totalCornerCount")
        if home is None or away is None:
            continue

        home_int = int(home)
        away_int = int(away)
        typ_str = str(df.at[idx, "Typ"]).strip() if "Typ" in df.columns else ""
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
    return updated, f"Rozliczono {updated} zakładów."


def settle_inverse_scorer(csv_path: str) -> Tuple[int, str]:
    """
    Rozlicza plik inverse_scorer_*.csv.
    Under X.5 → WIN jeśli totalCornerCount < X.5.
    BTTS No   → WIN jeśli homeGoalCount == 0 lub awayGoalCount == 0.
    """
    import re as _re
    if not os.path.isfile(csv_path):
        return 0, f"Plik nie istnieje: {csv_path}"

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "ID" not in df.columns:
        return 0, "Brak wymaganej kolumny ID."

    if "Liga" not in df.columns:
        return 0, "Brak kolumny Liga — nie można pobrać danych z API."

    if "Wynik" not in df.columns:
        df["Wynik"] = ""

    mask_pending = (
        df["Wynik"].astype(str).str.strip().str.lower().isin(["", "nan", "oczekujacy"])
        | df["Wynik"].isna()
    )
    if not mask_pending.any():
        return 0, "Wszystkie zakłady są już rozliczone."

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
    return updated, f"Rozliczono {updated} zakładów."


def settle_daily_scorer(csv_path: str) -> Tuple[int, str]:
    """
    Rozlicza plik scorer_*.csv (Daily Scorer).
    - Wynik: 1/0 zależnie od Typ (Corners Over X.5, BTTS Yes, Over 2.5, Home Win).
    - Wymaga kolumny ID (match_id). Jeśli brak ID, wiersze są pomijane.
    """
    if not os.path.isfile(csv_path):
        return 0, f"Plik nie istnieje: {csv_path}"

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty or "Liga" not in df.columns:
        return 0, "Brak wymaganych kolumn (Liga)."
    if "ID" not in df.columns:
        return 0, "Brak kolumny ID. Uruchom ponownie daily_scorer.py, aby wyeksportować z ID."

    if "Godzina" in df.columns:
        df = df[df["Godzina"].astype(str).str.match(r"^\d{1,2}:\d{2}$", na=False)].copy()

    mask_pending = (df["Wynik"].astype(str).str.strip().str.lower() == "oczekujacy")
    if not mask_pending.any():
        return 0, "Wszystkie zakłady są już rozliczone."

    matches_by_id = _fetch_matches_by_id(df)
    updated = 0
    for idx in df[mask_pending].index:
        mid = _normalize_id(df.at[idx, "ID"])
        if mid is None or mid not in matches_by_id:
            continue
        m = matches_by_id[mid]
        status = str(m.get("status", "")).lower()
        if status != "complete":
            continue

        home = m.get("homeGoalCount")
        away = m.get("awayGoalCount")
        total_goals = m.get("totalGoalCount")
        total_corners = m.get("totalCornerCount")
        if home is None or away is None:
            continue

        home_int = int(home)
        away_int = int(away)
        typ_str = str(df.at[idx, "Typ"]).strip() if "Typ" in df.columns else ""
        typ_upper = typ_str.upper()

        win = 0
        if "CORNERS" in typ_upper and "OVER" in typ_upper:
            line = _extract_line_from_typ(typ_str)
            if line is not None and total_corners is not None:
                win = 1 if int(total_corners) > line else 0
        elif "BTTS" in typ_upper and ("YES" in typ_upper or "TAK" in typ_upper):
            win = 1 if (home_int > 0 and away_int > 0) else 0
        elif "OVER" in typ_upper and "2.5" in typ_str:
            tg = int(total_goals) if total_goals is not None else home_int + away_int
            win = 1 if tg > 2 else 0
        elif "HOME" in typ_upper or typ_str == "1" or "RESULT 1" in typ_upper:
            win = 1 if home_int > away_int else 0

        df.at[idx, "Wynik"] = win
        if "Rezultat" in df.columns:
            df.at[idx, "Rezultat"] = f"{home_int}:{away_int}"
        corners_col = next((c for c in df.columns if c.upper() == "CORNERS"), None)
        if corners_col and total_corners is not None:
            df.at[idx, corners_col] = int(total_corners)
        updated += 1

    if updated > 0:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return updated, f"Rozliczono {updated} zakładów."
