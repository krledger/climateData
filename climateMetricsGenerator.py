#!/usr/bin/env python3
# metrics_from_parquet_multi.py — build metrics for scenarios in ./metricsDataFiles/<SCENARIO>
# Inputs:  raw_daily.parquet  +  raw_daily.json  (from importer)
# Outputs: metrics.parquet, metrics.csv, metrics.json  in each scenario folder
# Notes:
# - Uses raw_daily.json ->  date.column  for the index
# - Uses raw_daily.json ->  columns mapping (exact Ravenswood names) when present
# - Falls back to data_layout scanning when a name is not in the mapping

import os, sys, json, traceback, warnings, re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

# ========= Fixed roots relative to this file =========
BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT_METRICS = BASE / "metricsDataFiles"

# ========= Settings =========
ELEV_DEFAULT = 270.0  # m AMSL for Ravenswood (psl→ps)
WET_DAY_MM = 1.0  # wet-day threshold for CDD
warnings.filterwarnings("ignore", category=FutureWarning)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def detect_data_files(folder: str) -> str:
    """
    Detect whether raw_daily or raw_monthly files exist in folder.
    Returns the base name: "raw_daily" or "raw_monthly"
    Prefers daily if both exist.
    """
    daily_parq = os.path.join(folder, "raw_daily.parquet")
    monthly_parq = os.path.join(folder, "raw_monthly.parquet")
    daily_json = os.path.join(folder, "raw_daily.json")
    monthly_json = os.path.join(folder, "raw_monthly.json")

    # Prefer daily if both exist
    if os.path.isfile(daily_parq) and os.path.isfile(daily_json):
        return "raw_daily"
    elif os.path.isfile(monthly_parq) and os.path.isfile(monthly_json):
        return "raw_monthly"
    else:
        raise SystemExit(
            f"Missing data files in {folder}. Need either raw_daily.parquet+json or raw_monthly.parquet+json")


# ---------- scenario discovery + menu ----------
def list_scenarios(root_metrics: str) -> List[str]:
    if not os.path.isdir(root_metrics):
        raise SystemExit(f"Invalid ROOT_METRICS: {root_metrics}")
    names = [n for n in os.listdir(root_metrics)
             if os.path.isdir(os.path.join(root_metrics, n)) and not n.startswith(".")]
    return sorted(names)


def prompt_select_scenarios(scens: list[str]) -> list[str]:
    if not scens:
        raise SystemExit("No scenarios found.")
    while True:
        print("\nSelect scenarios to process:")
        for i, s in enumerate(scens, 1):
            print(f"{i}. {s}")
        print(f"{len(scens) + 1}. All")
        raw = input("Enter number(s) (e.g. 1 or 1,3 or All): ").strip().lower()
        if raw in {"all", "a"} or raw == str(len(scens) + 1):
            return scens
        picks = set()
        for tok in re.split(r"[,\s]+", raw):
            if tok.isdigit():
                k = int(tok)
                if 1 <= k <= len(scens):
                    picks.add(scens[k - 1])
        if picks:
            return sorted(picks)
        print("Please select at least one valid number, or type All.\n")


# ---------------- JSON-driven input helpers ----------------
def load_raw_schema(folder: str, base_name: str) -> dict:
    """Load either raw_daily.json or raw_monthly.json"""
    info_path = os.path.join(folder, f"{base_name}.json")
    if not os.path.isfile(info_path):
        sys.exit(f"Missing sidecar JSON: {info_path}")
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    # Basic validations for updated schema
    if "date" not in info or not isinstance(info["date"], dict) or not info["date"].get("column"):
        sys.exit(f"{base_name}.json must include {{'date': {{'column': '<index column name>'}}}}")
    if "parquet" not in info or "index_name" not in info["parquet"]:
        # tolerate older files but warn
        log(f"⚠ {base_name}.json missing parquet.index_name, will infer from 'date.column'.")

    # For simplified format compatibility, handle both old (with data_layout) and new (without)
    if "data_layout" not in info or "column_order" not in info.get("data_layout", {}):
        # Use parquet.column_order from simplified format
        if "parquet" in info and "column_order" in info["parquet"]:
            log(f"ℹ Using simplified format (parquet.column_order)")
        else:
            sys.exit(f"{base_name}.json missing both data_layout.column_order and parquet.column_order")

    return info


def read_data_with_schema(folder: str, base_name: str) -> Tuple[pd.DataFrame, dict]:
    """Read either raw_daily.parquet or raw_monthly.parquet with its JSON schema"""
    parq = os.path.join(folder, f"{base_name}.parquet")
    if not os.path.isfile(parq):
        sys.exit(f"Input not found: {parq}")

    info = load_raw_schema(folder, base_name)

    # Get column order from either data_layout or parquet section
    if "data_layout" in info and "column_order" in info["data_layout"]:
        order = info["data_layout"]["column_order"]
    else:
        order = info.get("parquet", {}).get("column_order", [])

    df = pd.read_parquet(parq, engine="pyarrow")

    # Set the index using the explicit date column from JSON
    date_col = info["date"]["column"]
    if date_col in df.columns:
        idx = pd.to_datetime(df[date_col], errors="raise")
        df = df.drop(columns=[date_col])
        df.index = idx
    else:
        # Some builds may have been written with an index already
        if str(df.index.name or "").lower() != str(date_col).lower():
            # As a hard requirement per updated contract, the date column must exist
            sys.exit(f"{base_name}.parquet missing required date column '{date_col}'")

    df.index.name = info.get("parquet", {}).get("index_name", date_col)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()

    # Reorder columns to the advertised order when available
    cols_present = [c for c in order if c in df.columns]
    if cols_present:
        df = df[cols_present]

    # Hard check: no missing days or nulls expected per prescreened data
    if df.index.duplicated().any():
        dup_n = int(df.index.duplicated().sum())
        raise ValueError(f"Index has {dup_n} duplicated dates.")
    if df.isna().any().any():
        n = int(df.isna().sum().sum())
        raise ValueError(f"Found {n} null values in raw data.")

    return df, info


def _first_or_none(xs: list[str], cols: set[str]) -> Optional[str]:
    for x in xs:
        if x in cols:
            return x
    return None


def find_col(info: dict, variable: str, region: str, unit_endswith: str | None = None) -> Optional[str]:
    """
    Updated resolver for simplified JSON format:
      1) If region is Ravenswood, prefer explicit mapping in info['columns'].
      2) Else scan data_layout columns by (variable, region, unit suffix) if present.
      3) Fallback to prefix match "<var>_<region>_".
      4) For simplified format, build expected column name from pattern.
    """
    # 1) explicit mapping for Ravenswood from importer (exact column names)
    if region.lower() == "ravenswood":
        mapping: Dict[str, str] = info.get("columns") or {}
        if isinstance(mapping, dict):
            mapped = mapping.get(variable)
            if mapped:
                return mapped

    # 2) scan structured layout (robust for Australia series and any others)
    cols_desc = info.get("data_layout", {}).get("columns", [])
    matches = []
    for c in cols_desc:
        if c.get("variable") == variable and c.get("region") == region:
            if unit_endswith is None or str(c.get("unit", "")).lower().endswith(unit_endswith.lower()):
                matches.append(c["name"])
    if matches:
        return matches[0]

    # 3) last resort: prefix search using parquet column_order
    pref = f"{variable}_{region}_"
    if "data_layout" in info:
        cand = [c["name"] for c in cols_desc if str(c.get("name", "")).startswith(pref)]
    else:
        # Simplified format - use parquet.column_order
        col_order = info.get("parquet", {}).get("column_order", [])
        cand = [c for c in col_order if c.startswith(pref)]

    return cand[0] if cand else None


# ---------- season helpers ----------
def add_season_fields(idx: pd.Index) -> pd.DataFrame:
    dti = pd.to_datetime(idx)
    m = dti.month.values
    season = np.select(
        [np.isin(m, [12, 1, 2]), np.isin(m, [3, 4, 5]), np.isin(m, [6, 7, 8]), np.isin(m, [9, 10, 11])],
        ["DJF", "MAM", "JJA", "SON"],
        default="UNK",
    )
    season = pd.Categorical(season, categories=["DJF", "MAM", "JJA", "SON", "UNK"], ordered=True)
    season_year = dti.year + (m == 12)
    out = pd.DataFrame({"Year": dti.year, "Season": season, "Season_Year": season_year}, index=idx)
    vc = pd.Series(season).value_counts(sort=False)
    log("Season rows → " + ", ".join(f"{k}:{int(v)}" for k, v in vc.items()))
    return out


def max_consecutive(bools: pd.Series) -> int:
    best = run = 0
    for v in bools.astype(bool).to_numpy():
        run = run + 1 if v else 0
        if run > best: best = run
    return int(best)


# ---------- label helpers ----------
_CANON_RX = re.compile(r"^(Temp|Wind|Rain|Humidity) \([^)]+, [^)]+\)$")


def _norm_5day(text: str) -> str:
    return re.sub(r"\b5\s*-\s*Day\b|\b5\s+Day\b", "5-Day", text, flags=re.I)


def canonical(name: str, typ: str, region: str) -> str:
    name = _norm_5day(re.sub(r"\s+", " ", str(name)).strip()).replace("(", "").replace(")", "")
    region = re.sub(r"\s+", " ", str(region)).strip()
    t = str(typ).lower()
    typ_norm = {"temp": "Temp", "wind": "Wind", "rain": "Rain", "humidity": "Humidity"}.get(t, t.title())
    return f"{typ_norm} ({name}, {region})"


def standardise_label(lbl: str) -> str:
    if not isinstance(lbl, str): return lbl
    s = _norm_5day(lbl.strip())
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(?<!\s)\(", " (", s)
    s = re.sub(r"\(\s*", "(", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\s*\)\s*$", ")", s)
    if _CANON_RX.match(s): return s
    m = re.match(r"^(?P<name>[^()]+?) \((?P<typ>Temp|Wind|Rain|Humidity), (?P<loc>[^)]+)\)$", s, flags=re.I)
    if m: return canonical(m["name"], m["typ"], m["loc"])
    m = re.match(r"Average (Temp|Temperature) \((?P<loc>[^)]+)\)$", s, flags=re.I)
    if m: return canonical("Average", "Temp", m["loc"])
    m = re.match(r"Average Wind \((?P<loc>[^)]+)\)$", s, flags=re.I)
    if m: return canonical("Average", "Wind", m["loc"])
    m = re.match(r"Total (Precipitation|Rainfall|Rain) \((?P<loc>[^)]+)\)$", s, flags=re.I)
    if m: return canonical("Total", "Rain", m["loc"])
    m = re.match(r"(Max Day|95th Percentile) \(Wind, (?P<loc>[^)]+)\)$", s, flags=re.I)
    if m: return canonical(m.group(1), "Wind", m["loc"])
    rain_names = r"(Max Day|Min Day|Max 5-Day|Min 5-Day|R10mm|R20mm|CDD)"
    m = re.match(rf"^{rain_names} \((?P<loc>[^)]+)\)$", s, flags=re.I)
    if m: return canonical(_norm_5day(m.group(1)), "Rain", m["loc"])
    if "precip" in s.lower():
        m = re.match(r"^(?P<name>[^()]+) \((?P<loc>[^)]+)\)$", s)
        if m: return canonical(m["name"], "Rain", m["loc"])
    return s


# ---------- humidity / VPD maths (kept; only used if inputs exist) ----------
def _psl_to_ps(psl_pa: pd.Series, tas_degC: pd.Series, z_m: float) -> pd.Series:
    g = 9.80665;
    M = 0.0289644;
    R = 8.314462618;
    L = 0.0065
    T0K = (pd.to_numeric(tas_degC, errors="coerce") + 273.15).clip(lower=200, upper=330)
    expo = (g * M) / (R * L)
    factor = (1.0 - (L * float(z_m)) / T0K) ** expo
    return pd.to_numeric(psl_pa, errors="coerce") * factor


def _sat_vap_pressure_pa(tC: pd.Series) -> pd.Series:
    T = pd.to_numeric(tC, errors="coerce")
    return 610.94 * np.exp((17.625 * T) / (T + 243.04))


def _vap_pressure_from_huss(q_kgkg: pd.Series, p_pa: pd.Series) -> pd.Series:
    q = pd.to_numeric(q_kgkg, errors="coerce")
    p = pd.to_numeric(p_pa, errors="coerce")
    return (q * p) / (0.622 + 0.378 * q)


def _rh_percent(q_kgkg: pd.Series, tas_degC: pd.Series, psl_pa: pd.Series, z_m: float) -> pd.Series:
    ps_pa = _psl_to_ps(psl_pa, tas_degC, z_m)
    e = _vap_pressure_from_huss(q_kgkg, ps_pa)
    es = _sat_vap_pressure_pa(tas_degC)
    return (100.0 * (e / es)).clip(lower=0.0, upper=100.0)


def _vpd_kpa(q_kgkg: pd.Series, tas_degC: pd.Series, psl_pa: pd.Series, z_m: float) -> pd.Series:
    ps_pa = _psl_to_ps(psl_pa, tas_degC, z_m)
    e = _vap_pressure_from_huss(q_kgkg, ps_pa)
    es = _sat_vap_pressure_pa(tas_degC)
    return ((es - e) / 1000.0).clip(lower=0.0)


# ---------- core metrics ----------
def summarise_metrics(df_daily: pd.DataFrame, wet_mm: float, info: dict) -> pd.DataFrame:
    log("Prepare daily frame…")
    df = df_daily.copy()
    df.index = pd.to_datetime(df.index)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Resolve columns using updated JSON
    col_tas_rav = find_col(info, "tas", "Ravenswood", unit_endswith="degC")
    col_tasmax_rav = find_col(info, "tasmax", "Ravenswood", unit_endswith="degC")
    col_pr_rav = find_col(info, "pr", "Ravenswood", unit_endswith="mm_day")
    col_wind_rav = find_col(info, "wind", "Ravenswood", unit_endswith="ms")
    col_huss_rav_gkg = find_col(info, "huss", "Ravenswood", unit_endswith="g_per_kg")
    col_huss_rav_kgkg = find_col(info, "huss", "Ravenswood", unit_endswith="kgkg") or find_col(info, "huss",
                                                                                               "Ravenswood",
                                                                                               unit_endswith="kg/kg")
    col_psl_rav_hpa = find_col(info, "psl", "Ravenswood", unit_endswith="hPa")
    col_psl_rav_pa = find_col(info, "psl", "Ravenswood", unit_endswith="Pa")

    col_pr_au = find_col(info, "pr", "Australia", unit_endswith="mm_day")
    col_tas_au = find_col(info, "tas", "Australia", unit_endswith="degC")
    col_tasmax_au = find_col(info, "tasmax", "Australia", unit_endswith="degC")
    col_wind_au = find_col(info, "wind", "Australia", unit_endswith="ms")

    # Humidity/VPD only if all inputs exist
    q_kgkg = None
    if col_huss_rav_kgkg and col_huss_rav_kgkg in df.columns:
        q_kgkg = pd.to_numeric(df[col_huss_rav_kgkg], errors="coerce")
    elif col_huss_rav_gkg and col_huss_rav_gkg in df.columns:
        q_kgkg = pd.to_numeric(df[col_huss_rav_gkg], errors="coerce") / 1000.0

    psl_pa = None
    if col_psl_rav_pa and col_psl_rav_pa in df.columns:
        psl_pa = pd.to_numeric(df[col_psl_rav_pa], errors="coerce")
    elif col_psl_rav_hpa and col_psl_rav_hpa in df.columns:
        psl_pa = pd.to_numeric(df[col_psl_rav_hpa], errors="coerce") * 100.0

    tas_rav = pd.to_numeric(df[col_tas_rav], errors="coerce") if (col_tas_rav and col_tas_rav in df.columns) else None

    if (q_kgkg is not None) and (psl_pa is not None) and (tas_rav is not None):
        log("Compute RH_Ravenswood_pct and VPD_Ravenswood_kPa…")
        df["RH_Ravenswood_pct"] = _rh_percent(q_kgkg, tas_rav, psl_pa, ELEV_DEFAULT)
        df["VPD_Ravenswood_kPa"] = _vpd_kpa(q_kgkg, tas_rav, psl_pa, ELEV_DEFAULT)
    else:
        missing_bits = []
        if q_kgkg is None: missing_bits.append("huss (any unit)")
        if psl_pa is None: missing_bits.append("psl (Pa or hPa)")
        if tas_rav is None: missing_bits.append("tas_Ravenswood_degC")
        if missing_bits:
            log(f"ℹ RH/VPD skipped.  Missing {missing_bits}")

    # ---- seasonal grouping meta ----
    idx = df.index
    years = np.unique(idx.year)
    log(f"Rows {len(df):,}  years {years.min()}–{years.max()}")
    meta = add_season_fields(idx)

    # ----- metric builders -----
    def add_precip(series: Optional[pd.Series], region: str):
        if series is None: return []
        log(f"[{region}] Precip metrics start…")
        rows = []
        s = pd.to_numeric(series, errors="coerce")
        r5 = s.rolling(5, min_periods=5).sum()
        years_arr = meta["Season_Year"];
        seasons = meta["Season"]
        dry = (s < WET_DAY_MM)
        g = s.groupby([years_arr, seasons], observed=True)
        g5 = r5.groupby([years_arr, seasons], observed=True)
        dryS = dry.groupby([years_arr, seasons], observed=True)
        for (yr, seas), grp in g:
            v = grp.dropna()
            v5 = g5.get_group((yr, seas)).dropna() if (yr, seas) in g5.groups else pd.Series(dtype=float)
            seas_str = str(seas)
            if not v.empty:
                rows += [
                    {"Year": int(yr), "Season": seas_str, "Data Type": canonical("Max Day", "Rain", region),
                     "Value": float(v.max())},
                    {"Year": int(yr), "Season": seas_str, "Data Type": canonical("Min Day", "Rain", region),
                     "Value": float(v.min())},
                    {"Year": int(yr), "Season": seas_str, "Data Type": canonical("Total", "Rain", region),
                     "Value": float(v.sum())},
                    {"Year": int(yr), "Season": seas_str, "Data Type": canonical("R10mm", "Rain", region),
                     "Value": float((v >= 10.0).sum())},
                    {"Year": int(yr), "Season": seas_str, "Data Type": canonical("R20mm", "Rain", region),
                     "Value": float((v >= 20.0).sum())},
                ]
            if not v5.empty:
                rows += [
                    {"Year": int(yr), "Season": seas_str, "Data Type": canonical("Max 5-Day", "Rain", region),
                     "Value": float(v5.max())},
                    {"Year": int(yr), "Season": seas_str, "Data Type": canonical("Min 5-Day", "Rain", region),
                     "Value": float(v5.min())},
                ]
            if (yr, seas) in dryS.groups:
                rows.append({"Year": int(yr), "Season": seas_str, "Data Type": canonical("CDD", "Rain", region),
                             "Value": float(max_consecutive(dryS.get_group((yr, seas))))})
        gY = s.groupby(idx.year);
        gY5 = r5.groupby(idx.year);
        dryY = dry.groupby(idx.year)
        for yr, grp in gY:
            v = grp.dropna()
            v5 = gY5.get_group(yr).dropna() if yr in gY5.groups else pd.Series(dtype=float)
            if not v.empty:
                rows += [
                    {"Year": int(yr), "Season": "Annual", "Data Type": canonical("Max Day", "Rain", region),
                     "Value": float(v.max())},
                    {"Year": int(yr), "Season": "Annual", "Data Type": canonical("Min Day", "Rain", region),
                     "Value": float(v.min())},
                    {"Year": int(yr), "Season": "Annual", "Data Type": canonical("Total", "Rain", region),
                     "Value": float(v.sum())},
                    {"Year": int(yr), "Season": "Annual", "Data Type": canonical("R10mm", "Rain", region),
                     "Value": float((v >= 10.0).sum())},
                    {"Year": int(yr), "Season": "Annual", "Data Type": canonical("R20mm", "Rain", region),
                     "Value": float((v >= 20.0).sum())},
                ]
            if not v5.empty:
                rows += [
                    {"Year": int(yr), "Season": "Annual", "Data Type": canonical("Max 5-Day", "Rain", region),
                     "Value": float(v5.max())},
                    {"Year": int(yr), "Season": "Annual", "Data Type": canonical("Min 5-Day", "Rain", region),
                     "Value": float(v5.min())},
                ]
            if yr in dryY.groups:
                rows.append({"Year": int(yr), "Season": "Annual", "Data Type": canonical("CDD", "Rain", region),
                             "Value": float(max_consecutive(dryY.get_group(yr)))})
        log(f"[{region}] Precip metrics done.")
        return rows

    def add_mean(series: Optional[pd.Series], region: str, label_typ: str):
        if series is None: return []
        rows = []
        s = pd.to_numeric(series, errors="coerce")
        years_arr = meta["Season_Year"];
        seasons = meta["Season"]
        g = s.groupby([years_arr, seasons], observed=True)
        for (yr, seas), grp in g:
            v = grp.dropna()
            if v.empty: continue
            rows.append({"Year": int(yr), "Season": str(seas),
                         "Data Type": canonical("Average", label_typ, region), "Value": float(v.mean())})
        for yr, grp in s.groupby(idx.year):
            v = grp.dropna()
            if not v.empty:
                rows.append({"Year": int(yr), "Season": "Annual",
                             "Data Type": canonical("Average", label_typ, region), "Value": float(v.mean())})
        return rows

    def add_wind_extras(series: Optional[pd.Series], region: str):
        if series is None: return []
        rows = []
        s = pd.to_numeric(series, errors="coerce")
        years_arr = meta["Season_Year"];
        seasons = meta["Season"]
        g = s.groupby([years_arr, seasons], observed=True)
        for (yr, seas), grp in g:
            v = grp.dropna()
            if v.empty: continue
            rows += [
                {"Year": int(yr), "Season": str(seas),
                 "Data Type": canonical("95th Percentile", "Wind", region), "Value": float(np.percentile(v, 95))},
                {"Year": int(yr), "Season": str(seas),
                 "Data Type": canonical("Max Day", "Wind", region), "Value": float(v.max())},
            ]
        for yr, grp in s.groupby(idx.year):
            v = grp.dropna()
            if not v.empty:
                rows += [
                    {"Year": int(yr), "Season": "Annual",
                     "Data Type": canonical("95th Percentile", "Wind", region), "Value": float(np.percentile(v, 95))},
                    {"Year": int(yr), "Season": "Annual",
                     "Data Type": canonical("Max Day", "Wind", region), "Value": float(v.max())},
                ]
        return rows

    def add_temp_metrics(region: str, col_tas: Optional[str], col_tasmax: Optional[str]):
        rows = []
        tas = pd.to_numeric(df[col_tas], errors="coerce") if col_tas and col_tas in df.columns else None
        smax = pd.to_numeric(df[col_tasmax], errors="coerce") if col_tasmax and col_tasmax in df.columns else None
        if tas is not None:
            rows += add_mean(tas, region, "Temp")
        if smax is None:
            return rows
        idx_local = df.index
        meta_local = add_season_fields(idx_local)
        r5 = smax.rolling(5, min_periods=5).mean()
        g = smax.groupby([meta_local["Season_Year"], meta_local["Season"]], observed=True)
        g5 = r5.groupby([meta_local["Season_Year"], meta_local["Season"]], observed=True)
        for (yr, seas), grp in g:
            v = grp.dropna()
            if v.empty: continue
            seas_str = str(seas)
            rows += [
                {"Year": int(yr), "Season": seas_str, "Data Type": canonical("Max Day", "Temp", region),
                 "Value": float(v.max())},
                {"Year": int(yr), "Season": seas_str, "Data Type": canonical("Avg Max", "Temp", region),
                 "Value": float(v.mean())},
            ]
            if (yr, seas) in g5.groups:
                v5 = g5.get_group((yr, seas)).dropna()
                if not v5.empty:
                    rows.append({"Year": int(yr), "Season": seas_str,
                                 "Data Type": canonical("5-Day Avg Max", "Temp", region), "Value": float(v5.max())})
        gY = smax.groupby(idx_local.year);
        gY5 = r5.groupby(idx_local.year)
        for yr, grp in gY:
            v = grp.dropna()
            if v.empty: continue
            rows += [
                {"Year": int(yr), "Season": "Annual", "Data Type": canonical("Max Day", "Temp", region),
                 "Value": float(v.max())},
                {"Year": int(yr), "Season": "Annual", "Data Type": canonical("Avg Max", "Temp", region),
                 "Value": float(v.mean())},
            ]
            if yr in gY5.groups:
                v5 = gY5.get_group(yr).dropna()
                if not v5.empty:
                    rows.append({"Year": int(yr), "Season": "Annual",
                                 "Data Type": canonical("5-Day Avg Max", "Temp", region), "Value": float(v5.max())})
        if tas is not None:
            for thr in (37.0, 40.0):
                hours = _hours_ge_from_tas_tasmax(tas, smax, thr)
                rows += _days_ge_3h(hours, meta_local["Season_Year"], meta_local["Season"], idx_local.year, region,
                                    f"{thr}°C")
        return rows

    def _hours_ge_from_tas_tasmax(tbar: pd.Series, tmax: pd.Series, thr: float) -> pd.Series:
        tb = pd.to_numeric(tbar, errors="coerce")
        tx = pd.to_numeric(tmax, errors="coerce")
        A = (tx - tb).clip(lower=1e-3)
        x = ((thr - tb) / A).clip(-1.0, 1.0)
        frac = 0.5 - np.arcsin(x) / np.pi
        return (24.0 * np.clip(frac, 0.0, 1.0)).astype(float)

    def _days_ge_3h(hours: pd.Series, years_arr, seasons, idx_year, region: str, thr_label: str):
        rows = []
        gS = hours.groupby([years_arr, seasons], observed=True)
        for (yr, seas), grp in gS:
            v = grp.dropna()
            if v.empty: continue
            rows.append({"Year": int(yr), "Season": str(seas),
                         "Data Type": canonical(f"Days ≥{thr_label} ≥3h est", "Temp", region),
                         "Value": float((v >= 3.0).sum())})
        gY = hours.groupby(idx_year)
        for yr, grp in gY:
            v = grp.dropna()
            if v.empty: continue
            rows.append({"Year": int(yr), "Season": "Annual",
                         "Data Type": canonical(f"Days ≥{thr_label} ≥3h est", "Temp", region),
                         "Value": float((v >= 3.0).sum())})
        return rows

    # ---- assemble all metric rows ----
    recs = []
    recs += add_precip(df.get(col_pr_au), "Australia")
    recs += add_precip(df.get(col_pr_rav), "Ravenswood")
    recs += add_temp_metrics("Australia", col_tas_au, col_tasmax_au)
    recs += add_temp_metrics("Ravenswood", col_tas_rav, col_tasmax_rav)
    recs += add_mean(df.get(col_wind_au), "Australia", "Wind")
    recs += add_mean(df.get(col_wind_rav), "Ravenswood", "Wind")
    recs += add_wind_extras(df.get(col_wind_au), "Australia")
    recs += add_wind_extras(df.get(col_wind_rav), "Ravenswood")
    if "RH_Ravenswood_pct" in df.columns:
        recs += add_mean(df["RH_Ravenswood_pct"], "Ravenswood", "Humidity")
    if "VPD_Ravenswood_kPa" in df.columns:
        recs += add_mean(df["VPD_Ravenswood_kPa"], "Ravenswood", "Humidity")

    out = pd.DataFrame.from_records(recs).sort_values(["Year", "Season", "Data Type"]).reset_index(drop=True)
    before = out["Data Type"].astype(str)
    out["Data Type"] = out["Data Type"].map(standardise_label)
    fixes = int((before != out["Data Type"]).sum())
    not_canon = sorted(set([t for t in out["Data Type"].unique() if not _CANON_RX.match(str(t))]))
    log(f"Metric labels unified → {fixes} renamed")
    if not_canon:
        log("⚠ Not canonical: " + " | ".join(not_canon))
    log(f"Metrics rows {out.shape[0]:,}")
    return out


# ---------------- schema for metrics.json ----------------
def describe_df_schema(df: pd.DataFrame) -> dict:
    cols = []
    for c in df.columns:
        s = df[c]
        cols.append({
            "name": c,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "nulls": int(s.isna().sum()),
            "min": (None if not pd.api.types.is_numeric_dtype(s) or s.dropna().empty
                    else float(np.nanmin(pd.to_numeric(s, errors="coerce").dropna().values))),
            "max": (None if not pd.api.types.is_numeric_dtype(s) or s.dropna().empty
                    else float(np.nanmax(pd.to_numeric(s, errors="coerce").dropna().values))),
        })
    return {"schema_version": "1.0", "index": {"name": "row", "kind": "tabular"}, "columns": cols,
            "rows": int(df.shape[0])}


# ---------- per-scenario driver ----------
def process_scenario(scen_name: str, scen_dir: str):
    log(f"=== Scenario {scen_name} ===")

    # Detect which file type is present
    base_name = detect_data_files(scen_dir)
    log(f"Detected input: {base_name}.parquet + {base_name}.json")

    in_parq = os.path.join(scen_dir, f"{base_name}.parquet")
    in_json = os.path.join(scen_dir, f"{base_name}.json")
    out_met = os.path.join(scen_dir, "metrics.parquet")
    out_csv = os.path.join(scen_dir, "metrics.csv")
    out_json = os.path.join(scen_dir, "metrics.json")

    size_mb = os.path.getsize(in_parq) / (1024 * 1024)
    log(f"Read {in_parq}  ({size_mb:.1f} MB)")
    df_data, info = read_data_with_schema(scen_dir, base_name)

    temporal_res = info.get("temporal_resolution", "daily")
    log(f"Loaded {temporal_res} data via {base_name}.json schema")

    log(f"Wet-day threshold {WET_DAY_MM} mm")
    metrics = summarise_metrics(df_data, WET_DAY_MM, info)

    log(f"Write {out_met}")
    metrics.to_parquet(out_met, engine="pyarrow", compression="zstd", index=False)
    log(f"Write {out_csv}")
    metrics.to_csv(out_csv, index=False)

    log(f"Write {out_json}")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "created": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "source": "metrics_from_parquet_multi",
            "input": {
                "file": f"{base_name}.parquet",
                "temporal_resolution": temporal_res
            },
            "parquet": {"path": os.path.abspath(out_met), "compression": "zstd"},
            "data_layout": describe_df_schema(metrics),
        }, f, indent=2)

    log(f"✅ Scenario {scen_name} complete.")


# ---------- main ----------
def main():
    import argparse
    p = argparse.ArgumentParser(description="Build metrics for scenarios in ./metricsDataFiles/<SCENARIO>")
    p.add_argument("--only", nargs="*", help="Optional list of scenario names to process, e.g. SSP1-26 SSP3-70")
    args = p.parse_args()

    try:
        all_scens = list_scenarios(str(ROOT_METRICS))
        if not all_scens:
            raise SystemExit("No scenario folders found under ./metricsDataFiles")

        if args.only:
            wanted = set(args.only)
            scens = [s for s in all_scens if s in wanted]
            missing = sorted(wanted.difference(scens))
            if missing:
                log(f"⚠ Skipping unknown scenarios: {missing}")
        else:
            scens = prompt_select_scenarios(all_scens)

        if not scens:
            raise SystemExit("No scenarios selected.")

        log(f"Scenarios: {scens}")
        for scen in scens:
            scen_dir = str((ROOT_METRICS / scen).resolve())
            try:
                process_scenario(scen, scen_dir)
            except SystemExit as e:
                log(f"❌ Scenario {scen} failed with exit {e.code}.  Continuing.")
            except Exception as e:
                log(f"❌ Scenario {scen} error: {e}")
                traceback.print_exc()
        log("🏁 All scenarios processed.")
    except SystemExit:
        raise
    except Exception as e:
        log(f"❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()