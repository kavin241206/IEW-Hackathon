"""
FIELDWISE AI — AI production optimisation for mature oil & gas fields
=====================================================================
IEW 2027 Hackathon · Problem Statement 3
"AI-Based Production Optimization and Reservoir Modelling for Mature Fields"

Run:   pip install -r requirements.txt
       streamlit run app.py

The app stays idle until a CSV / Excel file is uploaded. Then it:
  1. maps & cleans the data (auditable report)
  2. trains an AI virtual flow meter (oil-rate model) and validates it on unseen dates
  3. fits decline curves and forecasts 180 days
  4. learns gas-lift performance curves and optimises lift-gas allocation
  5. screens artificial-lift equipment and raises early warnings
  6. ranks wells for engineering attention, with reasons

Layout: 1 config · 2 ingestion & cleaning · 3 analytics · 4 optimiser · 5 UI kit · 6 pages · 7 main
Internal units: liquids m³/d · gas Sm³/d · pressure bar · choke size 1/64 in
"""
from __future__ import annotations

import hashlib
import html
import io
import re
from dataclasses import astuple, dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
APP = "FIELDWISE AI"
AUTHOR = "Kavinkarthick"

CORE_COLUMNS = ["Date", "Well", "Oil_Rate", "Water_Rate", "Gas_Rate",
                "Wellhead_Pressure", "Bottomhole_Pressure", "Injection_Rate"]
SENSOR_COLUMNS = ["Pump_Vibration", "Motor_Temperature", "Motor_Current", "Pump_Frequency", "Energy_kWh"]
OPTIONAL_COLUMNS = ["Choke_Size"] + SENSOR_COLUMNS
CORE_NUMERIC = CORE_COLUMNS[2:]
NUMERIC_COLUMNS = CORE_NUMERIC + OPTIONAL_COLUMNS
RATE_COLUMNS = ["Oil_Rate", "Water_Rate", "Gas_Rate", "Injection_Rate"]
PRESSURE_COLUMNS = ["Wellhead_Pressure", "Bottomhole_Pressure"]
SETPOINTS = ["Choke_Size"]   # changed in steps: carried forward, never interpolated or capped

DESCRIPTIONS = {
    "Date": "Observation date", "Well": "Unique well ID", "Oil_Rate": "Oil production rate",
    "Water_Rate": "Water production rate", "Gas_Rate": "Produced gas rate",
    "Wellhead_Pressure": "Flowing wellhead / tubing-head pressure", "Bottomhole_Pressure": "Flowing bottom-hole pressure",
    "Injection_Rate": "Lift-gas (or injection) rate", "Choke_Size": "Production choke (bean) size",
    "Pump_Vibration": "Pump vibration", "Motor_Temperature": "Motor temperature", "Motor_Current": "Motor current",
    "Pump_Frequency": "VFD / pump frequency", "Energy_kWh": "Energy used in the interval",
}

ALIASES = {
    "Date": ["date", "datetime", "timestamp", "time", "sample date", "observation date", "production date"],
    "Well": ["well", "well id", "well name", "wellno", "well no", "uwi", "api", "well identifier"],
    "Oil_Rate": ["oil rate", "oil", "oil production", "oil production rate", "qo", "oil m3 d", "oil rate (m3/d)",
                 "oil rate (m³/d)", "bopd", "oil bopd"],
    "Water_Rate": ["water rate", "water", "water production", "water production rate", "qw", "water m3 d",
                   "water rate (m3/d)", "water rate (m³/d)", "bwpd"],
    "Gas_Rate": ["gas rate", "gas", "gas production", "gas production rate", "qg", "gas mscf d", "gas rate (mcf/d)"],
    "Wellhead_Pressure": ["wellhead pressure", "whp", "tubing head pressure", "thp", "well head pressure", "fthp"],
    "Bottomhole_Pressure": ["bottomhole pressure", "bottom hole pressure", "bhp", "flowing bottomhole pressure",
                            "fbhp", "pwf", "downhole pressure"],
    "Injection_Rate": ["injection rate", "injection", "gas lift rate", "lift gas rate", "lift gas", "gas injection",
                       "water injection rate", "qinj", "glir"],
    "Choke_Size": ["choke size", "choke", "bean", "bean size", "choke 64ths", "choke (1/64 in)", "choke (64ths)",
                   "ck", "choke diameter"],
    "Pump_Vibration": ["pump vibration", "vibration", "vibration rms"],
    "Motor_Temperature": ["motor temperature", "motor temp", "motor temperature c"],
    "Motor_Current": ["motor current", "current", "motor amps", "motor current a"],
    "Pump_Frequency": ["pump frequency", "frequency", "vfd frequency", "frequency hz"],
    "Energy_kWh": ["energy kwh", "energy", "power consumption kwh", "energy consumed", "energy consumption"],
}

_LIQ = {"m³/d": 1.0, "bbl/d": 0.1589872949, "STB/d": 0.1589872949, "m³/h": 24.0, "L/s": 86.4, "L/min": 1.44}
_GAS = {"Sm³/d": 1.0, "Mscf/d": 28.3168466, "scf/d": 0.0283168466, "MMSCFD": 28316.8466, "Nm³/h": 24.0}
_PRES = {"bar": 1.0, "kPa": 0.01, "MPa": 10.0, "psi": 0.0689475729, "kgf/cm²": 0.980665}
UNITS = {  # standard = source × factor (°F also gets an offset)
    "Oil_Rate": _LIQ, "Water_Rate": _LIQ, "Gas_Rate": _GAS, "Injection_Rate": {**_GAS, **_LIQ},
    "Wellhead_Pressure": _PRES, "Bottomhole_Pressure": _PRES,
    "Choke_Size": {"1/64 in": 1.0, "in": 64.0, "mm": 64.0 / 25.4},
    "Motor_Temperature": {"°C": 1.0, "°F": 5.0 / 9.0}, "Motor_Current": {"A": 1.0, "mA": 0.001},
    "Pump_Frequency": {"Hz": 1.0, "rpm": 1.0 / 60.0}, "Pump_Vibration": {"mm/s": 1.0, "in/s": 25.4},
    "Energy_kWh": {"kWh": 1.0, "Wh": 0.001, "MWh": 1000.0},
}
STD_UNIT = {"Oil_Rate": "m³/d", "Water_Rate": "m³/d", "Gas_Rate": "Sm³/d", "Injection_Rate": "Sm³/d",
            "Wellhead_Pressure": "bar", "Bottomhole_Pressure": "bar", "Choke_Size": "1/64 in",
            "Motor_Temperature": "°C", "Motor_Current": "A", "Pump_Frequency": "Hz", "Pump_Vibration": "mm/s",
            "Energy_kWh": "kWh"}
MISSING = {"", "nan", "none", "null", "n/a", "na", "unknown", "—", "-", "<na>", "#n/a", "nat"}
M3_TO_BBL = 6.289811
GRID_CO2 = 0.71   # kg CO2 per kWh — Indian grid average; adjustable in Settings

PRIORITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "WATCH": 2, "STABLE": 3}
COLORS = {"teal": "#2DD4BF", "blue": "#60A5FA", "amber": "#FBBF24", "red": "#F87171", "violet": "#A78BFA",
          "muted": "#64748B", "grid": "rgba(148,163,184,0.12)", "text": "#CBD5E1"}
PRIORITY_COLORS = {"CRITICAL": COLORS["red"], "HIGH": "#FB923C", "WATCH": COLORS["amber"], "STABLE": COLORS["teal"]}


# =============================================================================
# 2. INGESTION & CLEANING
# =============================================================================
def _norm(name) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").replace("-", " ").split())


def _strip_units(name) -> str:
    """'Oil Rate (bbl/d)' → 'oil rate';  'THP [psi]' → 'thp';  'Choke_in' stays recognisable."""
    return _norm(re.sub(r"[\(\[\{].*?[\)\]\}]", " ", str(name)))


_LOOKUP = {_norm(a): canon for canon, names in ALIASES.items() for a in [canon, *names]}
# Keyword fallback for headers no alias list anticipated (first match wins, most specific first)
_KEYWORDS = [
    ("Choke_Size", ("choke", "bean")), ("Injection_Rate", ("lift gas", "gas lift", "injection", "glir")),
    ("Bottomhole_Pressure", ("bottom hole", "bottomhole", "bhp", "downhole press", "pwf")),
    ("Wellhead_Pressure", ("wellhead", "well head", "tubing head", "thp", "whp")),
    ("Pump_Vibration", ("vibration",)), ("Motor_Temperature", ("motor temp", "winding temp")),
    ("Motor_Current", ("amps", "current")), ("Pump_Frequency", ("frequency", "hz", "vfd")),
    ("Energy_kWh", ("kwh", "energy", "power")), ("Water_Rate", ("water", "bwpd", "qw")),
    ("Gas_Rate", ("gas", "qg", "mscf")), ("Oil_Rate", ("oil", "bopd", "qo", "crude")),
    ("Well", ("well",)), ("Date", ("date", "time", "day")),
]
# Unit hints found in header text → unit option
_UNIT_HINTS = [
    (r"\bmmscf", "MMSCFD"), (r"\bmscf|\bmcf", "Mscf/d"), (r"\bscf", "scf/d"), (r"nm3|nm³", "Nm³/h"),
    (r"\bbopd|\bbwpd|\bbbl|\bstb", "bbl/d"), (r"m3\s*/\s*h|m³\s*/\s*h", "m³/h"), (r"\bl\s*/\s*s\b", "L/s"),
    (r"\bpsi", "psi"), (r"\bkpa", "kPa"), (r"\bmpa", "MPa"), (r"kgf|kg\s*/\s*cm", "kgf/cm²"), (r"\bbar", "bar"),
    (r"°\s*f|\bdeg\s*f\b|\bdegf\b|fahrenheit", "°F"), (r"°\s*c|\bdeg\s*c\b|celsius", "°C"), (r"\brpm", "rpm"),
    (r"\bmwh", "MWh"), (r"\bkwh", "kWh"), (r"\bwh\b", "Wh"), (r"in\s*/\s*s|ips", "in/s"),
    (r"64", "1/64 in"), (r"\binch|\bin\b|\"", "in"), (r"\bmm\b", "mm"), (r"\bma\b", "mA"),
]


def detect_unit(header: str, field: str):
    """Guess the source unit from the header text, e.g. 'THP (psi)' → psi. None if no confident hint."""
    text = str(header).lower()
    opts = UNITS.get(field, {})
    for pattern, unit in _UNIT_HINTS:
        if unit in opts and re.search(pattern, text):
            return unit
    return None


def map_columns(raw: pd.DataFrame):
    """Return (frame with FIELDWISE columns, {field: source header}, ignored headers)."""
    rename, mapped = {}, {}
    for col in raw.columns:                         # 1) exact alias, with or without unit text
        canon = _LOOKUP.get(_norm(col)) or _LOOKUP.get(_strip_units(col))
        if canon and canon not in mapped:
            rename[col], mapped[canon] = canon, str(col)
    for canon, words in _KEYWORDS:                  # 2) keyword fallback, only if unambiguous
        if canon in mapped:
            continue
        hits = [c for c in raw.columns if c not in rename and any(w in _strip_units(c) for w in words)
                and not ("ratio" in _norm(c) or "cut" in _norm(c) or "cum" in _norm(c).split())]
        if len(hits) == 1:
            rename[hits[0]], mapped[canon] = canon, str(hits[0])
    frame = raw.rename(columns=rename)
    keep = [c for c in CORE_COLUMNS + OPTIONAL_COLUMNS if c in frame.columns]
    return frame[keep].copy(), mapped, [str(c) for c in raw.columns if c not in rename]


def _to_number(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.mask(s.str.lower().isin(MISSING))
    return pd.to_numeric(s, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)


def clean_data(raw: pd.DataFrame, unit_choices: dict):
    """Auditable cleaning. Returns (clean frame | None, report)."""
    frame, mapped, ignored = map_columns(raw)
    rep = {"input_rows": len(raw), "mapped": mapped, "ignored": ignored, "bad_keys": 0, "duplicates": 0,
           "invalid": 0, "imputed": {}, "capped": 0, "dropped_optional": [], "conversions": [],
           "injection_missing": False}
    if "Injection_Rate" not in frame.columns:
        frame["Injection_Rate"] = 0.0
        rep["injection_missing"] = True
    missing = [c for c in CORE_COLUMNS if c not in frame.columns]
    if missing:
        rep["missing"] = missing
        return None, rep

    try:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce", format="mixed")
    except (TypeError, ValueError):
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    well = frame["Well"].astype(str).str.strip().str.upper()
    frame["Well"] = well.mask(well.isin({"", "NAN", "NONE", "<NA>", "NULL"}))
    n0 = len(frame)
    frame = frame.dropna(subset=["Date", "Well"])
    rep["bad_keys"] = n0 - len(frame)

    numeric = [c for c in NUMERIC_COLUMNS if c in frame.columns]
    miss_frac = {}
    for col in numeric:
        frame[col] = _to_number(frame[col])
        miss_frac[col] = float(frame[col].isna().mean())

    for col, unit in unit_choices.items():
        if col not in frame.columns or unit not in UNITS.get(col, {}):
            continue
        target = "m³/d" if (col == "Injection_Rate" and unit in _LIQ) else STD_UNIT[col]
        if unit == target:
            continue
        offset = -32.0 * 5.0 / 9.0 if unit == "°F" else 0.0
        frame[col] = frame[col] * UNITS[col][unit] + offset
        rep["conversions"].append({"Field": col, "From": unit, "To": target})

    for col in [c for c in RATE_COLUMNS + PRESSURE_COLUMNS + SETPOINTS if c in frame.columns]:
        bad = frame[col] < 0
        rep["invalid"] += int(bad.sum())
        frame.loc[bad, col] = np.nan

    for col in [c for c in OPTIONAL_COLUMNS if c in frame.columns]:
        if miss_frac.get(col, 0) > 0.6:
            frame = frame.drop(columns=col)
            numeric.remove(col)
            rep["dropped_optional"].append(col)

    n0 = len(frame)
    frame = frame.drop_duplicates()
    if frame.duplicated(["Well", "Date"]).any():
        frame = frame.groupby(["Well", "Date"], as_index=False)[numeric].mean()
    rep["duplicates"] = n0 - len(frame)
    frame = frame.sort_values(["Well", "Date"]).reset_index(drop=True)

    # Imputation (fully vectorised): setpoints carried forward; measurements linearly interpolated within each well
    wells = frame["Well"]
    pos = pd.Series(np.arange(len(frame), dtype=float), index=frame.index)
    for col in numeric:
        n_miss = int(frame[col].isna().sum())
        if not n_miss:
            continue
        if col in SETPOINTS:
            frame[col] = frame.groupby("Well")[col].ffill()
            frame[col] = frame.groupby("Well")[col].bfill()
        else:
            ok = frame[col].notna()
            prev_v = frame[col].groupby(wells).ffill()
            next_v = frame[col].groupby(wells).bfill()
            prev_p = pos.where(ok).groupby(wells).ffill()
            next_p = pos.where(ok).groupby(wells).bfill()
            span = (next_p - prev_p).replace(0, np.nan)
            interp = prev_v + (next_v - prev_v) * (pos - prev_p) / span
            frame[col] = frame[col].fillna(interp).fillna(prev_v).fillna(next_v)
            if col in CORE_NUMERIC:
                frame[col] = frame[col].fillna(frame[col].median())
        rep["imputed"][col] = n_miss - int(frame[col].isna().sum())

    # Per-well 3×IQR capping (vectorised) — measurements only, never setpoints
    cap_cols = [c for c in numeric if c not in SETPOINTS]
    if cap_cols:
        grp = frame.groupby("Well")[cap_cols]
        q = grp.quantile([0.25, 0.75])
        q1 = q.xs(0.25, level=1).reindex(wells).to_numpy()
        q3 = q.xs(0.75, level=1).reindex(wells).to_numpy()
        iqr = q3 - q1
        enough = (grp.transform("count").to_numpy() >= 10) & (iqr > 0)
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        nonneg = np.array([c in RATE_COLUMNS + PRESSURE_COLUMNS for c in cap_cols])
        lo = np.where(nonneg, np.maximum(lo, 0.0), lo)
        vals = frame[cap_cols].to_numpy(dtype=float)
        out = enough & ((vals < lo) | (vals > hi))
        rep["capped"] = int(out.sum())
        frame[cap_cols] = np.where(out, np.clip(vals, lo, hi), vals)

    frame = frame.dropna(subset=CORE_NUMERIC).reset_index(drop=True)
    if len(frame) < 30:
        rep["error"] = f"Only {len(frame)} usable rows after cleaning — at least 30 are needed."
        return None, rep
    if frame["Date"].nunique() < 10:
        rep["error"] = "At least 10 distinct dates are needed to train and validate the models."
        return None, rep
    rep["output_rows"] = len(frame)
    return frame, rep


def _read_excel(data: bytes) -> dict:
    """Fastest available Excel engine: calamine (Rust, ~10× faster) → openpyxl."""
    last = None
    for engine in ("calamine", "openpyxl"):
        try:
            return pd.read_excel(io.BytesIO(data), sheet_name=None, engine=engine)
        except (ImportError, ValueError) as exc:
            last = exc
    if isinstance(last, ImportError):
        raise ValueError("Excel support is not installed on this server (add 'openpyxl' to requirements.txt), "
                         "or upload the data as CSV.") from last
    raise ValueError(f"Could not read the workbook ({last}). Try saving it as .xlsx or CSV.") from last


def _read_csv(data: bytes) -> pd.DataFrame:
    """Fastest available CSV parser: pyarrow (multi-threaded) → pandas C engine; falls back on odd encodings."""
    try:
        import pyarrow  # noqa: F401
        return pd.read_csv(io.BytesIO(data), engine="pyarrow")
    except Exception:  # noqa: BLE001 — any pyarrow problem → robust default parser
        pass
    try:
        return pd.read_csv(io.BytesIO(data), low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(data), encoding="cp1252", low_memory=False)


# Heavy results are cached with cache_resource and keyed by a cheap file token (not by hashing the data):
# no hashing of large DataFrames and no copying of results on every Streamlit rerun.
@st.cache_resource(show_spinner=False, max_entries=3)
def read_file(key: str, _data: bytes, name: str) -> dict:
    """Return {sheet name: frame}. A CSV gives a single 'data' sheet."""
    if not _data:
        raise ValueError("The file is empty.")
    if name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        sheets = {k: v for k, v in _read_excel(_data).items() if not v.empty}
        if not sheets:
            raise ValueError("The workbook has no non-empty sheets.")
        return sheets
    try:
        return {"data": _read_csv(_data)}
    except pd.errors.EmptyDataError as exc:
        raise ValueError("No readable columns found in the CSV.") from exc


@st.cache_resource(show_spinner=False, max_entries=3)
def clean_cached(key: str, _sheets: dict, sheet: str, units: tuple):
    return clean_data(_sheets[sheet], dict(units))


def template_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-02"], "Well": ["W-001", "W-001"], "Oil_Rate": [85.0, 84.6],
        "Water_Rate": [174.0, 175.2], "Gas_Rate": [7200.0, 7150.0], "Wellhead_Pressure": [42.0, 41.8],
        "Bottomhole_Pressure": [118.0, 117.6], "Injection_Rate": [14000, 14200], "Choke_Size": [28, 28],
        "Pump_Vibration": [2.1, 2.2], "Motor_Temperature": [72.0, 72.4], "Motor_Current": [54.0, 53.5],
        "Pump_Frequency": [48.0, 48.0], "Energy_kWh": [390.0, 388.0]})


def schema_frame() -> pd.DataFrame:
    return pd.DataFrame([{"Column": c, "Needed": "Required" if c in CORE_COLUMNS else "Optional",
                          "Unit": STD_UNIT.get(c, "—"), "Meaning": DESCRIPTIONS[c]}
                         for c in CORE_COLUMNS + OPTIONAL_COLUMNS])


@st.cache_data(show_spinner=False)
def template_xlsx() -> bytes | None:
    """Excel template; None if no Excel writer is installed (the CSV template still works)."""
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            template_frame().to_excel(xw, index=False, sheet_name="Production")
            schema_frame().to_excel(xw, index=False, sheet_name="Guide")
        return buf.getvalue()
    except ImportError:
        return None


@dataclass
class Field:
    df: pd.DataFrame
    name: str
    report: dict
    lift_is_gas: bool
    key: str = ""

    @property
    def sensors(self):
        return [c for c in SENSOR_COLUMNS if c in self.df.columns]

    @property
    def has_choke(self):
        return "Choke_Size" in self.df.columns

    @property
    def has_lift(self):
        return not self.report.get("injection_missing") and self.df["Injection_Rate"].sum() > 0

    @property
    def inj_unit(self):
        return "Sm³/d" if self.lift_is_gas else "m³/d"


# =============================================================================
# 3. ANALYTICS
# =============================================================================
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["Well", "Date"]).copy()
    d["Liquid_Rate"] = d["Oil_Rate"] + d["Water_Rate"]
    d["Water_Cut"] = (100 * d["Water_Rate"] / d["Liquid_Rate"].replace(0, np.nan)).fillna(0.0)
    d["Delta_P"] = d["Bottomhole_Pressure"] - d["Wellhead_Pressure"]
    d["Days"] = (d["Date"] - d.groupby("Well")["Date"].transform("min")).dt.total_seconds() / 86400
    return d.reset_index(drop=True)


# ---- 3a. AI virtual flow meter ------------------------------------------------
def _add_reference(d: pd.DataFrame, interval: int) -> pd.DataFrame:
    """Periodic separator well tests: every `interval` observations the measured rate is known.
    In between, the model infers the rate from pressures, choke, lift gas and pump signals."""
    d = d.copy()
    test = d.groupby("Well").cumcount() % interval == 0
    for src, ref in [("Oil_Rate", "Ref_Oil"), ("Water_Cut", "Ref_WC"), ("Liquid_Rate", "Ref_Liquid"),
                     ("Bottomhole_Pressure", "Ref_BHP"), ("Wellhead_Pressure", "Ref_WHP"), ("Days", "Ref_Day")]:
        d[ref] = d[src].where(test)
        d[ref] = d.groupby("Well")[ref].ffill()
    d["Days_Since_Test"] = d["Days"] - d["Ref_Day"]
    d["dBHP_Since_Test"] = d["Bottomhole_Pressure"] - d["Ref_BHP"]
    d["dWHP_Since_Test"] = d["Wellhead_Pressure"] - d["Ref_WHP"]
    d["Well_Code"] = d["Well"].astype("category").cat.codes
    return d


def vfm_features(d: pd.DataFrame) -> list:
    f = ["Wellhead_Pressure", "Bottomhole_Pressure", "Delta_P", "Injection_Rate", "Days", "Well_Code",
         "Ref_Oil", "Ref_WC", "Ref_Liquid", "Days_Since_Test", "dBHP_Since_Test", "dWHP_Since_Test"]
    if "Choke_Size" in d.columns:
        f.insert(3, "Choke_Size")
    return f + [c for c in ("Motor_Current", "Pump_Frequency", "Motor_Temperature") if c in d.columns]


class VFMEnsemble:
    """Gradient boosting + extra trees; both learn oil as a multiple of the last well-test rate.
    Sized for speed: same blind-test accuracy as a 4× larger ensemble at ~1/6 of the training time."""

    def __init__(self, features, n_wells):
        self.features = features
        cat = [features.index("Well_Code")] if n_wells <= 250 else None
        self.gb = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                                l2_regularization=1.0, categorical_features=cat, random_state=42)
        self.et = ExtraTreesRegressor(n_estimators=60, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=42)

    def fit(self, d: pd.DataFrame):
        y = d["Oil_Rate"] / d["Ref_Oil"].clip(lower=1.0)
        self.gb.fit(d[self.features], y)
        sub = d.sample(min(len(d), 40000), random_state=42)
        self.et.fit(sub[self.features], y.loc[sub.index])
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        ratio = 0.5 * self.gb.predict(d[self.features]) + 0.5 * self.et.predict(d[self.features])
        return np.maximum(ratio, 0) * d["Ref_Oil"].clip(lower=1.0).to_numpy()


def _scores(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = np.abs(y - p)
    return {"R²": r2_score(y, p) if len(y) > 1 else np.nan, "MAE (m³/d)": mean_absolute_error(y, p),
            "Accuracy (%)": 100 * (1 - err.sum() / max(np.abs(y).sum(), 1e-9)),
            "Within ±10 % (%)": 100 * np.mean(err <= 0.10 * np.maximum(np.abs(y), 1e-9))}


@st.cache_resource(show_spinner=False, max_entries=3)
def train_vfm(key: str, _df: pd.DataFrame, interval: int):
    """Chronological validation (last 20 % of dates unseen) + final model on all history."""
    d = _add_reference(enrich(_df), interval).dropna(subset=["Ref_Oil"])
    feats = vfm_features(d)
    d = d.dropna(subset=feats)
    split = d["Date"].quantile(0.8)
    train, test = d[d["Date"] <= split], d[d["Date"] > split]
    if len(train) < 20 or len(test) < 5:
        raise ValueError("Not enough history for a chronological train/test split.")
    n_wells = d["Well"].nunique()
    val_model = VFMEnsemble(feats, n_wells).fit(train)
    test = test.assign(Pred=val_model.predict(test))
    metrics = pd.DataFrame([
        {"Dataset": "Training (seen)", **_scores(train["Oil_Rate"], val_model.predict(train)), "Rows": len(train)},
        {"Dataset": "Validation (unseen later dates)", **_scores(test["Oil_Rate"], test["Pred"]), "Rows": len(test)},
        {"Dataset": "Baseline · carry last well test", **_scores(test["Oil_Rate"], test["Ref_Oil"]), "Rows": len(test)},
    ])
    importance = pd.DataFrame({"Feature": feats, "Importance": val_model.et.feature_importances_})

    final = VFMEnsemble(feats, n_wells).fit(d)
    d = d.assign(Pred=final.predict(d), Split=np.where(d["Date"] <= split, "Train", "Validation"), Pred_Unseen=np.nan)
    d.loc[test.index, "Pred_Unseen"] = test["Pred"]
    d["_ae"], d["_aeu"] = (d["Oil_Rate"] - d["Pred"]).abs(), (d["Oil_Rate"] - d["Pred_Unseen"]).abs()
    d["_yu"] = d["Oil_Rate"].where(d["Pred_Unseen"].notna())
    agg = d.groupby("Well", sort=True)[["_ae", "Oil_Rate", "_aeu", "_yu"]].sum(min_count=1)
    per_well = pd.DataFrame({"Well": agg.index,
                             "History accuracy (%)": 100 * (1 - agg["_ae"] / agg["Oil_Rate"].clip(lower=1e-9)).to_numpy(),
                             "Unseen accuracy (%)": 100 * (1 - agg["_aeu"] / agg["_yu"].clip(lower=1e-9)).to_numpy()})
    return {"metrics": metrics, "history": _scores(d["Oil_Rate"], d["Pred"]),
            "pred": d[["Date", "Well", "Oil_Rate", "Pred", "Pred_Unseen", "Split"]].reset_index(drop=True),
            "importance": importance.sort_values("Importance"), "per_well": per_well, "split": split}


# ---- 3b. Decline curve analysis (Arps) --------------------------------------------
def _arps(t, qi, di, b):
    return qi / np.power(1 + b * di * t, 1 / b)


@st.cache_resource(show_spinner=False, max_entries=3)
def fit_declines(key: str, _df: pd.DataFrame, horizon: int = 180):
    df = _df
    params, curves = [], []
    notes = {}
    for well, g in df.sort_values("Date").groupby("Well", sort=True):
        if "Choke_Size" in g.columns:   # a bean change starts a new decline segment
            change = np.flatnonzero(g["Choke_Size"].diff().fillna(0).to_numpy() != 0)
            if len(change) and len(g) - change[-1] >= 30:
                g = g.iloc[change[-1]:]
                notes[well] = f"fitted since choke change on {g['Date'].iloc[0]:%d %b %Y}"
        s = g.set_index("Date")["Oil_Rate"].rolling(7, min_periods=1).mean()
        t = (s.index - s.index[0]).total_seconds().to_numpy() / 86400
        y = s.to_numpy()
        if len(y) < 15 or y.max() <= 0:
            continue
        try:
            (qi, di, b), _ = curve_fit(_arps, t, y, p0=[y[:7].mean(), 1e-3, 0.5],
                                       bounds=([1e-6, 1e-7, 0.01], [y.max() * 3, 0.2, 1.0]), maxfev=20000)
        except (RuntimeError, ValueError):
            slope = np.polyfit(t, np.log(np.maximum(y, 1e-3)), 1)[0]
            qi, di, b = y[0], max(-slope, 1e-7), 0.01
        fit = _arps(t, qi, di, b)
        r2 = 1 - np.sum((y - fit) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
        d_now = di / (1 + b * di * t[-1])
        tf = t[-1] + np.arange(1, horizon + 1)
        qf = _arps(tf, qi, di, b)
        params.append({"Well": well, "Current rate (m³/d)": float(_arps(t[-1], qi, di, b)), "b-factor": b,
                       "Decline (%/yr)": 100 * (1 - np.power(1 + b * d_now * 365, -1 / b)),
                       "Next 180 d (m³)": float(qf.sum()), "Fit R²": r2, "Basis": notes.get(well, "full history")})
        curves.append(pd.DataFrame({"Well": well, "Date": s.index, "Fitted": fit}))
        curves.append(pd.DataFrame({"Well": well, "Date": s.index[-1] + pd.to_timedelta(np.arange(1, horizon + 1), "D"),
                                    "Forecast": qf}))
    return pd.DataFrame(params), (pd.concat(curves, ignore_index=True) if curves else pd.DataFrame())


# ---- 3c. Gas-lift performance curves ------------------------------------------------
@dataclass
class LiftCurve:
    """oil = b0 + b1·x + b2·x² + b3·days + b4·choke   (x = lift gas / scale), evaluated today."""
    well: str
    beta: np.ndarray
    scale: float
    days_now: float
    choke_now: float
    x_now: float
    x_min: float
    x_max: float
    r2: float
    shape: str

    def oil(self, inj):
        x = np.asarray(inj, float) / self.scale
        return np.maximum(self.beta[0] + self.beta[1] * x + self.beta[2] * x ** 2 + self.beta[3] * self.days_now
                          + self.beta[4] * self.choke_now, 0.0)

    def marginal(self, inj):
        """Extra oil (m³/d) per 1,000 units of lift gas at this setting."""
        x = np.asarray(inj, float) / self.scale
        return (self.beta[1] + 2 * self.beta[2] * x) / self.scale * 1000


@st.cache_resource(show_spinner=False, max_entries=3)
def fit_lift_curves(key: str, _df: pd.DataFrame, window: int = 150):
    df = _df
    d = enrich(df)
    pos = d.loc[d["Injection_Rate"] > 0, "Injection_Rate"]
    scale = 1000.0 if (len(pos) and pos.median() > 2000) else 1.0
    curves, skipped = {}, {}
    for well, g in d.groupby("Well", sort=True):
        g = g.tail(window)
        x = g["Injection_Rate"].to_numpy() / scale
        if len(g) < 20 or np.std(x) < 1e-6 or x.max() <= 0:
            skipped[well] = "no lift-gas variation to learn from"
            continue
        ch = g["Choke_Size"].to_numpy() if "Choke_Size" in g.columns else np.zeros(len(g))
        y = g["Oil_Rate"].to_numpy()
        X = np.column_stack([np.ones(len(g)), x, x ** 2, g["Days"].to_numpy(), ch])
        keep = [0, 1, 2, 3] + ([4] if np.std(ch) > 1e-6 else [])
        beta = np.zeros(5)
        beta[keep] = np.linalg.lstsq(X[:, keep], y, rcond=None)[0]
        shape = "Concave · diminishing returns"
        if beta[2] >= 0:   # no curvature in the data → linear response, bounded to observed range
            keep = [k for k in keep if k != 2]
            beta = np.zeros(5)
            beta[keep] = np.linalg.lstsq(X[:, keep], y, rcond=None)[0]
            shape = "Linear · within observed range"
        r2 = 1 - np.sum((y - X @ beta) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
        curves[well] = LiftCurve(well, beta, scale, float(g["Days"].iloc[-1]), float(ch[-1]),
                                 float(g["Injection_Rate"].iloc[-1]), float(g["Injection_Rate"].min()),
                                 float(g["Injection_Rate"].max()), float(r2), shape)
    return curves, skipped


# ---- 3d. Equipment health ------------------------------------------------------------
Z_ALERT, Z_WARN = 4.0, 2.5   # robust σ from the well's own healthy baseline


@st.cache_resource(show_spinner=False, max_entries=3)
def screen_equipment(key: str, _df: pd.DataFrame, sensors: tuple):
    df = _df
    """Per-well robust z-scores vs own baseline → Isolation Forest → alert + early warning (trend to threshold)."""
    sensors = list(sensors)
    if len(sensors) < 2:
        return None, None
    d = df[["Date", "Well"] + sensors].dropna().sort_values(["Well", "Date"]).reset_index(drop=True)
    if len(d) < 40:
        return None, None
    z = np.zeros((len(d), len(sensors)))
    base = np.zeros(len(d), bool)
    for _, idx in d.groupby("Well").groups.items():
        g = d.loc[idx, sensors]
        n = max(10, int(0.6 * len(g)))
        b = g.iloc[:n]
        spread = (b.quantile(0.75) - b.quantile(0.25)) / 1.349
        spread = spread.where(spread > 1e-9, b.std()).replace(0, 1).fillna(1)
        z[idx] = ((g - b.median()) / spread).to_numpy()
        base[idx[:n]] = True
    iso = IsolationForest(n_estimators=200, random_state=42).fit(z[base])
    d["Score"] = -iso.score_samples(z)
    thr = float(np.quantile(d.loc[base, "Score"], 0.995))
    calm = float(np.median(d.loc[base, "Score"]))
    d["Alert"] = d["Score"] > thr
    zcols = [f"z_{s}" for s in sensors]
    d[zcols] = z
    rows = []
    for well, g in d.groupby("Well", sort=True):
        now = float(g["Score"].tail(3).mean())
        tail = g.tail(10)
        slope = np.polyfit(np.arange(len(tail)), tail["Score"].to_numpy(), 1)[0] if len(tail) > 4 else 0.0
        days_to = (thr - now) / slope if (slope > 1e-6 and now < thr) else np.nan
        zl = g[zcols].tail(3).mean()
        drv = zl.abs().idxmax()
        z_now = float(abs(zl[drv]))
        z_slope = np.polyfit(np.arange(len(tail)), tail[drv].abs().to_numpy(), 1)[0] if len(tail) > 4 else 0.0
        rising = (np.isfinite(days_to) and days_to <= 14 and now >= calm + 0.7 * (thr - calm)) or \
                 (z_now >= Z_WARN and z_slope > 0)
        status = "ALERT" if (now > thr or z_now >= Z_ALERT) else ("EARLY WARNING" if rising else "NORMAL")
        if status == "EARLY WARNING" and not np.isfinite(days_to) and z_slope > 0:
            days_to = (Z_ALERT - z_now) / z_slope
        rows.append({"Well": well, "Status": status,
                     "Health": float(np.clip(min(100 * (1 - (now - calm) / max(thr - calm, 1e-9)), 100 * (1 - z_now / Z_ALERT)), 0, 100)),
                     "Days to alert": days_to, "Main signal": f"{drv[2:].replace('_', ' ')} {zl[drv]:+.1f}σ",
                     "Score": now})
    return d, pd.DataFrame(rows).assign(Threshold=thr)


# ---- 3e. Well ranking --------------------------------------------------------------------
@dataclass
class Settings:
    wc_watch: float = 70.0
    wc_high: float = 90.0
    decline_watch: float = 15.0
    decline_high: float = 30.0
    test_interval: int = 14
    co2: float = GRID_CO2
    oil_price: float = 70.0


def rank_wells(df, dca, eq, s: Settings) -> pd.DataFrame:
    d = enrich(df)
    latest = d.groupby("Well").tail(1).drop(columns=["Oil_Rate", "Water_Rate", "Liquid_Rate", "Water_Cut"])
    cols = ["Oil_Rate", "Water_Rate", "Liquid_Rate"] + (["Energy_kWh"] if "Energy_kWh" in d.columns else [])
    recent = d.groupby("Well").tail(7).groupby("Well")[cols].mean().reset_index()
    latest = latest.drop(columns=[c for c in cols if c in latest.columns]).merge(recent, on="Well")
    latest["Water_Cut"] = 100 * latest["Water_Rate"] / latest["Liquid_Rate"].replace(0, np.nan)
    if "Energy_kWh" in latest.columns:
        latest["kWh_per_bbl"] = latest["Energy_kWh"] / (latest["Oil_Rate"] * M3_TO_BBL).replace(0, np.nan)
    latest = latest.merge(dca[["Well", "Decline (%/yr)"]], on="Well", how="left") if not dca.empty \
        else latest.assign(**{"Decline (%/yr)": np.nan})
    if eq is not None:
        latest = latest.merge(eq[["Well", "Status", "Main signal", "Days to alert", "Health"]], on="Well", how="left")
    else:
        latest = latest.assign(Status="NO DATA", **{"Main signal": "", "Days to alert": np.nan, "Health": np.nan})

    def score(r):
        pts, why = 0, []
        if r["Water_Cut"] >= s.wc_high:
            pts += 2; why.append(f"water cut {r['Water_Cut']:.0f}%")
        elif r["Water_Cut"] >= s.wc_watch:
            pts += 1; why.append(f"water cut {r['Water_Cut']:.0f}%")
        dec = r["Decline (%/yr)"]
        if pd.notna(dec) and dec >= s.decline_high:
            pts += 2; why.append(f"steep decline {dec:.0f}%/yr")
        elif pd.notna(dec) and dec >= s.decline_watch:
            pts += 1; why.append(f"decline {dec:.0f}%/yr")
        if r["Status"] == "ALERT":
            pts += 3; why.append(f"equipment alert ({r['Main signal']})")
        elif r["Status"] == "EARLY WARNING":
            pts += 2; why.append(f"early warning, ~{r['Days to alert']:.0f} d to alert ({r['Main signal']})")
        level = "CRITICAL" if pts >= 4 else "HIGH" if pts >= 3 else "WATCH" if pts >= 1 else "STABLE"
        return pd.Series({"Score": pts, "Priority": level, "Why": ", ".join(why) or "within limits"})

    latest = pd.concat([latest.reset_index(drop=True), latest.apply(score, axis=1).reset_index(drop=True)], axis=1)
    return latest.sort_values(["Score", "Oil_Rate"], ascending=[False, False]).reset_index(drop=True)


@st.cache_resource(show_spinner=False, max_entries=6)
def rank_cached(key: str, _df, _dca, _eq, _s: Settings, settings_key: tuple):
    return rank_wells(_df, _dca, _eq, _s)


# =============================================================================
# 4. LIFT-GAS OPTIMISER — exact dynamic programme over discretised gas levels
# =============================================================================
def optimise_lift(curves: dict, wells: list, budget: float, min_frac: float, max_frac: float, units: int = 120):
    unit = budget / units
    options = []
    for w in wells:
        c = curves[w]
        lo = max(min_frac * c.x_now, 0.8 * c.x_min)
        hi = max(min(max_frac * c.x_now, 1.15 * c.x_max), lo)
        lv = np.arange(int(np.floor(lo / unit)), int(np.ceil(hi / unit)) + 1)
        options.append((w, lv, c.oil(lv * unit)))
    dp = np.full(units + 1, -np.inf)
    dp[0] = 0.0
    picks = []
    for _, lv, val in options:
        new, arg = np.full(units + 1, -np.inf), np.full(units + 1, -1)
        for i, (lvl, v) in enumerate(zip(lv, val)):
            if lvl > units:
                continue
            cand = np.full(units + 1, -np.inf)
            cand[lvl:] = dp[:units + 1 - lvl] + v
            better = cand > new
            new[better], arg[better] = cand[better], i
        picks.append(arg)
        dp = new
    if not np.isfinite(dp).any():
        return None
    b = int(np.argmax(dp))
    chosen = [0] * len(options)
    for i in range(len(options) - 1, -1, -1):
        chosen[i] = picks[i][b]
        b -= options[i][1][chosen[i]]
    rows = []
    for (w, lv, val), i in zip(options, chosen):
        c = curves[w]
        new_inj = float(lv[i] * unit)
        rows.append({"Well": w, "Current lift gas": c.x_now, "Optimised lift gas": new_inj,
                     "Oil now (model)": float(c.oil(c.x_now)), "Oil optimised": float(val[i]),
                     "Δ Oil": float(val[i] - c.oil(c.x_now)), "Marginal gain / 1k gas": float(c.marginal(new_inj)),
                     "Curve": c.shape})
    return pd.DataFrame(rows)


# =============================================================================
# 5. UI KIT
# =============================================================================
def _w():
    try:
        major, minor = (int(p) for p in st.__version__.split(".")[:2])
        return {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}
    except Exception:  # noqa: BLE001
        return {"use_container_width": True}


W = _w()

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
:root{--bg:#070D14;--panel:#0E1621;--line:#1E2A3A;--text:#E2E8F0;--muted:#94A3B8;--dim:#64748B;
      --teal:#2DD4BF;--amber:#FBBF24;--red:#F87171;--blue:#60A5FA;}
html,body,.stApp{font-family:'Inter',system-ui,sans-serif;}
.stApp{background:radial-gradient(1200px 600px at 90% -10%,rgba(45,212,191,.08),transparent 60%),
       radial-gradient(900px 500px at -10% 110%,rgba(96,165,250,.07),transparent 60%),var(--bg);color:var(--text);}
.block-container{padding:2.2rem 2.2rem 3rem;max-width:1480px;}
header[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"]{background:#0A121C;border-right:1px solid var(--line);}
/* All custom blocks are plain divs (no Streamlit heading anchors), with explicit line-heights → no text overlap */
.fw-brand{display:flex;align-items:center;gap:10px;margin:0 0 14px;}
.fw-logo{width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,#2DD4BF,#0EA5E9);display:grid;
         place-items:center;font-weight:800;color:#04201C;font-size:16px;flex:0 0 36px;}
.fw-brand-name{font-weight:800;letter-spacing:.16em;font-size:13px;color:var(--text);line-height:1.3;}
.fw-brand-sub{font-size:11px;color:var(--dim);line-height:1.3;}
.fw-hero{border:1px solid var(--line);border-radius:22px;padding:36px 40px;margin-bottom:22px;
         background:linear-gradient(135deg,rgba(45,212,191,.12),rgba(14,22,33,.92) 50%),var(--panel);}
.fw-eyebrow{color:var(--teal);font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;line-height:1.4;margin-bottom:12px;}
.fw-hero-title{font-size:clamp(26px,3vw,40px);font-weight:800;line-height:1.18;margin:0 0 14px;color:#F8FAFC;}
.fw-hero-text{font-size:16px;line-height:1.65;color:var(--muted);max-width:880px;margin:0;}
.fw-page{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap;
         margin:0 0 20px;padding-bottom:16px;border-bottom:1px solid var(--line);}
.fw-page-title{font-size:28px;font-weight:800;line-height:1.25;color:#F8FAFC;}
.fw-page-sub{color:var(--muted);font-size:14px;line-height:1.55;margin-top:6px;max-width:860px;}
.fw-pill{display:inline-block;padding:6px 12px;border-radius:999px;font-size:12px;font-weight:600;line-height:1.3;
         background:rgba(45,212,191,.12);color:var(--teal);border:1px solid rgba(45,212,191,.28);white-space:nowrap;}
.fw-kpi{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px 18px;height:116px;
        display:flex;flex-direction:column;justify-content:space-between;overflow:hidden;margin-bottom:14px;}
.fw-kpi-label{color:var(--muted);font-size:11.5px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;line-height:1.3;
              white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.fw-kpi-value{font-family:'JetBrains Mono',monospace;font-size:clamp(17px,1.6vw,25px);font-weight:700;color:#F8FAFC;
              line-height:1.25;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.fw-kpi-sub{color:var(--dim);font-size:12px;line-height:1.3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.fw-kpi.good .fw-kpi-value{color:var(--teal);} .fw-kpi.warn .fw-kpi-value{color:var(--amber);} .fw-kpi.bad .fw-kpi-value{color:var(--red);}
.fw-card-h{margin:20px 0 8px;}
.fw-card-title{font-size:15px;font-weight:700;color:#F1F5F9;line-height:1.35;}
.fw-card-sub{font-size:12.5px;color:var(--dim);line-height:1.4;margin-top:2px;}
.fw-feature{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:20px;min-height:168px;margin-bottom:8px;}
.fw-feature-icon{font-size:22px;line-height:1;margin-bottom:12px;}
.fw-feature-title{font-weight:700;font-size:15px;color:#F1F5F9;line-height:1.35;margin-bottom:6px;}
.fw-feature-text{font-size:13.5px;line-height:1.55;color:var(--muted);}
.fw-step{display:flex;gap:14px;align-items:flex-start;padding:12px 0;border-bottom:1px dashed var(--line);}
.fw-step-n{flex:0 0 30px;height:30px;border-radius:9px;background:rgba(45,212,191,.12);color:var(--teal);font-weight:800;
           display:grid;place-items:center;font-size:13px;}
.fw-step-t{font-weight:600;color:#F1F5F9;font-size:14px;line-height:1.4;}
.fw-step-d{color:var(--muted);font-size:13px;line-height:1.5;margin-top:2px;}
.fw-note{border:1px solid var(--line);border-left:3px solid var(--teal);background:var(--panel);border-radius:10px;
         padding:13px 16px;color:var(--muted);font-size:13.5px;line-height:1.6;margin:12px 0;}
.fw-note.warn{border-left-color:var(--amber);} .fw-note b{color:#F1F5F9;}
.fw-insight{background:linear-gradient(135deg,rgba(45,212,191,.10),rgba(96,165,250,.06));border:1px solid rgba(45,212,191,.25);
            border-radius:14px;padding:14px 18px;margin:4px 0 18px;color:var(--text);font-size:14px;line-height:1.6;}
.fw-insight b{color:var(--teal);}
.fw-foot{color:var(--dim);font-size:11px;text-align:center;margin-top:22px;padding-top:12px;border-top:1px solid var(--line);}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden;}
[data-testid="stPlotlyChart"]{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:8px 8px 2px;}
</style>
"""


# Fragments rerun only the page body when a widget on the page changes (not the whole app).
_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None) or (lambda f: f)


def esc(x) -> str:
    return html.escape(str(x))


def page_header(title, sub="", pill=""):
    pill_html = f'<span class="fw-pill">{esc(pill)}</span>' if pill else ""
    st.markdown(f'<div class="fw-page"><div><div class="fw-page-title">{esc(title)}</div>'
                f'<div class="fw-page-sub">{esc(sub)}</div></div>{pill_html}</div>', unsafe_allow_html=True)


def card_header(title, sub=""):
    sub_html = f'<div class="fw-card-sub">{esc(sub)}</div>' if sub else ""
    st.markdown(f'<div class="fw-card-h"><div class="fw-card-title">{esc(title)}</div>{sub_html}</div>',
                unsafe_allow_html=True)


def kpi(label, value, sub="", tone=""):
    st.markdown(f'<div class="fw-kpi {tone}" title="{esc(label)}: {esc(value)} · {esc(sub)}">'
                f'<div class="fw-kpi-label">{esc(label)}</div><div class="fw-kpi-value">{esc(value)}</div>'
                f'<div class="fw-kpi-sub">{esc(sub)}</div></div>', unsafe_allow_html=True)


def kpi_row(items):
    for col, item in zip(st.columns(len(items)), items):
        with col:
            kpi(*item)


def note(body_html, warn=False):
    st.markdown(f'<div class="fw-note{" warn" if warn else ""}">{body_html}</div>', unsafe_allow_html=True)


def insight(body_html):
    st.markdown(f'<div class="fw-insight">{body_html}</div>', unsafe_allow_html=True)


def chart(fig, height=330, legend=True, **layout):
    """Uniform chart style. Titles sit in card_header (outside the plot), legends get their own top band."""
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color=COLORS["text"]), title=None,
        margin=dict(l=8, r=8, t=40 if legend else 14, b=8), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11), title=None),
        hoverlabel=dict(bgcolor="#0E1621", bordercolor="#1E2A3A", font=dict(family="Inter")),
        colorway=[COLORS[c] for c in ("teal", "blue", "amber", "violet", "red")], **layout)
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False, automargin=True)
    st.plotly_chart(fig, config={"displayModeBar": False}, **W)


def table(frame: pd.DataFrame, decimals=1, height=None):
    out = frame.copy()
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(decimals)
    st.dataframe(out, hide_index=True, **W, **({"height": height} if height else {}))


def fmt(v, spec=",.1f", unit=""):
    try:
        return f"{v:{spec}}{unit}" if v is not None and np.isfinite(v) else "—"
    except (TypeError, ValueError):
        return "—"


# =============================================================================
# 6. PAGES
# =============================================================================
def page_landing():
    st.markdown('<div class="fw-hero"><div class="fw-eyebrow">IEW 2027 · AI production optimisation for mature fields</div>'
                '<div class="fw-hero-title">More barrels, fewer failures, less energy per barrel.</div>'
                '<p class="fw-hero-text">FIELDWISE AI learns every well from its own operating history — pressures, '
                'choke size, lift gas and pump telemetry — then estimates rates between well tests, forecasts decline, '
                're-allocates lift gas for maximum oil and flags artificial-lift problems before they become failures. '
                'Upload your field data in the sidebar to begin.</p></div>', unsafe_allow_html=True)
    feats = [("🛢️", "AI virtual flow meter", "Daily oil rate for every well from pressures, choke and lift gas — tested on dates it never saw."),
             ("📉", "Decline & forecast", "Arps decline curve per well with a 180-day production forecast."),
             ("⚖️", "Lift-gas optimiser", "Learns each well's gas-lift curve and redistributes gas for maximum field oil."),
             ("🛠️", "Equipment early warning", "Per-well anomaly detection on pump signals with projected days-to-alert.")]
    for col, (icon, title, text) in zip(st.columns(4), feats):
        with col:
            st.markdown(f'<div class="fw-feature"><div class="fw-feature-icon">{icon}</div><div class="fw-feature-title">'
                        f'{title}</div><div class="fw-feature-text">{text}</div></div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        card_header("Get started", "everything stays idle until your data arrives")
        steps = [("Download a template", "Excel or CSV — or use your own export; common header names are recognised."),
                 ("Upload in the sidebar", "CSV, XLSX or XLS. Choose the sheet and declare units (psi, bbl/d, inch chokes …)."),
                 ("Check the quality gate", "See exactly what was mapped, converted, repaired or ignored."),
                 ("Act on the insights", "Models train automatically and every page updates from your data.")]
        st.markdown("".join(f'<div class="fw-step"><div class="fw-step-n">{i}</div><div><div class="fw-step-t">{t}</div>'
                            f'<div class="fw-step-d">{d}</div></div></div>' for i, (t, d) in enumerate(steps, 1)),
                    unsafe_allow_html=True)
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            xlsx = template_xlsx()
            if xlsx is not None:
                st.download_button("⬇ Excel template", xlsx, "fieldwise_template.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", **W)
            else:
                st.caption("Excel support not installed — use the CSV template.")
        with c2:
            st.download_button("⬇ CSV template", template_frame().to_csv(index=False).encode(),
                               "fieldwise_template.csv", "text/csv", **W)
    with right:
        card_header("Data columns", "one row = one well on one date · Choke_Size and sensors are optional")
        table(schema_frame(), height=530)


def field_briefing(f: Field, ranked, vfm, dca, curves) -> list:
    """Plain-language interpretation of the analytics — what an engineer would say in a morning meeting."""
    out = []
    daily = f.df.groupby("Date")["Oil_Rate"].sum().sort_index()
    if len(daily) >= 60:
        now, before = daily.tail(14).mean(), daily.iloc[-44:-30].mean()
        chg = (now / before - 1) * 100 if before else 0
        out.append(f"Field oil is <b>{now:,.0f} m³/d</b> (≈ {now * M3_TO_BBL:,.0f} bbl/d), "
                   f"<b>{chg:+.1f}%</b> versus a month earlier.")
    wc = 100 * ranked["Water_Rate"].sum() / max(ranked["Liquid_Rate"].sum(), 1e-9)
    hi_wc = ranked.loc[ranked["Water_Cut"] >= 90, "Well"].tolist()
    out.append(f"Field water cut is <b>{wc:.0f}%</b>" + (f"; {len(hi_wc)} well(s) above 90% ({esc(', '.join(hi_wc[:4]))}"
               f"{'…' if len(hi_wc) > 4 else ''}) are candidates for water shut-off or choke-back." if hi_wc else "."))
    eq_bad = ranked[ranked["Status"].isin(["ALERT", "EARLY WARNING"])]
    if len(eq_bad):
        parts = []
        for _, r in eq_bad.head(4).iterrows():
            days = r["Days to alert"]
            state = "alert" if r["Status"] == "ALERT" else (f"warning (~{days:.0f} d to alert)" if pd.notna(days) else "warning")
            parts.append(f"<b>{esc(r['Well'])}</b> {state}")
        out.append("Artificial-lift equipment needs attention: " + ", ".join(parts) + " — plan inspection before failure.")
    elif f.sensors:
        out.append("No equipment anomalies — all pumps within their own normal range.")
    if curves and f.has_lift:
        res = optimise_lift(curves, sorted(curves), sum(c.x_now for c in curves.values()), 0.5, 1.6)
        if res is not None:
            gain = res["Oil optimised"].sum() - res["Oil now (model)"].sum()
            usd = gain * M3_TO_BBL * 365 * f.report.get("oil_price", 70) / 1e6
            if gain > 0.05:
                out.append(f"Re-allocating the same lift gas could add <b>{gain:+,.1f} m³/d</b> "
                           f"(≈ {gain * M3_TO_BBL:+,.0f} bbl/d, ~${usd:,.2f}M/yr) — see Lift-Gas Optimiser.")
    if not dca.empty:
        worst = dca.sort_values("Decline (%/yr)", ascending=False).head(2)
        out.append(f"Median decline is <b>{dca['Decline (%/yr)'].median():.0f}%/yr</b>; fastest: " +
                   ", ".join(f"{esc(r.Well)} ({r['Decline (%/yr)']:.0f}%/yr)" for _, r in worst.iterrows()) + ".")
    if vfm:
        va = vfm["metrics"].iloc[1]
        out.append(f"The AI rate model is <b>{va['Accuracy (%)']:.1f}%</b> accurate on unseen dates — reliable enough to "
                   f"track every well daily between well tests.")
    if f.report.get("auto_units"):
        out.append("Units auto-detected from headers: " + esc(", ".join(f.report["auto_units"])) + ".")
    return out


def page_overview(f: Field, ranked, vfm, dca, fc, curves=None):
    page_header("Field Command Centre",
                f"{f.name} · {f.df['Date'].min():%d %b %Y} → {f.df['Date'].max():%d %b %Y}",
                f"{f.df['Well'].nunique()} wells · {len(f.df):,} records")
    oil, water = ranked["Oil_Rate"].sum(), ranked["Water_Rate"].sum()
    wc = 100 * water / max(oil + water, 1e-9)
    acc = vfm["metrics"].iloc[1]["Accuracy (%)"] if vfm else np.nan
    alerts, warns = int((ranked["Status"] == "ALERT").sum()), int((ranked["Status"] == "EARLY WARNING").sum())
    kpi_row([("Field oil · 7-day avg", f"{oil:,.0f} m³/d", f"≈ {oil * M3_TO_BBL:,.0f} bbl/d"),
             ("Field water cut", f"{wc:.1f}%", "liquid-weighted", "warn" if wc >= 80 else ""),
             ("AI model accuracy", fmt(acc, ".1f", "%"), "blind test on later dates", "good" if acc >= 90 else "warn"),
             ("Equipment", f"{alerts} alert · {warns} warning" if f.sensors else "no sensors", "artificial lift",
              "bad" if alerts else ("warn" if warns else "good")),
             ("Wells needing action", int(ranked["Priority"].isin(["CRITICAL", "HIGH"]).sum()), "critical + high",
              "bad" if (ranked["Priority"] == "CRITICAL").any() else "")])
    if "kWh_per_bbl" in ranked.columns:
        field_kwh_bbl = ranked["Energy_kWh"].sum() / max(oil * M3_TO_BBL, 1e-9)
        kpi_row([("Energy intensity", fmt(field_kwh_bbl, ".2f", " kWh/bbl"), "field average · ESG"),
                 ("Carbon intensity", fmt(field_kwh_bbl * f.report.get("co2", GRID_CO2), ".2f", " kg CO₂/bbl"), "grid electricity"),
                 ("Field energy", fmt(ranked["Energy_kWh"].sum(), ",.0f", " kWh/d"), "7-day avg"),
                 ("Median decline", fmt(dca["Decline (%/yr)"].median() if not dca.empty else np.nan, ".1f", " %/yr"), "Arps"),
                 ("Lift gas in use", fmt(ranked["Injection_Rate"].sum(), ",.0f", f" {f.inj_unit}"), "latest day")])
    top = ranked.iloc[0]
    bullets = field_briefing(f, ranked, vfm, dca, curves or {})
    insight("<b>AI field briefing</b><ul style='margin:8px 0 0 18px;padding:0'>" +
            "".join(f"<li style='margin:3px 0'>{x}</li>" for x in bullets) +
            f"<li style='margin:3px 0'>Top priority: <b>{esc(top['Well'])}</b> ({esc(top['Priority'])}) — {esc(top['Why'])}.</li></ul>")
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        card_header("Field production · measured vs AI", "daily totals across all wells")
        t = f.df.groupby("Date")[["Oil_Rate", "Water_Rate"]].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t.Date, y=t.Water_Rate, name="Water", line=dict(color=COLORS["blue"], width=1.6)))
        fig.add_trace(go.Scatter(x=t.Date, y=t.Oil_Rate, name="Oil · measured", line=dict(color=COLORS["teal"], width=2.2)))
        if vfm:
            p = vfm["pred"].groupby("Date")["Pred"].sum().reset_index()
            fig.add_trace(go.Scatter(x=p.Date, y=p.Pred, name="Oil · AI", line=dict(color=COLORS["amber"], width=1.6, dash="dot")))
        chart(fig, 370, yaxis_title="m³/d")
    with right:
        card_header("Attention queue", "ranked by water cut, decline and equipment risk")
        table(ranked[["Well", "Priority", "Oil_Rate", "Water_Cut", "Why"]].rename(
            columns={"Oil_Rate": "Oil m³/d", "Water_Cut": "WC %"}), 1, height=388)
    a, b = st.columns(2, gap="large")
    with a:
        card_header("Oil vs water cut", "bubble = liquid rate · colour = priority")
        fig = px.scatter(ranked, x="Water_Cut", y="Oil_Rate", size="Liquid_Rate", color="Priority", text="Well",
                         color_discrete_map=PRIORITY_COLORS, hover_data={"Why": True})
        fig.update_traces(textposition="top center", textfont=dict(size=10))
        chart(fig, 350, xaxis_title="Water cut (%)", yaxis_title="Oil (m³/d)")
    with b:
        card_header("180-day oil outlook", "sum of per-well decline curves")
        if not fc.empty:
            hist = f.df.groupby("Date")["Oil_Rate"].sum().rolling(7, min_periods=1).mean()
            fut = fc.dropna(subset=["Forecast"]).groupby("Date")["Forecast"].sum()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="History · 7-day avg", line=dict(color=COLORS["teal"])))
            fig.add_trace(go.Scatter(x=fut.index, y=fut.values, name="Forecast", line=dict(color=COLORS["amber"], dash="dash")))
            chart(fig, 350, yaxis_title="m³/d")


@_fragment
def page_model(f: Field, vfm):
    page_header("AI Production Model", "Virtual flow meter — daily oil rate for every well from pressures, choke size, "
                "lift gas and pump signals, between periodic well tests.", "Gradient boosting + extra trees")
    if vfm is None:
        st.error("The model could not be trained — the dataset needs more history per well.")
        return
    m = vfm["metrics"]
    tr, va, base = m.iloc[0], m.iloc[1], m.iloc[2]
    gain = (base["MAE (m³/d)"] - va["MAE (m³/d)"]) / max(base["MAE (m³/d)"], 1e-9) * 100
    kpi_row([("Historical match", f"{vfm['history']['Accuracy (%)']:.1f}%", f"R² {vfm['history']['R²']:.4f} · all history", "good"),
             ("Blind-test accuracy", f"{va['Accuracy (%)']:.1f}%", f"R² {va['R²']:.3f} · unseen later dates",
              "good" if va["Accuracy (%)"] >= 90 else "warn"),
             ("Within ±10 %", f"{va['Within ±10 % (%)']:.0f}%", "of blind-test predictions"),
             ("Mean error", f"{va['MAE (m³/d)']:.2f} m³/d", f"≈ {va['MAE (m³/d)'] * M3_TO_BBL:.1f} bbl/d per well-day"),
             ("vs last well test", f"{gain:+.0f}%", "error reduction vs industry practice", "good" if gain > 0 else "warn")])
    insight(f"Trained on <b>{int(tr['Rows']):,}</b> well-days and tested blind on <b>{int(va['Rows']):,}</b> later well-days. "
            f"With a well test every <b>{f.report.get('interval', 14)}</b> observations, the AI fills every day in between "
            f"at <b>{va['Accuracy (%)']:.1f}%</b> volumetric accuracy — fewer separator tests, earlier detection of rate loss.")
    p = vfm["pred"]
    well = st.selectbox("Show", ["All wells · field total"] + sorted(p["Well"].unique()), key="vfm_well")
    g = p if well.startswith("All") else p[p["Well"] == well]
    g = g.groupby("Date")[["Oil_Rate", "Pred", "Pred_Unseen"]].sum(min_count=1).reset_index()
    a, b = st.columns([1.45, 1], gap="large")
    with a:
        card_header("Historical match", "shaded band = blind-test period, never seen during training")
        fig = go.Figure()
        fig.add_vrect(x0=vfm["split"], x1=p["Date"].max(), fillcolor=COLORS["amber"], opacity=0.07, line_width=0)
        fig.add_trace(go.Scatter(x=g.Date, y=g.Oil_Rate, name="Measured", line=dict(color=COLORS["teal"], width=2.2)))
        fig.add_trace(go.Scatter(x=g.Date, y=g.Pred, name="AI · full-history model", line=dict(color=COLORS["amber"], width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=g.Date, y=g.Pred_Unseen, name="AI · blind test", line=dict(color=COLORS["violet"], width=2)))
        chart(fig, 390, yaxis_title="Oil (m³/d)")
    with b:
        card_header("Predicted vs measured", "each dot is one well-day")
        s = p.sample(min(len(p), 4000), random_state=1)
        fig = px.scatter(s, x="Oil_Rate", y="Pred", color="Split", opacity=0.55, render_mode="webgl",
                         color_discrete_map={"Train": COLORS["blue"], "Validation": COLORS["amber"]})
        hi = float(max(s["Oil_Rate"].max(), s["Pred"].max()))
        fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", line=dict(color=COLORS["muted"], dash="dash"),
                                 showlegend=False))
        chart(fig, 390, xaxis_title="Measured (m³/d)", yaxis_title="AI (m³/d)")
    a, b = st.columns([1, 1.25], gap="large")
    with a:
        card_header("What drives the prediction", "share of the model's decisions using each input")
        imp = vfm["importance"].assign(Feature=lambda x: x["Feature"].str.replace("_", " "))
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[COLORS["teal"]])
        chart(fig, 430, legend=False, xaxis_tickformat=".0%", yaxis_title=None)
    with b:
        card_header("Scorecard", "seen data vs unseen data vs carrying the last well test forward")
        table(m, 3)
        card_header("Accuracy by well")
        table(vfm["per_well"], 1, height=250)
    note("<b>Honest accuracy.</b> Field meters carry ~1–3 % uncertainty, so a model that reproduced history at exactly "
         "100 % would be memorising meter noise and fail on new days. FIELDWISE shows both the full-history match and a "
         "blind test on later dates — the number judges and engineers can trust. Same-day oil, water and gas rates are "
         "never used as inputs.")


@_fragment
def page_decline(f: Field, dca, fc):
    page_header("Decline Curves & Forecast", "Arps decline fitted to each well's 7-day average oil rate.", "180-day horizon")
    if dca.empty:
        st.info("Not enough history per well to fit decline curves.")
        return
    worst = dca.loc[dca["Decline (%/yr)"].idxmax()]
    kpi_row([("Field rate now", f"{dca['Current rate (m³/d)'].sum():,.0f} m³/d", "decline-curve basis"),
             ("Median decline", f"{dca['Decline (%/yr)'].median():.1f} %/yr", "effective annual"),
             ("Steepest decline", str(worst["Well"]), f"{worst['Decline (%/yr)']:.1f} %/yr", "warn"),
             ("Next 180 days", f"{dca['Next 180 d (m³)'].sum():,.0f} m³", f"≈ {dca['Next 180 d (m³)'].sum() * M3_TO_BBL / 1e3:,.0f}k bbl"),
             ("Median fit R²", f"{dca['Fit R²'].median():.2f}", "curve quality")])
    well = st.selectbox("Well", sorted(dca["Well"]), key="dca_well")
    a, b = st.columns([1.45, 1], gap="large")
    with a:
        g, fw = f.df[f.df["Well"] == well], fc[fc["Well"] == well]
        card_header(f"{well} · decline curve", "measured · Arps fit · forecast")
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=g.Date, y=g.Oil_Rate, name="Measured", mode="markers",
                                 marker=dict(size=4, color=COLORS["teal"], opacity=.55)))
        fig.add_trace(go.Scatter(x=fw.Date, y=fw.Fitted, name="Arps fit", line=dict(color=COLORS["blue"], width=2)))
        fig.add_trace(go.Scatter(x=fw.Date, y=fw.Forecast, name="Forecast", line=dict(color=COLORS["amber"], width=2, dash="dash")))
        chart(fig, 390, yaxis_title="Oil (m³/d)")
    with b:
        card_header("Annual decline by well", "longer bar = faster production loss")
        fig = px.bar(dca.sort_values("Decline (%/yr)"), x="Decline (%/yr)", y="Well", orientation="h",
                     color="Decline (%/yr)", color_continuous_scale=["#2DD4BF", "#FBBF24", "#F87171"])
        fig.update_coloraxes(showscale=False)
        chart(fig, 390, legend=False, yaxis_title=None)
    card_header("Decline parameters")
    table(dca, 3)


@_fragment
def page_lift(f: Field, curves, skipped):
    page_header("Lift-Gas Optimiser", "Each well's gas-lift performance curve is learned from its history; an exact "
                "optimiser redistributes the available gas to maximise total field oil.", f"lift gas in {f.inj_unit}")
    if not f.has_lift:
        st.info("No lift-gas / injection column was found, so allocation optimisation is not available for this dataset.")
        return
    if not curves:
        st.info("Lift-gas rates do not vary enough in the history to learn each well's response.")
        return
    wells = sorted(curves)
    cur_total = float(sum(curves[w].x_now for w in wells))
    c1, c2, c3 = st.columns([1.3, 1, 1])
    with c1:
        budget = st.number_input(f"Total lift gas available ({f.inj_unit})", min_value=0.0, value=round(cur_total, 0),
                                 step=float(max(1.0, round(cur_total * 0.02))))
    with c2:
        lo = st.slider("Min per well (× current)", 0.0, 1.0, 0.5, 0.05)
    with c3:
        hi = st.slider("Max per well (× current)", 1.0, 3.0, 1.6, 0.1)
    res = optimise_lift(curves, wells, budget, lo, hi) if budget > 0 else None
    if res is None:
        st.warning("No feasible allocation — widen the per-well limits or increase the gas available.")
        return
    now, new = res["Oil now (model)"].sum(), res["Oil optimised"].sum()
    kpi_row([("Oil · current allocation", f"{now:,.1f} m³/d", "model basis"),
             ("Oil · optimised", f"{new:,.1f} m³/d", "same gas limit", "good"),
             ("Extra oil", f"{new - now:+,.1f} m³/d", f"≈ {(new - now) * M3_TO_BBL:+,.0f} bbl/d", "good" if new >= now else "bad"),
             ("Annual value", f"${(new - now) * M3_TO_BBL * 365 * f.report.get('oil_price', 70) / 1e6:+,.2f}M",
              f"at ${f.report.get('oil_price', 70):,.0f}/bbl · {(new / now - 1) * 100 if now else 0:+.1f}% oil", "good" if new >= now else "bad"),
             ("Gas used", f"{res['Optimised lift gas'].sum():,.0f}", f"of {budget:,.0f} {f.inj_unit}")])
    move = (res["Optimised lift gas"] - res["Current lift gas"])
    give, take = res.loc[move.idxmin()], res.loc[move.idxmax()]
    insight(f"Gas moves from low-return to high-return wells until each earns about the same extra oil per unit of gas. "
            f"Largest moves: <b>{esc(give.Well)}</b> {give['Current lift gas']:,.0f} → {give['Optimised lift gas']:,.0f} and "
            f"<b>{esc(take.Well)}</b> {take['Current lift gas']:,.0f} → {take['Optimised lift gas']:,.0f} {f.inj_unit}.")
    pick = st.multiselect("Wells on the curve chart", wells, default=wells[:min(5, len(wells))], key="glpc")
    a, b = st.columns([1.15, 1], gap="large")
    with a:
        card_header("Gas-lift performance curves", "● current setting · ◆ optimised setting")
        fig = go.Figure()
        palette = [COLORS[c] for c in ("teal", "blue", "amber", "violet", "red")] + ["#34D399", "#F472B6", "#E2E8F0"]
        for i, w in enumerate(pick):
            c, col = curves[w], palette[i % len(palette)]
            xs = np.linspace(max(0.0, 0.5 * c.x_min), 1.3 * c.x_max, 60)
            r = res[res.Well == w].iloc[0]
            fig.add_trace(go.Scatter(x=xs, y=c.oil(xs), name=w, line=dict(color=col, width=2)))
            fig.add_trace(go.Scatter(x=[c.x_now], y=[r["Oil now (model)"]], mode="markers", showlegend=False,
                                     marker=dict(color=col, size=9, line=dict(color="#0E1621", width=1))))
            fig.add_trace(go.Scatter(x=[r["Optimised lift gas"]], y=[r["Oil optimised"]], mode="markers", showlegend=False,
                                     marker=dict(color=col, size=12, symbol="diamond", line=dict(color="white", width=1))))
        chart(fig, 410, xaxis_title=f"Lift gas ({f.inj_unit})", yaxis_title="Oil (m³/d)")
    with b:
        card_header("Allocation by well", "current vs optimised")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=res.Well, y=res["Current lift gas"], name="Current", marker_color=COLORS["muted"]))
        fig.add_trace(go.Bar(x=res.Well, y=res["Optimised lift gas"], name="Optimised", marker_color=COLORS["teal"]))
        chart(fig, 410, barmode="group", yaxis_title=f"Lift gas ({f.inj_unit})")
    card_header("Recommendation table", "marginal gain = extra m³/d of oil per additional 1,000 units of lift gas")
    table(res, 3)
    if skipped:
        st.caption("Held at current settings: " + "; ".join(f"{w} ({r})" for w, r in skipped.items()))
    st.download_button("⬇ Download recommendations", res.to_csv(index=False).encode(), "fieldwise_lift_gas_plan.csv", "text/csv")
    note("<b>Engineer in the loop.</b> Apply changes in steps, confirm with a well test and respect compressor, "
         "separator and gas-handling limits. FIELDWISE never sends commands to the field.", warn=True)


@_fragment
def page_equipment(f: Field, rows, summ):
    page_header("Equipment Health", "Each well is compared with its own healthy baseline; an Isolation Forest scores "
                "unusual sensor combinations and the recent trend projects the days left before an alert.", "artificial lift")
    if rows is None:
        st.info("Upload at least two equipment sensor columns (vibration, motor temperature, current, frequency, energy) "
                "to enable equipment screening. No synthetic sensor values are ever generated.")
        return
    n_a, n_w = int((summ.Status == "ALERT").sum()), int((summ.Status == "EARLY WARNING").sum())
    kpi_row([("Alerts", n_a, "anomalous now", "bad" if n_a else "good"),
             ("Early warnings", n_w, "trend reaches alert ≤ 14 days", "warn" if n_w else "good"),
             ("Healthy wells", int((summ.Status == "NORMAL").sum()), f"of {len(summ)}"),
             ("Signals used", len(f.sensors), ", ".join(s.split("_")[-1].lower() for s in f.sensors))])
    a, b = st.columns([1, 1.35], gap="large")
    with a:
        card_header("Fleet status", "highest risk first")
        table(summ.sort_values("Score", ascending=False)[["Well", "Status", "Health", "Days to alert", "Main signal"]], 0,
              height=420)
    with b:
        order = summ.sort_values("Score", ascending=False)["Well"].tolist()
        well = st.selectbox("Well", order, key="eq_well")
        g = rows[rows.Well == well]
        card_header(f"{well} · anomaly score", "dashed line = alert threshold · red = anomalous day")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g.Date, y=g.Score, name="Score", line=dict(color=COLORS["teal"], width=2),
                                 fill="tozeroy", fillcolor="rgba(45,212,191,.08)"))
        fig.add_hline(y=summ["Threshold"].iloc[0], line_dash="dash", line_color=COLORS["red"])
        al = g[g.Alert]
        fig.add_trace(go.Scatter(x=al.Date, y=al.Score, mode="markers", name="Anomalous",
                                 marker=dict(color=COLORS["red"], size=6)))
        chart(fig, 350, yaxis_title="Anomaly score")
    for col, s in zip(st.columns(min(4, len(f.sensors))), f.sensors[:4]):
        with col:
            card_header(s.replace("_", " "), STD_UNIT.get(s, ""))
            chart(go.Figure(go.Scatter(x=g.Date, y=g[s], line=dict(color=COLORS["blue"], width=1.6))), 210, legend=False)


@_fragment
def page_well(f: Field, ranked, vfm):
    page_header("Well Explorer", "Everything about one well on one screen.")
    well = st.selectbox("Well (ranked by priority)", ranked["Well"].tolist(), key="explore_well")
    r = ranked[ranked["Well"] == well].iloc[0]
    g = enrich(f.df[f.df["Well"] == well])
    tone = {"CRITICAL": "bad", "HIGH": "warn", "WATCH": "warn", "STABLE": "good"}[r["Priority"]]
    kpi_row([("Oil · 7-day avg", f"{r['Oil_Rate']:.1f} m³/d", f"≈ {r['Oil_Rate'] * M3_TO_BBL:,.0f} bbl/d"),
             ("Water cut", f"{r['Water_Cut']:.1f}%", "7-day avg", "warn" if r["Water_Cut"] >= 80 else ""),
             ("Choke size", fmt(r.get("Choke_Size", np.nan), ".0f", "/64 in"), "current bean"),
             ("Decline", fmt(r.get("Decline (%/yr)", np.nan), ".1f", " %/yr"), "Arps"),
             ("Priority", r["Priority"], f"score {r['Score']}", tone)])
    insight(f"<b>{esc(well)}:</b> {esc(r['Why'])}.")
    a, b = st.columns(2, gap="large")
    with a:
        card_header("Oil, water and AI estimate")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g.Date, y=g.Water_Rate, name="Water", line=dict(color=COLORS["blue"], width=1.4)))
        fig.add_trace(go.Scatter(x=g.Date, y=g.Oil_Rate, name="Oil", line=dict(color=COLORS["teal"], width=2)))
        if vfm:
            pw = vfm["pred"][vfm["pred"]["Well"] == well]
            fig.add_trace(go.Scatter(x=pw.Date, y=pw.Pred, name="Oil · AI", line=dict(color=COLORS["amber"], width=1.4, dash="dot")))
        chart(fig, 350, yaxis_title="m³/d")
    with b:
        card_header("Pressures and choke size", "choke on the right axis" if f.has_choke else "")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g.Date, y=g.Bottomhole_Pressure, name="Bottom-hole", line=dict(color=COLORS["violet"])))
        fig.add_trace(go.Scatter(x=g.Date, y=g.Wellhead_Pressure, name="Wellhead", line=dict(color=COLORS["blue"])))
        extra = {}
        if f.has_choke:
            fig.add_trace(go.Scatter(x=g.Date, y=g.Choke_Size, name="Choke", yaxis="y2",
                                     line=dict(color=COLORS["amber"], shape="hv", dash="dot")))
            extra = {"yaxis2": dict(title="Choke (1/64 in)", overlaying="y", side="right", showgrid=False)}
        chart(fig, 350, yaxis_title="bar", **extra)
    a, b = st.columns(2, gap="large")
    with a:
        card_header("Water cut")
        chart(go.Figure(go.Scatter(x=g.Date, y=g.Water_Cut, line=dict(color=COLORS["blue"], width=2), fill="tozeroy",
                                   fillcolor="rgba(96,165,250,.08)")), 280, legend=False, yaxis_title="%")
    with b:
        card_header("Lift gas", f.inj_unit)
        chart(go.Figure(go.Scatter(x=g.Date, y=g.Injection_Rate, line=dict(color=COLORS["teal"], width=1.6))),
              280, legend=False, yaxis_title=f.inj_unit)


def page_quality(f: Field):
    r = f.report
    page_header("Data & Quality Gate", "What was recognised, converted, repaired or ignored — fully auditable.", f.name)
    kpi_row([("Input rows", f"{r['input_rows']:,}", "as uploaded"), ("Clean rows", f"{len(f.df):,}", "used by the models", "good"),
             ("Duplicates merged", f"{r['duplicates']:,}", "exact + same well/date"),
             ("Invalid values", f"{r['invalid']:,}", "negative rates / pressures"),
             ("Outliers capped", f"{r['capped']:,}", "per-well 3×IQR")])
    a, b = st.columns(2, gap="large")
    with a:
        card_header("Column mapping", "your header → FIELDWISE field")
        table(pd.DataFrame([{"FIELDWISE field": k, "Your column": v} for k, v in r["mapped"].items()]), height=430)
    with b:
        card_header("Cleaning log")
        items = [f"Rows without a valid date or well ID removed: <b>{r['bad_keys']}</b>",
                 f"Ignored columns: <b>{esc(', '.join(r['ignored']) or 'none')}</b>"]
        if r["imputed"]:
            items.append("Values filled: " + ", ".join(f"<b>{esc(k)}</b> {v}" for k, v in r["imputed"].items()))
        if r["conversions"]:
            items.append("Units converted: " + ", ".join(f"<b>{esc(c['Field'])}</b> {esc(c['From'])} → {esc(c['To'])}"
                                                         for c in r["conversions"]))
        if r["dropped_optional"]:
            items.append("Dropped (more than 60 % empty): <b>" + esc(", ".join(r["dropped_optional"])) + "</b>")
        items.append("Choke size: " + ("<b>present</b> — gaps carried forward per well, never interpolated"
                                       if f.has_choke else "<b>not supplied</b> — models run without it"))
        if r["injection_missing"]:
            items.append("No lift-gas column — lift-gas optimisation is disabled.")
        note("<br>".join("• " + i for i in items))
        opt = [c for c in OPTIONAL_COLUMNS if c in f.df.columns]
        note(f"<b>Optional fields found ({len(opt)}/{len(OPTIONAL_COLUMNS)}):</b> {esc(', '.join(opt) or 'none')}")
    card_header("Clean data preview", "first 200 rows")
    table(f.df.head(200), 2, height=360)
    st.download_button("⬇ Download clean data (CSV)", f.df.to_csv(index=False).encode(), "fieldwise_clean.csv", "text/csv")


def page_method():
    page_header("Methodology", "How FIELDWISE AI answers Problem Statement 3.")
    rows = [("Manual data collation", "Ingestion & quality gate",
             "Recognises common header variants, converts units (psi, bbl/d, inch chokes …), repairs and logs every change."),
            ("Periodic well testing", "AI virtual flow meter",
             "Gradient boosting + extra trees infer daily oil from pressures, choke size, lift gas and pump signals between tests."),
            ("Declining reservoir pressure", "Decline curves", "Arps decline per well, 180-day forecast and decline ranking."),
            ("Sub-optimal lift-gas allocation", "Lift-gas optimiser",
             "Per-well gas-lift performance curves + exact dynamic programme under the field gas limit."),
            ("Frequent pump burnouts", "Equipment early warning",
             "Per-well baselines, Isolation Forest scoring and trend projection to days-to-alert."),
            ("Energy per barrel · ESG", "Energy & carbon intensity", "kWh/bbl and kg CO₂/bbl from equipment energy."),
            ("Reactive troubleshooting", "Attention queue", "Explainable ranking with the reason behind every flag.")]
    table(pd.DataFrame(rows, columns=["Pain point in the brief", "FIELDWISE module", "How it works"]), 0)
    card_header("Validation discipline")
    note("Models are always tested on the <b>latest 20 % of dates</b>, which they never see during training (no look-ahead). "
         "Same-day oil, water and gas rates are never model inputs, so the AI works on days without a well test.")
    card_header("Technology")
    st.code("Python · pandas · NumPy · SciPy · scikit-learn · Plotly · Streamlit", language="text")


# =============================================================================
# 7. MAIN
# =============================================================================
PAGES = {"Command Centre": "📊", "AI Production Model": "🧠", "Decline & Forecast": "📉", "Lift-Gas Optimiser": "⚖️",
         "Equipment Health": "🛠️", "Well Explorer": "🔎", "Data & Quality": "🧹", "Methodology": "📘"}


def _file_token(up, data: bytes) -> str:
    """Cheap, stable identity for an upload (avoids hashing the whole file on every rerun)."""
    head = hashlib.md5(data[:1 << 20] + data[-(1 << 20):]).hexdigest()
    return f"{up.name}|{len(data)}|{head}"


def sidebar_upload():
    st.sidebar.markdown("**Field data**")
    up = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls", "xlsm"])
    if up is None:
        return None, None
    data = up.getvalue()
    token = _file_token(up, data)
    try:
        sheets = read_file(token, data, up.name)
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not read {up.name}: {exc}"
    sheet = list(sheets)[0]
    if len(sheets) > 1:
        sheet = st.sidebar.selectbox("Sheet", list(sheets))
    sample, mapped, _ = map_columns(sheets[sheet].head(200))
    units, detected = {}, {}
    for col in UNITS:
        if col in sample.columns:
            detected[col] = detect_unit(mapped.get(col, ""), col) or STD_UNIT[col]
    auto = [f"{c.replace('_', ' ')}: {u}" for c, u in detected.items() if u != STD_UNIT[c]]
    with st.sidebar.expander("Source units" + (f" · {len(auto)} auto-detected" if auto else ""), expanded=False):
        st.caption("Detected from your column headers where possible — change any that are wrong.")
        for col, guess in detected.items():
            keys = list(UNITS[col])
            units[col] = st.selectbox(col.replace("_", " "), keys, index=keys.index(guess), key=f"u_{token}_{col}")
    try:
        df, rep = clean_cached(f"{token}|{sheet}", sheets, sheet, tuple(sorted(units.items())))
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not process {up.name}: {exc}"
    if df is None:
        if rep.get("missing"):
            return None, ("Required columns not found: " + ", ".join(rep["missing"]) + ". Recognised: " +
                          (", ".join(rep["mapped"]) or "none") + ". Rename the headers or start from the template.")
        return None, rep.get("error", "The data could not be cleaned.")
    rep = {**rep, "auto_units": auto}
    key = f"{token}|{sheet}|{sorted(units.items())}"
    return Field(df=df, name=up.name, report=rep, lift_is_gas=units.get("Injection_Rate", "Sm³/d") in _GAS, key=key), None


def sidebar_settings() -> Settings:
    with st.sidebar.expander("Settings", expanded=False):
        return Settings(test_interval=st.select_slider("Well-test interval (observations)", options=[7, 14, 21, 30], value=14),
                        wc_watch=st.slider("Water cut · watch (%)", 30, 95, 70),
                        wc_high=st.slider("Water cut · high (%)", 50, 99, 90),
                        decline_watch=st.slider("Decline · watch (%/yr)", 5, 50, 15),
                        decline_high=st.slider("Decline · high (%/yr)", 10, 80, 30),
                        co2=st.number_input("Grid CO₂ factor (kg/kWh)", min_value=0.0, max_value=2.0, value=GRID_CO2, step=0.01),
                        oil_price=st.number_input("Oil price (USD/bbl)", min_value=0.0, max_value=300.0, value=70.0, step=1.0))


def main():
    st.set_page_config(page_title=APP, page_icon="🛢️", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown('<div class="fw-brand"><div class="fw-logo">F</div><div><div class="fw-brand-name">FIELDWISE AI</div>'
                        '<div class="fw-brand-sub">Mature-field optimisation</div></div></div>', unsafe_allow_html=True)
    field, err = sidebar_upload()
    footer = f'<div class="fw-foot">designed by {esc(AUTHOR)}</div>'

    if field is None:              # dormant until data is supplied
        if err:
            st.error(err)
        page_landing()
        st.sidebar.markdown(footer, unsafe_allow_html=True)
        return

    settings = sidebar_settings()
    field.report = {**field.report, "interval": settings.test_interval, "co2": settings.co2, "oil_price": settings.oil_price}
    page = st.sidebar.radio("Navigate", list(PAGES), format_func=lambda p: f"{PAGES[p]}  {p}")
    st.sidebar.markdown(footer, unsafe_allow_html=True)

    df, k = field.df, field.key
    with st.spinner("Reading the field and training models…"):
        try:
            vfm = train_vfm(k, df, settings.test_interval)
        except ValueError:
            vfm = None
        dca, fc = fit_declines(k, df)
        curves, skipped = fit_lift_curves(k, df) if field.has_lift else ({}, {})
        eq_rows, eq_sum = screen_equipment(k, df, tuple(field.sensors))
        ranked = rank_cached(k, df, dca, eq_sum, settings, astuple(settings))

    if page == "Command Centre":
        page_overview(field, ranked, vfm, dca, fc, curves)
    elif page == "AI Production Model":
        page_model(field, vfm)
    elif page == "Decline & Forecast":
        page_decline(field, dca, fc)
    elif page == "Lift-Gas Optimiser":
        page_lift(field, curves, skipped)
    elif page == "Equipment Health":
        page_equipment(field, eq_rows, eq_sum)
    elif page == "Well Explorer":
        page_well(field, ranked, vfm)
    elif page == "Data & Quality":
        page_quality(field)
    else:
        page_method()


if __name__ == "__main__":
    main()
