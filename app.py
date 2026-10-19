"""
FIELDWISE AI — Mature-field production optimisation & choke intelligence
=========================================================================
IEW 2027 Hackathon · Problem Statement 3
"AI-Based Production Optimization and Reservoir Modelling for Mature Fields"

Run:  streamlit run app.py

File layout (top to bottom)
  1. Configuration ............ column schema, aliases, units, physical constants
  2. Choke & well physics ...... Gilbert choke correlation, unit helpers
  3. Data ingestion & cleaning . alias mapping, unit conversion, auditable cleaning
  4. Demo field generator ...... physically consistent 12-well gas-lift field
  5. Analytics ................. choke diagnostics, hybrid nodal well models,
                                 ML proxy, equipment anomaly screening, scoring
  6. Optimiser ................. joint choke-size + lift-gas allocation (DP)
  7. UI helpers & styling
  8. Pages
  9. main()

Internal standard units
  liquid rates m³/d · gas rates Sm³/d · pressures bar · choke size 1/64 in
"""
from __future__ import annotations

import html
import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
APP_TITLE = "FIELDWISE AI"
AUTHOR = "Kavinkarthick"

CORE_COLUMNS = ["Date", "Well", "Oil_Rate", "Water_Rate", "Gas_Rate",
                "Wellhead_Pressure", "Bottomhole_Pressure", "Injection_Rate"]
CHOKE_COLUMNS = ["Choke_Size", "Flowline_Pressure"]
SENSOR_COLUMNS = ["Pump_Vibration", "Motor_Temperature", "Motor_Current", "Pump_Frequency", "Energy_kWh"]
OPTIONAL_COLUMNS = CHOKE_COLUMNS + SENSOR_COLUMNS
CORE_NUMERIC = [c for c in CORE_COLUMNS if c not in ("Date", "Well")]
NUMERIC_COLUMNS = CORE_NUMERIC + OPTIONAL_COLUMNS
RATE_COLUMNS = ["Oil_Rate", "Water_Rate", "Gas_Rate", "Injection_Rate"]
PRESSURE_COLUMNS = ["Wellhead_Pressure", "Bottomhole_Pressure", "Flowline_Pressure"]
# Setpoints change in steps: never interpolate or outlier-cap them.
SETPOINT_COLUMNS = ["Choke_Size"]

COLUMN_DESCRIPTIONS = {
    "Date": "Observation date / timestamp",
    "Well": "Unique well identifier",
    "Oil_Rate": "Oil production rate",
    "Water_Rate": "Water production rate",
    "Gas_Rate": "Produced (formation) gas rate",
    "Wellhead_Pressure": "Flowing wellhead / tubing-head pressure (upstream of choke)",
    "Bottomhole_Pressure": "Flowing bottom-hole pressure (downhole gauge)",
    "Injection_Rate": "Lift-gas (or injection) rate allocated to the well",
    "Choke_Size": "Production choke (bean) size — industry convention 1/64 inch",
    "Flowline_Pressure": "Pressure downstream of the choke (flowline / manifold)",
    "Pump_Vibration": "Pump vibration",
    "Motor_Temperature": "Motor temperature",
    "Motor_Current": "Motor current",
    "Pump_Frequency": "VFD / pump frequency",
    "Energy_kWh": "Energy consumed over the observation interval",
}

# Unknown columns (GR, RHOB, NPHI, resistivity …) are deliberately ignored.
COLUMN_ALIASES = {
    "Date": ["date", "datetime", "timestamp", "time", "sample_date", "observation_date", "production_date"],
    "Well": ["well", "well_id", "well name", "well_name", "wellno", "well_no", "uwi", "api", "well identifier"],
    "Oil_Rate": ["oil_rate", "oil rate", "oil production", "oil production rate", "qo", "oil_volumetric_rate",
                 "oil_m3_d", "oil rate (m3/d)", "oil rate (m³/d)", "oil_bopd", "bopd"],
    "Water_Rate": ["water_rate", "water rate", "water production", "water production rate", "qw",
                   "water_volumetric_rate", "water_m3_d", "water rate (m3/d)", "water rate (m³/d)", "bwpd"],
    "Gas_Rate": ["gas_rate", "gas rate", "gas production", "gas production rate", "qg", "gas_mscf_d",
                 "gas rate (mcf/d)", "formation gas"],
    "Wellhead_Pressure": ["wellhead_pressure", "wellhead pressure", "whp", "tubing_head_pressure", "thp",
                          "well head pressure", "fthp", "flowing tubing head pressure"],
    "Bottomhole_Pressure": ["bottomhole_pressure", "bottom hole pressure", "bhp", "flowing_bottomhole_pressure",
                            "fbhp", "bottom-hole pressure", "pwf", "downhole pressure"],
    "Injection_Rate": ["injection_rate", "injection rate", "injection", "gas_lift_rate", "gas lift rate",
                       "lift_gas_rate", "lift gas rate", "water_injection_rate", "qinj", "glir"],
    "Choke_Size": ["choke_size", "choke size", "choke", "bean", "bean size", "bean_size", "choke_64ths",
                   "choke (1/64 in)", "choke (64ths)", "ck", "choke_diameter", "choke diameter"],
    "Flowline_Pressure": ["flowline_pressure", "flowline pressure", "flp", "downstream pressure",
                          "downstream_pressure", "manifold pressure", "separator pressure", "fl pressure",
                          "pressure downstream of choke"],
    "Pump_Vibration": ["pump_vibration", "vibration", "vibration_rms", "pump vibration"],
    "Motor_Temperature": ["motor_temperature", "motor temp", "motor_temperature_c", "motor temperature", "motor_temp"],
    "Motor_Current": ["motor_current", "current", "motor amps", "motor_current_a", "motor current"],
    "Pump_Frequency": ["pump_frequency", "frequency", "vfd_frequency", "pump frequency", "frequency_hz"],
    "Energy_kWh": ["energy_kwh", "energy", "power_consumption_kwh", "energy consumed", "energy consumption"],
}

_LIQ = {"m³/d": (1.0, 0.0), "bbl/d": (0.1589872949, 0.0), "STB/d": (0.1589872949, 0.0), "m³/h": (24.0, 0.0),
        "L/s": (86.4, 0.0), "L/min": (1.44, 0.0), "ft³/d": (0.0283168466, 0.0)}
_GAS = {"Sm³/d": (1.0, 0.0), "Mscf/d": (28.3168466, 0.0), "scf/d": (0.0283168466, 0.0),
        "MMSCFD": (28316.8466, 0.0), "Nm³/h": (24.0, 0.0)}
_PRES = {"bar": (1.0, 0.0), "kPa": (0.01, 0.0), "MPa": (10.0, 0.0), "psi": (0.0689475729, 0.0),
         "kgf/cm²": (0.980665, 0.0)}
# value_standard = value_source × factor + offset
UNIT_OPTIONS = {
    "Oil_Rate": _LIQ, "Water_Rate": _LIQ, "Gas_Rate": _GAS,
    "Injection_Rate": {**_LIQ, **_GAS},
    "Wellhead_Pressure": _PRES, "Bottomhole_Pressure": _PRES, "Flowline_Pressure": _PRES,
    "Choke_Size": {"1/64 in": (1.0, 0.0), "in": (64.0, 0.0), "mm": (64.0 / 25.4, 0.0)},
    "Motor_Temperature": {"°C": (1.0, 0.0), "°F": (5.0 / 9.0, -32.0 * 5.0 / 9.0)},
    "Motor_Current": {"A": (1.0, 0.0), "mA": (0.001, 0.0)},
    "Pump_Frequency": {"Hz": (1.0, 0.0), "rpm": (1.0 / 60.0, 0.0)},
    "Pump_Vibration": {"mm/s": (1.0, 0.0), "in/s": (25.4, 0.0)},
    "Energy_kWh": {"kWh": (1.0, 0.0), "Wh": (0.001, 0.0), "MWh": (1000.0, 0.0)},
}
STANDARD_UNITS = {"Oil_Rate": "m³/d", "Water_Rate": "m³/d", "Gas_Rate": "Sm³/d", "Injection_Rate": "m³/d",
                  "Wellhead_Pressure": "bar", "Bottomhole_Pressure": "bar", "Flowline_Pressure": "bar",
                  "Choke_Size": "1/64 in", "Motor_Temperature": "°C", "Motor_Current": "A", "Pump_Frequency": "Hz",
                  "Pump_Vibration": "mm/s", "Energy_kWh": "kWh"}
GAS_UNITS = set(_GAS)
MISSING_TOKENS = {"", "nan", "none", "null", "n/a", "na", "unknown", "—", "-", "<na>", "#n/a", "nat"}

PRIORITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "WATCH": 2, "STABLE": 3}
PRIORITY_COLORS = {"CRITICAL": "#E5484D", "HIGH": "#F0883E", "WATCH": "#E7B85B", "STABLE": "#3FB6A8"}
STATUS_COLORS = {"ALERT": "#E5484D", "NORMAL": "#3FB6A8", "NO DATA": "#56707C"}

# =============================================================================
# 2. CHOKE & WELL PHYSICS
# =============================================================================
# Gilbert (1954) critical-flow choke correlation, oilfield units:
#     P_wh [psi] = a · R^b · q [STB/d] / S^c      R = GLR [Mscf/STB], S = choke [1/64 in]
GILBERT_A, GILBERT_B, GILBERT_C = 435.0, 0.546, 1.89
M3_TO_STB = 6.289811
SM3_TO_MSCF = 0.0353147
BAR_TO_PSI = 14.503774
CRITICAL_FLOW_RATIO = 0.55   # P_downstream / P_upstream below this ⇒ critical (sonic) flow


def glr_mscf_per_stb(gas_sm3d, liquid_m3d):
    """Gas-liquid ratio in Mscf/STB, bounded to a physically sensible range."""
    gas = np.asarray(gas_sm3d, dtype=float)
    liq = np.maximum(np.asarray(liquid_m3d, dtype=float), 1e-3)
    return np.clip(gas * SM3_TO_MSCF / (liq * M3_TO_STB), 0.01, 50.0)


def gilbert_k(glr, choke_64, a=GILBERT_A):
    """Choke resistance k [bar per m³/d] so that WHP = k · q_liquid (critical flow)."""
    s = np.maximum(np.asarray(choke_64, dtype=float), 1e-6)
    return a * np.power(glr, GILBERT_B) * M3_TO_STB / np.power(s, GILBERT_C) / BAR_TO_PSI


def gilbert_coefficient(whp_bar, liquid_m3d, glr, choke_64):
    """Back-calculate the Gilbert 'a' coefficient from measurements (435 for an ideal choke)."""
    q = np.maximum(np.asarray(liquid_m3d, dtype=float), 1e-6) * M3_TO_STB
    return (np.asarray(whp_bar, dtype=float) * BAR_TO_PSI * np.power(np.asarray(choke_64, dtype=float), GILBERT_C)
            / (np.power(glr, GILBERT_B) * q))


def gilbert_liquid_rate(whp_bar, glr, choke_64, a=GILBERT_A):
    """Virtual flow meter: liquid rate [m³/d] implied by WHP, choke size and GLR."""
    q_stb = (np.asarray(whp_bar, dtype=float) * BAR_TO_PSI * np.power(np.asarray(choke_64, dtype=float), GILBERT_C)
             / (a * np.power(glr, GILBERT_B)))
    return q_stb / M3_TO_STB


# =============================================================================
# 3. DATA INGESTION & CLEANING
# =============================================================================
def _norm(name) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").replace("-", " ").split())


def map_industrial_columns(raw: pd.DataFrame):
    """Map recognised headers to FIELDWISE names; return (selected frame, mapping, ignored columns)."""
    lookup = {_norm(alias): canon for canon, names in COLUMN_ALIASES.items() for alias in [canon, *names]}
    rename, mapped = {}, {}
    for col in raw.columns:
        canon = lookup.get(_norm(col))
        if canon and canon not in mapped:
            rename[col] = canon
            mapped[canon] = str(col)
    frame = raw.rename(columns=rename)
    selected = [c for c in CORE_COLUMNS + OPTIONAL_COLUMNS if c in frame.columns]
    ignored = [str(c) for c in raw.columns if c not in rename]
    return frame[selected].copy(), mapped, ignored


def _parse_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.mask(s.str.lower().isin(MISSING_TOKENS))
    return pd.to_numeric(s, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)


def _parse_dates(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce")


def clean_industrial_data(raw: pd.DataFrame, unit_choices: dict | None = None):
    """Conservative, auditable cleaning. Returns (clean frame | None, report)."""
    unit_choices = unit_choices or {}
    frame, mapped, ignored = map_industrial_columns(raw)
    report = {"input_rows": len(raw), "mapped_columns": mapped, "ignored_columns": ignored,
              "rows_without_date_or_well_removed": 0, "duplicates_removed": 0, "conflicting_timestamps_merged": 0,
              "non_numeric_values": 0, "invalid_values_removed": 0, "imputed_by_column": {},
              "outlier_values_capped": 0, "dropped_optional_columns": [], "unit_conversions": [],
              "injection_assumed_zero": False, "notes": []}

    if "Injection_Rate" not in frame.columns:
        frame["Injection_Rate"] = 0.0
        report["injection_assumed_zero"] = True
        report["notes"].append("Injection_Rate was absent and set to 0. Lift-gas allocation is disabled; "
                               "choke optimisation still works if Choke_Size is supplied.")
    missing = [c for c in CORE_COLUMNS if c not in frame.columns]
    if missing:
        report["missing_required"] = missing
        return None, report

    # Keys
    frame["Date"] = _parse_dates(frame["Date"])
    well = frame["Well"].astype(str).str.strip().str.upper()
    frame["Well"] = well.mask(well.isin({"", "NAN", "NONE", "<NA>", "NULL"}))
    before = len(frame)
    frame = frame.dropna(subset=["Date", "Well"]).copy()
    report["rows_without_date_or_well_removed"] = before - len(frame)

    # Numeric parsing
    numeric = [c for c in NUMERIC_COLUMNS if c in frame.columns]
    original_missing = {}
    for col in numeric:
        had_value = ~frame[col].astype(str).str.strip().str.lower().isin(MISSING_TOKENS) & frame[col].notna()
        parsed = _parse_numeric(frame[col])
        report["non_numeric_values"] += int((had_value & parsed.isna()).sum())
        frame[col] = parsed
        original_missing[col] = float(parsed.isna().mean())

    # Unit standardisation (after parsing, before any statistics)
    for col, unit in unit_choices.items():
        if col not in frame.columns or unit not in UNIT_OPTIONS.get(col, {}):
            continue
        target = "Sm³/d" if (col == "Injection_Rate" and unit in GAS_UNITS) else STANDARD_UNITS[col]
        if unit != target:
            factor, offset = UNIT_OPTIONS[col][unit]
            frame[col] = frame[col] * factor + offset
            report["unit_conversions"].append({"Field": col, "Source unit": unit, "Standard unit": target,
                                               "Conversion": f"value × {factor:.6g} + {offset:.6g}"})

    # Physical validity: negative rates, pressures or choke sizes are impossible
    for col in [c for c in RATE_COLUMNS + PRESSURE_COLUMNS + SETPOINT_COLUMNS if c in frame.columns]:
        bad = frame[col] < 0
        report["invalid_values_removed"] += int(bad.sum())
        frame.loc[bad, col] = np.nan
    if "Choke_Size" in frame.columns:   # > 4 inch bean is not a production choke
        bad = frame["Choke_Size"] > 256
        report["invalid_values_removed"] += int(bad.sum())
        frame.loc[bad, "Choke_Size"] = np.nan

    # Optional columns that are mostly empty are not trustworthy evidence
    for col in [c for c in OPTIONAL_COLUMNS if c in frame.columns]:
        if original_missing.get(col, 0) > 0.6:
            frame = frame.drop(columns=col)
            report["dropped_optional_columns"].append(col)
            numeric.remove(col)

    # Duplicates
    before = len(frame)
    frame = frame.drop_duplicates()
    report["duplicates_removed"] = before - len(frame)
    before = len(frame)
    if frame.duplicated(["Well", "Date"]).any():
        frame = frame.groupby(["Well", "Date"], as_index=False)[numeric].mean()
    report["conflicting_timestamps_merged"] = before - len(frame)
    frame = frame.sort_values(["Well", "Date"]).reset_index(drop=True)

    # Imputation: setpoints are carried forward; measurements are interpolated in time
    for col in numeric:
        n_missing = int(frame[col].isna().sum())
        if n_missing == 0:
            continue
        grouped = frame.groupby("Well")[col]
        if col in SETPOINT_COLUMNS:
            frame[col] = grouped.ffill()
            frame[col] = frame.groupby("Well")[col].bfill()
        else:
            frame[col] = grouped.transform(lambda s: s.interpolate(limit_direction="both"))
            if col in CORE_NUMERIC and frame[col].isna().any():
                frame[col] = frame[col].fillna(frame[col].median())
        report["imputed_by_column"][col] = n_missing - int(frame[col].isna().sum())

    # Per-well outlier capping (3×IQR) — measurements only, never setpoints
    for col in [c for c in numeric if c not in SETPOINT_COLUMNS]:
        for _, idx in frame.groupby("Well").groups.items():
            s = frame.loc[idx, col]
            if s.notna().sum() < 10:
                continue
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr <= 0:
                continue
            lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
            if col in RATE_COLUMNS or col in PRESSURE_COLUMNS:
                lo = max(lo, 0.0)
            mask = (s < lo) | (s > hi)
            if mask.any():
                report["outlier_values_capped"] += int(mask.sum())
                frame.loc[idx, col] = s.clip(lo, hi)

    frame = frame.dropna(subset=CORE_NUMERIC)
    if len(frame) < 20:
        report["error"] = f"Only {len(frame)} usable rows remain after cleaning; at least 20 are needed."
        return None, report
    if frame["Date"].nunique() < 2:
        report["error"] = "At least two distinct dates are required for chronological validation."
        return None, report

    report["output_rows"] = len(frame)
    report["notes"] += [
        "Unrecognised columns were ignored; no missing production targets were fabricated.",
        "Choke size is treated as a setpoint: gaps are carried forward per well, never interpolated or capped.",
        "Missing measurements were interpolated within each well; core fields fall back to the field median.",
        "Measurement outliers were capped per well at 3×IQR fences. Review against engineering limits.",
    ]
    return frame.reset_index(drop=True), report


@st.cache_data(show_spinner="Reading CSV…", max_entries=2)
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    if not file_bytes:
        raise ValueError("The uploaded CSV is empty.")
    try:
        return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("No readable CSV columns. Check that it is a real CSV, not an Excel file renamed .csv.") from exc
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(file_bytes), encoding="cp1252", low_memory=False)


@st.cache_data(show_spinner="Cleaning and standardising data…", max_entries=4)
def clean_uploaded(file_bytes: bytes, unit_items: tuple):
    return clean_industrial_data(read_csv_bytes(file_bytes), dict(unit_items))


# =============================================================================
# 4. DEMO FIELD GENERATOR (physically consistent nodal model)
# =============================================================================
@st.cache_data
def make_demo_data() -> pd.DataFrame:
    """12 gas-lifted wells, 240 days. Each well follows linear IPR + tubing loss + Gilbert choke.
    Embedded stories: W-003 bean-up (day 120), W-010 bean-down for sand control (day 160),
    W-006 choke erosion, W-009 choke restriction (sand/scale), W-008 live equipment anomaly."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-03-01", periods=240, freq="D")
    chokes = [24, 28, 24, 20, 36, 28, 24, 32, 40, 40, 24, 32]
    rows = []
    for wi in range(12):
        well = f"W-{wi + 1:03d}"
        pr0, J, d0, d1 = 215 - 3 * wi, 2.0 + 0.3 * ((wi * 7) % 5), 92 + 2.5 * wi, 0.02
        lift_eff, lift_sat = 0.8 + 0.08 * ((wi * 5) % 7), 0.022 + 0.002 * (wi % 3)
        a_true = GILBERT_A * (1 + rng.normal(0, 0.03))
        wc0, gor = 0.45 + 0.035 * wi, 70 + 7 * wi
        base_inj = 12000 + 600 * ((wi * 3) % 5)
        q = 150.0
        for t, d in enumerate(dates):
            choke = chokes[wi]
            if wi == 2 and t >= 120:
                choke = 32
            if wi == 9 and t >= 160:
                choke = 32
            eff = choke
            if wi == 5:
                eff = choke * (1 + 0.0012 * t)
            if wi == 8 and t > 150:
                eff = choke * (1 - 0.0016 * (t - 150))
            inj = max(0.0, base_inj + 4000 * np.sin(t / 15 + wi) + rng.normal(0, 300))
            x = inj / 1000
            pr = pr0 - 0.035 * t
            tub0 = d0 - lift_eff * x + lift_sat * x ** 2
            wc = float(np.clip(wc0 + 0.00035 * t + rng.normal(0, 0.008), 0.05, 0.97))
            for _ in range(40):
                glr = glr_mscf_per_stb(q * (1 - wc) * gor + inj, q)
                k = gilbert_k(glr, eff, a_true)
                q = 0.5 * q + 0.5 * max(J * (pr - tub0) / (1 + J * (k + d1)), 1.0)
            q_meas = q * (1 + rng.normal(0, 0.01))
            oil, water = q_meas * (1 - wc), q_meas * wc
            gas = oil * gor * (1 + rng.normal(0, 0.02))
            whp = float(k * q) * (1 + rng.normal(0, 0.02))
            bhp = float(k * q) + tub0 + d1 * q + rng.normal(0, 0.5)
            flp = 14 + 0.5 * np.sin(t / 9) + rng.normal(0, 0.3)
            vib = max(0.1, 2.0 + 0.0008 * t + 0.10 * wi + rng.normal(0, .2))
            temp = 68 + 0.005 * t + 0.7 * wi + 2 * np.sin(t / 17 + wi) + rng.normal(0, 1.5)
            current = max(1, 40 + 0.08 * q + rng.normal(0, 2.0))
            freq = float(np.clip(48 + 2 * np.sin(t / 22 + wi) + rng.normal(0, 1), 35, 60))
            energy = max(1, current * freq * 0.15 + rng.normal(0, 3))
            if (wi == 2 and 170 <= t <= 182) or (wi == 7 and t >= 228):
                vib += 2.4
                temp += 9
            rows.append([d, well, oil, water, gas, whp, bhp, inj, choke, flp, vib, temp, current, freq, energy])
    return pd.DataFrame(rows, columns=CORE_COLUMNS + OPTIONAL_COLUMNS)


def csv_template() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": ["2026-01-01", "2026-01-02"], "Well": ["W-001", "W-001"],
        "Oil_Rate": [85.0, 84.2], "Water_Rate": [174.0, 175.0], "Gas_Rate": [7200.0, 7150.0],
        "Wellhead_Pressure": [42.0, 41.8], "Bottomhole_Pressure": [118.0, 117.6], "Injection_Rate": [14000, 14200],
        "Choke_Size": [28, 28], "Flowline_Pressure": [14.0, 14.1],
        "Pump_Vibration": [2.1, 2.2], "Motor_Temperature": [72.0, 72.4], "Motor_Current": [54.0, 53.5],
        "Pump_Frequency": [48.0, 48.0], "Energy_kWh": [390.0, 388.0],
    })


# =============================================================================
# 5. ANALYTICS
# =============================================================================
@dataclass
class DataBundle:
    df: pd.DataFrame
    source: str
    report: dict | None = None
    error: str | None = None
    inj_is_gas: bool = True
    is_demo: bool = True

    @property
    def has_choke(self) -> bool:
        return "Choke_Size" in self.df.columns and self.df["Choke_Size"].notna().any()

    @property
    def has_flp(self) -> bool:
        return "Flowline_Pressure" in self.df.columns and self.df["Flowline_Pressure"].notna().any()

    @property
    def sensors(self) -> list[str]:
        return [c for c in SENSOR_COLUMNS if c in self.df.columns and self.df[c].notna().any()]

    @property
    def injection_available(self) -> bool:
        return not (self.report or {}).get("injection_assumed_zero", False) and self.df["Injection_Rate"].sum() > 0


@dataclass
class Settings:
    wc_watch: float = 70.0
    wc_high: float = 85.0
    decline_watch: float = -5.0
    decline_high: float = -15.0
    choke_tol: float = 15.0
    critical_ratio: float = CRITICAL_FLOW_RATIO


def add_derived(df: pd.DataFrame, inj_is_gas: bool) -> pd.DataFrame:
    out = df.copy()
    out["Liquid_Rate"] = out["Oil_Rate"] + out["Water_Rate"]
    out["Water_Cut"] = (100 * out["Water_Rate"] / out["Liquid_Rate"].replace(0, np.nan)).fillna(0.0)
    out["GOR"] = out["Gas_Rate"] / out["Oil_Rate"].replace(0, np.nan)
    total_gas = out["Gas_Rate"] + (out["Injection_Rate"] if inj_is_gas else 0.0)
    out["GLR"] = glr_mscf_per_stb(total_gas, out["Liquid_Rate"])
    out["Days"] = (out["Date"] - out.groupby("Well")["Date"].transform("min")).dt.total_seconds() / 86400.0
    return out


# ---- 5a. Choke diagnostics -------------------------------------------------
@st.cache_data(show_spinner="Running choke diagnostics…", max_entries=4)
def build_choke_frame(df: pd.DataFrame, inj_is_gas: bool, has_flp: bool) -> pd.DataFrame:
    """Row-level choke metrics: Gilbert coefficient, choke health index, VFM liquid rate, flow regime."""
    d = add_derived(df, inj_is_gas)
    valid = (d["Choke_Size"] > 0) & (d["Liquid_Rate"] > 0) & (d["Wellhead_Pressure"] > 0)
    d["Gilbert_Coeff"] = np.where(valid, gilbert_coefficient(d["Wellhead_Pressure"], d["Liquid_Rate"],
                                                             d["GLR"], d["Choke_Size"].fillna(1)), np.nan)
    d["Gilbert_WHP"] = np.where(valid, gilbert_k(d["GLR"], d["Choke_Size"].fillna(1)) * d["Liquid_Rate"], np.nan)
    parts = []
    for _, g in d.groupby("Well", sort=False):
        g = g.sort_values("Date").copy()
        n_base = max(5, int(0.3 * len(g)))
        base = g["Gilbert_Coeff"].iloc[:n_base].median()
        g["Coeff_Baseline"] = base
        g["Choke_Health"] = g["Gilbert_Coeff"].rolling(7, min_periods=3).median() / base if base and base > 0 else np.nan
        # Periodic well tests (every 14 observations) supply the GLR used by the virtual flow meter
        g["GLR_Test"] = g["GLR"].where(np.arange(len(g)) % 14 == 0).ffill()
        ok = (g["Choke_Size"] > 0) & np.isfinite(g["GLR_Test"]) & (base > 0)
        g["VFM_Liquid"] = np.where(ok, gilbert_liquid_rate(g["Wellhead_Pressure"], g["GLR_Test"],
                                                           g["Choke_Size"].fillna(1), base if base > 0 else 1), np.nan)
        parts.append(g)
    d = pd.concat(parts, ignore_index=True)
    if has_flp:
        d["Pressure_Ratio"] = d["Flowline_Pressure"] / d["Wellhead_Pressure"].replace(0, np.nan)
    else:
        d["Pressure_Ratio"] = np.nan
    return d


def choke_status(health: float, tol_pct: float) -> str:
    if not np.isfinite(health):
        return "Insufficient data"
    if health < 1 - tol_pct / 100:
        return "Possible erosion / washout"
    if health > 1 + tol_pct / 100:
        return "Possible restriction (sand / scale / hydrate)"
    return "Consistent with Gilbert"


def choke_summary(cf: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    rows = []
    for well, g in cf.groupby("Well", sort=True):
        g = g.sort_values("Date")
        last = g.iloc[-1]
        recent = g.tail(30)
        vfm_ok = recent["VFM_Liquid"].notna() & (recent["Liquid_Rate"] > 0)
        vfm_err = (np.mean(np.abs(recent.loc[vfm_ok, "VFM_Liquid"] - recent.loc[vfm_ok, "Liquid_Rate"])
                           / recent.loc[vfm_ok, "Liquid_Rate"]) * 100) if vfm_ok.any() else np.nan
        ratio = last.get("Pressure_Ratio", np.nan)
        if not np.isfinite(ratio):
            regime = "Unknown (no flowline pressure)"
        elif ratio <= settings.critical_ratio:
            regime = "Critical"
        else:
            regime = "Sub-critical"
        health = float(last["Choke_Health"]) if np.isfinite(last["Choke_Health"]) else np.nan
        rows.append({"Well": well, "Choke_Size": last["Choke_Size"],
                     "Choke_Changes": int((g["Choke_Size"].diff().fillna(0) != 0).sum()),
                     "Choke_Health": health, "Choke_Status": choke_status(health, settings.choke_tol),
                     "Flow_Regime": regime, "Pressure_Ratio": ratio, "VFM_Error_Pct": vfm_err})
    return pd.DataFrame(rows)


def choke_change_log(cf: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    events = []
    for well, g in cf.groupby("Well", sort=True):
        g = g.sort_values("Date").reset_index(drop=True)
        changes = g.index[g["Choke_Size"].diff().fillna(0) != 0]
        for i in changes:
            before, after = g.iloc[max(0, i - window):i], g.iloc[i + 1:i + 1 + window]
            if before.empty or after.empty:
                continue
            frm, to = g.loc[i - 1, "Choke_Size"], g.loc[i, "Choke_Size"]
            d_oil = after["Oil_Rate"].mean() - before["Oil_Rate"].mean()
            events.append({"Well": well, "Date": g.loc[i, "Date"].date(), "From (1/64 in)": frm, "To (1/64 in)": to,
                           "Δ Oil (m³/d)": d_oil,
                           "Δ Water cut (pts)": after["Water_Cut"].mean() - before["Water_Cut"].mean(),
                           "Δ WHP (bar)": after["Wellhead_Pressure"].mean() - before["Wellhead_Pressure"].mean(),
                           "Δ Drawdown proxy BHP (bar)": after["Bottomhole_Pressure"].mean() - before["Bottomhole_Pressure"].mean(),
                           "Oil per 1/64 in": d_oil / (to - frm) if to != frm else np.nan})
    return pd.DataFrame(events)


# ---- 5b. Hybrid physics-guided well model (IPR + tubing + Gilbert choke) ----
@dataclass
class WellModel:
    """Per-well nodal model, parameters learned from operating history.
         IPR:     q = J · (Pr − BHP)
         Tubing:  BHP − WHP = d0 + d1·q + d2·x + d3·x²      (x = injection / inj_scale)
         Choke:   WHP = k(GLR, S, A) · q                     (Gilbert, critical flow)"""
    well: str
    J: float
    pr_now: float
    pr_slope: float          # bar/day
    t_last: float
    d: np.ndarray
    A: float
    wc: float
    gor: float
    inj_is_gas: bool
    inj_scale: float
    has_choke: bool
    whp_now: float
    choke_now: float
    inj_now: float
    inj_max_obs: float
    choke_min_obs: float
    choke_max_obs: float
    flp_now: float
    ipr_r2: float
    tubing_r2: float
    quality: str
    notes: list = field(default_factory=list)

    def solve(self, choke, inj, pr=None, iters: int = 40) -> dict:
        choke = np.atleast_1d(np.asarray(choke, dtype=float))
        inj = np.atleast_1d(np.asarray(inj, dtype=float))
        choke, inj = np.broadcast_arrays(choke, inj)
        pr = np.full(choke.shape, self.pr_now) if pr is None else np.broadcast_to(np.asarray(pr, float), choke.shape)
        x = inj / self.inj_scale
        tub0 = self.d[0] + self.d[2] * x + self.d[3] * x ** 2
        wc = min(max(self.wc, 0.0), 0.99)
        if self.has_choke:
            q = np.maximum(self.J * (pr - tub0) / (1 + self.J * self.d[1]), 1.0)
            k = np.zeros_like(q)
            for _ in range(iters):
                gas = q * (1 - wc) * self.gor + (inj if self.inj_is_gas else 0.0)
                k = gilbert_k(glr_mscf_per_stb(gas, q), np.where(choke > 0, choke, 1.0), self.A)
                q_new = np.maximum(self.J * (pr - tub0) / (1 + self.J * (k + self.d[1])), 0.0)
                q = 0.5 * q + 0.5 * q_new
            q = np.where(choke > 0, q, 0.0)
            whp = np.where(choke > 0, k * q, np.nan)
        else:
            whp = np.full(choke.shape, self.whp_now)
            q = np.maximum(self.J * (pr - whp - tub0) / (1 + self.J * self.d[1]), 0.0)
        bhp = (np.nan_to_num(whp, nan=0.0) + tub0 + self.d[1] * q)
        return {"liquid": q, "oil": q * (1 - wc), "water": q * wc, "whp": whp, "bhp": bhp, "drawdown": pr - bhp}


def _lstsq(X, y, ridge=1e-6):
    X, y = np.asarray(X, float), np.asarray(y, float)
    pen = ridge * np.eye(X.shape[1])
    pen[0, 0] = 0.0
    beta = np.linalg.solve(X.T @ X + pen, X.T @ y)
    pred = X @ beta
    ss = np.sum((y - y.mean()) ** 2)
    return beta, (1 - np.sum((y - pred) ** 2) / ss) if ss > 0 else 0.0


def fit_well_model(g: pd.DataFrame, inj_is_gas: bool, use_choke: bool, inj_scale: float, window: int = 120):
    """Calibrate one well's nodal model from its (recent) history. Returns WellModel or (None, reason)."""
    g = g.sort_values("Date")
    fit = g.tail(window)
    fit = fit[(fit["Liquid_Rate"] > 0) & (fit["Bottomhole_Pressure"] > fit["Wellhead_Pressure"])]
    if len(fit) < 15:
        return None, "fewer than 15 valid observations"
    notes = []
    # 3-observation centred rolling means suppress gauge/meter noise before regression (errors-in-variables)
    sm = fit[["Liquid_Rate", "Bottomhole_Pressure", "Wellhead_Pressure", "Injection_Rate"]].rolling(3, center=True, min_periods=1).mean()
    q, bhp, whp = sm["Liquid_Rate"].to_numpy(), sm["Bottomhole_Pressure"].to_numpy(), sm["Wellhead_Pressure"].to_numpy()
    x = sm["Injection_Rate"].to_numpy() / inj_scale
    t = fit["Days"].to_numpy()

    # Tubing / lift response
    y = bhp - whp
    cols = [np.ones_like(q), q]
    use_x = np.std(x) > 1e-6
    if use_x:
        cols += [x, x ** 2]
    beta, tub_r2 = _lstsq(np.column_stack(cols), y)
    if beta[1] < 0:   # friction cannot be negative — refit without it
        cols.pop(1)
        b2, tub_r2 = _lstsq(np.column_stack(cols), y)
        beta = np.insert(b2, 1, 0.0)
    d = np.zeros(4)
    d[:len(beta)] = beta
    if use_x and d[3] < 0:  # no diminishing return in data → linear lift response
        b2, tub_r2 = _lstsq(np.column_stack([np.ones_like(q), q, x]), y)
        d = np.array([b2[0], max(b2[1], 0.0), b2[2], 0.0])
        notes.append("lift response fitted as linear (no diminishing return visible)")

    # IPR with declining reservoir pressure: q = c0 + c1·t − J·BHP
    c, ipr_r2 = _lstsq(np.column_stack([np.ones_like(q), t, bhp]), q)
    J = -c[2]
    quality = "Good" if ipr_r2 > 0.6 and tub_r2 > 0.5 else "Fair"
    pr_now = (c[0] + c[1] * t[-1]) / J if J > 0 else np.nan
    pr_slope = float(np.clip(c[1] / J, -1.0, 0.2)) if J > 0 else 0.0
    if not (0.05 < J < 100) or not np.isfinite(pr_now) or not (bhp[-30:].max() < pr_now < 3 * bhp.max()):
        pr_now = 1.25 * float(np.median(bhp[-30:]))
        J = float(np.median(q[-30:] / np.maximum(pr_now - bhp[-30:], 1.0)))
        pr_slope = 0.0
        quality = "Low (fallback IPR)"
        notes.append("IPR not identifiable from data; reservoir pressure assumed 1.25 × median BHP")

    # Choke coefficient (current condition, so erosion/restriction is reflected)
    A = GILBERT_A
    has_choke = use_choke and "Choke_Size" in fit.columns and (fit["Choke_Size"] > 0).sum() >= 10
    if has_choke:
        ok = fit["Choke_Size"] > 0
        a_rows = gilbert_coefficient(whp[ok], q[ok], fit.loc[ok, "GLR"], fit.loc[ok, "Choke_Size"])
        a_recent = np.asarray(a_rows)[-30:]
        A = float(np.nanmedian(a_recent)) if np.isfinite(a_recent).any() else GILBERT_A

    recent = g.tail(14)
    wc = float(np.clip((recent["Water_Rate"].sum() / max(recent["Liquid_Rate"].sum(), 1e-6)), 0, 0.99))
    gor = float(recent["Gas_Rate"].sum() / max(recent["Oil_Rate"].sum(), 1e-6))
    last = g.iloc[-1]
    choke_hist = g["Choke_Size"] if "Choke_Size" in g.columns else pd.Series([np.nan])
    flp = float(last["Flowline_Pressure"]) if "Flowline_Pressure" in g.columns else np.nan
    return WellModel(
        well=str(last["Well"]), J=float(J), pr_now=float(pr_now), pr_slope=pr_slope, t_last=float(t[-1]), d=d,
        A=A, wc=wc, gor=gor, inj_is_gas=inj_is_gas, inj_scale=inj_scale, has_choke=has_choke,
        whp_now=float(np.median(whp[-7:])), choke_now=float(last.get("Choke_Size", np.nan)),
        inj_now=float(last["Injection_Rate"]), inj_max_obs=float(g["Injection_Rate"].max()),
        choke_min_obs=float(choke_hist[choke_hist > 0].min()) if (choke_hist > 0).any() else np.nan,
        choke_max_obs=float(choke_hist.max()) if choke_hist.notna().any() else np.nan,
        flp_now=flp, ipr_r2=float(ipr_r2), tubing_r2=float(tub_r2), quality=quality, notes=notes), None


def _inj_scale(df: pd.DataFrame) -> float:
    med = float(df.loc[df["Injection_Rate"] > 0, "Injection_Rate"].median()) if (df["Injection_Rate"] > 0).any() else 1.0
    return 1000.0 if med > 2000 else 1.0


@st.cache_data(show_spinner="Calibrating hybrid well models…", max_entries=4)
def fit_well_models(df: pd.DataFrame, inj_is_gas: bool, use_choke: bool):
    d = add_derived(df, inj_is_gas)
    scale = _inj_scale(d)
    models, skipped = {}, {}
    for well, g in d.groupby("Well", sort=True):
        m, reason = fit_well_model(g, inj_is_gas, use_choke, scale)
        if m is None:
            skipped[well] = reason
        else:
            models[well] = m
    return models, skipped


@st.cache_data(show_spinner="Validating models on held-out data…", max_entries=4)
def validate_models(df: pd.DataFrame, inj_is_gas: bool, use_choke: bool):
    """Chronological 80/20 split. Compares persistence, ML proxy (Random Forest) and hybrid nodal model."""
    d = add_derived(df, inj_is_gas).sort_values(["Well", "Date"]).reset_index(drop=True)
    d["Oil_Lag1"] = d.groupby("Well")["Oil_Rate"].shift(1)
    d["WC_Lag1"] = d.groupby("Well")["Water_Cut"].shift(1)
    d["GOR_Lag1"] = d.groupby("Well")["GOR"].shift(1)
    split = d["Date"].quantile(0.8)
    scale = _inj_scale(d)

    # --- ML proxy: setpoints + measured pressures + lagged fluid state (no same-day rate leakage)
    feats = ["Wellhead_Pressure", "Bottomhole_Pressure", "Injection_Rate", "WC_Lag1", "GOR_Lag1"]
    if use_choke and "Choke_Size" in d.columns:
        feats.insert(3, "Choke_Size")
    ml = d.dropna(subset=feats + ["Oil_Lag1"])
    train, test = ml[ml["Date"] <= split], ml[ml["Date"] > split]
    if len(train) < 10 or len(test) < 2:
        raise ValueError("Not enough rows for chronological train/test validation.")
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1)
    rf.fit(train[feats], train["Oil_Rate"])
    test = test.copy()
    test["RF_Pred"] = rf.predict(test[feats])
    test["Persistence_Pred"] = test["Oil_Lag1"]

    # --- Hybrid model: calibrate on train only, forecast test using actual setpoints
    test["Hybrid_Pred"] = np.nan
    for well, g in d.groupby("Well", sort=False):
        m, _ = fit_well_model(g[g["Date"] <= split], inj_is_gas, use_choke, scale)
        idx = test.index[test["Well"] == well]
        if m is None or len(idx) == 0:
            continue
        rows = test.loc[idx]
        pr = m.pr_now + m.pr_slope * (rows["Days"].to_numpy() - m.t_last)
        choke = rows["Choke_Size"].to_numpy() if m.has_choke else np.full(len(rows), np.nan)
        out = m.solve(choke, rows["Injection_Rate"].to_numpy(), pr=pr)
        test.loc[idx, "Hybrid_Pred"] = out["oil"]

    def metrics(col):
        ok = test[col].notna()
        y, p = test.loc[ok, "Oil_Rate"], test.loc[ok, col]
        if len(y) < 2:
            return {"MAE (m³/d)": np.nan, "R²": np.nan, "MAPE (%)": np.nan, "Coverage (%)": 0.0}
        return {"MAE (m³/d)": mean_absolute_error(y, p), "R²": r2_score(y, p),
                "MAPE (%)": float(np.mean(np.abs(y - p) / np.maximum(y, 1e-6)) * 100),
                "Coverage (%)": 100 * ok.mean()}

    table = pd.DataFrame([
        {"Model": "Persistence (yesterday's rate)", **metrics("Persistence_Pred")},
        {"Model": "ML proxy · Random Forest", **metrics("RF_Pred")},
        {"Model": "Hybrid nodal model (physics + learned parameters)", **metrics("Hybrid_Pred")},
    ])
    importance = pd.DataFrame({"Feature": feats, "Importance": rf.feature_importances_}).sort_values("Importance")
    return table, test[["Date", "Well", "Oil_Rate", "Persistence_Pred", "RF_Pred", "Hybrid_Pred"]], importance


# ---- 5c. Equipment anomaly screening ----------------------------------------
@st.cache_data(show_spinner="Screening equipment signals…", max_entries=4)
def screen_equipment(df: pd.DataFrame, sensors: tuple):
    """Per-well robust z-scores vs each well's own baseline (first 60 %), then Isolation Forest.
    Returns row-level scores and a per-well latest summary with the dominant signal."""
    sensors = list(sensors)
    if len(sensors) < 2:
        return None, None
    d = df[["Date", "Well"] + sensors].dropna().sort_values(["Well", "Date"]).copy()
    if len(d) < 30:
        return None, None
    z = pd.DataFrame(index=d.index, columns=sensors, dtype=float)
    d["Baseline"] = False
    for _, idx in d.groupby("Well").groups.items():
        g = d.loc[idx]
        n = max(10, int(0.6 * len(g)))
        base = g.iloc[:n]
        d.loc[base.index, "Baseline"] = True
        med = base[sensors].median()
        spread = (base[sensors].quantile(0.75) - base[sensors].quantile(0.25)) / 1.349
        spread = spread.where(spread > 1e-9, base[sensors].std().fillna(1.0)).replace(0, 1.0)
        z.loc[idx] = ((g[sensors] - med) / spread).to_numpy()
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    iso.fit(z.loc[d["Baseline"]])
    d["Anomaly_Score"] = -iso.score_samples(z)
    threshold = float(np.quantile(d.loc[d["Baseline"], "Anomaly_Score"], 0.995))
    d["Row_Alert"] = d["Anomaly_Score"] > threshold
    for s in sensors:
        d[f"z_{s}"] = z[s].astype(float)
    summary = []
    for well, g in d.groupby("Well", sort=True):
        tail = g.tail(3)
        score = float(tail["Anomaly_Score"].mean())
        zl = tail[[f"z_{s}" for s in sensors]].mean()
        driver = zl.abs().idxmax().replace("z_", "")
        summary.append({"Well": well, "Anomaly_Score": score, "Anomaly": "ALERT" if score > threshold else "NORMAL",
                        "Driver": f"{driver.replace('_', ' ')} ({zl[f'z_{driver}']:+.1f}σ)"})
    return d, pd.DataFrame(summary)


# ---- 5d. Well snapshot & priority scoring -----------------------------------
def build_latest(df, cf_summary, eq_summary, models, settings: Settings) -> pd.DataFrame:
    d = df.sort_values(["Well", "Date"])
    latest = d.groupby("Well", as_index=False).tail(1).copy()
    latest["Liquid_Rate"] = latest["Oil_Rate"] + latest["Water_Rate"]
    latest["Water_Cut"] = (100 * latest["Water_Rate"] / latest["Liquid_Rate"].replace(0, np.nan)).fillna(0)
    trend = []
    for well, g in d.groupby("Well"):
        r = g.tail(30)["Oil_Rate"].rolling(5, min_periods=1).mean()
        trend.append({"Well": well, "Oil_Trend_Pct": (r.iloc[-1] / r.iloc[0] - 1) * 100 if len(r) > 1 and r.iloc[0] > 0 else 0.0})
    latest = latest.merge(pd.DataFrame(trend), on="Well", how="left")
    if eq_summary is not None:
        latest = latest.merge(eq_summary, on="Well", how="left")
        latest["Anomaly"] = latest["Anomaly"].fillna("NO DATA")
    else:
        latest["Anomaly"], latest["Anomaly_Score"], latest["Driver"] = "NO DATA", np.nan, ""
    if cf_summary is not None:
        latest = latest.merge(cf_summary.drop(columns=["Choke_Size"]), on="Well", how="left")
    else:
        latest["Choke_Health"], latest["Choke_Status"], latest["Flow_Regime"] = np.nan, "No choke data", "—"
    latest["Drawdown"] = latest["Well"].map({w: m.pr_now for w, m in models.items()}) - latest["Bottomhole_Pressure"]

    def score(r):
        pts, why = 0, []
        if r["Water_Cut"] >= settings.wc_high:
            pts += 2; why.append(f"water cut {r['Water_Cut']:.0f}% ≥ {settings.wc_high:.0f}%")
        elif r["Water_Cut"] >= settings.wc_watch:
            pts += 1; why.append(f"water cut {r['Water_Cut']:.0f}%")
        if r["Oil_Trend_Pct"] <= settings.decline_high:
            pts += 2; why.append(f"oil {r['Oil_Trend_Pct']:+.0f}% over 30 obs")
        elif r["Oil_Trend_Pct"] <= settings.decline_watch:
            pts += 1; why.append(f"oil {r['Oil_Trend_Pct']:+.0f}% over 30 obs")
        if r["Anomaly"] == "ALERT":
            pts += 2; why.append(f"equipment anomaly: {r.get('Driver', '')}")
        status = str(r.get("Choke_Status", ""))
        if status.startswith("Possible"):
            pts += 1; why.append(status.lower())
        if str(r.get("Flow_Regime", "")).startswith("Sub"):
            pts += 1; why.append("choke in sub-critical flow")
        level = "CRITICAL" if pts >= 4 else "HIGH" if pts >= 2 else "WATCH" if pts == 1 else "STABLE"
        return pd.Series({"Score": pts, "Priority": level, "Reasons": "; ".join(why) or "no trigger"})

    latest = pd.concat([latest.reset_index(drop=True), latest.apply(score, axis=1).reset_index(drop=True)], axis=1)
    return latest


# =============================================================================
# 6. OPTIMISER — joint choke size + lift-gas allocation
# =============================================================================
@dataclass
class OptConfig:
    budget: float
    optimise_injection: bool
    optimise_choke: bool
    max_choke_step: int = 8
    choke_increment: int = 2
    inj_min_frac: float = 0.5
    inj_max_frac: float = 1.5
    water_penalty: float = 0.0
    enforce_critical: bool = True
    critical_ratio: float = CRITICAL_FLOW_RATIO
    max_drawdown: float | None = None
    units: int = 80


def _candidate_grid(m: WellModel, cfg: OptConfig, unit_size: float):
    if cfg.optimise_choke and m.has_choke and np.isfinite(m.choke_now):
        lo = max(8.0, m.choke_min_obs - 4, m.choke_now - cfg.max_choke_step)
        hi = min(128.0, m.choke_max_obs + 8, m.choke_now + cfg.max_choke_step)
        chokes = np.unique(np.concatenate([np.arange(np.ceil(lo), hi + 0.1, cfg.choke_increment), [m.choke_now]]))
    else:
        chokes = np.array([m.choke_now])
    if cfg.optimise_injection and unit_size > 0:
        lo_u = int(np.floor(cfg.inj_min_frac * m.inj_now / unit_size))
        hi_u = int(np.ceil(min(cfg.inj_max_frac * m.inj_now, 1.2 * m.inj_max_obs) / unit_size))
        levels = np.arange(max(lo_u, 0), max(hi_u, lo_u) + 1)
    else:
        levels = np.array([0])
    return chokes, levels


def optimise_field(models: dict, wells: list, cfg: OptConfig):
    """Exact dynamic programme over wells: each well picks one (choke, injection-level) pair;
    total injection ≤ budget; objective = Σ oil − λ·Σ water, subject to flow-regime and drawdown limits."""
    unit = cfg.budget / cfg.units if cfg.optimise_injection and cfg.budget > 0 else 0.0
    per_well, notes = [], []
    for w in wells:
        m = models[w]
        chokes, levels = _candidate_grid(m, cfg, unit)
        C, L = np.meshgrid(chokes, levels, indexing="ij")
        inj = L * unit if unit > 0 else np.full(C.shape, m.inj_now)
        out = m.solve(C.ravel(), inj.ravel())
        feasible = np.ones(C.size, bool)
        if cfg.enforce_critical and m.has_choke and np.isfinite(m.flp_now):
            feasible &= (m.flp_now / np.maximum(out["whp"], 1e-6)) <= cfg.critical_ratio
        if cfg.max_drawdown:
            feasible &= out["drawdown"] <= cfg.max_drawdown
        value = np.where(feasible, out["oil"] - cfg.water_penalty * out["water"], -np.inf).reshape(C.shape)
        best_choke_idx = np.argmax(value, axis=0)       # best choke for each injection level
        best_val = value[best_choke_idx, np.arange(len(levels))]
        if not np.isfinite(best_val).any():
            notes.append(f"{w}: no candidate met the constraints — current settings kept.")
        per_well.append({"well": w, "levels": levels, "values": best_val, "chokes": chokes[best_choke_idx]})

    if unit > 0:
        U = cfg.units
        dp = np.full(U + 1, -np.inf)
        dp[0] = 0.0
        choice = []
        for pw in per_well:
            new, arg = np.full(U + 1, -np.inf), np.full(U + 1, -1, int)
            for li, (lv, val) in enumerate(zip(pw["levels"], pw["values"])):
                if not np.isfinite(val) or lv > U:
                    continue
                shifted = np.full(U + 1, -np.inf)
                shifted[lv:] = dp[:U + 1 - lv] + val
                better = shifted > new
                new[better], arg[better] = shifted[better], li
            choice.append(arg)
            dp = new
        if not np.isfinite(dp).any():
            return None, notes + ["No feasible allocation within the lift-gas budget. Relax constraints or increase budget."]
        b = int(np.nanargmax(np.where(np.isfinite(dp), dp, -np.inf)))
        picks = [None] * len(per_well)
        for i in range(len(per_well) - 1, -1, -1):
            li = choice[i][b]
            if li < 0:
                return None, notes + ["Allocation could not be reconstructed — relax constraints."]
            picks[i] = li
            b -= per_well[i]["levels"][li]
    else:
        picks = [int(np.argmax(np.where(np.isfinite(pw["values"]), pw["values"], -np.inf))) for pw in per_well]

    rows = []
    for pw, li in zip(per_well, picks):
        m = models[pw["well"]]
        feasible = np.isfinite(pw["values"][li])
        new_choke = pw["chokes"][li] if feasible else m.choke_now
        new_inj = pw["levels"][li] * unit if (unit > 0 and feasible) else m.inj_now
        cur = m.solve(m.choke_now, m.inj_now)
        new = m.solve(new_choke, new_inj)
        rows.append({"Well": pw["well"], "Current choke": m.choke_now, "Recommended choke": new_choke,
                     "Current injection": m.inj_now, "Recommended injection": new_inj,
                     "Model oil now": cur["oil"][0], "Model oil recommended": new["oil"][0],
                     "Δ Oil": new["oil"][0] - cur["oil"][0], "Δ Water": new["water"][0] - cur["water"][0],
                     "WHP recommended (bar)": new["whp"][0], "Drawdown recommended (bar)": new["drawdown"][0]})
    return pd.DataFrame(rows), notes


# =============================================================================
# 7. UI HELPERS & STYLING
# =============================================================================
def _stretch() -> dict:
    try:
        major, minor = (int(p) for p in st.__version__.split(".")[:2])
        return {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}
    except Exception:
        return {"use_container_width": True}


W = _stretch()

CSS = """
<style>
.stApp {background:radial-gradient(circle at 84% 7%,rgba(73,174,158,.13),transparent 25%),radial-gradient(circle at 12% 88%,rgba(45,103,133,.13),transparent 30%),linear-gradient(135deg,#061017 0%,#091720 50%,#071018 100%);color:#EAF2F7;}
.stApp:before {content:"";position:fixed;inset:0;pointer-events:none;opacity:.16;background-image:linear-gradient(rgba(112,171,184,.055) 1px,transparent 1px),linear-gradient(90deg,rgba(112,171,184,.055) 1px,transparent 1px);background-size:42px 42px;}
.block-container{padding-top:1.2rem;max-width:1450px;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#07131b 0%,#091821 100%);border-right:1px solid #20323d;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"]{gap:5px;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label{border-radius:9px;padding:7px 9px;}
.hero{padding:28px 34px;border:1px solid #233541;border-radius:20px;background:linear-gradient(135deg,#0c1d28,#0a141b);margin-bottom:20px;}
.brand{font-size:14px;letter-spacing:4px;font-weight:800;color:#77D6C8;}
.hero h1{font-size:38px;margin:8px 0 6px;}
.hero p{font-size:16px;color:#AFC1CB;max-width:950px;}
.badge{display:inline-block;padding:6px 10px;border-radius:20px;background:#14322f;color:#86E3D2;font-size:12px;font-weight:700;margin-right:6px;}
.badge.demo{background:#2b2412;color:#E7B85B;}
.kpi{background:#0d1b24;border:1px solid #223541;border-radius:15px;padding:16px;min-height:104px;margin-bottom:8px;}
.kpi-label{color:#8EA3AE;font-size:12px;text-transform:uppercase;letter-spacing:1.2px;}
.kpi-value{font-size:26px;font-weight:800;margin-top:6px;overflow-wrap:anywhere;}
.kpi-sub{color:#718792;font-size:12px;margin-top:4px;}
.kpi.good .kpi-value{color:#6EE7C8;} .kpi.bad .kpi-value{color:#FF8A8A;} .kpi.warn .kpi-value{color:#F2C46D;}
.card{background:#0d1b24;border:1px solid #223541;border-radius:15px;padding:18px;margin-bottom:10px;}
.section-kicker{color:#65CFC0;text-transform:uppercase;letter-spacing:2px;font-size:11px;font-weight:800;margin-top:12px;}
.section-title{margin-top:2px;}
.alert{border-left:4px solid #65CFC0;background:#0d2026;padding:14px 16px;border-radius:8px;margin:8px 0;}
.warn{border-left-color:#E7B85B;background:#251f11;}
.small{color:#8EA3AE;font-size:13px;}
</style>
"""


def kpi(label, value, sub="", tone=""):
    st.markdown(f'<div class="kpi {tone}"><div class="kpi-label">{html.escape(str(label))}</div>'
                f'<div class="kpi-value">{html.escape(str(value))}</div>'
                f'<div class="kpi-sub">{html.escape(str(sub))}</div></div>', unsafe_allow_html=True)


def section(title, kicker=""):
    st.markdown(f'<div class="section-kicker">{html.escape(kicker)}</div>'
                f'<h3 class="section-title">{html.escape(title)}</h3>', unsafe_allow_html=True)


def callout(body_html: str, warn: bool = False):
    st.markdown(f'<div class="alert{" warn" if warn else ""}">{body_html}</div>', unsafe_allow_html=True)


def style_fig(fig, height=320, **layout):
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=10, r=10, t=45, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), **layout)
    return fig


def show_fig(fig, height=320, **layout):
    st.plotly_chart(style_fig(fig, height, **layout), **W)


def show_df(frame: pd.DataFrame, decimals: int = 2, **kw):
    out = frame.copy()
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(decimals)
    st.dataframe(out, hide_index=True, **W, **kw)


def fmt(v, spec=".1f", unit=""):
    try:
        return f"{v:{spec}}{unit}" if v is not None and np.isfinite(v) else "—"
    except (TypeError, ValueError):
        return "—"


def needs_choke(bundle: DataBundle) -> bool:
    if bundle.has_choke:
        return True
    st.info("No **Choke_Size** column was found in the uploaded data. Add a choke/bean-size column "
            "(e.g. `Choke_Size`, `Bean`, `Choke (1/64 in)`) to unlock choke diagnostics, virtual flow metering "
            "and choke optimisation.")
    return False


# =============================================================================
# 8. PAGES
# =============================================================================
def page_command_centre(b: DataBundle, latest, cf):
    section("Field Operations Command Centre", "EXECUTIVE VIEW")
    oil, water = latest["Oil_Rate"].sum(), latest["Water_Rate"].sum()
    wc = 100 * water / (oil + water) if oil + water else 0
    c = st.columns(5)
    with c[0]: kpi("Producing wells", latest["Well"].nunique(), "latest observation per well")
    with c[1]: kpi("Total oil rate", f"{oil:,.0f} m³/d", f"≈ {oil * M3_TO_STB:,.0f} bbl/d")
    with c[2]: kpi("Field water cut", f"{wc:.1f}%", "liquid-weighted", "warn" if wc > 80 else "")
    n_alert = int((latest["Anomaly"] == "ALERT").sum())
    with c[3]: kpi("Equipment alerts", n_alert if b.sensors else "—", "anomaly screen", "bad" if n_alert else "good")
    n_choke = int(latest["Choke_Status"].astype(str).str.startswith("Possible").sum())
    with c[4]: kpi("Choke flags", n_choke if b.has_choke else "—", "erosion / restriction", "bad" if n_choke else "good")

    if "Energy_kWh" in b.sensors:
        liq = latest["Oil_Rate"] + latest["Water_Rate"]
        se_liq = latest["Energy_kWh"].sum() / max(liq.sum(), 1e-6)
        se_oil = latest["Energy_kWh"].sum() / max(oil, 1e-6)
        c2 = st.columns(3)
        with c2[0]: kpi("Specific energy", f"{se_oil:.2f} kWh/m³ oil", "ESG · energy per unit oil")
        with c2[1]: kpi("Specific energy (liquid)", f"{se_liq:.2f} kWh/m³", "all lifted liquid")
        with c2[2]: kpi("Critical-flow wells", f"{(latest['Flow_Regime'] == 'Critical').sum()} / {len(latest)}"
                        if b.has_flp else "—", "choke isolates well from flowline")

    left, right = st.columns([1.3, 1])
    with left:
        section("Field production trend", "PRODUCTION SURVEILLANCE")
        trend = b.df.groupby("Date")[["Oil_Rate", "Water_Rate"]].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend.Date, y=trend.Oil_Rate, name="Oil", line=dict(color="#6EE7C8")))
        fig.add_trace(go.Scatter(x=trend.Date, y=trend.Water_Rate, name="Water", line=dict(color="#5B9BD5")))
        show_fig(fig, 340, yaxis_title="Rate (m³/d)")
    with right:
        section("Wells requiring attention", "SCREENING QUEUE")
        q = latest.assign(_r=latest["Priority"].map(PRIORITY_ORDER)).sort_values(["_r", "Score"], ascending=[True, False])
        show_df(q[["Well", "Priority", "Oil_Rate", "Water_Cut", "Reasons"]].head(8)
                .rename(columns={"Oil_Rate": "Oil (m³/d)", "Water_Cut": "WC (%)"}), 1)

    section("Choke & equipment signals", "OPERATIONS")
    a, bcol = st.columns(2)
    with a:
        if b.has_choke:
            fig = px.bar(latest, x="Well", y="Choke_Health", color="Choke_Status",
                         title="Choke health index (1.0 = behaves like nominal bean)")
            fig.add_hline(y=1.0, line_dash="dash", line_color="#8EA3AE")
            show_fig(fig, 300)
        else:
            st.info("Upload Choke_Size to see choke health.")
    with bcol:
        if b.sensors:
            fig = px.bar(latest, x="Well", y="Anomaly_Score", color="Anomaly", color_discrete_map=STATUS_COLORS,
                         title="Equipment anomaly score (latest 3 observations)", hover_data=["Driver"])
            show_fig(fig, 300)
        else:
            st.info("No equipment sensor columns uploaded.")
    st.caption("Flags are screening outputs that prioritise investigation; they are not confirmed failures.")


def page_data(b: DataBundle):
    section("Data Input & Quality Gate", "INGESTION")
    if b.error:
        st.error(b.error)
    elif not b.is_demo:
        st.success(f"Mapped, cleaned and loaded **{b.source}**.")
    else:
        st.info("Showing the built-in demo field. Upload a CSV in the sidebar — messy headers, extra columns and "
                "non-standard units are handled.")
    c = st.columns(4)
    with c[0]: kpi("Rows loaded", f"{len(b.df):,}", "well-date observations")
    with c[1]: kpi("Wells", b.df["Well"].nunique(), "unique IDs")
    with c[2]: kpi("History", f"{b.df['Date'].min():%d %b %Y} → {b.df['Date'].max():%d %b %Y}", "date range")
    avail = [c for c in OPTIONAL_COLUMNS if c in b.df.columns]
    with c[3]: kpi("Optional fields", f"{len(avail)} / {len(OPTIONAL_COLUMNS)}", "choke + sensors")

    t1, t2, t3 = st.tabs(["Schema", "Cleaning report", "Preview"])
    with t1:
        schema = pd.DataFrame([{"Column": col, "Required": "Yes" if col in CORE_COLUMNS else "Optional",
                                "Standard unit": STANDARD_UNITS.get(col, "—"), "Meaning": COLUMN_DESCRIPTIONS[col],
                                "Present": "✓" if col in b.df.columns else "✗"} for col in CORE_COLUMNS + OPTIONAL_COLUMNS])
        show_df(schema)
        st.caption("Choke size convention: 1/64 inch (e.g. 32 = ½ in). Sizes in inches or mm are converted. "
                   "Injection_Rate in a gas unit (Sm³/d, Mscf/d …) is treated as lift gas and included in the choke GLR.")
    with t2:
        r = b.report
        if not r:
            st.info("The cleaning report appears after a CSV upload.")
        else:
            c = st.columns(4)
            with c[0]: kpi("Input rows", f"{r.get('input_rows', 0):,}")
            with c[1]: kpi("Duplicates removed", r.get("duplicates_removed", 0) + r.get("conflicting_timestamps_merged", 0))
            with c[2]: kpi("Invalid values", r.get("invalid_values_removed", 0) + r.get("non_numeric_values", 0), "negative / non-numeric")
            with c[3]: kpi("Outliers capped", r.get("outlier_values_capped", 0), "per-well 3×IQR")
            if r.get("mapped_columns"):
                st.markdown("**Column mapping**")
                show_df(pd.DataFrame([{"FIELDWISE field": k, "Source column": v} for k, v in r["mapped_columns"].items()]))
            st.markdown("**Unit conversions**")
            if r.get("unit_conversions"):
                show_df(pd.DataFrame(r["unit_conversions"]))
            else:
                st.caption("No conversion needed — selected units already match the standard.")
            if r.get("imputed_by_column"):
                st.markdown("**Values imputed per column**")
                show_df(pd.DataFrame([{"Column": k, "Imputed": v} for k, v in r["imputed_by_column"].items()]))
            st.markdown(f"**Ignored columns:** {', '.join(r.get('ignored_columns', [])) or 'none'}")
            if r.get("dropped_optional_columns"):
                st.warning("Dropped (> 60 % missing): " + ", ".join(r["dropped_optional_columns"]))
            for n in r.get("notes", []):
                st.caption("• " + n)
    with t3:
        show_df(b.df.head(50))
        st.download_button("Download cleaned data (CSV)", b.df.to_csv(index=False).encode(),
                           file_name="fieldwise_cleaned_data.csv", mime="text/csv")


def page_well(b: DataBundle, latest, cf, models):
    section("Well Intelligence", "WELL-LEVEL SURVEILLANCE")
    well = st.selectbox("Select well", sorted(latest["Well"]), key="well_intel")
    row = latest[latest["Well"] == well].iloc[0]
    hist = add_derived(b.df[b.df["Well"] == well], b.inj_is_gas).sort_values("Date")
    c = st.columns(6)
    with c[0]: kpi("Oil rate", f"{row.Oil_Rate:.1f} m³/d", f"30-obs trend {row.Oil_Trend_Pct:+.1f}%")
    with c[1]: kpi("Water cut", f"{row.Water_Cut:.1f}%", "latest")
    with c[2]: kpi("Choke", fmt(row.get("Choke_Size", np.nan), ".0f", "/64\""), "bean size")
    with c[3]: kpi("WHP / BHP", f"{row.Wellhead_Pressure:.0f} / {row.Bottomhole_Pressure:.0f} bar", "latest")
    with c[4]: kpi("Drawdown", fmt(row.Drawdown, ".0f", " bar"), "Pr (model) − BHP")
    tone = {"CRITICAL": "bad", "HIGH": "warn", "WATCH": "warn", "STABLE": "good"}[row.Priority]
    with c[5]: kpi("Priority", row.Priority, f"score {row.Score}", tone)

    a, bcol = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.Date, y=hist.Oil_Rate, name="Oil (m³/d)", line=dict(color="#6EE7C8")))
        if b.has_choke:
            fig.add_trace(go.Scatter(x=hist.Date, y=hist.Choke_Size, name="Choke (1/64 in)", yaxis="y2",
                                     line=dict(color="#F2C46D", shape="hv", dash="dot")))
            fig.update_layout(yaxis2=dict(title="Choke (1/64 in)", overlaying="y", side="right", showgrid=False))
        show_fig(fig, 330, title="Oil rate vs choke setting", yaxis_title="Oil (m³/d)")
    with bcol:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.Date, y=hist.Water_Cut, name="Water cut (%)", line=dict(color="#5B9BD5")))
        show_fig(fig, 330, title="Water-cut history", yaxis_title="%")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.Date, y=hist.Bottomhole_Pressure, name="BHP"))
    fig.add_trace(go.Scatter(x=hist.Date, y=hist.Wellhead_Pressure, name="WHP"))
    if b.has_flp:
        fig.add_trace(go.Scatter(x=hist.Date, y=hist.Flowline_Pressure, name="Flowline (downstream of choke)"))
    show_fig(fig, 300, title="Pressure profile", yaxis_title="bar")
    section("Screening indicators", "ENGINEERING REVIEW")
    callout("• " + html.escape(row.Reasons).replace("; ", "<br>• "))
    m = models.get(well)
    if m:
        st.caption(f"Hybrid model: J = {m.J:.2f} m³/d/bar · reservoir pressure ≈ {m.pr_now:.0f} bar "
                   f"({m.pr_slope * 30:+.1f} bar/month) · choke coefficient A = {m.A:.0f} · fit quality: {m.quality}.")


def page_choke(b: DataBundle, latest, cf, cf_sum, models, settings: Settings):
    section("Choke Performance & Virtual Flow Metering", "CHOKE INTELLIGENCE")
    if not needs_choke(b):
        return
    st.write("The production choke sets the back-pressure on each well. FIELDWISE AI calibrates the **Gilbert "
             "critical-flow correlation** per well, tracks how the choke *really* behaves versus its nominal bean "
             "size, estimates rate from pressure alone (virtual flow meter) and predicts the response to a bean change.")
    c = st.columns(4)
    with c[0]: kpi("Wells with choke data", int(cf_sum["Choke_Health"].notna().sum()), f"of {len(cf_sum)}")
    flagged = cf_sum["Choke_Status"].str.startswith("Possible").sum()
    with c[1]: kpi("Choke condition flags", int(flagged), f"±{settings.choke_tol:.0f}% deviation", "bad" if flagged else "good")
    with c[2]: kpi("Sub-critical chokes", int((cf_sum["Flow_Regime"] == "Sub-critical").sum()) if b.has_flp else "—",
                   f"ratio > {settings.critical_ratio:.2f}")
    with c[3]: kpi("VFM median error", fmt(cf_sum["VFM_Error_Pct"].median(), ".1f", "%"), "last 30 obs vs measured")

    tabs = st.tabs(["Fleet choke health", "Well choke diagnostics", "Nodal & sensitivity", "Choke change log"])
    with tabs[0]:
        show_df(cf_sum.rename(columns={"Choke_Size": "Choke (1/64 in)", "Choke_Changes": "Changes",
                                       "Choke_Health": "Health index", "Pressure_Ratio": "P_down/P_up",
                                       "VFM_Error_Pct": "VFM error (%)"}), 2)
        fig = px.scatter(latest, x="Choke_Size", y="Oil_Rate", color="Choke_Status", size="Water_Rate",
                         hover_name="Well", title="Oil rate vs current choke size (bubble = water rate)")
        show_fig(fig, 360, xaxis_title="Choke (1/64 in)", yaxis_title="Oil (m³/d)")
        st.caption("Health index = rolling median of the calibrated Gilbert coefficient ÷ the well's own baseline. "
                   "< 1: the bean flows more than its nominal size (erosion/washout). "
                   "> 1: it flows less (sand, scale, wax or hydrate restriction).")
    with tabs[1]:
        well = st.selectbox("Well", sorted(cf["Well"].unique()), key="choke_well")
        g = cf[cf["Well"] == well].sort_values("Date")
        s = cf_sum[cf_sum["Well"] == well].iloc[0]
        c = st.columns(4)
        with c[0]: kpi("Choke", fmt(s.Choke_Size, ".0f", "/64 in"), f"{s.Choke_Changes} change(s) in history")
        tone = "bad" if str(s.Choke_Status).startswith("Possible") else "good"
        with c[1]: kpi("Health index", fmt(s.Choke_Health, ".2f"), s.Choke_Status, tone)
        with c[2]: kpi("Flow regime", s.Flow_Regime.split(" (")[0], f"P_down/P_up = {fmt(s.Pressure_Ratio, '.2f')}")
        with c[3]: kpi("VFM error", fmt(s.VFM_Error_Pct, ".1f", "%"), "last 30 observations")
        a, bcol = st.columns(2)
        with a:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=g.Date, y=g.Choke_Health, name="Health index", line=dict(color="#F2C46D")))
            for yv in (1 - settings.choke_tol / 100, 1 + settings.choke_tol / 100):
                fig.add_hline(y=yv, line_dash="dot", line_color="#E5484D")
            fig.add_hline(y=1.0, line_dash="dash", line_color="#8EA3AE")
            show_fig(fig, 320, title="Choke health index over time")
        with bcol:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=g.Date, y=g.Liquid_Rate, name="Measured liquid", line=dict(color="#6EE7C8")))
            fig.add_trace(go.Scatter(x=g.Date, y=g.VFM_Liquid, name="Virtual flow meter", line=dict(color="#F0883E", dash="dot")))
            show_fig(fig, 320, title="Virtual flow meter (WHP + choke + well-test GLR)", yaxis_title="m³/d")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g.Date, y=g.Wellhead_Pressure, name="Measured WHP"))
        fig.add_trace(go.Scatter(x=g.Date, y=g.Gilbert_WHP, name="Gilbert WHP (nominal bean, A = 435)", line=dict(dash="dot")))
        if b.has_flp:
            fig.add_trace(go.Scatter(x=g.Date, y=g.Flowline_Pressure / settings.critical_ratio,
                                     name="Critical-flow limit", line=dict(dash="dash", color="#E5484D")))
        show_fig(fig, 300, title="Upstream choke pressure: measured vs correlation", yaxis_title="bar")
        st.caption("Below the critical-flow limit the choke no longer isolates the well from flowline pressure, so "
                   "Gilbert-type estimates and the VFM become unreliable.")
    with tabs[2]:
        _nodal_tab(b, models, settings)
    with tabs[3]:
        log = choke_change_log(cf)
        if log.empty:
            st.info("No choke changes found in the history.")
        else:
            show_df(log, 2)
            st.caption("Measured response = mean of 7 observations after the change minus 7 before. "
                       "These real step-tests are the best evidence for choke decisions and are used to sanity-check the model.")


def _nodal_tab(b: DataBundle, models, settings: Settings):
    usable = sorted(w for w, m in models.items() if m.has_choke)
    if not usable:
        st.info("Not enough choke history to calibrate a nodal model.")
        return
    well = st.selectbox("Well", usable, key="nodal_well")
    m = models[well]
    chokes = np.arange(8, max(64, int(m.choke_now) + 16) + 1, 2, dtype=float)
    out = m.solve(chokes, np.full(chokes.shape, m.inj_now))
    sens = pd.DataFrame({"Choke": chokes, "Oil": out["oil"], "Water": out["water"], "WHP": out["whp"],
                         "Drawdown": out["drawdown"]})
    cur = m.solve(m.choke_now, m.inj_now)
    a, bcol = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sens.Choke, y=sens.Oil, name="Oil", line=dict(color="#6EE7C8")))
        fig.add_trace(go.Scatter(x=sens.Choke, y=sens.Water, name="Water", line=dict(color="#5B9BD5")))
        fig.add_trace(go.Scatter(x=[m.choke_now], y=cur["oil"], mode="markers", name="Current",
                                 marker=dict(size=13, color="#F2C46D", symbol="diamond")))
        if np.isfinite(m.choke_min_obs):
            fig.add_vrect(x0=m.choke_min_obs, x1=m.choke_max_obs, fillcolor="#3FB6A8", opacity=0.08,
                          line_width=0, annotation_text="history range")
        show_fig(fig, 340, title="Choke sensitivity (current lift gas)", xaxis_title="Choke (1/64 in)", yaxis_title="m³/d")
    with bcol:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sens.Choke, y=sens.WHP, name="WHP"))
        fig.add_trace(go.Scatter(x=sens.Choke, y=sens.Drawdown, name="Drawdown"))
        if np.isfinite(m.flp_now):
            fig.add_hline(y=m.flp_now / settings.critical_ratio, line_dash="dash", line_color="#E5484D",
                          annotation_text="critical-flow WHP limit")
        show_fig(fig, 340, title="Pressure response to choke", xaxis_title="Choke (1/64 in)", yaxis_title="bar")

    # Nodal plot: IPR vs outflow (tubing + choke) for a few bean sizes
    qmax = m.J * m.pr_now
    q = np.linspace(1, max(qmax * 0.98, 10), 120)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q, y=m.pr_now - q / m.J, name="Inflow (IPR)", line=dict(color="#6EE7C8", width=3)))
    x = m.inj_now / m.inj_scale
    tub0 = m.d[0] + m.d[2] * x + m.d[3] * x ** 2
    for s_ in sorted({max(8, m.choke_now - 8), m.choke_now, m.choke_now + 8}):
        gas = q * (1 - m.wc) * m.gor + (m.inj_now if m.inj_is_gas else 0)
        whp = gilbert_k(glr_mscf_per_stb(gas, q), s_, m.A) * q
        fig.add_trace(go.Scatter(x=q, y=whp + tub0 + m.d[1] * q, name=f"Outflow · {s_:.0f}/64\"",
                                 line=dict(dash="solid" if s_ == m.choke_now else "dot")))
    fig.add_trace(go.Scatter(x=cur["liquid"], y=cur["bhp"], mode="markers", name="Operating point",
                             marker=dict(size=13, color="#F2C46D", symbol="diamond")))
    show_fig(fig, 380, title=f"{well} · nodal analysis at the bottom-hole node", xaxis_title="Liquid rate (m³/d)",
             yaxis_title="Flowing BHP (bar)", yaxis_range=[0, m.pr_now * 1.05])
    st.caption("Intersection of inflow and outflow = predicted operating point. A larger bean lowers the outflow "
               "curve, raising rate and drawdown — watch drawdown for sand-prone completions.")


def page_equipment(b: DataBundle, latest, eq_rows):
    section("Equipment Health & Anomaly Screening", "ARTIFICIAL-LIFT SURVEILLANCE")
    if eq_rows is None:
        st.info("At least two equipment sensor columns (vibration, motor temperature, current, frequency, energy) "
                "are needed. No synthetic sensor data is generated — alerts are only raised on real measurements.")
        return
    st.write("Each well is compared with **its own** baseline (first 60 % of history) using robust z-scores; an "
             "Isolation Forest then scores unusual sensor combinations. The dominant signal is reported for each alert.")
    c = st.columns(3)
    with c[0]: kpi("Wells flagged", int((latest["Anomaly"] == "ALERT").sum()), "latest 3 observations",
                   "bad" if (latest["Anomaly"] == "ALERT").any() else "good")
    with c[1]: kpi("Wells screened", int((latest["Anomaly"] != "NO DATA").sum()))
    with c[2]: kpi("Sensor variables", len(b.sensors), ", ".join(s.replace("_", " ").lower() for s in b.sensors))
    show_df(latest[["Well", "Anomaly", "Anomaly_Score", "Driver"] + b.sensors].sort_values("Anomaly_Score", ascending=False), 2)
    well = st.selectbox("Inspect well", sorted(latest["Well"]), key="eq_well")
    g = eq_rows[eq_rows["Well"] == well]
    cols = st.columns(2)
    for i, s in enumerate(b.sensors[:4]):
        with cols[i % 2]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=g.Date, y=g[s], name=s.replace("_", " "), line=dict(color="#6EE7C8")))
            al = g[g["Row_Alert"]]
            fig.add_trace(go.Scatter(x=al.Date, y=al[s], mode="markers", name="Anomalous",
                                     marker=dict(color="#E5484D", size=7)))
            show_fig(fig, 260, title=f"{well} · {s.replace('_', ' ')}")
    callout("<b>Engineering caution</b><br>An anomaly is not proof of imminent failure. Confirm with equipment "
            "limits, maintenance logs and operating context (e.g. a choke change also shifts motor load).", warn=True)


def page_optimisation(b: DataBundle, latest, models, settings: Settings):
    section("Multi-Well Production Optimisation", "CHOKE + LIFT-GAS SCENARIO LAB")
    if not models:
        st.warning("No well could be calibrated, so optimisation is unavailable.")
        return
    inj_ok, choke_ok = b.injection_available, b.has_choke
    if not inj_ok and not choke_ok:
        st.warning("Neither lift-gas/injection nor choke data are available — nothing to optimise.")
        return
    st.write("A **hybrid nodal model** per well (learned IPR + tubing/lift response + calibrated Gilbert choke) "
             "predicts oil, water, WHP and drawdown for any choke / lift-gas combination. An exact dynamic "
             "programme then searches every combination across wells under the field lift-gas limit.")
    wells_all = sorted(models)
    wells = st.multiselect("Wells included", wells_all, default=wells_all)
    if not wells:
        st.info("Select at least one well.")
        return
    cur_total = float(sum(models[w].inj_now for w in wells))
    t_opt, t_manual = st.tabs(["Automatic optimiser", "Manual scenario"])
    with t_opt:
        c = st.columns(3)
        with c[0]:
            opt_choke = st.checkbox("Optimise choke size", value=choke_ok, disabled=not choke_ok)
            max_step = st.slider("Max choke change per well (1/64 in)", 2, 16, 8, 2, disabled=not choke_ok)
        with c[1]:
            opt_inj = st.checkbox("Optimise lift-gas allocation", value=inj_ok, disabled=not inj_ok)
            budget = st.number_input("Lift-gas available (field total)", min_value=0.0, value=round(cur_total, 1),
                                     step=max(1.0, round(cur_total * 0.05, 0)), disabled=not inj_ok)
        with c[2]:
            penalty = st.slider("Water-handling penalty (m³ oil per m³ water)", 0.0, 0.5, 0.05, 0.01,
                                help="Discourages solutions that buy oil with a lot of extra water.")
            enforce = st.checkbox("Keep chokes in critical flow", value=b.has_flp, disabled=not b.has_flp)
            dd_on = st.checkbox("Limit drawdown (sand control)", value=True)
        dd_now = latest["Drawdown"].dropna()
        dd_limit = None
        if dd_on and not dd_now.empty:
            dd_max = float(max(dd_now.max() * 1.6, 20.0))
            dd_limit = st.slider("Max drawdown (bar)", 5.0, dd_max,
                                 float(min(max(round(dd_now.max() * 1.1, 0), 5.0), dd_max)), 1.0)
        if st.button("RUN OPTIMISER", type="primary", **W):
            cfg = OptConfig(budget=budget, optimise_injection=opt_inj and inj_ok, optimise_choke=opt_choke and choke_ok,
                            max_choke_step=max_step, water_penalty=penalty, enforce_critical=enforce,
                            critical_ratio=settings.critical_ratio, max_drawdown=dd_limit)
            res, notes = optimise_field(models, wells, cfg)
            for n in notes:
                st.warning(n)
            if res is not None:
                _show_scenario_result(res, cur_total if cfg.optimise_injection else None)
    with t_manual:
        base = pd.DataFrame({"Well": wells,
                             "Current choke": [models[w].choke_now for w in wells],
                             "Proposed choke": [models[w].choke_now for w in wells],
                             "Current injection": [models[w].inj_now for w in wells],
                             "Proposed injection": [round(models[w].inj_now, 1) for w in wells]})
        edited = st.data_editor(base, hide_index=True, **W, key="manual_editor",
                                disabled=["Well", "Current choke", "Current injection"],
                                column_config={"Proposed choke": st.column_config.NumberColumn(min_value=0, max_value=128, step=1),
                                               "Proposed injection": st.column_config.NumberColumn(min_value=0)})
        if st.button("EVALUATE SCENARIO", type="primary", **W):
            rows = []
            for _, r in edited.iterrows():
                m = models[r["Well"]]
                ch = float(r["Proposed choke"]) if choke_ok and pd.notna(r["Proposed choke"]) else m.choke_now
                inj = float(r["Proposed injection"]) if inj_ok and pd.notna(r["Proposed injection"]) else m.inj_now
                cur, new = m.solve(m.choke_now, m.inj_now), m.solve(ch, inj)
                rows.append({"Well": r["Well"], "Current choke": m.choke_now, "Recommended choke": ch,
                             "Current injection": m.inj_now, "Recommended injection": inj,
                             "Model oil now": cur["oil"][0], "Model oil recommended": new["oil"][0],
                             "Δ Oil": new["oil"][0] - cur["oil"][0], "Δ Water": new["water"][0] - cur["water"][0],
                             "WHP recommended (bar)": new["whp"][0], "Drawdown recommended (bar)": new["drawdown"][0]})
            res = pd.DataFrame(rows)
            total_new = res["Recommended injection"].sum()
            if inj_ok and total_new > cur_total * 1.001:
                st.warning(f"Proposed lift gas {total_new:,.0f} exceeds the current total {cur_total:,.0f}.")
            if b.has_flp:
                flp = {w: models[w].flp_now for w in wells}
                sub = res[res.apply(lambda r: flp[r.Well] / max(r["WHP recommended (bar)"], 1e-6) > settings.critical_ratio
                                    if np.isfinite(r["WHP recommended (bar)"]) else False, axis=1)]
                if not sub.empty:
                    st.warning("Sub-critical choke flow predicted for: " + ", ".join(sub.Well))
            _show_scenario_result(res, cur_total if inj_ok else None)
    callout("<b>Before field use:</b> recommendations are model-based scenarios. Step-test choke changes "
            "(2–4/64 in at a time), confirm with a well test, and respect sand, erosion-velocity, compressor and "
            "separator limits. No commands are sent to the field.", warn=True)


def _show_scenario_result(res: pd.DataFrame, cur_total):
    now, new = res["Model oil now"].sum(), res["Model oil recommended"].sum()
    c = st.columns(4)
    with c[0]: kpi("Oil — current (model)", f"{now:,.1f} m³/d", "selected wells")
    with c[1]: kpi("Oil — scenario", f"{new:,.1f} m³/d", "hybrid-model estimate")
    with c[2]: kpi("Δ Oil", f"{new - now:+,.1f} m³/d", f"{(new / now - 1) * 100 if now else 0:+.1f}% · ≈ {(new - now) * M3_TO_STB:+,.0f} bbl/d",
                   "good" if new >= now else "bad")
    with c[3]: kpi("Δ Water", f"{res['Δ Water'].sum():+,.1f} m³/d",
                   f"lift gas used {res['Recommended injection'].sum():,.0f}" + (f" / {cur_total:,.0f}" if cur_total else ""))
    show_df(res, 1)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=res.Well, y=res["Model oil now"], name="Current", marker_color="#56707C"))
    fig.add_trace(go.Bar(x=res.Well, y=res["Model oil recommended"], name="Scenario", marker_color="#6EE7C8"))
    show_fig(fig, 340, barmode="group", yaxis_title="Oil (m³/d)", title="Oil rate per well")
    st.download_button("Download recommendations (CSV)", res.to_csv(index=False).encode(),
                       file_name="fieldwise_recommendations.csv", mime="text/csv")


def page_analytics(b: DataBundle, latest):
    section("Field Analytics", "WELL COMPARISON")
    metrics = ["Oil_Rate", "Water_Cut", "Injection_Rate", "Bottomhole_Pressure", "Wellhead_Pressure", "Drawdown"]
    if b.has_choke:
        metrics += ["Choke_Size", "Choke_Health"]
    metrics += [s for s in b.sensors]
    metric = st.selectbox("Metric", metrics)
    fig = px.bar(latest.sort_values(metric), x="Well", y=metric, color="Priority", color_discrete_map=PRIORITY_COLORS,
                 hover_data=["Oil_Rate", "Water_Cut", "Reasons"], title=f"Latest {metric.replace('_', ' ')} by well")
    show_fig(fig, 380)
    section("Opportunity quadrant", "WHERE TO ACT")
    fig = px.scatter(latest, x="Water_Cut", y="Oil_Rate", size="Liquid_Rate", color="Priority", text="Well",
                     color_discrete_map=PRIORITY_COLORS, hover_data=["Reasons"],
                     title="High oil & low water cut = protect · high water cut & declining = candidate for choke-back / WSO")
    fig.update_traces(textposition="top center")
    show_fig(fig, 430, xaxis_title="Water cut (%)", yaxis_title="Oil (m³/d)")


def page_models(b: DataBundle, models, skipped):
    section("Model Validation & Explainability", "TRUST THE NUMBERS")
    try:
        table, preds, imp = validate_models(b.df, b.inj_is_gas, b.has_choke)
    except ValueError as e:
        st.error(str(e))
        return
    st.write("All models are tested on the **latest 20 % of dates**, never seen in training.")
    show_df(table, 3)
    a, bcol = st.columns(2)
    with a:
        p = preds.dropna(subset=["Hybrid_Pred"])
        fig = px.scatter(p, x="Oil_Rate", y="Hybrid_Pred", color="Well", opacity=.6, title="Hybrid model: actual vs predicted")
        lo, hi = float(p[["Oil_Rate", "Hybrid_Pred"]].min().min()) if len(p) else 0, float(p[["Oil_Rate", "Hybrid_Pred"]].max().max()) if len(p) else 1
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="#8EA3AE"))
        show_fig(fig, 360, xaxis_title="Actual oil (m³/d)", yaxis_title="Predicted oil (m³/d)", showlegend=False)
    with bcol:
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h", title="ML proxy feature importance")
        show_fig(fig, 360)
    st.caption("The hybrid model forecasts rate from setpoints (choke, lift gas) and learned well physics alone — it does "
               "not see same-day measured pressures — which is what makes it usable for 'what-if' optimisation. "
               "Persistence is a hard benchmark for one-day-ahead accuracy but cannot answer what-if questions.")
    section("Learned well parameters", "PHYSICS THE AI DISCOVERED")
    rows = [{"Well": w, "Productivity index J (m³/d/bar)": m.J, "Reservoir pressure (bar)": m.pr_now,
             "Pressure trend (bar/month)": m.pr_slope * 30, "Choke coefficient A": m.A if m.has_choke else np.nan,
             "Water cut": m.wc * 100, "GOR (Sm³/m³)": m.gor, "IPR R²": m.ipr_r2, "Lift/tubing R²": m.tubing_r2,
             "Quality": m.quality, "Notes": "; ".join(m.notes)} for w, m in sorted(models.items())]
    show_df(pd.DataFrame(rows), 2)
    if skipped:
        st.warning("Not calibrated: " + "; ".join(f"{w} ({r})" for w, r in skipped.items()))


def page_methodology():
    section("How FIELDWISE AI Works", "ARCHITECTURE")
    steps = [
        ("01", "Industrial ingestion", "Alias-based header mapping, explicit unit conversion (incl. choke size in 1/64 in, inches or mm), unrelated log curves ignored."),
        ("02", "Auditable cleaning", "Date/ID repair, duplicate merging, physical-validity checks, setpoint-aware imputation (choke carried forward), per-well 3×IQR capping."),
        ("03", "Choke intelligence", "Per-well Gilbert calibration → choke health index (erosion vs restriction), critical-flow check, virtual flow meter, choke step-test log."),
        ("04", "Hybrid well model", "Learned IPR with declining reservoir pressure + tubing/lift response + Gilbert choke, solved as a nodal system."),
        ("05", "ML benchmark", "Random Forest proxy and persistence baseline, chronological hold-out validation."),
        ("06", "Equipment screening", "Per-well robust z-scores + Isolation Forest with driver attribution — no synthetic sensors."),
        ("07", "Optimisation", "Exact dynamic programme over choke size × lift-gas level per well, with lift-gas budget, critical-flow, drawdown (sand) and water-handling terms."),
        ("08", "Engineer in the loop", "Explainable recommendations; nothing is sent to field controllers."),
    ]
    for n, t, d in steps:
        st.markdown(f'<div class="card"><b>{n} · {t}</b><br><span class="small">{d}</span></div>', unsafe_allow_html=True)
    section("Key equations", "PHYSICS")
    st.latex(r"P_{wh}\,[\text{psi}] = \frac{A\,R^{0.546}\,q_L\,[\text{STB/d}]}{S^{1.89}}\qquad A_{\text{ideal}}=435,\;S\text{ in }1/64''")
    st.latex(r"q_L = J\,(P_r - P_{wf}),\qquad P_{wf} = P_{wh} + d_0 + d_1 q_L + d_2 x + d_3 x^2")
    st.latex(r"\Rightarrow\; q_L = \frac{J\,(P_r - d_0 - d_2x - d_3x^2)}{1 + J\,(k(S,R,A) + d_1)},\quad k = P_{wh}/q_L")
    section("Prototype boundaries", "IMPORTANT")
    callout("<b>Decision support, not autonomous control.</b><br>Gilbert assumes critical flow and a fixed-bean choke; "
            "linear IPR is a simplification for solution-gas or below-bubble-point wells; lift response is empirical. "
            "Validate with well tests, step-tests and facility constraints before acting.", warn=True)
    section("Technology stack", "IMPLEMENTATION")
    st.code("Python · Pandas · NumPy · Scikit-learn · Plotly · Streamlit", language="text")


# =============================================================================
# 9. MAIN
# =============================================================================
def sidebar_data() -> DataBundle:
    st.sidebar.markdown("### Data ingestion")
    st.sidebar.download_button("Download CSV template", csv_template().to_csv(index=False).encode(),
                               file_name="fieldwise_template.csv", mime="text/csv", **W)
    up = st.sidebar.file_uploader("Upload field CSV", type=["csv"],
                                  help="Messy headers and extra columns are fine; known aliases are mapped.")
    demo = DataBundle(df=make_demo_data(), source="Built-in demo field", inj_is_gas=True, is_demo=True)
    if up is None:
        return demo
    data = up.getvalue()
    unit_choices = {}
    try:
        sample, _, _ = map_industrial_columns(pd.read_csv(io.BytesIO(data), nrows=500, low_memory=False))
        with st.sidebar.expander("Source units → standard units", expanded=True):
            st.caption("Declare the unit used in your file for each recognised field.")
            for col, opts in UNIT_OPTIONS.items():
                if col in sample.columns:
                    keys = list(opts)
                    unit_choices[col] = st.selectbox(col, keys, index=keys.index(STANDARD_UNITS[col]), key=f"unit_{col}",
                                                     help="Pick a gas unit for lift gas." if col == "Injection_Rate" else None)
    except Exception as exc:  # noqa: BLE001
        st.sidebar.warning(f"Could not inspect units: {exc}")
    try:
        cleaned, report = clean_uploaded(data, tuple(sorted(unit_choices.items())))
        if cleaned is None:
            if report.get("missing_required"):
                raise ValueError("Required operational fields not found: " + ", ".join(report["missing_required"]) +
                                 ". Production optimisation needs date, well ID, oil/water/gas rates, wellhead and "
                                 "bottom-hole pressure. Pure well-log curves (GR, RHOB, NPHI …) cannot drive it.")
            raise ValueError(report.get("error", "Data could not be cleaned."))
        inj_gas = unit_choices.get("Injection_Rate", "m³/d") in GAS_UNITS
        return DataBundle(df=cleaned, source=up.name, report=report, inj_is_gas=inj_gas, is_demo=False)
    except Exception as exc:  # noqa: BLE001
        demo.error = f"Upload not accepted — {exc}"
        return demo


def sidebar_settings() -> Settings:
    with st.sidebar.expander("Screening thresholds"):
        return Settings(
            wc_watch=st.slider("Water cut — watch (%)", 30, 95, 70),
            wc_high=st.slider("Water cut — high (%)", 40, 99, 85),
            decline_watch=st.slider("Oil decline — watch (%)", -30, 0, -5),
            decline_high=st.slider("Oil decline — high (%)", -50, -1, -15),
            choke_tol=st.slider("Choke health tolerance (±%)", 5, 40, 15),
            critical_ratio=st.slider("Critical-flow pressure ratio", 0.40, 0.70, CRITICAL_FLOW_RATIO, 0.01),
        )


PAGES = ["Command Centre", "Data & Quality", "Well Intelligence", "Choke Performance", "Equipment Health",
         "Production Optimisation", "Field Analytics", "Model Validation", "Methodology"]


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="⛽", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    bundle = sidebar_data()
    settings = sidebar_settings()
    page = st.sidebar.radio("Navigation", PAGES)
    st.sidebar.markdown(f'<div style="margin-top:24px;padding-top:10px;border-top:1px solid #20323d;color:#6F858F;'
                        f'font-size:11px;text-align:center;">designed by {AUTHOR}</div>', unsafe_allow_html=True)

    badge = '<span class="badge demo">DEMO DATA</span>' if bundle.is_demo else '<span class="badge">LIVE UPLOAD</span>'
    st.markdown(f'<div class="hero"><div class="brand">{APP_TITLE}</div><span class="badge">MATURE FIELD OPERATIONS '
                f'INTELLIGENCE</span>{badge}<h1>Optimise chokes & lift gas. Anticipate failures.</h1><p>Hybrid '
                f'physics-AI decision support: choke performance & virtual flow metering, equipment anomaly screening '
                f'and multi-well choke + lift-gas optimisation for brownfield assets.</p></div>', unsafe_allow_html=True)
    if bundle.error:
        st.warning(bundle.error + " Showing the built-in demo field instead.")

    df = bundle.df
    cf = build_choke_frame(df, bundle.inj_is_gas, bundle.has_flp) if bundle.has_choke else None
    cf_sum = choke_summary(cf, settings) if cf is not None else None
    eq_rows, eq_sum = screen_equipment(df, tuple(bundle.sensors))
    models, skipped = fit_well_models(df, bundle.inj_is_gas, bundle.has_choke)
    latest = build_latest(df, cf_sum, eq_sum, models, settings)

    if page == "Command Centre":
        page_command_centre(bundle, latest, cf)
    elif page == "Data & Quality":
        page_data(bundle)
    elif page == "Well Intelligence":
        page_well(bundle, latest, cf, models)
    elif page == "Choke Performance":
        page_choke(bundle, latest, cf, cf_sum, models, settings)
    elif page == "Equipment Health":
        page_equipment(bundle, latest, eq_rows)
    elif page == "Production Optimisation":
        page_optimisation(bundle, latest, models, settings)
    elif page == "Field Analytics":
        page_analytics(bundle, latest)
    elif page == "Model Validation":
        page_models(bundle, models, skipped)
    else:
        page_methodology()


if __name__ == "__main__":
    main()
