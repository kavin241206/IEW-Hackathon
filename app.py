import streamlit as st
import io
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="FIELDWISE AI", page_icon="⛽", layout="wide")

REQUIRED_COLUMNS = ["Date", "Well", "Oil_Rate", "Water_Rate", "Gas_Rate", "Wellhead_Pressure", "Bottomhole_Pressure", "Injection_Rate"]
NUMERIC_COLUMNS = ["Oil_Rate", "Water_Rate", "Gas_Rate", "Wellhead_Pressure", "Bottomhole_Pressure", "Injection_Rate"]
OPTIONAL_SENSOR_COLUMNS = ["Pump_Vibration", "Motor_Temperature", "Motor_Current", "Pump_Frequency", "Energy_kWh"]
FEATURES = ["Water_Rate", "Gas_Rate", "Wellhead_Pressure", "Bottomhole_Pressure", "Injection_Rate"]

# Common industrial naming variants. Unknown columns (e.g. GR, RHOB, NPHI, resistivity)
# are intentionally ignored by the production dashboard rather than fed into its models.
COLUMN_ALIASES = {
    "Date": ["date", "datetime", "timestamp", "time", "sample_date", "observation_date", "production_date"],
    "Well": ["well", "well_id", "well name", "well_name", "wellno", "well_no", "uwi", "api", "well identifier"],
    "Oil_Rate": ["oil_rate", "oil rate", "oil production", "oil production rate", "qo", "oil_volumetric_rate", "oil_m3_d", "oil rate (m3/d)", "oil rate (m³/d)"],
    "Water_Rate": ["water_rate", "water rate", "water production", "water production rate", "qw", "water_volumetric_rate", "water_m3_d", "water rate (m3/d)", "water rate (m³/d)"],
    "Gas_Rate": ["gas_rate", "gas rate", "gas production", "gas production rate", "qg", "gas_mscf_d", "gas rate (mcf/d)"],
    "Wellhead_Pressure": ["wellhead_pressure", "wellhead pressure", "whp", "tubing_head_pressure", "thp", "well head pressure"],
    "Bottomhole_Pressure": ["bottomhole_pressure", "bottom hole pressure", "bhp", "flowing_bottomhole_pressure", "fbhp", "bottom-hole pressure"],
    "Injection_Rate": ["injection_rate", "injection rate", "injection", "gas_lift_rate", "gas lift rate", "lift_gas_rate", "lift gas rate", "water_injection_rate", "qinj"],
    "Pump_Vibration": ["pump_vibration", "vibration", "vibration_rms", "pump vibration"],
    "Motor_Temperature": ["motor_temperature", "motor temp", "motor_temperature_c", "motor temperature", "motor_temp"],
    "Motor_Current": ["motor_current", "current", "motor amps", "motor_current_a", "motor current"],
    "Pump_Frequency": ["pump_frequency", "frequency", "vfd_frequency", "pump frequency", "frequency_hz"],
    "Energy_kWh": ["energy_kwh", "energy", "energy_kwh", "power_consumption_kwh", "energy consumed", "energy consumption"]
}

def _key(name):
    return " ".join(str(name).strip().lower().replace("_", " ").replace("-", " ").split())

def map_industrial_columns(raw):
    """Map recognised source headers to FIELDWISE names; discard unneeded columns."""
    df = raw.copy()
    aliases = {_key(alias): canonical for canonical, names in COLUMN_ALIASES.items() for alias in [canonical, *names]}
    rename = {}
    mapped = {}
    for col in df.columns:
        canonical = aliases.get(_key(col))
        if canonical and canonical not in mapped:
            rename[col] = canonical
            mapped[canonical] = col
    df = df.rename(columns=rename)
    selected = [c for c in REQUIRED_COLUMNS + OPTIONAL_SENSOR_COLUMNS if c in df.columns]
    ignored = [str(c) for c in raw.columns if c not in mapped.values()]
    return df[selected].copy(), mapped, ignored

UNIT_OPTIONS = {
    "Oil_Rate": {"m³/d": (1.0, 0.0), "bbl/d": (0.1589872949, 0.0), "STB/d": (0.1589872949, 0.0), "m³/h": (24.0, 0.0), "L/s": (86.4, 0.0), "L/min": (1.44, 0.0), "ft³/d": (0.0283168466, 0.0)},
    "Water_Rate": {"m³/d": (1.0, 0.0), "bbl/d": (0.1589872949, 0.0), "STB/d": (0.1589872949, 0.0), "m³/h": (24.0, 0.0), "L/s": (86.4, 0.0), "L/min": (1.44, 0.0), "ft³/d": (0.0283168466, 0.0)},
    "Gas_Rate": {"Sm³/d": (1.0, 0.0), "Mscf/d": (28.3168466, 0.0), "scf/d": (0.0283168466, 0.0), "MMSCFD": (28316.8466, 0.0), "Nm³/h": (24.0, 0.0)},
    "Injection_Rate": {"m³/d": (1.0, 0.0), "bbl/d": (0.1589872949, 0.0), "STB/d": (0.1589872949, 0.0), "m³/h": (24.0, 0.0), "L/s": (86.4, 0.0), "ft³/d": (0.0283168466, 0.0), "Sm³/d": (1.0, 0.0), "Mscf/d": (28.3168466, 0.0), "scf/d": (0.0283168466, 0.0), "MMSCFD": (28316.8466, 0.0)},
    "Wellhead_Pressure": {"bar": (1.0, 0.0), "kPa": (0.01, 0.0), "MPa": (10.0, 0.0), "psi": (0.0689475729, 0.0), "kgf/cm²": (0.980665, 0.0)},
    "Bottomhole_Pressure": {"bar": (1.0, 0.0), "kPa": (0.01, 0.0), "MPa": (10.0, 0.0), "psi": (0.0689475729, 0.0), "kgf/cm²": (0.980665, 0.0)},
    "Motor_Temperature": {"°C": (1.0, 0.0), "°F": (5.0/9.0, -32.0*5.0/9.0)},
    "Motor_Current": {"A": (1.0, 0.0), "mA": (0.001, 0.0)},
    "Pump_Frequency": {"Hz": (1.0, 0.0), "rpm": (1.0/60.0, 0.0)},
    "Pump_Vibration": {"mm/s": (1.0, 0.0), "in/s": (25.4, 0.0)},
    "Energy_kWh": {"kWh": (1.0, 0.0), "Wh": (0.001, 0.0), "MWh": (1000.0, 0.0)},
}
STANDARD_UNITS = {"Oil_Rate": "m³/d", "Water_Rate": "m³/d", "Gas_Rate": "Sm³/d", "Injection_Rate": "m³/d", "Wellhead_Pressure": "bar", "Bottomhole_Pressure": "bar", "Motor_Temperature": "°C", "Motor_Current": "A", "Pump_Frequency": "Hz", "Pump_Vibration": "mm/s", "Energy_kWh": "kWh"}

def clean_industrial_data(raw, unit_choices=None):
    """Conservative, auditable cleanup. Converts explicitly selected source units to standard units."""
    frame, mapped, ignored = map_industrial_columns(raw)
    unit_choices = unit_choices or {}
    conversion_log = []
    for col, unit in unit_choices.items():
        if col not in frame.columns or col not in UNIT_OPTIONS or unit not in UNIT_OPTIONS[col]:
            continue
        factor, offset = UNIT_OPTIONS[col][unit]
        target_unit = STANDARD_UNITS[col]
        if col == "Injection_Rate" and unit in {"Sm³/d", "Mscf/d", "scf/d", "MMSCFD"}:
            target_unit = "Sm³/d"
        if unit != target_unit:
            vals = pd.to_numeric(frame[col].astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")
            frame[col] = vals * factor + offset
            conversion_log.append({"Field": col, "Source unit": unit, "Standard unit": target_unit, "Conversion": f"value × {factor:.10g} + {offset:.10g}"})
    report = {"input_rows": len(raw), "mapped_columns": mapped, "ignored_columns": ignored, "duplicates_removed": 0,
              "rows_without_date_or_well_removed": 0, "invalid_negative_rates_removed": 0,
              "numeric_values_imputed": 0, "outlier_values_capped": 0, "unit_conversions": conversion_log, "notes": []}
    # Injection is not present in every production export. Keep the file usable for surveillance,
    # but explicitly mark missing injection as assumed zero; optimisation results must not be treated as field recommendations.
    if "Injection_Rate" not in frame.columns:
        frame["Injection_Rate"] = 0.0
        report["notes"].append("Injection_Rate was absent and set to 0 for dashboard compatibility; allocation optimisation is not valid until real injection/lift-gas data and units are supplied.")
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        report["missing_required"] = missing
        return None, report
