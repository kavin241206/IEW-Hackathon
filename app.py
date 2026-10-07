import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="FIELDWISE AI V3", page_icon="⛽", layout="wide")

# ---------- Synthetic demo data ----------
@st.cache_data
def make_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    wells = [f"W-{i:03d}" for i in range(1, 13)]
    rows = []
    for wi, well in enumerate(wells):
        base_oil = 145 - wi * 5 + rng.normal(0, 4)
        base_water = 30 + wi * 4
        for t, d in enumerate(dates):
            decline = np.exp(-t / (260 + wi * 15))
            injection = 55 + 10*np.sin(t/25 + wi/3) + rng.normal(0, 3)
            bhp = 190 - wi*2 + 7*np.sin(t/30 + wi) + rng.normal(0, 2)
            whp = 105 - wi + 4*np.sin(t/18) + rng.normal(0, 1.5)
            water = max(5, base_water + 0.055*t + 0.16*injection + rng.normal(0, 3))
            gas = max(20, 65 + 12*np.sin(t/20 + wi) + rng.normal(0, 5))
            oil = max(15, base_oil*decline + 0.26*injection + 0.15*bhp - 20
                       - 0.10*water + rng.normal(0, 4))
            rows.append([d, well, oil, water, gas, whp, bhp, injection])
    return pd.DataFrame(rows, columns=[
        "Date","Well","Oil_Rate","Water_Rate","Gas_Rate",
        "Wellhead_Pressure","Bottomhole_Pressure","Injection_Rate"
    ])

df = make_data()
features = ["Water_Rate","Gas_Rate","Wellhead_Pressure","Bottomhole_Pressure","Injection_Rate"]

def train_model():
    split_date = df["Date"].quantile(0.80)
    tr = df[df.Date <= split_date]
    te = df[df.Date > split_date]
    model = RandomForestRegressor(
        n_estimators=180, max_depth=10, min_samples_leaf=3, random_state=42
    )
    model.fit(tr[features], tr["Oil_Rate"])
    pred = model.predict(te[features])
    return model, mean_absolute_error(te["Oil_Rate"], pred), te.assign(Predicted_Oil=pred)

model, mae, test_df = train_model()
latest = df.sort_values("Date").groupby("Well").tail(1).copy()
latest["Water_Cut"] = latest["Water_Rate"]/(latest["Water_Rate"]+latest["Oil_Rate"])*100

# ---------- Helpers ----------
def priority(row):
    wc = row["Water_Cut"]
    if row["Oil_Rate"] < 80 and wc > 40:
        return "CRITICAL"
    if row["Oil_Rate"] < 95 or wc > 35:
        return "HIGH"
    if wc > 28 or row["Oil_Rate"] < 110:
        return "WATCH"
    return "STABLE"

latest["Priority"] = latest.apply(priority, axis=1)

def kpi_card(label, value, sub=""):
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

def section(title, kicker=""):
    st.markdown(f'<div class="section-kicker">{kicker}</div><h2 class="section-title">{title}</h2>', unsafe_allow_html=True)

# ---------- Styling ----------
st.markdown("""
<style>
.stApp { background: #071018; color: #EAF2F7; }
.block-container { padding-top: 1.2rem; max-width: 1450px; }
.hero {
  padding: 34px 38px; border: 1px solid #233541; border-radius: 20px;
  background: linear-gradient(135deg,#0c1d28,#0a141b);
  margin-bottom: 22px;
}
.brand { font-size: 14px; letter-spacing: 4px; font-weight: 800; color:#77D6C8; }
.hero h1 { font-size: 44px; margin: 8px 0 6px; }
.hero p { font-size: 17px; color:#AFC1CB; max-width: 850px; }
.badge { display:inline-block; padding:6px 10px; border-radius:20px; background:#14322f; color:#86E3D2; font-size:12px; font-weight:700; }
.kpi { background:#0d1b24; border:1px solid #223541; border-radius:15px; padding:18px; min-height:112px; }
.kpi-label { color:#8EA3
