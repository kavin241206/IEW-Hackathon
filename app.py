import streamlit as st
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

# ---------- Styling ----------
st.markdown("""
<style>
.stApp {background:radial-gradient(circle at 84% 7%,rgba(73,174,158,.13),transparent 25%),radial-gradient(circle at 12% 88%,rgba(45,103,133,.13),transparent 30%),linear-gradient(135deg,#061017 0%,#091720 50%,#071018 100%);color:#EAF2F7;}
.stApp:before {content:"";position:fixed;inset:0;pointer-events:none;opacity:.16;background-image:linear-gradient(rgba(112,171,184,.055) 1px,transparent 1px),linear-gradient(90deg,rgba(112,171,184,.055) 1px,transparent 1px);background-size:42px 42px;}
.block-container{padding-top:1.2rem;max-width:1450px;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#07131b 0%,#091821 100%);border-right:1px solid #20323d;}
[data-testid="stSidebar"]>div:first-child{padding-top:1.2rem;}
[data-testid="stSidebar"] .stRadio>label{color:#8EA3AE;font-size:11px;text-transform:uppercase;letter-spacing:1.4px;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"]{gap:5px;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label{border-radius:9px;padding:7px 9px;}
.hero{padding:30px 36px;border:1px solid #233541;border-radius:20px;background:linear-gradient(135deg,#0c1d28,#0a141b);margin-bottom:22px;}
.brand{font-size:14px;letter-spacing:4px;font-weight:800;color:#77D6C8;}
.hero h1{font-size:40px;margin:8px 0 6px;}
.hero p{font-size:16px;color:#AFC1CB;max-width:900px;}
.badge{display:inline-block;padding:6px 10px;border-radius:20px;background:#14322f;color:#86E3D2;font-size:12px;font-weight:700;}
.kpi{background:#0d1b24;border:1px solid #223541;border-radius:15px;padding:17px;min-height:108px;}
.kpi-label{color:#8EA3AE;font-size:12px;text-transform:uppercase;letter-spacing:1.2px;}
.kpi-value{font-size:27px;font-weight:800;margin-top:6px;overflow-wrap:anywhere;}
.kpi-sub{color:#718792;font-size:12px;margin-top:4px;}
.card{background:#0d1b24;border:1px solid #223541;border-radius:15px;padding:18px;}
.section-kicker{color:#65CFC0;text-transform:uppercase;letter-spacing:2px;font-size:11px;font-weight:800;margin-top:10px;}
.section-title{margin-top:3px;}
.alert{border-left:4px solid #65CFC0;background:#0d2026;padding:14px 16px;border-radius:8px;}
.warn{border-left-color:#E7B85B;background:#251f11;}
.small{color:#8EA3AE;font-size:13px;}
</style>
""", unsafe_allow_html=True)

# ---------- Demo data ----------
@st.cache_data
def make_demo_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    wells = [f"W-{i:03d}" for i in range(1, 13)]
    rows = []
    for wi, well in enumerate(wells):
        base_oil = 145 - wi * 5 + rng.normal(0, 4)
        base_water = 30 + wi * 4
        for t, d in enumerate(dates):
            decline = np.exp(-t / (260 + wi * 15))
            injection = max(5, 55 + 10*np.sin(t/25 + wi/3) + rng.normal(0, 3))
            bhp = 190 - wi*2 + 7*np.sin(t/30 + wi) + rng.normal(0, 2)
            whp = 105 - wi + 4*np.sin(t/18) + rng.normal(0, 1.5)
            water = max(5, base_water + 0.055*t + 0.16*injection + rng.normal(0, 3))
            gas = max(20, 65 + 12*np.sin(t/20 + wi) + rng.normal(0, 5))
            oil = max(15, base_oil*decline + 0.26*injection + 0.15*bhp - 20 - 0.10*water + rng.normal(0, 4))
            # Synthetic equipment signals for demonstrating monitoring only.
            vib = max(0.1, 2.0 + 0.004*t + 0.10*wi + rng.normal(0, .22))
            temp = 68 + 0.025*t + 0.7*wi + 2*np.sin(t/17+wi) + rng.normal(0, 1.6)
            current = max(1, 35 + 0.22*oil + 0.05*water + rng.normal(0, 3))
            freq = np.clip(48 + 2*np.sin(t/22+wi) + rng.normal(0, 1), 35, 60)
            energy = max(1, current*freq*0.15 + rng.normal(0, 3))
            # Insert a few illustrative abnormal periods in sensor signals.
            if wi in (2, 7) and 125 <= t <= 137:
                vib += 2.2
                temp += 9
            rows.append([d, well, oil, water, gas, whp, bhp, injection, vib, temp, current, freq, energy])
    cols = REQUIRED_COLUMNS + OPTIONAL_SENSOR_COLUMNS
    return pd.DataFrame(rows, columns=cols)

# ---------- Upload and validation ----------
st.sidebar.markdown("## FIELDWISE AI")
st.sidebar.caption("Production optimisation & equipment intelligence")
template = pd.DataFrame({
    "Date": ["2026-01-01", "2026-01-02"], "Well": ["W-001", "W-001"],
    "Oil_Rate": [85.0, 84.2], "Water_Rate": [74.0, 75.0], "Gas_Rate": [62.0, 61.5],
    "Wellhead_Pressure": [100.0, 99.5], "Bottomhole_Pressure": [172.0, 171.5], "Injection_Rate": [55.0, 56.0],
    "Pump_Vibration": [2.1, 2.2], "Motor_Temperature": [72.0, 72.4], "Motor_Current": [54.0, 53.5],
    "Pump_Frequency": [48.0, 48.0], "Energy_kWh": [390.0, 388.0]
})
st.sidebar.download_button("Download CSV template", template.to_csv(index=False).encode(), file_name="fieldwise_operations_template.csv", mime="text/csv", use_container_width=True)
uploaded_file = st.sidebar.file_uploader("Upload field data (CSV)", type=["csv"], help="Required production columns plus optional equipment sensor columns.")

@st.cache_data
def load_uploaded(file_bytes):
    return pd.read_csv(file_bytes)

source = "Built-in illustrative dataset"
upload_error = None
df = make_demo_data()
if uploaded_file is not None:
    try:
        raw = pd.read_csv(uploaded_file)
        missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))
        keep = REQUIRED_COLUMNS + [c for c in OPTIONAL_SENSOR_COLUMNS if c in raw.columns]
        raw = raw[keep].copy()
        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
        raw["Well"] = raw["Well"].astype(str).str.strip()
        for c in keep:
            if c not in ("Date", "Well"):
                raw[c] = pd.to_numeric(raw[c], errors="coerce")
        raw = raw.replace([np.inf, -np.inf], np.nan)
        if raw[keep].isna().any().any():
            raise ValueError("Required or supplied optional columns contain blank/non-numeric values. Clean the file and upload again.")
        if len(raw) < 20:
            raise ValueError("Please upload at least 20 valid historical rows.")
        if raw["Date"].nunique() < 2:
            raise ValueError("The dataset must contain at least two distinct dates.")
        if (raw[["Oil_Rate", "Water_Rate", "Injection_Rate"]] < 0).any().any():
            raise ValueError("Oil, water and injection rates must be non-negative.")
        # Create optional sensor fields only if the uploaded data does not contain them.
        rng = np.random.default_rng(7)
        for c, base, spread in [("Pump_Vibration", 2.0, .25), ("Motor_Temperature", 72.0, 2.0), ("Motor_Current", 50.0, 4.0), ("Pump_Frequency", 48.0, 1.5), ("Energy_kWh", 380.0, 15.0)]:
            if c not in raw.columns:
                raw[c] = np.maximum(0, rng.normal(base, spread, len(raw)))
        df = raw.sort_values(["Date", "Well"]).reset_index(drop=True)
        source = f"User-uploaded data · {uploaded_file.name}"
    except Exception as e:
        upload_error = str(e)
        df = make_demo_data()

# ---------- ML model and derived metrics ----------
def train_proxy(frame):
    split = frame["Date"].quantile(.8)
    train = frame[frame["Date"] <= split]
    test = frame[frame["Date"] > split]
    if len(train) < 10 or len(test) < 2:
        raise ValueError("Not enough rows for chronological train/test validation.")
    model = RandomForestRegressor(n_estimators=160, max_depth=10, min_samples_leaf=3, random_state=42)
    model.fit(train[FEATURES], train["Oil_Rate"])
    pred = model.predict(test[FEATURES])
    return model, mean_absolute_error(test["Oil_Rate"], pred), test.assign(Predicted_Oil=pred)

try:
    model, mae, test_df = train_proxy(df)
except Exception as e:
    upload_error = str(e)
    df = make_demo_data()
    source = "Built-in illustrative dataset (uploaded data could not be modelled)"
    model, mae, test_df = train_proxy(df)

latest = df.sort_values("Date").groupby("Well", as_index=False).tail(1).copy()
latest["Water_Cut"] = (latest["Water_Rate"] / (latest["Water_Rate"] + latest["Oil_Rate"]).replace(0, np.nan) * 100).fillna(0)

def priority(r):
    if r["Oil_Rate"] < 80 and r["Water_Cut"] > 40: return "CRITICAL"
    if r["Oil_Rate"] < 95 or r["Water_Cut"] > 35: return "HIGH"
    if r["Water_Cut"] > 28 or r["Oil_Rate"] < 110: return "WATCH"
    return "STABLE"
latest["Priority"] = latest.apply(priority, axis=1)

sensor_cols = [c for c in OPTIONAL_SENSOR_COLUMNS if c in df.columns]
sensor_features = ["Pump_Vibration", "Motor_Temperature", "Motor_Current", "Pump_Frequency", "Energy_kWh"]
# Fit an anomaly model on the latest well observations. This is screening, not a certified failure forecast.
try:
    iso = IsolationForest(contamination=0.12, random_state=42)
    iso.fit(latest[sensor_features])
    latest["Anomaly"] = np.where(iso.predict(latest[sensor_features]) == -1, "ALERT", "NORMAL")
    latest["Anomaly_Score"] = -iso.score_samples(latest[sensor_features])
except Exception:
    latest["Anomaly"] = "UNAVAILABLE"
    latest["Anomaly_Score"] = 0.0

# ---------- UI helpers ----------
def kpi(label, value, sub=""):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)
def section(title, kicker=""):
    st.markdown(f'<div class="section-kicker">{kicker}</div><h2 class="section-title">{title}</h2>', unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", ["Command Centre", "Data Input", "Well Intelligence", "Equipment Health", "Production Optimisation", "Field Analytics", "Explainable AI", "Methodology"])
st.sidebar.markdown('<div style="margin-top:28px;padding-top:10px;border-top:1px solid #20323d;color:#6F858F;font-size:11px;letter-spacing:.5px;text-align:center;">designed by Kavinkarthick</div>', unsafe_allow_html=True)

st.markdown('''<div class="hero"><div class="brand">FIELDWISE AI</div><span class="badge">MATURE FIELD OPERATIONS INTELLIGENCE</span><h1>Optimise production. Anticipate equipment issues.</h1><p>An AI-assisted decision-support prototype combining well surveillance, equipment anomaly screening and multi-well operating-scenario comparison for mature oil and gas fields.</p></div>''', unsafe_allow_html=True)

# ---------- Command Centre ----------
if page == "Command Centre":
    section("Field Operations Command Centre", "EXECUTIVE VIEW")
    if upload_error: st.warning(f"Uploaded data was not accepted: {upload_error}. The built-in illustrative dataset is being used.")
    elif source.startswith("User-uploaded"): st.success(f"Dashboard data source: {uploaded_file.name}")
    oil_total = latest["Oil_Rate"].sum()
    water_total = latest["Water_Rate"].sum()
    weighted_wc = 100*water_total/(oil_total+water_total) if oil_total+water_total else 0
    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Producing wells", f"{latest['Well'].nunique()}", "latest observation per well")
    with c2: kpi("Total oil rate", f"{oil_total:,.1f} m³/d", "field total")
    with c3: kpi("Field water cut", f"{weighted_wc:.1f}%", "liquid-rate weighted")
    with c4: kpi("Equipment alerts", int((latest["Anomaly"] == "ALERT").sum()), "anomaly-screening flags")
    left,right = st.columns([1.35,1])
    with left:
        section("Field production trend", "PRODUCTION SURVEILLANCE")
        trend = df.groupby("Date")[["Oil_Rate", "Water_Rate"]].sum().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend.Date, y=trend.Oil_Rate, name="Oil rate", mode="lines"))
        fig.add_trace(go.Scatter(x=trend.Date, y=trend.Water_Rate, name="Water rate", mode="lines"))
        fig.update_layout(template="plotly_dark", height=330, margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Rate (m³/d)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        section("Wells requiring attention", "SCREENING QUEUE")
        order = {"CRITICAL":0,"HIGH":1,"WATCH":2,"STABLE":3}
        queue = latest.copy(); queue["_rank"] = queue.Priority.map(order)
        queue = queue.sort_values(["_rank","Water_Cut"], ascending=[True,False]).head(8)
        st.dataframe(queue[["Well","Oil_Rate","Water_Cut","Priority","Anomaly"]].rename(columns={"Oil_Rate":"Oil rate (m³/d)","Water_Cut":"Water cut (%)"}), hide_index=True, use_container_width=True)
    section("Operational signals", "EQUIPMENT & ENERGY")
    a,b,c = st.columns(3)
    with a:
        fig = px.bar(latest, x="Well", y="Pump_Vibration", color="Anomaly", title="Latest pump vibration (demo units)")
        fig.update_layout(template="plotly_dark", height=280); st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.scatter(latest, x="Motor_Temperature", y="Motor_Current", color="Anomaly", size="Oil_Rate", hover_name="Well", title="Motor signal screening")
        fig.update_layout(template="plotly_dark", height=280); st.plotly_chart(fig, use_container_width=True)
    with c:
        fig = px.bar(latest, x="Well", y="Energy_kWh", color="Priority", title="Energy indicator (source units)")
        fig.update_layout(template="plotly_dark", height=280); st.plotly_chart(fig, use_container_width=True)
    st.caption("Equipment flags are anomaly-screening outputs. They are not confirmed failures; validate against equipment-specific limits and maintenance records.")

# ---------- Data Input ----------
elif page == "Data Input":
    section("Bring Your Field Data", "DATA IMPORT")
    if upload_error: st.error(upload_error)
    elif source.startswith("User-uploaded"): st.success(f"Successfully loaded {uploaded_file.name}; accepted data power the dashboard and oil-rate proxy model.")
    else: st.info("Upload a CSV from the sidebar. Required columns cover well production, pressure and injection. Sensor columns are optional but enable more meaningful equipment screening.")
    a,b,c = st.columns(3)
    with a: kpi("Rows loaded", f"{len(df):,}", "well-date observations")
    with b: kpi("Wells", f"{df['Well'].nunique()}", "unique well IDs")
    with c: kpi("History", f"{df['Date'].min():%d %b %Y} – {df['Date'].max():%d %b %Y}", "available date range")
    st.markdown("### Required columns")
    req_desc = ["Observation date","Unique well identifier","Oil production rate (m³/d)","Water production rate (m³/d)","Gas rate (keep units consistent)","Wellhead pressure (keep units consistent)","Bottom-hole pressure (keep units consistent)","Injection or lift-gas rate (keep units consistent)"]
    st.dataframe(pd.DataFrame({"Column":REQUIRED_COLUMNS,"Meaning":req_desc}), hide_index=True, use_container_width=True)
    st.markdown("### Optional equipment columns")
    opt_desc = ["Pump vibration measurement","Motor temperature","Motor current","Pump operating frequency","Energy consumed over the observation interval"]
    st.dataframe(pd.DataFrame({"Column":OPTIONAL_SENSOR_COLUMNS,"Meaning":opt_desc}), hide_index=True, use_container_width=True)
    st.caption("One row should represent one well at one timestamp. Optional sensor columns are important for equipment monitoring. If omitted, the prototype fills them with illustrative values; those generated values must not be interpreted as real equipment evidence.")
    st.markdown("### Data preview")
    st.dataframe(df.head(25), hide_index=True, use_container_width=True)
    st.download_button("Download currently loaded data", df.to_csv(index=False).encode(), file_name="fieldwise_loaded_data.csv", mime="text/csv")

# ---------- Well Intelligence ----------
elif page == "Well Intelligence":
    section("Well Intelligence", "WELL-LEVEL SURVEILLANCE")
    well = st.selectbox("Select well", latest.Well.tolist())
    hist = df[df.Well == well].sort_values("Date").copy()
    row = latest[latest.Well == well].iloc[0]
    wc = row.Water_Cut
    recent = hist.tail(min(30, len(hist)))
    oil_change = (recent.Oil_Rate.iloc[-1]/recent.Oil_Rate.iloc[0]-1)*100 if len(recent)>1 and recent.Oil_Rate.iloc[0] else 0
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: kpi("Oil rate", f"{row.Oil_Rate:.1f} m³/d", "latest")
    with c2: kpi("Water rate", f"{row.Water_Rate:.1f} m³/d", "latest")
    with c3: kpi("Water cut", f"{wc:.1f}%", "latest")
    with c4: kpi("Bottom-hole pressure", f"{row.Bottomhole_Pressure:.1f}", "source pressure units")
    with c5: kpi("Priority", row.Priority, "rule-based screen")
    a,b = st.columns(2)
    with a:
        fig = px.line(hist, x="Date", y="Oil_Rate", title="Oil production history")
        fig.update_layout(template="plotly_dark", height=320, yaxis_title="Oil rate (m³/d)"); st.plotly_chart(fig, use_container_width=True)
    with b:
        hist["Water_Cut"] = 100*hist.Water_Rate/(hist.Water_Rate+hist.Oil_Rate).replace(0,np.nan)
        fig = px.line(hist, x="Date", y="Water_Cut", title="Water-cut history")
        fig.update_layout(template="plotly_dark", height=320, yaxis_title="Water cut (%)"); st.plotly_chart(fig, use_container_width=True)
    section("Screening indicators", "ENGINEERING REVIEW")
    reasons=[]
    if oil_change < -5: reasons.append(f"Oil rate changed by {oil_change:.1f}% across the latest {len(recent)} observations.")
    if wc > 35: reasons.append("Water cut is elevated under the prototype's screening threshold.")
    if row.Anomaly == "ALERT": reasons.append("The latest equipment-sensor pattern is flagged as anomalous by the screening model.")
    if not reasons: reasons.append("No configured screening trigger detected; continue normal surveillance.")
    st.markdown('<div class="alert">' + "<br>".join("• "+x for x in reasons) + '</div>', unsafe_allow_html=True)
    st.caption("Screening indicators identify what to investigate; they do not diagnose the reservoir or confirm a mechanical failure.")

# ---------- Equipment Health ----------
elif page == "Equipment Health":
    section("Equipment Health & Anomaly Screening", "ARTIFICIAL-LIFT SURVEILLANCE")
    st.markdown("The prototype uses Isolation Forest to flag unusual combinations of the available sensor variables. It is an anomaly screen—not a labelled failure-prediction model.")
    a,b,c = st.columns(3)
    with a: kpi("Wells flagged", int((latest.Anomaly=="ALERT").sum()), "latest observations")
    with b: kpi("Wells screened", len(latest), "one latest row per well")
    with c: kpi("Sensor variables", len(sensor_features), "vibration, temperature, current, frequency, energy")
    table = latest[["Well","Pump_Vibration","Motor_Temperature","Motor_Current","Pump_Frequency","Energy_kWh","Anomaly","Anomaly_Score"]].sort_values("Anomaly_Score",ascending=False)
    st.dataframe(table.rename(columns={"Pump_Vibration":"Vibration","Motor_Temperature":"Motor temp","Motor_Current":"Motor current","Pump_Frequency":"Frequency","Energy_kWh":"Energy","Anomaly_Score":"Anomaly score"}), hide_index=True, use_container_width=True)
    well = st.selectbox("Inspect equipment signals", latest.Well.tolist(), key="equipment_well")
    hist = df[df.Well==well].sort_values("Date")
    m1,m2 = st.columns(2)
    with m1:
        fig=px.line(hist,x="Date",y="Pump_Vibration",title=f"{well} · pump vibration")
        fig.update_layout(template="plotly_dark",height=300); st.plotly_chart(fig,use_container_width=True)
    with m2:
        fig=px.line(hist,x="Date",y="Motor_Temperature",title=f"{well} · motor temperature")
        fig.update_layout(template="plotly_dark",height=300); st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="alert warn"><b>Engineering caution</b><br>An anomaly is not proof of imminent failure. Confirm with equipment-specific thresholds, operating context, maintenance logs and engineering review. If no sensor columns were uploaded, the app uses synthetic placeholder signals for demonstration.</div>', unsafe_allow_html=True)

# ---------- Production Optimisation ----------
elif page == "Production Optimisation":
    section("Multi-Well Production Optimisation", "SCENARIO LAB")
    st.write("Explore alternative allocations of the available injection/lift-gas resource across selected wells. The oil-response model is a proxy; results are scenario estimates, not operating instructions.")
    st.caption("This prototype treats Injection_Rate as the allocation variable. For a gas-lift deployment, upload actual lift-gas allocation data and use consistent units throughout.")
    wells = latest.Well.tolist()
    selected = st.multiselect("Wells included in allocation", wells, default=wells[:min(5,len(wells))])
    if not selected:
        st.info("Select at least one well to build an allocation scenario.")
    else:
        subset = latest[latest.Well.isin(selected)].copy().reset_index(drop=True)
        current_total = float(subset.Injection_Rate.sum())
        st.markdown(f"**Current total allocation:** {current_total:.1f} source units per day across {len(subset)} selected wells.")
        total_available = st.number_input("Available total allocation (same units as Injection_Rate)", min_value=0.0, value=float(round(current_total,1)), step=5.0)
        weights = np.array([float(subset.loc[i,"Oil_Rate"]) for i in range(len(subset))])
        weights = weights / weights.sum() if weights.sum() else np.ones(len(subset))/len(subset)
        allocation = weights * total_available
        proposed_df = subset[["Well","Oil_Rate","Water_Rate","Water_Cut","Injection_Rate"]].copy()
        proposed_df["Proposed_Allocation"] = allocation
        proposed_df["Proposed_Allocation"] = np.round(proposed_df["Proposed_Allocation"],2)
        edited = st.data_editor(proposed_df[["Well","Injection_Rate","Proposed_Allocation"]].rename(columns={"Injection_Rate":"Current allocation","Proposed_Allocation":"Proposed allocation"}), hide_index=True, use_container_width=True, disabled=["Well","Current allocation"], column_config={"Proposed allocation":st.column_config.NumberColumn(min_value=0, step=1)})
        if st.button("EVALUATE ALLOCATION SCENARIO", type="primary", use_container_width=True):
            alloc = edited["Proposed allocation"].astype(float).to_numpy()
            if np.any(alloc < 0):
                st.error("Allocations must be non-negative.")
            elif not np.isclose(alloc.sum(), total_available, rtol=0.01, atol=0.1):
                st.warning(f"Proposed allocations sum to {alloc.sum():.2f}, not the available total {total_available:.2f}. Adjust them to match the total resource limit.")
            else:
                preds=[]
                for i,r in subset.iterrows():
                    vals=[[r[f] for f in FEATURES]]
                    vals[0][FEATURES.index("Injection_Rate")] = alloc[i]
                    preds.append(float(model.predict(pd.DataFrame(vals,columns=FEATURES))[0]))
                result = pd.DataFrame({"Well":subset.Well,"Current oil (m³/d)":subset.Oil_Rate.values,"Predicted oil (m³/d)":preds,"Current allocation":subset.Injection_Rate.values,"Proposed allocation":alloc,"Water cut (%)":subset.Water_Cut.values})
                current_oil = float(result["Current oil (m³/d)"].sum())
                predicted_oil = float(result["Predicted oil (m³/d)"].sum())
                delta = predicted_oil-current_oil
                k1,k2,k3 = st.columns(3)
                with k1: kpi("Current oil", f"{current_oil:.1f} m³/d", "selected wells")
                with k2: kpi("Proxy-predicted oil", f"{predicted_oil:.1f} m³/d", "scenario estimate")
                with k3: kpi("Estimated difference", f"{delta:+.1f} m³/d", "not a validated production gain")
                st.dataframe(result, hide_index=True, use_container_width=True)
                fig=go.Figure()
                fig.add_trace(go.Bar(x=result.Well,y=result["Current oil (m³/d)"],name="Current oil"))
                fig.add_trace(go.Bar(x=result.Well,y=result["Predicted oil (m³/d)"],name="Scenario oil"))
                fig.update_layout(template="plotly_dark",barmode="group",height=350,yaxis_title="Oil rate (m³/d)")
                st.plotly_chart(fig,use_container_width=True)
                st.markdown('<div class="alert warn"><b>Before operational use:</b> the allocation logic here is a prototype demonstration. Validate model response, lift-gas or injection constraints, pressure limits, compressor/facility capacity and uncertainty before considering any operating change.</div>',unsafe_allow_html=True)

# ---------- Field Analytics ----------
elif page == "Field Analytics":
    section("Field Analytics", "WELL COMPARISON")
    metric=st.selectbox("Visualise metric",["Oil_Rate","Water_Cut","Injection_Rate","Bottomhole_Pressure","Pump_Vibration","Motor_Temperature","Energy_kWh"])
    plot=latest.copy(); y=metric
    if metric=="Water_Cut": title="Current water cut by well"; y="Water_Cut"
    else: title=f"Latest {metric.replace('_',' ')} by well"
    fig=px.scatter(plot,x="Well",y=y,size="Oil_Rate",color="Priority",hover_data=["Oil_Rate","Water_Cut","Injection_Rate","Anomaly"],title=title)
    fig.update_layout(template="plotly_dark",height=390); st.plotly_chart(fig,use_container_width=True)
    section("Well opportunity map", "FIELD VIEW")
    rng=np.random.default_rng(8); mapdf=latest.copy(); mapdf["X_demo"]=rng.uniform(0,100,len(mapdf)); mapdf["Y_demo"]=rng.uniform(0,60,len(mapdf))
    fig=px.scatter(mapdf,x="X_demo",y="Y_demo",text="Well",color="Priority",size="Oil_Rate",hover_data=["Oil_Rate","Water_Cut","Anomaly"])
    fig.update_traces(textposition="top center"); fig.update_layout(template="plotly_dark",height=420,xaxis_title="Field coordinate X (demo)",yaxis_title="Field coordinate Y (demo)")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Coordinates are synthetic unless actual field coordinates are uploaded and integrated.")

# ---------- Explainable AI ----------
elif page == "Explainable AI":
    section("Model Transparency", "PROXY MODEL & SCREENING")
    imp=pd.Series(model.feature_importances_,index=FEATURES).sort_values(ascending=True).reset_index(); imp.columns=["Feature","Importance"]
    fig=px.bar(imp,x="Importance",y="Feature",orientation="h",title="Oil-rate model feature importance")
    fig.update_layout(template="plotly_dark",height=330); st.plotly_chart(fig,use_container_width=True)
    st.caption("Feature importance describes the model's global use of inputs; it does not establish physical causality.")
    section("Held-out validation", "CHRONOLOGICAL TEST DATA")
    a,b=st.columns(2)
    with a: kpi("Test-set MAE", f"{mae:.2f}", "oil-rate units; same units as Oil_Rate")
    with b: kpi("Test observations", f"{len(test_df):,}", "held-out later observations")
    fig=px.scatter(test_df,x="Oil_Rate",y="Predicted_Oil",opacity=.55,title="Actual vs predicted oil rate")
    lo=min(test_df.Oil_Rate.min(),test_df.Predicted_Oil.min()); hi=max(test_df.Oil_Rate.max(),test_df.Predicted_Oil.max())
    fig.add_shape(type="line",x0=lo,y0=lo,x1=hi,y1=hi,line=dict(dash="dash"))
    fig.update_layout(template="plotly_dark",height=380,xaxis_title="Actual oil rate",yaxis_title="Predicted oil rate")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Validation results depend on the uploaded data quality, date coverage and representativeness. Synthetic-demo metrics do not establish real-field performance.")

# ---------- Methodology ----------
elif page == "Methodology":
    section("How FIELDWISE AI Works", "TECHNICAL ARCHITECTURE")
    steps=[
        ("01","Field data ingestion","Load production, pressure, injection and optional artificial-lift sensor data."),
        ("02","Data validation","Check required columns, timestamps, numeric values and consistent units."),
        ("03","Production proxy model","Random Forest estimates oil rate from available well operating variables."),
        ("04","Equipment anomaly screening","Isolation Forest flags unusual combinations of vibration, temperature, current, frequency and energy indicators."),
        ("05","Multi-well scenario evaluation","Compare alternative allocations under a user-defined total resource limit."),
        ("06","Performance comparison","Review oil-rate estimates, water-cut context, anomaly flags and validation metrics."),
        ("07","Engineer review","Use results to support investigation; no autonomous field-control commands are issued.")]
    for n,t,d in steps: st.markdown(f'<div class="card" style="margin-bottom:10px"><b>{n} · {t}</b><br><span class="small">{d}</span></div>',unsafe_allow_html=True)
    section("Technology stack", "IMPLEMENTATION")
    st.code("Python · Pandas · NumPy · Scikit-learn · Plotly · Streamlit", language="text")
    section("Current prototype boundaries", "IMPORTANT")
    st.markdown('<div class="alert warn"><b>Prototype, not a certified operational system.</b><br>The oil-rate model is a statistical proxy. Equipment monitoring uses anomaly detection, not a trained failure-timing predictor. The allocation routine is a demonstration scenario tool and does not prove an optimal gas-lift schedule. Validate against real field data, equipment-specific limits, reservoir behaviour, facilities constraints and engineering review before operational use.</div>',unsafe_allow_html=True)
    section("Innovation statement", "WHY THIS IS DIFFERENT")
    st.markdown('<div class="alert"><b>FIELDWISE AI connects production surveillance, equipment anomaly screening and multi-well operating-scenario comparison in one engineer-facing workflow.</b><br><br>It aims to help engineers identify where to investigate and which operating alternatives merit further evaluation—without replacing physics-based reservoir simulation or human operational approval.</div>',unsafe_allow_html=True)
