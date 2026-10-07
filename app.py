
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# =========================================================
# FIELDWISE AI — V2
# =========================================================
st.set_page_config(
    page_title="FIELDWISE AI | Mature Field Intelligence",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- DATA --------------------------
@st.cache_data
def create_demo_data():
    rng = np.random.default_rng(24)
    dates = pd.date_range("2023-01-01", "2026-09-01", freq="MS")
    wells = [f"W-{i:03d}" for i in range(1, 25)]
    rows = []

    for wi, well in enumerate(wells):
        base_oil = rng.uniform(155, 335)
        base_water = rng.uniform(260, 590)
        base_gas = rng.uniform(90, 255)
        base_inj = rng.uniform(560, 900)
        base_bhp = rng.uniform(1280, 1780)

        for m, date in enumerate(dates):
            decline = np.exp(-0.0108 * m)
            season = 1 + 0.022*np.sin(m/4.2 + wi)
            injection = np.clip(
                base_inj + 48*np.sin(m/5 + wi) + rng.normal(0, 16),
                450, 1100
            )
            bhp = base_bhp - 1.35*m + 0.34*(injection-base_inj) + rng.normal(0, 13)
            water = max(
                base_water*(1 + 0.0087*m)
                + 0.11*max(injection-base_inj, 0)
                + rng.normal(0, 20), 80
            )
            oil = max(
                base_oil*decline*season
                + 0.070*(injection-base_inj)
                + 0.022*(bhp-base_bhp)
                + rng.normal(0, 7), 42
            )
            gas = max(base_gas*(0.94 + 0.06*decline) + rng.normal(0, 10), 20)
            whp = max(350, bhp - 430 + rng.normal(0, 12))
            wc = water/(water+oil)

            rows.append([
                date, well, oil, water, gas, whp, bhp,
                injection, wc
            ])

    return pd.DataFrame(rows, columns=[
        "Date","Well_ID","Oil_Rate","Water_Rate","Gas_Rate",
        "Wellhead_Pressure","Bottomhole_Pressure","Injection_Rate","Water_Cut"
    ])

df = create_demo_data()

FEATURES = [
    "Water_Rate", "Gas_Rate", "Wellhead_Pressure",
    "Bottomhole_Pressure", "Injection_Rate"
]

@st.cache_resource
def train_model(data):
    ordered = data.sort_values("Date")
    split = ordered["Date"].quantile(0.80)
    train = ordered[ordered["Date"] <= split]
    test = ordered[ordered["Date"] > split]

    model = RandomForestRegressor(
        n_estimators=260,
        max_depth=13,
        min_samples_leaf=3,
        random_state=42
    )
    model.fit(train[FEATURES], train["Oil_Rate"])
    pred = model.predict(test[FEATURES])
    mae = mean_absolute_error(test["Oil_Rate"], pred)
    return model, mae

model, mae = train_model(df)

# ----------------------- HELPERS -------------------------
def latest_snapshot():
    return df[df["Date"] == df["Date"].max()].copy()

def well_stats(well):
    w = df[df["Well_ID"] == well].sort_values("Date")
    latest = w.iloc[-1]
    recent = w.tail(6)["Oil_Rate"].mean()
    previous = w.iloc[-12:-6]["Oil_Rate"].mean()
    decline = (recent-previous)/previous*100
    return w, latest, decline

def priority_for(well):
    w, latest, decline = well_stats(well)
    score = 0
    if decline < -8: score += 2
    if latest["Water_Cut"] > 0.70: score += 2
    elif latest["Water_Cut"] > 0.60: score += 1

    if score >= 3:
        return "HIGH", score
    if score >= 2:
        return "MEDIUM", score
    return "LOW", score

def scenario_table(well, low=450, high=1100):
    w, latest, _ = well_stats(well)
    records = []
    for inj in range(low, high+1, 25):
        X = pd.DataFrame([{
            "Water_Rate": latest["Water_Rate"],
            "Gas_Rate": latest["Gas_Rate"],
            "Wellhead_Pressure": latest["Wellhead_Pressure"],
            "Bottomhole_Pressure": latest["Bottomhole_Pressure"],
            "Injection_Rate": inj
        }])
        oil = float(model.predict(X)[0])

        # Demonstration-only water response heuristic.
        delta = inj-latest["Injection_Rate"]
        water = max(
            50,
            latest["Water_Rate"]
            + 0.10*max(delta, 0)
            - 0.035*max(-delta, 0)
        )
        wc = water/(water+oil)
        records.append([inj, oil, water, wc])
    return pd.DataFrame(records, columns=[
        "Injection_Rate","Predicted_Oil","Estimated_Water","Estimated_Water_Cut"
    ])

# ------------------------ CSS -----------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.hero {
    padding: 28px 30px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,.22);
    background: linear-gradient(135deg, rgba(80,110,150,.13), rgba(40,40,40,.04));
    margin-bottom: 20px;
}
.hero h1 {font-size: 46px; margin: 0;}
.hero p {font-size: 18px; margin: 8px 0 0; opacity: .78;}
.section-title {font-size: 25px; font-weight: 750; margin-top: 8px;}
.card {
    border: 1px solid rgba(128,128,128,.22);
    border-radius: 15px;
    padding: 17px;
    background: rgba(128,128,128,.035);
}
.alert {
    border: 1px solid rgba(210,75,75,.45);
    border-radius: 14px;
    padding: 16px;
    background: rgba(210,75,75,.08);
}
.good {
    border: 1px solid rgba(75,160,100,.45);
    border-radius: 14px;
    padding: 16px;
    background: rgba(75,160,100,.08);
}
.muted {opacity:.65; font-size: 12px;}
</style>
""", unsafe_allow_html=True)

# ----------------------- SIDEBAR --------------------------
st.sidebar.markdown("## 🛢️ FIELDWISE AI")
st.sidebar.caption("Mature-field production intelligence")

page = st.sidebar.radio(
    "WORKSPACE",
    [
        "Command Center",
        "Well Intelligence",
        "Scenario Lab",
        "Field Analytics",
        "Methodology",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("### Prototype status")
st.sidebar.success("● DEMONSTRATION READY")
st.sidebar.caption("Synthetic data • ML proxy • decision-support workflow")
st.sidebar.divider()
st.sidebar.caption("FIELDWISE AI V2")
st.sidebar.caption("Prototype for FIPI / India Energy Week Hackathon")

# =========================================================
# COMMAND CENTER
# =========================================================
if page == "Command Center":
    snap = latest_snapshot()
    total_oil = snap["Oil_Rate"].sum()
    avg_wc = snap["Water_Cut"].mean()
    high = sum(priority_for(w)[0] == "HIGH" for w in df["Well_ID"].unique())

    st.markdown("""
    <div class="hero">
      <h1>FIELDWISE AI</h1>
      <p>Turning field history into smarter production decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Field Command Center")
    st.caption("A single decision-support view for mature-field performance, well risk and scenario screening.")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Active Wells", f"{df['Well_ID'].nunique()}")
    c2.metric("Current Oil", f"{total_oil:,.0f} BOPD")
    c3.metric("Average Water Cut", f"{avg_wc*100:.1f}%")
    c4.metric("High-Priority Wells", f"{high}")

    monthly = df.groupby("Date", as_index=False).agg(
        Oil=("Oil_Rate","sum"),
        Water=("Water_Rate","sum")
    )
    monthly["Water_Cut"] = monthly["Water"]/(monthly["Water"]+monthly["Oil"])

    left,right = st.columns(2)
    with left:
        fig = px.area(monthly, x="Date", y="Oil", title="Field Oil Production")
        fig.update_layout(height=340, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.line(monthly, x="Date", y="Water_Cut", title="Field Water-Cut Trend")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=340, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### AI Opportunity Queue")
    records = []
    for well in df["Well_ID"].unique():
        w, latest, decline = well_stats(well)
        p,_ = priority_for(well)
        records.append([
            well, p, latest["Oil_Rate"], latest["Water_Cut"]*100,
            decline, latest["Injection_Rate"]
        ])
    q = pd.DataFrame(records, columns=[
        "Well","Priority","Oil (BOPD)","Water Cut (%)",
        "Recent Oil Change (%)","Injection (bbl/day)"
    ])
    order = {"HIGH":0,"MEDIUM":1,"LOW":2}
    q["_o"] = q["Priority"].map(order)
    q = q.sort_values(["_o","Water Cut (%)"], ascending=[True,False]).drop(columns="_o")
    st.dataframe(q.head(10), use_container_width=True, hide_index=True)

    st.info(
        "Decision workflow: identify the wells requiring attention → inspect drivers → "
        "screen operating scenarios → present a recommendation for engineer review."
    )

# =========================================================
# WELL INTELLIGENCE
# =========================================================
elif page == "Well Intelligence":
    st.markdown("## 🧠 AI Well Intelligence")
    st.caption("Move from field-level trends to a well-level engineering view.")

    well = st.selectbox("Select well", sorted(df["Well_ID"].unique()))
    w, latest, decline = well_stats(well)
    priority,_ = priority_for(well)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Oil Rate", f"{latest['Oil_Rate']:.0f} BOPD")
    c2.metric("Water Rate", f"{latest['Water_Rate']:.0f} BWPD")
    c3.metric("Water Cut", f"{latest['Water_Cut']*100:.1f}%")
    c4.metric("BHP", f"{latest['Bottomhole_Pressure']:.0f} psi")
    c5.metric("Recent Oil Change", f"{decline:.1f}%")

    if priority == "HIGH":
        st.markdown(
            '<div class="alert"><b>🔴 HIGH PRIORITY</b><br>'
            'Prototype screening suggests this well deserves engineering review.</div>',
            unsafe_allow_html=True
        )
    elif priority == "MEDIUM":
        st.warning("🟠 MEDIUM PRIORITY — review the well trend and operating context.")
    else:
        st.markdown(
            '<div class="good"><b>🟢 LOW PRIORITY</b><br>'
            'No strong warning signal from the prototype indicators.</div>',
            unsafe_allow_html=True
        )

    a,b = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w["Date"], y=w["Oil_Rate"], mode="lines+markers", name="Oil"))
        fig.update_layout(title="Oil Production History", yaxis_title="BOPD", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w["Date"], y=w["Water_Cut"], mode="lines+markers", name="Water Cut"))
        fig.update_layout(title="Water-Cut History", yaxis_title="Water Cut", xaxis_title="")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔍 AI Driver Assessment")
    drivers = {
        "Production decline": max(0, min(100, -decline*3)),
        "Water contribution": max(0, min(100, latest["Water_Cut"]*100)),
        "Pressure condition": max(0, min(100, (1800-latest["Bottomhole_Pressure"])/8)),
    }
    for name,val in drivers.items():
        st.write(f"**{name}** — {val:.0f}/100")
        st.progress(int(val))

    reasons=[]
    if decline < -8:
        reasons.append("Recent oil production is declining.")
    if latest["Water_Cut"] > .70:
        reasons.append("Water cut is high.")
    elif latest["Water_Cut"] > .60:
        reasons.append("Water cut is elevated.")
    if not reasons:
        reasons.append("No major prototype warning indicator was detected.")

    st.markdown("### AI-generated engineering note")
    st.write(" • " + "\n • ".join(reasons))

    st.caption(
        "This is screening support, not a replacement for reservoir/production engineering judgement."
    )

# =========================================================
# SCENARIO LAB
# =========================================================
elif page == "Scenario Lab":
    st.markdown("## 🎛️ Scenario Lab")
    st.caption("Rapidly screen alternative injection conditions with the trained proxy model.")

    well = st.selectbox("Select well", sorted(df["Well_ID"].unique()), key="scenario")
    w, latest, decline = well_stats(well)

    c1,c2,c3 = st.columns(3)
    c1.metric("Current Injection", f"{latest['Injection_Rate']:.0f} bbl/day")
    c2.metric("Current Oil", f"{latest['Oil_Rate']:.0f} BOPD")
    c3.metric("Current Water Cut", f"{latest['Water_Cut']*100:.1f}%")

    inj = st.slider(
        "Proposed Injection Rate",
        450, 1100,
        int(round(latest["Injection_Rate"])),
        10
    )

    run = st.button("▶ RUN AI SCENARIO", type="primary", use_container_width=True)

    if run:
        table = scenario_table(well)
        chosen = table.iloc[(table["Injection_Rate"]-inj).abs().argsort()[:1]].iloc[0]
        best = table.loc[table["Predicted_Oil"].idxmax()]

        delta_oil = chosen["Predicted_Oil"]-latest["Oil_Rate"]
        delta_wc = chosen["Estimated_Water_Cut"]-latest["Water_Cut"]

        st.divider()
        st.markdown("### Scenario Outcome")

        x1,x2,x3 = st.columns(3)
        x1.metric("Predicted Oil", f"{chosen['Predicted_Oil']:.0f} BOPD", f"{delta_oil:+.0f}")
        x2.metric("Estimated Water", f"{chosen['Estimated_Water']:.0f} BWPD")
        x3.metric("Estimated Water Cut", f"{chosen['Estimated_Water_Cut']*100:.1f}%", f"{delta_wc*100:+.1f} pp")

        fig = px.line(
            table, x="Injection_Rate", y="Predicted_Oil",
            markers=True, title="Proxy Response Curve"
        )
        fig.add_vline(x=inj, line_dash="dash")
        fig.add_vline(x=float(best["Injection_Rate"]), line_dash="dot")
        fig.update_layout(
            xaxis_title="Injection Rate (bbl/day)",
            yaxis_title="Predicted Oil (BOPD)"
        )
        st.plotly_chart(fig, use_container_width=True)

        if delta_oil > 0 and chosen["Estimated_Water_Cut"] <= latest["Water_Cut"]:
            tone = "good"
            text = (
                f"The tested scenario shows a potentially favourable response. "
                f"Within the screened range, the highest predicted oil response occurs "
                f"near **{best['Injection_Rate']:.0f} bbl/day**."
            )
        elif delta_oil > 0:
            tone = "alert"
            text = (
                f"Oil response improves in the tested scenario, but estimated water contribution "
                f"also rises. The result should be reviewed against field constraints."
            )
        else:
            tone = "alert"
            text = (
                "The tested condition does not improve predicted oil response. "
                "Review alternative operating conditions."
            )

        st.markdown(f'<div class="{tone}"><b>💡 AI Recommendation</b><br>{text}</div>',
                    unsafe_allow_html=True)

        st.caption(
            "Prototype only: numerical scenario outputs are generated from synthetic demonstration "
            "data and a prototype model. They are not field-operating recommendations."
        )

    st.divider()
    st.markdown("### Model validation")
    st.write(f"Held-out demonstration-period MAE: **{mae:.1f} BOPD**")
    st.caption(
        "Real deployment would require validated field data, uncertainty analysis, "
        "engineering constraints and independent field validation."
    )

# =========================================================
# FIELD ANALYTICS
# =========================================================
elif page == "Field Analytics":
    st.markdown("## 📊 Field Analytics")
    st.caption("Use historical behaviour to understand where production potential may be changing.")

    metric = st.selectbox(
        "Field metric",
        ["Oil Rate", "Water Cut", "Injection Rate", "Bottomhole Pressure"]
    )

    agg = df.groupby("Date", as_index=False).agg(
        Oil_Rate=("Oil_Rate","sum"),
        Water_Rate=("Water_Rate","sum"),
        Injection_Rate=("Injection_Rate","sum"),
        Bottomhole_Pressure=("Bottomhole_Pressure","mean")
    )
    agg["Water_Cut"] = agg["Water_Rate"]/(agg["Water_Rate"]+agg["Oil_Rate"])

    col_map = {
        "Oil Rate":"Oil_Rate",
        "Water Cut":"Water_Cut",
        "Injection Rate":"Injection_Rate",
        "Bottomhole Pressure":"Bottomhole_Pressure"
    }
    y = col_map[metric]
    fig = px.line(agg, x="Date", y=y, markers=True, title=f"Field {metric}")
    if metric == "Water Cut":
        fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Well opportunity map")
    latest = latest_snapshot()
    latest["Priority"] = latest["Well_ID"].map(lambda x: priority_for(x)[0])
    latest["Oil_per_1000_BWPD"] = latest["Oil_Rate"]/(latest["Water_Rate"]/1000)

    fig = px.scatter(
        latest,
        x="Water_Cut", y="Oil_Rate",
        size="Injection_Rate",
        hover_name="Well_ID",
        text="Well_ID",
        symbol="Priority",
        title="Current Well Positioning: Oil Rate vs Water Cut",
        labels={"Water_Cut":"Water Cut","Oil_Rate":"Oil Rate (BOPD)"}
    )
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Interpretation: wells with lower oil performance and higher water contribution "
        "may warrant closer engineering review. This screening view does not diagnose "
        "the underlying reservoir mechanism."
    )

# =========================================================
# METHODOLOGY
# =========================================================
else:
    st.markdown("## 🧩 How FIELDWISE AI Works")
    st.caption("The prototype follows the proposed data-to-decision workflow.")

    stages = [
        ("01", "Historical Field Data", "Oil, water, gas, pressure and injection observations."),
        ("02", "Data Processing", "Prepare noisy/missing operational data for modelling."),
        ("03", "ML Proxy Model", "Learn relationships between operating conditions and oil response."),
        ("04", "Well Intelligence", "Screen wells for decline and elevated water contribution."),
        ("05", "Scenario Lab", "Rapidly evaluate alternative operating/injection conditions."),
        ("06", "Optimisation", "Compare predicted outcomes within the screened scenario space."),
        ("07", "Engineer Decision Support", "Present interpretable outputs for engineering review."),
    ]

    for num,title,desc in stages:
        st.markdown(f"### {num} — {title}")
        st.write(desc)
        if num != "07":
            st.markdown("↓")

    st.divider()
    st.markdown("### Technology Stack")
    st.code("""
Data                  → CSV / field-history format
Processing            → Python + Pandas
ML Proxy               → Scikit-Learn / Random Forest prototype
Visual Analytics      → Plotly
Application            → Streamlit
Deployment target      → Streamlit-compatible hosting
""")

    st.markdown("### What is innovative here?")
    st.write(
        "The prototype connects four activities in one decision-support workflow: "
        "historical field behaviour, well-level diagnostics, rapid scenario prediction "
        "and optimisation-oriented recommendations."
    )

    st.warning(
        "Prototype limitation: the current dataset is synthetic. The model is a demonstration "
        "of the proposed workflow, not a validated reservoir model. A production implementation "
        "must be calibrated and validated with appropriate field data and engineering constraints."
    )

st.divider()
st.caption(
    "FIELDWISE AI V2 • AI decision-support prototype for mature oil fields • "
    "Synthetic demonstration data"
)
