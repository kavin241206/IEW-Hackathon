
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="FIELDWISE AI",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Synthetic demonstration data
# -----------------------------
@st.cache_data
def create_demo_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", "2026-09-01", freq="MS")
    wells = [f"W-{i:03d}" for i in range(1, 25)]

    rows = []
    for wi, well in enumerate(wells):
        base_oil = rng.uniform(150, 330)
        base_water = rng.uniform(250, 600)
        base_gas = rng.uniform(80, 260)
        base_inj = rng.uniform(550, 900)
        base_bhp = rng.uniform(1250, 1750)

        for m, date in enumerate(dates):
            decline = np.exp(-0.0105 * m)
            seasonal = 1 + 0.025 * np.sin(m / 4 + wi)

            injection = np.clip(
                base_inj + 45*np.sin(m/5 + wi) + rng.normal(0, 18),
                450, 1100
            )
            bhp = base_bhp - 1.4*m + 0.32*(injection-base_inj) + rng.normal(0, 14)

            water_rate = (
                base_water * (1 + 0.0085*m)
                + 0.11*max(injection-base_inj, 0)
                + rng.normal(0, 22)
            )
            water_rate = max(water_rate, 80)

            oil_rate = (
                base_oil * decline * seasonal
                + 0.065*(injection-base_inj)
                + 0.025*(bhp-base_bhp)
                + rng.normal(0, 7)
            )
            oil_rate = max(oil_rate, 45)

            gas_rate = max(
                base_gas * (0.93 + 0.07*decline) + rng.normal(0, 10), 20
            )

            water_cut = water_rate / (water_rate + oil_rate)

            whp = max(350, bhp - 430 + rng.normal(0, 12))

            rows.append([
                date, well, oil_rate, water_rate, gas_rate,
                whp, bhp, injection, water_cut
            ])

    df = pd.DataFrame(rows, columns=[
        "Date", "Well_ID", "Oil_Rate", "Water_Rate", "Gas_Rate",
        "Wellhead_Pressure", "Bottomhole_Pressure",
        "Injection_Rate", "Water_Cut"
    ])
    return df


df = create_demo_data()

# -----------------------------
# ML proxy model
# -----------------------------
FEATURES = [
    "Water_Rate",
    "Gas_Rate",
    "Wellhead_Pressure",
    "Bottomhole_Pressure",
    "Injection_Rate"
]
TARGET = "Oil_Rate"

@st.cache_resource
def train_proxy_model(data):
    # Chronological split: earlier records train, later records validate.
    ordered = data.sort_values("Date")
    split_date = ordered["Date"].quantile(0.80)

    train = ordered[ordered["Date"] <= split_date]
    test = ordered[ordered["Date"] > split_date]

    model = RandomForestRegressor(
        n_estimators=220,
        max_depth=12,
        random_state=42,
        min_samples_leaf=3
    )
    model.fit(train[FEATURES], train[TARGET])
    pred = model.predict(test[FEATURES])
    mae = mean_absolute_error(test[TARGET], pred)

    return model, mae

model, validation_mae = train_proxy_model(df)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0;
}
.subtitle {
    font-size: 18px;
    opacity: 0.75;
    margin-bottom: 25px;
}
.card {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.25);
    background: rgba(128,128,128,0.06);
}
.priority-high {
    padding: 14px;
    border-radius: 12px;
    border: 1px solid rgba(220,60,60,0.45);
    background: rgba(220,60,60,0.08);
}
.small-note {
    font-size: 12px;
    opacity: 0.65;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("FIELDWISE AI")
st.sidebar.caption("Production intelligence for mature oil fields")

page = st.sidebar.radio(
    "Navigate",
    ["Executive Dashboard", "AI Well Intelligence", "Scenario Optimisation", "How It Works"]
)

st.sidebar.divider()
st.sidebar.info(
    "Prototype mode\n\n"
    "This demonstration uses synthetic data to show the proposed decision-support workflow."
)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">FIELDWISE AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Turning field history into smarter production decisions</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Executive Dashboard
# -----------------------------
if page == "Executive Dashboard":
    latest = df[df["Date"] == df["Date"].max()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Wells", len(wells) if "wells" in globals() else df["Well_ID"].nunique())
    c2.metric("Current Oil Production", f"{latest['Oil_Rate'].sum():,.0f} BOPD")
    c3.metric("Average Water Cut", f"{latest['Water_Cut'].mean()*100:.1f}%")
    high_priority = int(
        (
            (latest["Water_Cut"] > latest["Water_Cut"].quantile(0.75)) |
            (latest["Oil_Rate"] < latest["Oil_Rate"].quantile(0.25))
        ).sum()
    )
    c4.metric("Wells Requiring Attention", high_priority)

    st.subheader("Field Performance")

    monthly = df.groupby("Date", as_index=False).agg(
        Oil_Rate=("Oil_Rate", "sum"),
        Water_Rate=("Water_Rate", "sum")
    )
    monthly["Water_Cut"] = monthly["Water_Rate"] / (
        monthly["Water_Rate"] + monthly["Oil_Rate"]
    )

    left, right = st.columns(2)

    with left:
        fig = px.line(
            monthly, x="Date", y="Oil_Rate",
            title="Field Oil Production Trend",
            labels={"Oil_Rate": "Oil Rate (BOPD)", "Date": ""}
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.line(
            monthly, x="Date", y="Water_Cut",
            title="Field Water Cut Trend",
            labels={"Water_Cut": "Water Cut", "Date": ""}
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Current Well Performance")
    display = latest[[
        "Well_ID", "Oil_Rate", "Water_Rate", "Water_Cut",
        "Bottomhole_Pressure", "Injection_Rate"
    ]].copy()
    display["Water_Cut"] = (display["Water_Cut"] * 100).round(1).astype(str) + "%"
    display.columns = [
        "Well", "Oil (BOPD)", "Water (BWPD)", "Water Cut",
        "BHP (psi)", "Injection (bbl/day)"
    ]
    st.dataframe(display.sort_values("Oil (BOPD)"), use_container_width=True, hide_index=True)

# -----------------------------
# AI Well Intelligence
# -----------------------------
elif page == "AI Well Intelligence":
    well = st.selectbox("Select a well", sorted(df["Well_ID"].unique()))
    w = df[df["Well_ID"] == well].sort_values("Date").copy()
    latest = w.iloc[-1]

    # Simple decline indicator using the last 6 months vs previous 6 months
    recent = w.tail(6)["Oil_Rate"].mean()
    previous = w.iloc[-12:-6]["Oil_Rate"].mean()
    decline_pct = (recent - previous) / previous * 100

    water_cut = latest["Water_Cut"]
    priority_score = 0
    if decline_pct < -8:
        priority_score += 2
    if water_cut > 0.70:
        priority_score += 2
    elif water_cut > 0.60:
        priority_score += 1

    if priority_score >= 3:
        priority = "HIGH PRIORITY"
        icon = "🔴"
    elif priority_score >= 2:
        priority = "MEDIUM PRIORITY"
        icon = "🟠"
    else:
        priority = "LOW PRIORITY"
        icon = "🟢"

    st.subheader(f"{well} — AI Well Assessment")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Oil Rate", f"{latest['Oil_Rate']:.0f} BOPD")
    c2.metric("Water Rate", f"{latest['Water_Rate']:.0f} BWPD")
    c3.metric("Water Cut", f"{water_cut*100:.1f}%")
    c4.metric("BHP", f"{latest['Bottomhole_Pressure']:.0f} psi")
    c5.metric("Oil Decline", f"{decline_pct:.1f}%")

    if priority == "HIGH PRIORITY":
        st.markdown(
            f'<div class="priority-high"><b>{icon} {priority}</b><br>'
            'The prototype flags this well for engineering review based on its '
            'production decline and/or water contribution.</div>',
            unsafe_allow_html=True
        )
    else:
        st.success(f"{icon} {priority} — no strong prototype warning signal.")

    left, right = st.columns(2)

    with left:
        fig = px.line(
            w, x="Date", y="Oil_Rate",
            title=f"{well}: Oil Production",
            labels={"Oil_Rate": "Oil Rate (BOPD)", "Date": ""}
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.line(
            w, x="Date", y="Water_Cut",
            title=f"{well}: Water Cut",
            labels={"Water_Cut": "Water Cut", "Date": ""}
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Why was this well flagged?")
    reasons = []
    if decline_pct < -8:
        reasons.append(f"Oil production has declined by approximately {abs(decline_pct):.1f}% over the recent comparison period.")
    if water_cut > 0.70:
        reasons.append(f"Water cut is high at approximately {water_cut*100:.1f}%.")
    elif water_cut > 0.60:
        reasons.append(f"Water cut is elevated at approximately {water_cut*100:.1f}%.")
    if not reasons:
        reasons.append("No major prototype warning signal was detected from the selected indicators.")

    for r in reasons:
        st.write("• " + r)

# -----------------------------
# Scenario Optimisation
# -----------------------------
elif page == "Scenario Optimisation":
    st.subheader("AI Scenario Optimisation")
    st.write(
        "Test alternative injection conditions and use the trained proxy model "
        "to rapidly estimate the corresponding oil response."
    )

    well = st.selectbox("Select well", sorted(df["Well_ID"].unique()), key="scenario_well")
    w = df[df["Well_ID"] == well].sort_values("Date")
    latest = w.iloc[-1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Injection", f"{latest['Injection_Rate']:.0f} bbl/day")
    c2.metric("Current Oil", f"{latest['Oil_Rate']:.0f} BOPD")
    c3.metric("Current Water Cut", f"{latest['Water_Cut']*100:.1f}%")

    injection = st.slider(
        "Test Injection Rate (bbl/day)",
        min_value=450,
        max_value=1100,
        value=int(round(latest["Injection_Rate"])),
        step=10
    )

    run = st.button("▶ RUN AI SCENARIO", type="primary", use_container_width=True)

    if run:
        # Hold other observed variables at the current well condition
        X = pd.DataFrame([{
            "Water_Rate": latest["Water_Rate"],
            "Gas_Rate": latest["Gas_Rate"],
            "Wellhead_Pressure": latest["Wellhead_Pressure"],
            "Bottomhole_Pressure": latest["Bottomhole_Pressure"],
            "Injection_Rate": injection
        }])

        predicted_oil = float(model.predict(X)[0])

        # Demonstration water response: modest sensitivity to injection change.
        # This is a prototype heuristic, not a calibrated reservoir model.
        delta_inj = injection - latest["Injection_Rate"]
        predicted_water = max(
            50,
            latest["Water_Rate"] + 0.10 * max(delta_inj, 0) - 0.035 * max(-delta_inj, 0)
        )
        predicted_wc = predicted_water / (predicted_water + predicted_oil)

        oil_change = predicted_oil - latest["Oil_Rate"]
        water_change = predicted_water - latest["Water_Rate"]

        st.divider()
        st.subheader("Scenario Result")

        a, b, c = st.columns(3)
        a.metric("Predicted Oil", f"{predicted_oil:.0f} BOPD", f"{oil_change:+.0f}")
        b.metric("Estimated Water", f"{predicted_water:.0f} BWPD", f"{water_change:+.0f}")
        c.metric("Estimated Water Cut", f"{predicted_wc*100:.1f}%")

        scenarios = []
        for inj in range(max(450, injection-200), min(1100, injection+200)+1, 25):
            XX = X.copy()
            XX["Injection_Rate"] = inj
            po = float(model.predict(XX)[0])
            pw = max(
                50,
                latest["Water_Rate"] + 0.10*max(inj-latest["Injection_Rate"], 0)
                - 0.035*max(latest["Injection_Rate"]-inj, 0)
            )
            scenarios.append([inj, po, pw, pw/(pw+po)])

        sc = pd.DataFrame(scenarios, columns=[
            "Injection_Rate", "Predicted_Oil", "Estimated_Water", "Estimated_Water_Cut"
        ])

        fig = px.line(
            sc, x="Injection_Rate", y="Predicted_Oil",
            markers=True,
            title="Proxy Model: Predicted Oil vs Injection Rate",
            labels={"Injection_Rate": "Injection Rate (bbl/day)", "Predicted_Oil": "Predicted Oil (BOPD)"}
        )
        fig.add_vline(x=injection, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

        best_idx = sc["Predicted_Oil"].idxmax()
        best = sc.loc[best_idx]

        if predicted_oil > latest["Oil_Rate"]:
            recommendation = (
                f"Tested scenario indicates an increase in predicted oil rate. "
                f"The proxy model identifies approximately {best['Injection_Rate']:.0f} bbl/day "
                f"as the highest-oil scenario within the tested range."
            )
        else:
            recommendation = (
                "The tested scenario does not improve predicted oil rate. "
                "The engineer should review alternative operating conditions."
            )

        st.info("💡 **AI Recommendation**\n\n" + recommendation)

        st.caption(
            "Prototype disclaimer: scenario outputs are based on synthetic demonstration data "
            "and a prototype model. They are not field-operating recommendations."
        )

    st.subheader("Model Validation")
    st.write(
        f"Prototype validation MAE on the held-out demonstration period: "
        f"**{validation_mae:.1f} BOPD**."
    )
    st.caption(
        "This metric is only a demonstration of the modelling workflow. "
        "Real deployment would require validated field data and engineering acceptance criteria."
    )

# -----------------------------
# How It Works
# -----------------------------
else:
    st.subheader("How FIELDWISE AI Works")

    steps = [
        ("1. Historical Field Data",
         "Production, pressure, injection and well-performance observations."),
        ("2. Data Quality & Processing",
         "Clean missing/noisy observations and prepare model-ready variables."),
        ("3. ML Proxy Model",
         "Learn the relationship between operating conditions and production response."),
        ("4. Well Intelligence",
         "Identify wells showing decline or elevated water contribution."),
        ("5. Scenario Prediction",
         "Rapidly screen alternative operating/injection conditions."),
        ("6. Optimisation",
         "Compare predicted outcomes and identify promising scenarios."),
        ("7. Engineer Recommendation",
         "Present the result as decision support rather than replacing engineering judgement.")
    ]

    for title, desc in steps:
        st.markdown(f"### {title}")
        st.write(desc)
        if title != steps[-1][0]:
            st.write("↓")

    st.divider()
    st.subheader("Prototype Architecture")
    st.code("""
Historical Field Data
        ↓
Python / Pandas
        ↓
ML Proxy Model
        ↓
Well Performance Analysis
        ↓
Scenario Prediction
        ↓
Optimisation
        ↓
Streamlit Dashboard
        ↓
Engineer Decision Support
""")

    st.warning(
        "The current application is a prototype using synthetic demonstration data. "
        "A production implementation would require validated field data, model calibration, "
        "uncertainty assessment, engineering constraints and field-level validation."
    )

st.divider()
st.caption(
    "FIELDWISE AI | Prototype for FIPI / India Energy Week Hackathon | "
    "Demonstration system — not a field-operating recommendation."
)
