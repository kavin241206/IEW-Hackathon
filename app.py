import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="Reser-AI | Production Optimization",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make it look more like a website and hide default Streamlit branding
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .metric-card {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #ff3333;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# DATA GENERATION (MOCK DATA FOR PROTOTYPE)
# ==========================================
@st.cache_data
def generate_field_data():
    dates = pd.date_range(start='2020-01-01', end='2026-10-01', freq='M')
    # Simulate mature field decline (exponential decline)
    time = np.arange(len(dates))
    base_oil = 10000 * np.exp(-0.02 * time) + np.random.normal(0, 200, len(dates))
    base_water = 2000 + 50 * time + np.random.normal(0, 100, len(dates)) # Water cut increasing
    
    df = pd.DataFrame({
        'Date': dates,
        'Oil_Production_bbl': np.maximum(base_oil, 0),
        'Water_Production_bbl': np.maximum(base_water, 0)
    })
    df['Water_Cut_%'] = (df['Water_Production_bbl'] / (df['Oil_Production_bbl'] + df['Water_Production_bbl'])) * 100
    return df

@st.cache_data
def get_well_data():
    wells = ['Well-A1', 'Well-B2', 'Well-C3', 'Well-D4', 'Well-E5']
    status = ['Active', 'Active', 'Shut-in', 'Active', 'Under-performing']
    oil_rate = [450, 320, 0, 510, 120]
    water_cut = [65, 82, 95, 45, 88]
    ai_recommendation = ['Maintain Choke', 'Increase Gas Lift', 'Evaluate for Workover', 'Optimal', 'Chemical Water Shut-off']
    ai_confidence = [92, 88, 75, 95, 81]
    
    return pd.DataFrame({
        'Well ID': wells,
        'Status': status,
        'Current Oil (bbl/d)': oil_rate,
        'Water Cut (%)': water_cut,
        'AI ML Intervention': ai_recommendation,
        'AI Confidence (%)': ai_confidence
    })

df_historical = generate_field_data()
df_wells = get_well_data()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2165/2165039.png", width=60)
st.sidebar.title("Reser-AI Platform")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", 
    ["📊 Field Dashboard", "🤖 AI Reservoir Modeling", "⚙️ Production Optimization"]
)

st.sidebar.markdown("---")
st.sidebar.info("Hackathon Prototype\n\n**Theme:** AI-Based Production Optimisation and Reservoir Modelling for Mature Fields.")

# ==========================================
# PAGE 1: FIELD DASHBOARD
# ==========================================
if page == "📊 Field Dashboard":
    st.title("Mature Field Overview Dashboard")
    st.markdown("Real-time monitoring of historical production and current field health indicators.")
    
    # Top Level Metrics
    col1, col2, col3, col4 = st.columns(4)
    latest = df_historical.iloc[-1]
    prev = df_historical.iloc[-2]
    
    col1.metric("Total Oil Rate", f"{latest['Oil_Production_bbl']:.0f} bbl/d", f"{latest['Oil_Production_bbl'] - prev['Oil_Production_bbl']:.0f} bbl/d")
    col2.metric("Field Water Cut", f"{latest['Water_Cut_%']:.1f} %", f"{latest['Water_Cut_%'] - prev['Water_Cut_%']:.1f} %", delta_color="inverse")
    col3.metric("Active Wells", "42", "-2")
    col4.metric("AI Health Score", "78/100", "+5")
    
    st.markdown("---")
    
    # Main Chart
    st.subheader("Historical Production vs Water Cut")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_historical['Date'], y=df_historical['Oil_Production_bbl'], fill='tozeroy', name='Oil Production', line=dict(color='#00b4d8')))
    fig.add_trace(go.Scatter(x=df_historical['Date'], y=df_historical['Water_Production_bbl'], fill='tonexty', name='Water Production', line=dict(color='#0077b6')))
    
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis_title="Year", yaxis_title="Barrels per Day (bbl)")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 2: AI RESERVOIR MODELING
# ==========================================
elif page == "🤖 AI Reservoir Modeling":
    st.title("AI Decline Curve Analysis (DCA) & Forecasting")
    st.markdown("Machine Learning predictions for remaining useful life and future production trends in mature zones.")
    
    # Simulate ML Prediction Data
    future_dates = pd.date_range(start='2026-10-01', end='2030-01-01', freq='M')
    time = np.arange(len(df_historical), len(df_historical) + len(future_dates))
    
    # Standard decline vs ML optimized decline
    standard_decline = 10000 * np.exp(-0.02 * time)
    ml_optimized = 10000 * np.exp(-0.015 * time) # slower decline due to optimization
    
    fig_pred = go.Figure()
    # Historical
    fig_pred.add_trace(go.Scatter(x=df_historical['Date'], y=df_historical['Oil_Production_bbl'], name='Historical Oil', line=dict(color='gray', width=2)))
    # Predictions
    fig_pred.add_trace(go.Scatter(x=future_dates, y=standard_decline, name='Standard Arps Decline', line=dict(color='red', dash='dash')))
    fig_pred.add_trace(go.Scatter(x=future_dates, y=ml_optimized, name='AI Optimized Forecast', line=dict(color='#00E676', width=3)))
    
    fig_pred.update_layout(title="Production Forecast: Standard vs AI Intervention", height=450, plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_pred, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("### Estimated AI Uplift\nBy applying AI-recommended artificial lift scheduling, the model predicts a **15% reduction in decline rate**, adding an estimated 250,000 bbls over the next 3 years.")
    with col2:
        st.info("### Reservoir Pressure Heatmap")
        # Generate dummy heatmap data for reservoir
        z = np.random.uniform(low=2000, high=3500, size=(10, 10))
        fig_heat = px.imshow(z, color_continuous_scale='Viridis', title="Simulated Subsurface Pressure Profile (psi)")
        fig_heat.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_heat, use_container_width=True)

# ==========================================
# PAGE 3: PRODUCTION OPTIMIZATION
# ==========================================
elif page == "⚙️ Production Optimization":
    st.title("Well-Level Intervention & Optimization")
    st.markdown("Actionable insights generated by neural networks to minimize water cut and maximize oil recovery.")
    
    st.subheader("High-Priority Well Recommendations")
    
    # Display the dataframe with stylized columns
    st.dataframe(
        df_wells.style.applymap(
            lambda x: 'background-color: #ff4b4b; color: white' if x == 'Under-performing' or x == 'Shut-in' else 'background-color: #00E676; color: black' if x == 'Active' else '',
            subset=['Status']
        ).applymap(
            lambda x: 'color: #00b4d8; font-weight: bold' if isinstance(x, str) else '',
            subset=['AI ML Intervention']
        ),
        use_container_width=True,
        height=220
    )
    
    st.markdown("---")
    st.subheader("Run Optimization Simulation")
    
    col1, col2, col3 = st.columns(3)
    target_well = col1.selectbox("Select Well to Simulate", df_wells['Well ID'])
    intervention_type = col2.selectbox("Select Action", ["Adjust Choke Size", "Chemical Water Shut-off", "Gas Lift Optimization", "Re-perforation"])
    
    if col3.button("Run ML Simulation 🚀"):
        with st.spinner('Running neural network reservoir simulation...'):
            import time
            time.sleep(1.5)
            st.success(f"Simulation Complete for {target_well}!")
            
            # Show simulated results
            st.write(f"**Predicted Outcome of {intervention_type}:**")
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Predicted Oil Production", "Up by 22%", "+112 bbl/d")
            res_col2.metric("Predicted Water Cut", "Down to 65%", "-17%")
            
            # Progress bar for visual effect
            st.progress(100, text="Confidence Level: 89%")
