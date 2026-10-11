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
        ("01","Industrial data ingestion","Recognise common header aliases, select required operational/sensor columns and ignore unrelated log curves."),
        ("02","Automatic data cleaning","Parse dates, standardise well IDs, remove exact duplicates, impute missing numeric values and cap statistical outliers for engineer review."),
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
