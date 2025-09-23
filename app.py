# Paste your full Streamlit dashboard code hereimport streamlit as st
import streamlit as st
st.set_page_config(page_title="ART Risk Prediction", layout="centered")

import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Load trained model
model = joblib.load('logistic_model.pkl')

# Page config
st.set_page_config(page_title="ART Risk Prediction", layout="centered")
# Header
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🧬 ART Risk Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("""
Welcome to the ART Risk Prediction Dashboard.  
This tool helps estimate whether a patient is likely to be **virally suppressed**, which is a key indicator of **low transmission risk**.  
It’s designed to support decision-making for both clinicians and non-clinical users.
""")

# Sidebar
st.sidebar.header("🔍 Filter Options")
maternal_filter = st.sidebar.selectbox("Show patients who are:", ["All", "Pregnant/Breastfeeding", "Not Maternal"])
st.sidebar.markdown("Use this to focus on maternal-risk patients.")

# Input section
st.markdown("### 📋 Enter Patient Details")

with st.expander("ℹ️ What do these inputs mean?"):
    st.markdown("""
    - **Age at Reporting**: Patient's age when last evaluated  
    - **Baseline CD4 Result**: Immune system strength at ART start  
    - **Time on ART (Months)**: Duration on treatment  
    - **Maternal Status**: Whether the patient is pregnant or breastfeeding  
    """)

age = st.number_input("👤 Age at Reporting", min_value=0, max_value=100, value=30)
cd4 = st.number_input("🧪 Baseline CD4 Result", min_value=0, max_value=2000, value=300)
months_on_art = st.number_input("📅 Time on ART (Months)", min_value=0.0, max_value=240.0, value=12.0)
maternal_status = st.selectbox("🤰 Maternal Status", ["Pregnant", "Breastfeeding", "Neither"])

maternal_flag = 1 if maternal_status in ["Pregnant", "Breastfeeding"] else 0

# Prediction
if st.button("🚀 Predict Suppression Status"):
    input_data = pd.DataFrame({
        'Age At Reporting': [age],
        'Baseline Cd4 Result': [cd4],
        'Time On ART (Months)': [months_on_art]
    })

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][prediction]

    status = "🟢 Suppressed (Low Risk)" if prediction == 1 else "🔴 Unsuppressed (High Risk)"
    color = "#2ECC71" if prediction == 1 else "#E74C3C"

    st.markdown(f"<h3 style='color:{color};'>Prediction: {status}</h3>", unsafe_allow_html=True)
    st.markdown(f"Confidence Score: **{prob:.2f}**")

    # Visual confidence gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Confidence (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': color},
               'steps': [
                   {'range': [0, 50], 'color': "#FADBD8"},
                   {'range': [50, 80], 'color': "#F9E79F"},
                   {'range': [80, 100], 'color': "#D5F5E3"}]}
    ))
    st.plotly_chart(fig)

    with st.expander("🧠 What does this mean?"):
        if prediction == 1:
            st.markdown("""
            The patient is likely **virally suppressed**, which means their risk of transmitting HIV is **very low**.  
            Continue monitoring and supporting adherence to treatment.
            """)
        else:
            st.markdown("""
            The patient is likely **not virally suppressed**, which may indicate **higher transmission risk**.  
            Consider clinical review, adherence support, and maternal care if applicable.
            """)

# Maternal filter reminder
if maternal_filter != "All":
    st.info(f"Dashboard is currently filtered to show patients who are **{'maternal' if maternal_filter == 'Pregnant/Breastfeeding' else 'not maternal'}**.")

# Footer
st.markdown("---")
st.markdown("🔐 This tool does not store any patient data. Predictions are based on anonymized clinical features.")
