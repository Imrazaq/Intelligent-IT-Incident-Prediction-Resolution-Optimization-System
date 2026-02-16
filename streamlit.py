import streamlit as st
import joblib

st.title("IT Incident Risk Predictor")

description = st.text_area("Enter Ticket Description")

if st.button("Predict"):
    vectorized = vectorizer.transform([description])
    prediction = model.predict(vectorized)

    if prediction[0] == 1:
        st.error("⚠ High Risk SLA Breach")
    else:
        st.success("Low Risk Ticket")

streamlit run app.py
