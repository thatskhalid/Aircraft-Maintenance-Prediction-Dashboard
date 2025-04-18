import streamlit as st
import pandas as pd
import pickle

# Load model and data
@st.cache_data
def load_model():
    with open('models/trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    return pd.read_csv('data/maintenance_data.csv')

model = load_model()
data = load_data()

# Streamlit app
st.title("Aircraft Maintenance Prediction Dashboard")

st.subheader("Input Aircraft Details")

# Example inputs
engine_hours = st.number_input("Engine Hours", min_value=0)
aircraft_age = st.number_input("Aircraft Age (Years)", min_value=0)
recent_failures = st.number_input("Number of Failures in Last Year", min_value=0)

if st.button("Predict Maintenance Risk"):
    features = [[engine_hours, aircraft_age, recent_failures]]
    prediction = model.predict(features)
    st.write(f"Predicted Maintenance Risk Level: {prediction[0]}")
