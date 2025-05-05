import streamlit as st
import pandas as pd
import pickle

# Load the trained model from .sav
with open('trained_model.sav', 'rb') as f:
    model = pickle.load(f)

st.title("Car Deal Evaluator 🚗")

st.markdown("Fill in the car details below to see if it's a good deal for you.")

# --- User Inputs ---
engine_hp = st.number_input("Engine HP", min_value=50, max_value=1000, value=200)
displacement = st.number_input("Engine Displacement (L)", min_value=1.0, max_value=10.0, value=2.5)
mpg = st.number_input("Combined MPG", min_value=5, max_value=100, value=25)
msrp = st.number_input("MSRP ($)", min_value=5000, max_value=300000, value=35000)

vehicle_size = st.selectbox("Vehicle Size", ['Compact', 'Midsize', 'Large'])
vehicle_style = st.selectbox("Vehicle Style", ['Sedan', 'SUV', 'Coupe', 'Convertible', 'Hatchback', 'Wagon'])

# --- Input Preparation ---
# One-hot encoding simulation (based on training encoding logic)
input_dict = {
    'Engine HP': engine_hp,
    'displacement': displacement,
    'combination_mpg': mpg,
    'MSRP': msrp,
    'Number of Doors': 4,  # Default assumption, or make it user input
}

# Add encoded columns for Vehicle Size
for size in ['Midsize', 'Large']:
    input_dict[f'Vehicle Size_{size}'] = 1 if vehicle_size == size else 0

# Add encoded columns for Vehicle Style
styles = ['Coupe', 'Convertible', 'Hatchback', 'SUV', 'Wagon']
for style in styles:
    input_dict[f'Vehicle Style_{style}'] = 1 if vehicle_style == style else 0

# Make sure all expected columns are present
input_df = pd.DataFrame([input_dict])
for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model.feature_names_in_]

# --- Prediction ---
if st.button("Check Deal"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ This is a GOOD deal!")
    else:
        st.warning("❌ This might NOT be a good deal.")