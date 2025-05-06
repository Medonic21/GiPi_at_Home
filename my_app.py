import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open('trained_model.sav', 'rb') as f:
    model = pickle.load(f)

# Load dataset used for predictions
df = pd.read_csv('Dataset.csv')  # Must include Make, Model, Year, and all model input features

st.title("Car Deal Evaluator 🚗")
st.markdown("Select your car to evaluate whether it's a good deal.")

# --- Dropdown Selections ---
make_options = sorted(df['Make'].unique())
selected_make = st.selectbox("Select Make", make_options)

# Filter models based on selected make
filtered_models = df[df['Make'] == selected_make]['Model'].unique()
selected_model = st.selectbox("Select Model", sorted(filtered_models))

# Filter years based on selected make and model
filtered_years = df[(df['Make'] == selected_make) & (df['Model'] == selected_model)]['Year'].unique()
selected_year = st.selectbox("Select Year", sorted(filtered_years, reverse=True))

# --- Matching the Row ---
matched_rows = df[
    (df['Make'] == selected_make) &
    (df['Model'] == selected_model) &
    (df['Year'] == selected_year)
]

if not matched_rows.empty:
    car_features = matched_rows.iloc[0]  # Take the first match if duplicates exist
    st.subheader("Auto-filled Car Attributes:")
    st.write(car_features)

    # Prepare input_dict based on the model’s features
    input_dict = {}
    for col in model.feature_names_in_:
        input_dict[col] = car_features.get(col, 0)

    input_df = pd.DataFrame([input_dict])

    # --- Prediction ---
    if st.button("Check Deal"):
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.success("✅ This is a GOOD deal!")
        else:
            st.warning("❌ This might NOT be a good deal.")
else:
    st.warning("No matching car found in the dataset.")