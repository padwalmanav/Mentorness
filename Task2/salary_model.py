import streamlit as st
import pandas as pd
import pickle
import json  # Import JSON to load the saved encodings

# Load the trained model
pickle_in = open("salary_model.pkl", "rb")
model = pickle.load(pickle_in)

# Load the label encodings from the JSON file
with open("label_encodings.json", "r") as f:
    mappings = json.load(f)

# Streamlit UI
st.title("Salary Prediction Web Application")
st.write("Enter the required information to predict the salary.")

# Extract categories from the loaded mappings
designation_categories = list(mappings["designation"].keys())
unit_categories = list(mappings["unit"].keys())

# Input fields for each feature in the dataset
designation = st.selectbox("Designation", designation_categories)
age = st.number_input("Age(18-65)", min_value=18, max_value=65, step=1)
unit = st.selectbox("Unit", unit_categories)
ratings = st.slider("Ratings", min_value=1.0, max_value=5.0, step=0.5)
past_exp = st.number_input("Past Experience (in years)(0-40)", min_value=0, max_value=40, step=1)

# Get the encoded values from the user input using the loaded mappings
encoded_designation = mappings["designation"][designation]
encoded_unit = mappings["unit"][unit]

# Create a DataFrame for model input
input_data = pd.DataFrame({
    "DESIGNATION": [encoded_designation],
    "AGE": [age],
    "UNIT": [encoded_unit],
    "RATINGS": [ratings],
    "PAST EXP": [past_exp]
})

# Button to predict
if st.button("Predict Salary"):
    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        st.success(f"The predicted salary is: Rs.{prediction:.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
