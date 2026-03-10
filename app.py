import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Customer Product Prediction")

# Input fields
age = st.number_input("Age", 1, 100, 25)
gender = st.selectbox("Gender", [0, 1])
income = st.number_input("Income", 0, 100000, 50000)
spending = st.number_input("Spending Score", 0, 100, 50)
browsing = st.number_input("Browsing Frequency", 0, 100, 10)
purchases = st.number_input("Number of Purchases", 0, 100, 5)
last_category = st.number_input("Last Category", 0, 4, 0)

product_map = {0:"Electronics", 1:"Fashion", 2:"Groceries", 3:"Home Appliances", 4:"Books"}

if st.button("Predict Product"):
    features = [age, gender, income, spending, browsing, purchases, last_category]
    scaled_features = scaler.transform([features])
    pred_code = model.predict(scaled_features)[0]
    st.success(f"Predicted Product Category: {product_map[pred_code]}")
