# streamlit_app.py
"""Streamlit application to serve the XGBoost sales prediction model.

Prerequisites (add to requirements.txt):
    streamlit
    pandas
    scikit-learn
    joblib
    xgboost

Place this file in the root of your GitHub repository alongside:
    - xgboost_sales_model.pkl
    - preprocessor.pkl

Run locally with:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# Load the preprocessor and trained model
@st.cache_resource
def load_assets():
    preprocessor = joblib.load("preprocessor.pkl")
    model = joblib.load("xgboost_sales_model.pkl")
    return preprocessor, model

preprocessor, model = load_assets()

st.title("🛍️ Sales Prediction ML System Built By Stanley")
st.write("Enter the details of a product sale and the model will estimate the future sales amount.")

# ------- User Input Section -------
col1, col2 = st.columns(2)

with col1:
    # Date input – we will extract month, day of week, week of year, year
    date_input = st.date_input("Date", datetime.today())
    month = date_input.month
    day_of_week = date_input.weekday()  # Monday=0
    week_of_year = int(date_input.strftime("%U"))
    year = date_input.year

    product_category = st.selectbox(
        "Product Category",
        options=["Electronics", "Furniture", "Accessories", "Sports", "Home Appliances"],
    )
    price = st.number_input("Price", min_value=0.0, value=500.0, step=0.01)
    promotion = st.checkbox("Promotion (Yes/No)")
    discount_rate = st.slider("Discount Rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    quantity = st.number_input("Quantity Sold", min_value=0, value=1, step=1)

with col2:
    previous_day_sales = st.number_input("Previous Day Sales", min_value=0.0, value=0.0, step=0.01)
    previous_week_sales = st.number_input("Previous Week Sales", min_value=0.0, value=0.0, step=0.01)
    rolling_7day_sales = st.number_input("Rolling 7‑Day Sales", min_value=0.0, value=0.0, step=0.01)

# Build input DataFrame matching training schema
input_dict = {
    "ProductID": ["P001"],  # placeholder – model may ignore it after encoding
    "ProductName": ["Placeholder"],  # placeholder
    "ProductCategory": [product_category],
    "Price": [price],
    "Promotion": ["Yes" if promotion else "No"],
    "DiscountRate": [discount_rate],
    "QuantitySold": [quantity],
    "PreviousDaySales": [previous_day_sales],
    "PreviousWeekSales": [previous_week_sales],
    "Rolling7DaySales": [rolling_7day_sales],
    "Month": [month],
    "DayOfWeek": [day_of_week],
    "WeekOfYear": [week_of_year],
    "Year": [year],
}

input_df = pd.DataFrame(input_dict)

# Add derived feature if you used it during training (FinalPrice)
if "DiscountRate" in input_df.columns and "Price" in input_df.columns:
    input_df["FinalPrice"] = input_df["Price"] * (1 - input_df["DiscountRate"])

# Prediction button
if st.button("Predict Sales"):
    # Transform with the same preprocessor used during training
    X_processed = preprocessor.transform(input_df)
    # XGBoost model expects a NumPy array
    pred = model.predict(X_processed)
    # Model likely predicts total sales amount (float)
    st.success(f"🔮 Estimated Sales: ${pred[0]:,.2f}")

st.caption("*Ensure the column names and encoding match those used during training. The placeholders for ProductID and ProductName are kept to satisfy the original schema.")
