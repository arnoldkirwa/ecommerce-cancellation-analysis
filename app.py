
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from datetime import date

st.set_page_config(
    page_title="E-Commerce Cancellation Risk",
    page_icon="🛒",
    layout="wide"
)

# Load the trained pipeline
MODEL_PATH = Path(__file__).parent / "cancellation_risk_model.joblib"
model = joblib.load(MODEL_PATH)

st.title("🛒 E-Commerce Order Cancellation Risk Analysis")
st.write(
    "Explore order characteristics using a machine learning model "
    "trained on the Brazilian Olist e-commerce dataset."
)

st.info(
    "Portfolio demonstration only. The model has low cancellation "
    "precision and is not suitable for operational decisions."
)

st.subheader("Model Performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Average Precision", "0.0154")
c2.metric("Cancellation Recall", "67.39%")
c3.metric("Cancellation Precision", "1.14%")
c4.metric("F1-score", "0.0224")

st.caption(
    "Held-out test results. Only 1.14% of orders flagged as canceled "
    "were actually canceled in the evaluation."
)

st.divider()
st.header("Try an Order Risk Estimate")
st.write(
    "Enter hypothetical order details below. These inputs are for "
    "demonstration; predictions are experimental."
)

with st.form("order_form"):
    col1, col2 = st.columns(2)

    with col1:
        purchase_date = st.date_input(
            "Purchase date",
            value=date(2026, 1, 15)
        )
        purchase_hour = st.slider("Purchase hour", 0, 23, 14)
        total_order_items = st.number_input(
            "Number of order items", min_value=1, max_value=100, value=1
        )
        total_item_price = st.number_input(
            "Total item price", min_value=0.0, value=100.0, step=10.0
        )
        total_freight_value = st.number_input(
            "Total freight value", min_value=0.0, value=20.0, step=5.0
        )
        primary_product_category = st.text_input(
            "Primary product category", value="beleza_saude"
        )

    with col2:
        average_item_price = st.number_input(
            "Average item price", min_value=0.0, value=100.0, step=10.0
        )
        distinct_product_categories = st.number_input(
            "Distinct product categories", min_value=1, max_value=100, value=1
        )
        total_distinct_products = st.number_input(
            "Distinct products", min_value=1, max_value=100, value=1
        )

    submitted = st.form_submit_button("Estimate cancellation risk")

if submitted:
    purchase_day_of_week = purchase_date.weekday()

    input_data = pd.DataFrame([{
        "purchase_year": purchase_date.year,
        "purchase_month": purchase_date.month,
        "purchase_day_of_week": purchase_day_of_week,
        "purchase_hour": purchase_hour,
        "is_weekend": int(purchase_day_of_week >= 5),
        "total_order_items": total_order_items,
        "total_item_price": total_item_price,
        "total_freight_value": total_freight_value,
        "average_item_price": average_item_price,
        "distinct_product_categories": distinct_product_categories,
        "total_distinct_products": total_distinct_products,
        "primary_product_category": primary_product_category
    }])

    try:
        probabilities = model.predict_proba(input_data)
        cancellation_probability = float(probabilities[0][1])

        st.subheader("Demo Result")
        st.metric(
            "Estimated cancellation probability",
            f"{cancellation_probability:.2%}"
        )
        st.progress(cancellation_probability)

        st.warning(
            "This is an experimental model score, not a reliable "
            "prediction of what will happen to an individual order. "
            "Its low precision and class imbalance limit practical use."
        )
    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
st.subheader("About This Project")
st.write(
    "This project follows CRISP-DM to investigate order cancellations, "
    "engineer features, compare classification models, and evaluate "
    "performance on an extremely imbalanced dataset."
)
st.write("**Current model:** Expanded Gradient Boosting")
st.write("**Dataset:** Brazilian E-Commerce Public Dataset by Olist")
st.write("**Purpose:** Portfolio demonstration, not production use.")
