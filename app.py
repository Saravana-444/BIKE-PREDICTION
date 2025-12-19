import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Bike Recommendation System",
    page_icon="🏍️",
    layout="centered"
)

st.title("🏍️ Bike Recommendation System")
st.write("Predict the **best bikes** based on your **budget & preferences**")

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_models():
    df = joblib.load("bike_data.pkl")
    scaler = joblib.load("scaler.pkl")
    return df, scaler

df, scaler = load_models()

# ------------------ USER INPUT ------------------
st.sidebar.header("🔧 Enter Your Preferences")

budget = st.sidebar.number_input(
    "💰 Budget (₹)",
    min_value=30000,
    max_value=500000,
    value=120000,
    step=5000
)

displacement = st.sidebar.number_input(
    "⚙️ Engine Capacity (cc)",
    min_value=90,
    max_value=1000,
    value=150
)

city_mileage = st.sidebar.number_input(
    "⛽ City Mileage (km/l)",
    min_value=10,
    max_value=80,
    value=45
)

highway_mileage = st.sidebar.number_input(
    "🛣️ Highway Mileage (km/l)",
    min_value=10,
    max_value=90,
    value=55
)

rating = st.sidebar.slider(
    "⭐ Minimum Rating",
    min_value=1.0,
    max_value=5.0,
    value=4.0,
    step=0.1
)

# ------------------ RECOMMENDATION LOGIC ------------------
def recommend_bikes(user_input, top_n=5):
    feature_cols = [
        'price',
        'displacement_cc',
        'city_mileage',
        'highway_mileage',
        'average_stars'
    ]

    data_features = df[feature_cols]
    scaled_data = scaler.transform(data_features)

    user_scaled = scaler.transform(user_input)
    similarity = cosine_similarity(user_scaled, scaled_data)

    top_indices = similarity[0].argsort()[-top_n:][::-1]
    return df.iloc[top_indices][
        ['bike', 'price', 'displacement_cc',
         'city_mileage', 'highway_mileage', 'average_stars']
    ]

# ------------------ BUTTON ------------------
if st.button("🔍 Recommend Bikes"):
    user_input = np.array([[
        budget,
        displacement,
        city_mileage,
        highway_mileage,
        rating
    ]])

    recommendations = recommend_bikes(user_input)

    st.success("✅ Top Bike Recommendations")
    st.dataframe(recommendations, use_container_width=True)

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("📌 Built using Machine Learning | KMeans + Cosine Similarity")
