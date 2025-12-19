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
def recommend_bikes(user_input):
    # --- Ensure 2D DataFrame with proper columns ---
    columns = ['Brand','Model','Year','Engine','Price']  # Replace with your actual dataset columns
    
    # If user_input is a dict or list
    if isinstance(user_input, dict):
        user_df = pd.DataFrame([user_input])
    elif isinstance(user_input, list):
        user_df = pd.DataFrame([user_input], columns=columns)
    else:
        raise ValueError("user_input must be a dict or list")

    # --- Scale features ---
    user_scaled = scaler.transform(user_df)

    # --- Replace NaN/Inf to avoid cosine_similarity error ---
    import numpy as np
    user_scaled = np.nan_to_num(user_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    scaled_data_clean = np.nan_to_num(scaled_data, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Ensure 2D shape ---
    if user_scaled.ndim == 1:
        user_scaled = user_scaled.reshape(1, -1)

    # --- Compute similarity ---
    similarity = cosine_similarity(user_scaled, scaled_data_clean)

    # --- Get top recommendations ---
    top_indices = similarity.argsort()[0][-5:][::-1]
    return bike_data.iloc[top_indices]


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend_bikes(user_input):
    # Preprocess and encode user input
    user_df = pd.DataFrame([user_input])
    user_scaled = scaler.transform(user_df)

    # Handle NaN or Inf
    user_scaled = np.nan_to_num(user_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    scaled_data_clean = np.nan_to_num(scaled_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure 2D shape
    if user_scaled.ndim == 1:
        user_scaled = user_scaled.reshape(1, -1)

    # Compute similarity safely
    similarity = cosine_similarity(user_scaled, scaled_data_clean)

    # Get top recommendations
    top_indices = similarity.argsort()[0][-5:][::-1]
    return bike_data.iloc[top_indices]

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
