import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load trained model
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("🏡 Advanced House Price Prediction App")
st.write("Enter details below to get an estimated house price.")

# Sidebar inputs
st.sidebar.header("Input Features")
sqft = st.sidebar.slider("Square Feet", min_value=500, max_value=5000, step=100)
bedrooms = st.sidebar.slider("Bedrooms", min_value=1, max_value=5, step=1)
bathrooms = st.sidebar.slider("Bathrooms", min_value=1, max_value=3, step=1)
age = st.sidebar.slider("House Age", min_value=0, max_value=100, step=1)
location = st.sidebar.selectbox("Location", ["Downtown", "Suburb", "Countryside"])

# Process input
input_data = pd.DataFrame([[sqft, bedrooms, bathrooms, age, location]],
                          columns=["sqft", "bedrooms", "bathrooms", "age", "location"])
input_data[["sqft", "bedrooms", "bathrooms", "age"]] = scaler.transform(input_data[["sqft", "bedrooms", "bathrooms", "age"]])
input_data["location"] = label_encoders["location"].transform(input_data["location"])

# Predict price
predicted_price = np.exp(model.predict(input_data))[0]

st.subheader("Predicted House Price: 💰")
st.write(f"**${predicted_price:,.2f}**")

# EDA Section
st.subheader("Exploratory Data Analysis")
if st.checkbox("Show Dataset Summary"):
    df = pd.read_csv("house_data.csv")
    st.write(df.describe())

# Correlation Heatmap
if st.checkbox("Show Correlation Heatmap"):
    df = pd.read_csv("house_data.csv")
    corr = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Feature Importance
if st.checkbox("Show Feature Importance"):
    feature_importance = pd.Series(model.feature_importances_, index=["sqft", "bedrooms", "bathrooms", "age", "location"])
    fig, ax = plt.subplots()
    feature_importance.sort_values().plot(kind='barh', ax=ax)
    st.pyplot(fig)

# Download Predictions
st.subheader("Download Predictions")
if st.button("Download CSV"):
    prediction_df = pd.DataFrame([[sqft, bedrooms, bathrooms, age, location, predicted_price]],
                                 columns=["sqft", "bedrooms", "bathrooms", "age", "location", "Predicted Price"])
    prediction_df.to_csv("prediction.csv", index=False)
    st.success("Download Complete!")
