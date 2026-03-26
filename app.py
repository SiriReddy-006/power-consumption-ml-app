import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

# ---------------- TITLE ----------------
st.title("⚡ Power Consumption Prediction App")
# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("📊 Dataset Preview")
        st.write(data.head())

        # ---------------- DATETIME PROCESSING ----------------
        if 'Datetime' in data.columns:
            data['Datetime'] = pd.to_datetime(data['Datetime'])

            data['hour'] = data['Datetime'].dt.hour
            data['day'] = data['Datetime'].dt.day
            data['month'] = data['Datetime'].dt.month

            data.drop('Datetime', axis=1, inplace=True)
        else:
            st.error("❌ Datetime column not found")
            st.stop()

        # ---------------- OUTLIER CAPPING ----------------
        numeric_data = data.select_dtypes(include=['number'])

        Q1 = numeric_data.quantile(0.25)
        Q3 = numeric_data.quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        data[numeric_data.columns] = numeric_data.clip(lower, upper, axis=1)


        # ---------------- FEATURES ----------------
        X = data.drop('PowerConsumption_Zone1', axis=1)
        y = data['PowerConsumption_Zone1']

        # ---------------- SPLIT ----------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ---------------- SCALING ----------------
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ---------------- MODEL SELECTION ----------------
        model_option = st.selectbox("Choose Model", ["Linear Regression", "KNN"])

        if model_option == "Linear Regression":
            model = LinearRegression()
        else:
            model = KNeighborsRegressor(n_neighbors=5)

        # ---------------- TRAIN ----------------
        model.fit(X_train, y_train)

        # ---------------- PREDICT ----------------
        y_pred = model.predict(X_test)

        # ---------------- EVALUATION ----------------
        r2 = r2_score(y_test, y_pred)

        st.subheader("📈 Model Performance")
        st.write("R2 Score:", r2)
        st.write("Accuracy:", round(r2 * 100, 2), "%")

        # ---------------- GRAPH ----------------
        st.subheader("📉 Power Consumption Trend")
        st.line_chart(data['PowerConsumption_Zone1'])

        # ---------------- USER INPUT ----------------
        st.subheader("🔮 Predict New Value")

        temp = st.number_input("Temperature")
        humidity = st.number_input("Humidity")
        wind = st.number_input("WindSpeed")
        gdf = st.number_input("GeneralDiffuseFlows")
        df = st.number_input("DiffuseFlows")
        zone2 = st.number_input("Zone2 Consumption")
        zone3 = st.number_input("Zone3 Consumption")
        hour = st.number_input("Hour")
        day = st.number_input("Day")
        month = st.number_input("Month")

        if st.button("Predict"):
            input_data = np.array([[temp, humidity, wind, gdf, df,
                                    zone2, zone3, hour, day, month]])

            input_scaled = scaler.transform(input_data)

            prediction = model.predict(input_scaled)

            st.success(f"⚡ Predicted Power Consumption: {prediction[0]:.2f}")

    except Exception as e:
        st.error(f"🚨 Error: {e}")

else:
    st.warning("⚠️ Please upload a CSV file")