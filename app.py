import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Energy AI", layout="wide")

st.title("⚡ AI Smart Energy Management System")

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/electricity_data.csv")
df['date'] = pd.to_datetime(df['date'])

# -----------------------------
# Cost Function
# -----------------------------
def calculate_bill(units):
    if units <= 100:
        return units * 2
    elif units <= 200:
        return units * 3
    elif units <= 300:
        return units * 5
    else:
        return units * 7

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "🔮 Prediction",
    "🚨 Anomaly",
    "💡 Insights",
    "⚙️ Simulator"
])

# =============================
# 📊 TAB 1: DASHBOARD (Premium)
# =============================
with tab1:
    st.subheader("📊 Overview Dashboard")

    latest_units = df['units'].iloc[-1]
    avg_units = int(df['units'].mean())
    latest_bill = calculate_bill(latest_units)

    col1, col2, col3 = st.columns(3)

    col1.metric("⚡ Current Usage", f"{latest_units} units")
    col2.metric("📊 Avg Usage", f"{avg_units} units")
    col3.metric("💰 Current Bill", f"₹{latest_bill}")

    st.markdown("---")

    st.subheader("📈 Usage Trend")
    st.line_chart(df.set_index('date')['units'])

# =============================
# 🔮 TAB 2: Prediction
# =============================
with tab2:
    st.subheader("🔮 Future Prediction")

    df_prophet = df.rename(columns={"date": "ds", "units": "y"})

    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=3, freq='ME')
    forecast = model.predict(future)

    forecast['bill'] = forecast['yhat'].apply(calculate_bill)

    col1, col2 = st.columns(2)

    with col1:
        st.write("### 📊 Predicted Units")
        st.write(forecast[['ds', 'yhat']].tail())

    with col2:
        st.write("### 💰 Predicted Bill")
        st.write(forecast[['ds', 'bill']].tail())

    st.markdown("---")

    fig1 = model.plot(forecast)
    st.pyplot(fig1)

# =============================
# 🚨 TAB 3: Anomaly
# =============================
with tab3:
    st.subheader("🚨 Anomaly Detection")

    model_anomaly = IsolationForest(contamination=0.15, random_state=42)
    df['anomaly'] = model_anomaly.fit_predict(df[['units']])

    anomalies = df[df['anomaly'] == -1]

    fig, ax = plt.subplots()
    ax.plot(df['date'], df['units'])
    ax.scatter(anomalies['date'], anomalies['units'], color='red')
    ax.set_title("Usage vs Anomalies")

    st.pyplot(fig)

    if not anomalies.empty:
        st.error("🚨 High anomaly detected! Sudden spike in usage.")
    else:
        st.success("✅ No abnormal usage detected")

# =============================
# 💡 TAB 4: Insights
# =============================
with tab4:
    st.subheader("💡 Smart Insights")

    df['prev_units'] = df['units'].shift(1)

    def suggest(row):
        if row['anomaly'] == -1:
            return "🚨 Sudden spike - Check appliances"
        elif row['units'] > 350:
            return "⚠️ High usage - Reduce AC usage"
        elif pd.notna(row['prev_units']) and row['units'] > row['prev_units']:
            return "📈 Increasing trend - Monitor usage"
        else:
            return "✅ Stable usage"

    df['suggestion'] = df.apply(suggest, axis=1)

    st.dataframe(df[['date', 'units', 'suggestion']], use_container_width=True)

# =============================
# ⚙️ TAB 5: Simulator
# =============================
with tab5:
    st.subheader("⚙️ Smart Simulation")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### 🔘 Usage Simulation")
        ac_hours = st.slider("AC usage (hours/day)", 0, 24, 5)

        extra_usage = ac_hours * 10
        predicted_units = df['units'].mean() + extra_usage

        st.metric("Estimated Units", int(predicted_units))
        st.metric("Estimated Bill", f"₹{int(calculate_bill(predicted_units))}")

    with col2:
        st.write("### 📉 Reduction Simulator")

        reduce_units = st.slider("Reduce units", 0, 200, 50)

        new_units = latest_units - reduce_units

        st.metric("New Units", new_units)
        st.metric("New Bill", f"₹{calculate_bill(new_units)}")

    st.markdown("---")

    st.subheader("🏠 Appliance Breakdown")

    ac = latest_units * 0.5
    fan = latest_units * 0.3
    lights = latest_units * 0.2

    st.write(f"AC: {int(ac)} units | Fan: {int(fan)} | Lights: {int(lights)}")