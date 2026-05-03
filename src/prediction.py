import pandas as pd
from prophet import Prophet

# Load data
df = pd.read_csv("data/electricity_data.csv")

# Convert columns for Prophet
df = df.rename(columns={
    "date": "ds",
    "units": "y"
})

df['ds'] = pd.to_datetime(df['ds'])

# Create model
model = Prophet()

# Train model
model.fit(df)

# Create future dates (next 3 months)
future = model.make_future_dataframe(periods=3, freq='ME')

# Predict
forecast = model.predict(future)

# Show predictions
print(forecast[['ds', 'yhat']].tail())