import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data
df = pd.read_csv("data/electricity_data.csv")

data = df['units'].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences
X = []
y = []

window = 3  # using past 3 months

for i in range(window, len(data_scaled)):
    X.append(data_scaled[i-window:i])
    y.append(data_scaled[i])

X = np.array(X)
y = np.array(y)

# Build model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X.shape[1], 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X, y, epochs=100, verbose=0)

# Predict next value
last_input = data_scaled[-window:]
last_input = last_input.reshape(1, window, 1)

pred = model.predict(last_input)

predicted_value = scaler.inverse_transform(pred)

print("Next Month Prediction (LSTM):", predicted_value[0][0])