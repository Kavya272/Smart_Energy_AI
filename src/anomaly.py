import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/electricity_data.csv")

# Prepare feature
X = df[['units']]

# Train model
model = IsolationForest(contamination=0.15, random_state=42)
df['anomaly'] = model.fit_predict(X)

# Mark anomalies
anomalies = df[df['anomaly'] == -1]

print("Anomalies detected:\n")
print(anomalies)

# Plot
plt.figure(figsize=(8,5))
plt.plot(df['units'], label="Normal Usage")
plt.scatter(anomalies.index, anomalies['units'], color='red', label='Anomaly')
plt.title("Electricity Usage with Anomalies")
plt.xlabel("Time Index")
plt.ylabel("Units")
plt.legend()
plt.grid()

plt.savefig("anomaly.png")
plt.show()