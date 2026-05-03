import pandas as pd
from sklearn.ensemble import IsolationForest

# Load data
df = pd.read_csv("data/electricity_data.csv")

# ---------------------------
# STEP 1: Add anomaly detection
# ---------------------------
model = IsolationForest(contamination=0.15, random_state=42)
df['anomaly'] = model.fit_predict(df[['units']])

# ---------------------------
# STEP 2: Add previous values
# ---------------------------
df['prev_units'] = df['units'].shift(1)

# ---------------------------
# STEP 3: Recommendation function
# ---------------------------
def suggest(row):
    units = row['units']
    prev = row['prev_units']
    anomaly = row['anomaly']

    # 🔴 ADD HERE (FIRST PRIORITY)
    if anomaly == -1:
        return "Unusual spike 🚨: Check appliances immediately"

    elif units > 350:
        return "High usage ⚠️: Reduce AC usage"

    elif pd.notna(prev) and units > prev:
        return "Usage increasing 📈: Monitor appliances"

    else:
        return "Usage stable ✅"

# Apply function
df['suggestion'] = df.apply(suggest, axis=1)

# Show results
print(df[['date', 'units', 'anomaly', 'suggestion']])