import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/electricity_data.csv")

df['date'] = pd.to_datetime(df['date'])

plt.figure(figsize=(8,5))
plt.plot(df['date'], df['units'], marker='o')
plt.title("Electricity Usage Over Time")
plt.xlabel("Date")
plt.ylabel("Units")
plt.grid()

plt.savefig("output.png")  # 👈 ADD THIS
plt.show()