import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np

# List of allowed food items (adjust as needed)
allowed_food_items = [
    'apple', 'banana', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake'
]

# Read the inventory log CSV file (must have columns: Timestamp, Label, Count)
df = pd.read_csv("inventory_log.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Dictionary to store forecasts for each food item
forecasts = {}

# Forecast settings
# Using minute-level aggregation since data is short; adjust period as needed.
resample_freq = 'T'   # 'T' = minute-level; change to 'H' or 'D' if you have more data.
forecast_periods = 10  # Forecast next 10 minutes

for item in allowed_food_items:
    # Filter data for the current item
    item_data = df[df['Label'] == item].copy()
    
    if item_data.empty:
        print(f"No data found for {item}, skipping forecast.")
        continue
    
    # Aggregate counts by minute
    aggregated = item_data.set_index('Timestamp').resample(resample_freq).sum().reset_index()
    # Rename columns for Prophet: 'ds' for dates, 'y' for value
    aggregated = aggregated.rename(columns={'Timestamp': 'ds', 'Count': 'y'})
    # Fill missing minutes with zeros
    aggregated = aggregated.set_index('ds').asfreq(resample_freq, fill_value=0).reset_index()
    
    # Check if we have at least 2 data points
    if aggregated.shape[0] < 2 or aggregated['y'].dropna().shape[0] < 2:
        print(f"Not enough data for {item} to forecast, skipping.")
        continue
    
    # Initialize and fit Prophet model
    model = Prophet()
    model.fit(aggregated)
    
    # Create future DataFrame for forecasting
    future = model.make_future_dataframe(periods=forecast_periods, freq=resample_freq)
    forecast = model.predict(future)
    
    forecasts[item] = forecast
    print(f"Forecast computed for {item}.")

# Plot all forecasts in one graph
plt.figure(figsize=(10, 6))
for item, forecast in forecasts.items():
    # Plot the forecasted yhat for each item
    plt.plot(forecast['ds'], forecast['yhat'], label=item)

plt.xlabel("Time")
plt.ylabel("Count")
plt.title("Forecasted Demand for Food Items")
plt.legend()
plt.tight_layout()
plt.savefig("combined_forecast.png")
plt.show()
