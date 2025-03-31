import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# List of allowed food items (adjust as needed)
allowed_food_items = [
    'apple', 'banana', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake'
]

# Read the waste log CSV file.
# It should have at least these columns: Timestamp, Label, Waste
df = pd.read_csv("waste_log.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Dictionary to store forecasts for each food item
forecasts = {}

# Forecast settings: using minute-level aggregation and forecasting next 10 minutes.
resample_freq = 'T'   # 'T' stands for minute-level frequency.
forecast_periods = 10  # Forecast horizon: next 10 minutes

for item in allowed_food_items:
    # Filter waste data for the current item
    item_data = df[df['Label'] == item].copy()
    
    if item_data.empty:
        print(f"No waste data found for {item}, skipping forecast.")
        continue

    # Aggregate waste by minute
    aggregated = item_data.set_index('Timestamp').resample(resample_freq).sum().reset_index()
    # Rename columns for Prophet: 'ds' for dates and 'y' for the waste value
    aggregated = aggregated.rename(columns={'Timestamp': 'ds', 'Waste': 'y'})
    # Fill in missing minutes with zero waste
    aggregated = aggregated.set_index('ds').asfreq(resample_freq, fill_value=0).reset_index()
    
    # Check if there are at least 2 data points (minimum for Prophet)
    if aggregated.shape[0] < 2 or aggregated['y'].dropna().shape[0] < 2:
        print(f"Not enough data for {item} to forecast waste, skipping.")
        continue

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(aggregated)
    
    # Create a DataFrame for future predictions (next 10 minutes)
    future = model.make_future_dataframe(periods=forecast_periods, freq=resample_freq)
    forecast = model.predict(future)
    
    forecasts[item] = forecast
    print(f"Forecast computed for waste of {item}.")

# Plot all forecasts in one combined graph
plt.figure(figsize=(10, 6))
for item, forecast in forecasts.items():
    plt.plot(forecast['ds'], forecast['yhat'], label=item)

plt.xlabel("Time")
plt.ylabel("Waste")
plt.title("Forecasted Waste for Food Items")
plt.legend()
plt.tight_layout()
plt.savefig("combined_waste_forecast.png")
plt.show()
