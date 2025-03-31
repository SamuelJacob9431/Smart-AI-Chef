import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# List of allowed food items (adjust as needed)
allowed_food_items = [
    'apple', 'banana', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake'
]

# Read the inventory log CSV file
# Expected CSV columns: Timestamp, Label, Count
df = pd.read_csv("inventory_log.csv")
# Convert the Timestamp column to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Loop over each allowed food item and forecast its demand
for item in allowed_food_items:
    print(f"Processing forecast for {item}...")

    # Filter rows for the current food item
    item_data = df[df['Label'] == item].copy()
    
    # Skip this item if no data is found
    if item_data.empty:
        print(f"  No data found for {item}, skipping forecast.")
        continue

    # Aggregate counts by day:
    daily = item_data.set_index('Timestamp').resample('D').sum().reset_index()
    # Rename columns to fit Prophet's requirements: 'ds' for dates and 'y' for values
    daily = daily.rename(columns={'Timestamp': 'ds', 'Count': 'y'})
    # Ensure every day is represented by filling missing days with 0 count
    daily = daily.set_index('ds').asfreq('D', fill_value=0).reset_index()

    # Check if there is enough data (at least 2 rows) to train Prophet
    if daily.shape[0] < 2 or daily['y'].dropna().shape[0] < 2:
        print(f"  Not enough data for {item} to forecast, skipping.")
        continue

    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(daily)

    # Create a DataFrame for the next 7 days
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    # Print forecast summary (last 5 rows)
    print(f"  Forecast for {item}:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Plot the forecast
    fig = model.plot(forecast)
    plt.title(f"{item.capitalize()} Demand Forecast")
    plt.xlabel("Date")
    plt.ylabel("Count")
    # Save the plot to a PNG file (e.g., apple_forecast.png)
    plt.savefig(f"{item}_forecast.png")
    plt.show()
    plt.close(fig)
