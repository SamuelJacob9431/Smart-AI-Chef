import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

import pandas as pd

# Step 1: Read the CSV file (make sure inventory_log.csv is in the same folder)
df = pd.read_csv("inventory_log.csv")

# Step 2: Convert the Timestamp column to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Step 3: Filter data for the food item you want to forecast (e.g., 'banana')
# Note: Adjust the label if needed (case-sensitive)
banana_data = df[df['Label'] == 'banana'].copy()

# Step 4: Aggregate the count per day
# Set 'Timestamp' as index, then resample daily and sum the 'Count' values
banana_daily = banana_data.set_index('Timestamp').resample('D').sum().reset_index()

# Step 5: Prepare DataFrame for Prophet: rename columns to 'ds' and 'y'
banana_daily = banana_daily.rename(columns={'Timestamp': 'ds', 'Count': 'y'})

# If some days have no data, fill in zeros
banana_daily = banana_daily.set_index('ds').asfreq('D', fill_value=0).reset_index()

# Check the first few rows of the preprocessed data
print(banana_daily.head())

from prophet import Prophet
import matplotlib.pyplot as plt

# Step 6: Initialize the Prophet model
model = Prophet()

# Step 7: Fit the model on the daily banana data
model.fit(banana_daily)

# Step 8: Create a DataFrame to hold future dates (e.g., forecast the next 7 days)
future = model.make_future_dataframe(periods=7)

# Step 9: Make predictions
forecast = model.predict(future)

# Display the last few rows of the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Step 10: Plot the forecast
fig = model.plot(forecast)
plt.title("Banana Demand Forecast")
plt.xlabel("Date")
plt.ylabel("Count")
plt.show()
