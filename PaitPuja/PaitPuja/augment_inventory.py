import pandas as pd
import numpy as np
from datetime import timedelta

# Define allowed food items (adjust as needed)
allowed_food_items = [
    'apple', 'banana', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake'
]

# Read the existing inventory log CSV file
# If it doesn't exist, create a new DataFrame with the required columns.
try:
    df = pd.read_csv("inventory_log.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
except FileNotFoundError:
    print("inventory_log.csv not found. Creating a new empty log.")
    df = pd.DataFrame(columns=["Timestamp", "Label", "Count"])

# Determine the last date in the log; if no data exists, use today's date (normalized to midnight).
if not df.empty:
    last_date = df['Timestamp'].max()
else:
    last_date = pd.Timestamp.today().normalize()

# Number of days of hypothetical data to add
n_days = 10  # Adjust as needed

# Create new rows for hypothetical data
new_rows = []

for day in range(1, n_days + 1):
    new_date = last_date + timedelta(days=day)
    for item in allowed_food_items:
        # Generate a random count between 0 and 20 (adjust the range as needed)
        count = np.random.randint(0, 21)
        new_rows.append({"Timestamp": new_date, "Label": item, "Count": count})

# Convert new rows to a DataFrame
df_new = pd.DataFrame(new_rows)

# Combine the original data and the new hypothetical data
df_augmented = pd.concat([df, df_new], ignore_index=True)

# Overwrite the existing inventory log CSV with the augmented data
df_augmented.to_csv("inventory_log.csv", index=False)

print("inventory_log.csv has been updated with hypothetical data.")
