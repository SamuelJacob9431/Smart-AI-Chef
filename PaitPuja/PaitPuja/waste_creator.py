import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Allowed items for waste
allowed_food_items = [
    'apple', 'banana', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake'
]

N_DAYS = 14  # how many days of data to simulate
MIN_RECORDS_PER_DAY = 3
MAX_RECORDS_PER_DAY = 7

# For each item, define an initial baseline and standard deviation for daily changes
# We'll simulate a random walk so it doesn't repeat a cycle
item_params_waste = {
    'apple':    {'baseline': 3,  'std': 1.0},
    'banana':   {'baseline': 4,  'std': 1.0},
    'orange':   {'baseline': 2,  'std': 0.8},
    'broccoli': {'baseline': 1,  'std': 0.5},
    'carrot':   {'baseline': 3,  'std': 1.2},
    'hot dog':  {'baseline': 1,  'std': 0.5},
    'pizza':    {'baseline': 2,  'std': 1.3},
    'donut':    {'baseline': 4,  'std': 1.0},
    'cake':     {'baseline': 2,  'std': 1.0},
}

# We'll generate data starting from "today" at 00:00
start_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

# Track current waste count for each item as a random walk
current_values = {}
for item in allowed_food_items:
    current_values[item] = item_params_waste[item]['baseline']

rows = []
for d in range(N_DAYS):
    # For each day
    day_start = start_date + timedelta(days=d)
    # Decide how many records we'll create for this day (random times)
    records_today = np.random.randint(MIN_RECORDS_PER_DAY, MAX_RECORDS_PER_DAY + 1)
    
    # Generate random timestamps within the day
    # e.g., between 0 and 24*60 = 1440 minutes
    random_minutes = np.random.choice(range(1440), size=records_today, replace=False)
    random_minutes.sort()  # so timestamps go in ascending order
    
    for minute_of_day in random_minutes:
        current_timestamp = day_start + timedelta(minutes=int(minute_of_day))
        
        # For each item, do a random walk update
        for item in allowed_food_items:
            std = item_params_waste[item]['std']
            # Each record can shift the waste count a bit
            fluctuation = np.random.normal(0, std)
            new_value = current_values[item] + fluctuation
            # Round to a non-negative integer
            new_value = max(int(round(new_value)), 0)
            current_values[item] = new_value
            
            rows.append({
                "Timestamp": current_timestamp,
                "Label": item,
                "Waste": new_value
            })

# Convert to DataFrame, sort by Timestamp
df = pd.DataFrame(rows).sort_values("Timestamp").reset_index(drop=True)

# Save to waste_log.csv
df.to_csv("waste_log.csv", index=False)
print("Created waste_log.csv with random, non-repeating data.")
