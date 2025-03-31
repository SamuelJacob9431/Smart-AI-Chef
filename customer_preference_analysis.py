import pandas as pd
import matplotlib.pyplot as plt

# Load order history
orders_df = pd.read_csv('order_history.csv')

# Step 1 - Calculate Total Orders per Dish
dish_popularity = orders_df.groupby('recipe_name')['quantity'].sum().reset_index()
dish_popularity.rename(columns={'quantity': 'total_orders'}, inplace=True)

# Step 2 - Sort by Most Preferred Dishes
dish_popularity.sort_values(by='total_orders', ascending=False, inplace=True)

# Step 3 - Normalize Preference Score (0 to 1)
max_orders = dish_popularity['total_orders'].max()
dish_popularity['preference_score'] = dish_popularity['total_orders'] / max_orders

print("\n🔵 Customer Preference Analysis:\n")
print(dish_popularity)

# Step 4 - Optional Visualization
plt.figure(figsize=(8,6))
plt.barh(dish_popularity['recipe_name'], dish_popularity['total_orders'])
plt.xlabel("Total Orders")
plt.title("Dish Popularity based on Customer Orders")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Step 5 - Save for Future AI Integration
dish_popularity.to_csv('customer_preference_scores.csv', index=False)
print("\n✅ Saved as customer_preference_scores.csv")
