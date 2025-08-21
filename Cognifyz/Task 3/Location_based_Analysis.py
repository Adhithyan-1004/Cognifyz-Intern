import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from tkinter import Tk, filedialog

# --------------------------
# Step 1: Upload CSV File
# --------------------------
Tk().withdraw()  # Hide tkinter window
file_path = filedialog.askopenfilename(
    title="Select Zomato Dataset",
    filetypes=[("CSV Files", "*.csv")]
)

if not file_path:
    print("❌ No file selected. Exiting...")
    exit()

df = pd.read_csv(file_path, encoding='latin1')
print("✅ Dataset loaded successfully!")
print("Columns:", df.columns.tolist())
print(df.head())

# --------------------------
# Step 2: Data Cleaning
# --------------------------
# Drop missing values in essential columns
df.dropna(subset=['City', 'Longitude', 'Latitude', 'Aggregate rating'], inplace=True)

# --------------------------
# Step 3: Top 10 Cities by Restaurant Count
# --------------------------
city_counts = df['City'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(
    x=city_counts.index[:10],
    y=city_counts.values[:10],
    hue=city_counts.index[:10],  # Hue fix for future seaborn versions
    palette="viridis",
    legend=False
)
plt.title("Top 10 Cities by Number of Restaurants", fontsize=16)
plt.ylabel("Number of Restaurants")
plt.xlabel("City")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --------------------------
# Step 4: Ratings Distribution
# --------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df['Aggregate rating'], bins=15, kde=True, color="purple")
plt.title("Ratings Distribution", fontsize=16)
plt.xlabel("Aggregate Rating")
plt.ylabel("Count")
plt.show()

# --------------------------
# Step 5: Price Range Distribution
# --------------------------
plt.figure(figsize=(6, 6))
df['Price range'].value_counts().plot.pie(
    autopct='%1.1f%%', 
    startangle=140,
    colors=sns.color_palette("pastel")
)
plt.title("Price Range Distribution")
plt.ylabel("")
plt.show()

# --------------------------
# Step 6: Interactive Map
# --------------------------
map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
restaurant_map = folium.Map(location=map_center, zoom_start=3)

for _, row in df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['Restaurant Name']} ({row['City']}) - Rating: {row['Aggregate rating']}",
        icon=folium.Icon(color="red" if row['Aggregate rating'] >= 4 else "blue")
    ).add_to(restaurant_map)

restaurant_map.save("restaurant_map.html")
print("🌍 Interactive map saved as 'restaurant_map.html'. Open it in a browser.")

# --------------------------
# Step 7: Heatmap (Optional)
# --------------------------
try:
    from folium.plugins import HeatMap
    heat_data = df[['Latitude', 'Longitude']].dropna().values.tolist()
    heat_map = folium.Map(location=map_center, zoom_start=3)
    HeatMap(heat_data).add_to(heat_map)
    heat_map.save("restaurant_heatmap.html")
    print("🔥 Heatmap saved as 'restaurant_heatmap.html'. Open it in a browser.")
except ImportError:
    print("⚠ Install folium.plugins for heatmap: pip install folium")
