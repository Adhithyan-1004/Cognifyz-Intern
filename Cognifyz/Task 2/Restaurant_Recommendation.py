import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tkinter import Tk, filedialog

# Step 1: File upload dialog
Tk().withdraw()  # Hide main tkinter window
file_path = filedialog.askopenfilename(
    title="Select Restaurant Dataset CSV",
    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
)

if not file_path:
    print("No file selected. Exiting.")
    exit()

# Step 2: Load dataset
df = pd.read_csv(file_path, encoding='utf-8')

print("\nDataset loaded successfully!")
print("Columns in dataset:", df.columns.tolist())
print(df.head())

# Step 3: Rename columns to match expected names
df.rename(columns={
    "Cuisines": "Cuisine",
    "Price range": "Price Range",
    "Aggregate rating": "Rating",
    "City": "Location"
}, inplace=True)

# Step 4: Fill missing values
df.fillna({"Cuisine": "", "Price Range": "", "Rating": 0, "Location": ""}, inplace=True)

# Step 5: Combine text features for similarity calculation
df["Combined_Features"] = (
    df["Cuisine"].astype(str) + " " +
    df["Location"].astype(str)
)

# Step 6: Vectorize
vectorizer = TfidfVectorizer(stop_words="english")
feature_matrix = vectorizer.fit_transform(df["Combined_Features"])

# Step 7: Get user preferences
user_cuisine = input("\nEnter preferred cuisine (e.g., Italian, Chinese): ").strip().lower()
user_location = input("Enter preferred city/location (or leave blank for any): ").strip().lower()
user_price = input("Enter preferred price range (1=Low, 2=Medium, 3=High, 4=Premium or leave blank): ").strip()

# Step 8: Create preference vector
user_pref = user_cuisine + " " + user_location
user_vector = vectorizer.transform([user_pref])

# Step 9: Calculate similarity
similarity_scores = cosine_similarity(user_vector, feature_matrix).flatten()

# Step 10: Add similarity column
df["Similarity"] = similarity_scores

# Step 11: Filter by price range if provided
if user_price.isdigit():
    df = df[df["Price Range"] == int(user_price)]

# Step 12: Sort recommendations
recommendations = df.sort_values(by=["Similarity", "Rating"], ascending=[False, False])

# Step 13: Display results
if recommendations.empty:
    print("\nNo recommendations found for your preferences.")
else:
    print("\nTop Recommended Restaurants for You:\n")
    print(recommendations[["Restaurant Name", "Cuisine", "Price Range", "Rating", "Location"]].head(10).to_string(index=False))
