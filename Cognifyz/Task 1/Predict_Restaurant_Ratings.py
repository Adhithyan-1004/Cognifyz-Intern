import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ================== FILE PICKER ==================
Tk().withdraw()
file_path = askopenfilename(
    title="Select your dataset CSV file",
    filetypes=[("CSV files", "*.csv")]
)
df = pd.read_csv(file_path, encoding='latin1')
print("✅ File loaded:", file_path)
print("Shape:", df.shape)

# ================== SAMPLE FOR SPEED ==================
# Remove or comment out next line for full dataset
df = df.sample(n=min(5000, len(df)), random_state=42)

# ================== ENCODING ==================
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col].astype(str))

# ================== SPLIT DATA ==================
X = df.drop(columns=['Aggregate rating'], errors='ignore')
y = df['Aggregate rating'] if 'Aggregate rating' in df.columns else df.iloc[:, -1]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================== MODELS ==================
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42),
    "Support Vector Regression": SVR(kernel='rbf'),
    "KNN Regression": KNeighborsRegressor(n_neighbors=5)
}

# ================== TRAIN & EVALUATE ==================
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = r2
    print(f"{name}: MSE={mse:.4f}, R²={r2:.4f}")

# ================== BEST MODEL ==================
best_model = max(results, key=results.get)
print("\n🏆 Best Model:", best_model, "with R² =", results[best_model])

# ================== PLOT RESULTS ==================
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color='skyblue')
plt.ylabel("R² Score")
plt.title("Model Comparison")
plt.xticks(rotation=30)
plt.show()

# ================== FEATURE IMPORTANCE (if available) ==================
if "Random Forest" in models:
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(importances)), importances, color='orange')
    plt.title("Feature Importances (Random Forest)")
    plt.show()
