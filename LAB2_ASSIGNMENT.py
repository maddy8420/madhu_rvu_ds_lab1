import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. Load dataset
df = pd.read_csv("Automobile - Automobile.csv")

# Ensure numeric columns are properly converted
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

# -----------------------------
# i. Delete the column horsepower
# -----------------------------
if "horsepower" in df.columns:
    df = df.drop(columns=["horsepower"])
    print("Dropped 'horsepower' column")

# -----------------------------
# ii. Impute missing values with median
# -----------------------------
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\nRemaining missing values:")
print(df.isnull().sum())

# -----------------------------
# iii. Apply Min-Max Scaling and Standardization
# -----------------------------
num_cols = df.select_dtypes(include=[np.number]).columns

# Min-Max Scaling (0–1)
minmax_scaler = MinMaxScaler()
df_minmax = df.copy()
df_minmax[num_cols] = minmax_scaler.fit_transform(df_minmax[num_cols])

# Standardization (Z-score)
standard_scaler = StandardScaler()
df_standard = df.copy()
df_standard[num_cols] = standard_scaler.fit_transform(df_standard[num_cols])

# Show results
print("\nAfter Min-Max Scaling:")
print(df_minmax.head())

print("\nAfter Standardization:")
print(df_standard.head())

# Save outputs
df_minmax.to_csv("Automobile_minmax.csv", index=False)
df_standard.to_csv("Automobile_standard.csv", index=False)

# -----------------------------
# Reasoning
# -----------------------------
print("\nPreprocessing complete! Both Min-Max and Standardization applied.")
print("Reasoning: Standardization (Z-score) is better for Automobile.csv "
      "because the dataset contains large variations in feature ranges and "
      "outliers. Standardization is more robust in such cases compared to Min-Max scaling.")
