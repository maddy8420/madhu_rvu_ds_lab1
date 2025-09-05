import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load dataset
data = 'F500.csv'
df = pd.read_csv(data)

# Dataset overview
print("Shape:", df.shape)
print("\nHead:\n", df.head())
print("\nInfo:\n")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

print("\nDescribe (numeric):\n", df.describe())
print("\nDescribe (categorical):\n", df.describe(include=['object']))
print("\nDescribe (all):\n", df.describe(include='all'))

# Central tendency
mean = df['Revenue (in millions)'].mean()
median = df['Revenue (in millions)'].median()
mode = df['Revenue (in millions)'].mode()[0]

print("\nMean:", mean)
print("Median:", median)
print("Mode:", mode)

# Distribution plot
sns.histplot(df['Revenue (in millions)'], bins=10, kde=True)
plt.title("Revenue Distribution")
plt.xlabel("Revenue (in millions)")
plt.ylabel("Frequency")
plt.show()

# Range, variance, std
min_val = df['Revenue (in millions)'].min()
max_val = df['Revenue (in millions)'].max()
range_val = max_val - min_val
variance = df['Revenue (in millions)'].var()
std_dev = df['Revenue (in millions)'].std()

print("\nMin:", min_val)
print("Max:", max_val)
print("Range:", range_val)
print("Variance:", variance)
print("Standard Deviation:", std_dev)

# Quartiles and IQR
Q1 = df['Revenue (in millions)'].quantile(0.25)
Q2 = df['Revenue (in millions)'].quantile(0.50)  # Median
Q3 = df['Revenue (in millions)'].quantile(0.75)
IQR = Q3 - Q1

print("\nQ1:", Q1)
print("Q2 (Median):", Q2)
print("Q3:", Q3)
print("IQR:", IQR)

# Boxplot
plt.boxplot(df['Revenue (in millions)'])
plt.title("Boxplot of Revenue")
plt.ylabel("Revenue (in millions)")
plt.show()

# Skewness & Kurtosis
skewness = df['Revenue (in millions)'].skew()
kurtosis = df['Revenue (in millions)'].kurt()

print("\nSkewness:", skewness)
print("Kurtosis:", kurtosis)
