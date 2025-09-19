import pandas as pd
from math import sqrt
from scipy.stats import norm

# Read the CSV file
df = pd.read_csv("lab4_ds_activity2.csv")

# Ensure Salary column is numeric
df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")

# Extract salary column as numpy array
salaries = df["Salary"].values

# Parameters
mu_0 = 50000   # Industry standard
sigma = 2500   # Known population std dev
n = len(salaries)
x_bar = salaries.mean()

# Standard error
se = sigma / sqrt(n)

# Z statistic
z = (x_bar - mu_0) / se

# Two-tailed test
p_two_tailed = 2 * (1 - norm.cdf(abs(z)))

# One-tailed test (greater than)
p_one_tailed = 1 - norm.cdf(z)

# Results
print("Sample mean (x̄):", round(x_bar, 2))
print("Z statistic:", round(z, 3))

print("\n--- Two-tailed Test ---")
print("p-value:", round(p_two_tailed, 4))
if p_two_tailed < 0.05:
    print("Reject H0 → Average salary differs from $50,000")
else:
    print("Fail to reject H0 → No significant difference")

print("\n--- One-tailed Test (greater than) ---")
print("p-value:", round(p_one_tailed, 4))
if p_one_tailed < 0.05:
    print("Reject H0 → Average salary is greater than $50,000")
else:
    print("Fail to reject H0 → Not significantly greater")
