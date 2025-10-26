import pandas as pd
from scipy import stats

# Read CSV file
df = pd.read_csv("lab4_ds_activity1.csv")

# Extract Salary column
salaries = df["Salary"].values
n = len(salaries)
industry_standard = 50000
alpha = 0.05

# --- Perform the Two-Tailed T-Test ---
print("### Two-Tailed Test Results")
# H0: The average salary = $50,000
# Ha: The average salary ≠ $50,000

t_statistic_two, p_value_two = stats.ttest_1samp(a=salaries, popmean=industry_standard)

print(f"Sample Mean: ${salaries.mean():.2f}")
print(f"T-statistic: {t_statistic_two:.4f}")
print(f"P-value: {p_value_two:.4f}")

if p_value_two < alpha:
    print(f"Conclusion: Reject H0 → Average salary significantly differs from ${industry_standard}")
else:
    print(f"Conclusion: Fail to reject H0 → No significant difference from ${industry_standard}")

print("-" * 50)

# --- Perform the One-Tailed T-Test (Salary > $50,000) ---
print("\n### One-Tailed Test Results (Is the salary greater than $50,000?)")
# H0: μ ≤ 50,000
# Ha: μ > 50,000

t_statistic_one, p_value_one = stats.ttest_1samp(a=salaries, popmean=industry_standard)

# For one-tailed (greater), divide p-value by 2 and check sign
if t_statistic_one > 0:
    p_value_one = p_value_one / 2
else:
    p_value_one = 1 - (p_value_one / 2)

print(f"Sample Mean: ${salaries.mean():.2f}")
print(f"T-statistic: {t_statistic_one:.4f}")
print(f"P-value (one-tailed): {p_value_one:.4f}")

if p_value_one < alpha:
    print(f"Conclusion: Reject H0 → Average salary is significantly greater than ${industry_standard}")
else:
    print(f"Conclusion: Fail to reject H0 → Not significantly greater than ${industry_standard}")
