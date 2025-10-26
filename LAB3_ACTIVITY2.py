import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
data = 'monthly_expense.csv'
df = pd.read_csv(data)
print("Shape:", df.shape)
print("\nHead:\n", df.head())
print("\nInfo:\n")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nDescribe (numeric):\n", df.describe())
print("\nDescribe (categorical):\n", df.describe(include=['object']))
print("\nDescribe (all):\n", df.describe(include='all'))

# Central tendency for Monthly Household Income
mean = df['Mthly_HH_Income'].mean()
median = df['Mthly_HH_Income'].median()
mode = df['Mthly_HH_Income'].mode()[0]
print("\nMonthly Household Income:")
print("Mean   :", mean)
print("Median :", median)
print("Mode   :", mode)

mean = df['Mthly_HH_Expense'].mean()
median = df['Mthly_HH_Expense'].median()
mode = df['Mthly_HH_Expense'].mode()[0]
print("\nMonthly Household Expenditure:")
print("Mean   :", mean)
print("Median :", median)
print("Mode   :", mode)

# Compare mean and median of Monthly Household Expense
mean_exp = df['Mthly_HH_Expense'].mean()
median_exp = df['Mthly_HH_Expense'].median()

print(f"Mean of Monthly Household Expense: {mean_exp}")
print(f"Median of Monthly Household Expense: {median_exp}")

if mean_exp > median_exp:
    print("The expenditure distribution is right skewed (positively skewed).")
elif mean_exp < median_exp:
    print("The expenditure distribution is left skewed (negatively skewed).")
else:
    print("The expenditure distribution is symmetric.")

# Range of monthly household income
income_range = df['Mthly_HH_Income'].max() - df['Mthly_HH_Income'].min()
print("Range of Monthly Household Income:", income_range)

# Average number of family members per household
avg_family_members = df['No_of_Fly_Members'].mean()
print("Average number of family members per household:", avg_family_members)

# Calculate dependency ratio for each household
df['Dependency_Ratio'] = (df['No_of_Fly_Members'] - df['No_of_Earning_Members']) / df['No_of_Fly_Members']

# Find the household with the highest dependency ratio
max_dep_idx = df['Dependency_Ratio'].idxmax()
max_dep_household = df.loc[max_dep_idx]

print("Household with the highest dependency ratio:")
print(max_dep_household[['No_of_Fly_Members', 'No_of_Earning_Members', 'Dependency_Ratio']])

# Average EMI or rent amount as a percentage of monthly income
avg_emi_percent = (df['Emi_or_Rent_Amt'] / df['Mthly_HH_Income']).mean() * 100
print(f"Average EMI or rent amount as a percentage of monthly income: {avg_emi_percent:.2f}%")

# Identify households where Emi_or_Rent_Amt exceeds 40% of Mthly_HH_Income
high_emi_households = df[df['Emi_or_Rent_Amt'] > 0.4 * df['Mthly_HH_Income']]

print("Households where EMI or Rent exceeds 40% of Monthly Income:")
print(high_emi_households[['Mthly_HH_Income', 'Emi_or_Rent_Amt']])

# Calculate disposable income for each household
df['Disposable_Income'] = df['Mthly_HH_Income'] - df['Mthly_HH_Expense'] - df['Emi_or_Rent_Amt']

# Find the household with the lowest disposable income
min_disp_idx = df['Disposable_Income'].idxmin()
lowest_disp_household = df.loc[min_disp_idx]

print("Household with the lowest disposable income:")
print(lowest_disp_household[['Mthly_HH_Income', 'Mthly_HH_Expense', 'Emi_or_Rent_Amt', 'Disposable_Income']])

# ...existing code...

# Check consistency between Annual_HH_Income and Mthly_HH_Income * 12
df['Calculated_Annual_Income'] = df['Mthly_HH_Income'] * 12
df['Income_Discrepancy'] = df['Annual_HH_Income'] - df['Calculated_Annual_Income']

# Find households with discrepancies
discrepancies = df[df['Income_Discrepancy'] != 0]
print("Households with inconsistent annual income:")
print(discrepancies[['Mthly_HH_Income', 'Annual_HH_Income', 'Calculated_Annual_Income', 'Income_Discrepancy']])

# Group households by Highest_Qualified_Member and calculate average monthly income in each group
avg_income_by_qualification = df.groupby('Highest_Qualified_Member')['Mthly_HH_Income'].mean()

print("Average monthly income by qualification group:")
print(avg_income_by_qualification)

# ...existing code...

# Group by qualification and calculate mean and median monthly income
mean_income_by_qualification = df.groupby('Highest_Qualified_Member')['Mthly_HH_Income'].mean()
median_income_by_qualification = df.groupby('Highest_Qualified_Member')['Mthly_HH_Income'].median()

print("Mean monthly income by qualification:")
print(mean_income_by_qualification)
print("\nMedian monthly income by qualification:")
print(median_income_by_qualification)


# Detect outliers in Monthly Household Income using Z-score method
income_mean = df['Mthly_HH_Income'].mean()
income_std = df['Mthly_HH_Income'].std()
df['Income_Zscore'] = (df['Mthly_HH_Income'] - income_mean) / income_std

# Outliers: Z-score > 3 or < -3
outliers = df[(df['Income_Zscore'] > 3) | (df['Income_Zscore'] < -3)]
print("Outliers in Monthly Household Income (Z-score method):")
print(outliers[['Mthly_HH_Income', 'Income_Zscore']])

# Compute correlation between Monthly Household Income and Monthly Household Expense
correlation = df['Mthly_HH_Income'].corr(df['Mthly_HH_Expense'])
print(f"Correlation between Monthly Income and Monthly Expense: {correlation:.2f}")

# Interpretation
if abs(correlation) > 0.7:
    print("The relationship is strong.")
elif abs(correlation) > 0.3:
    print("The relationship is moderate.")
else:
    print("The relationship is weak.")

# Is there a significant correlation between No_of_Earning_Members and Mthly_HH_Income?
correlation = df['No_of_Earning_Members'].corr(df['Mthly_HH_Income'])
print(f"Correlation between No_of_Earning_Members and Monthly Income: {correlation:.2f}")

if abs(correlation) > 0.7:
    print("The relationship is strong.")
elif abs(correlation) > 0.3:
    print("The relationship is moderate.")
else:
    print("The relationship is weak.")
