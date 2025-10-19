import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('lab5_ds.csv')

# 1. Line Plot - Unit price vs Total
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x='Unit price', y='Total')
plt.title('Line Plot: Unit price vs Total')
plt.xlabel('Unit price')
plt.ylabel('Total')
plt.show()

# 2. Scatter Plot - Unit price vs Total
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Unit price', y='Total')
plt.title('Scatter Plot: Unit price vs Total')
plt.xlabel('Unit price')
plt.ylabel('Total')
plt.show()

# 3. Linear Regression Plot - Unit price vs Quantity
plt.figure(figsize=(8, 5))
sns.regplot(data=df, x='Unit price', y='Quantity', line_kws={'color': 'red'})
plt.title('Linear Regression: Unit price vs Quantity')
plt.xlabel('Unit price')
plt.ylabel('Quantity')
plt.show()

# 4. Bar Plot (Gender) - Male vs Female
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Gender')
plt.title('Bar Plot: Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# 5. Bar Plot (Customer Type) - Member vs Normal
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Customer type')
plt.title('Bar Plot: Customer Type')
plt.xlabel('Customer type')
plt.ylabel('Count')
plt.show()

# 6. Bar Plot (Average Unit Price by Gender)
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Gender', y='Unit price', estimator='mean', ci=None)
plt.title('Bar Plot: Average Unit price by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Unit price')
plt.show()

# 7. Box Plot (City-wise Unit Price)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='City', y='Unit price')
plt.title('Box Plot: City-wise Unit price')
plt.xlabel('City')
plt.ylabel('Unit price')
plt.show()

# 8. Swarm Plot - Customer type vs Quantity
plt.figure(figsize=(8, 5))
sns.swarmplot(data=df, x='Customer type', y='Quantity')
plt.title('Swarm Plot: Customer type vs Quantity')
plt.xlabel('Customer type')
plt.ylabel('Quantity')
plt.show()

# 9. Violin Plot - Customer type vs Quantity
plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x='Customer type', y='Quantity')
plt.title('Violin Plot: Customer type vs Quantity')
plt.xlabel('Customer type')
plt.ylabel('Quantity')
plt.show()

# 10. Subplots - Swarm Plot + Violin Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.swarmplot(data=df, x='Customer type', y='Quantity', ax=axes[0])
axes[0].set_title('Swarm Plot')

sns.violinplot(data=df, x='Customer type', y='Quantity', ax=axes[1])
axes[1].set_title('Violin Plot')

plt.tight_layout()
plt.show()
