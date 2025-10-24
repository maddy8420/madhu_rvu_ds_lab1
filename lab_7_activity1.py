import pandas as pd
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()

# Show feature and target names
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Add target column
df['target'] = iris.target
print("\nDataset with target column:")
print(df.head())

# Display subset where target == 1 or 2
print("\nRows where target == 1:")
print(df[df.target == 1].head())

print("\nRows where target == 2:")
print(df[df.target == 2].head())

# Add flower name column
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
print("\nDataset with flower name:")
print(df.head())

# View rows 45–55
print("\nRows 45–55:")
print(df[45:55])

# Split by class
df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

print("\nUnique flower names in each subset:")
print("df0:", df0.flower_name.unique())
print("df1:", df1.flower_name.unique())
print("df2:", df2.flower_name.unique())
