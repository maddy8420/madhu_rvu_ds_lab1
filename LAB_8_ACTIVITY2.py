import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import math

# Read the CSV file
df = pd.read_csv("insurance.csv")

# Display first few rows
print(df.head())

# Visualize data
plt.scatter(df.age, df.bought_insurance, marker='+', color='red')
plt.xlabel('Age')
plt.ylabel('Bought Insurance')
plt.title('Insurance Purchase vs Age')
plt.show()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, train_size=0.8, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Model accuracy
print("Model accuracy:", model.score(X_test, y_test))

# Coefficient and intercept
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Prediction function using trained model parameters
def prediction_function(age):
    z = model.coef_[0][0] * age + model.intercept_[0]
    y = sigmoid(z)
    return y

# Example predictions
ages = [25, 35, 43, 52]
for age in ages:
    prob = prediction_function(age)
    prediction = 1 if prob >= 0.5 else 0
    print(f"Age: {age}, Probability of buying insurance: {prob:.4f}, Prediction: {prediction}")
