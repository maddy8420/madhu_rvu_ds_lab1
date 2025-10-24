import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

df = pd.read_csv("HR_comma_sep.csv")
print(" Data loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())
print("\nSummary statistics:")
print(df.describe())

numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()['left'].sort_values(ascending=False)
print("\nCorrelation with 'left':")
print(corr)

plt.figure(figsize=(7,5))
sns.barplot(x='salary', y='left', data=df, estimator=lambda x: np.mean(x)*100)
plt.title("Impact of Salary on Employee Retention (%)")
plt.ylabel("Percentage of Employees Who Left")
plt.xlabel("Salary Level")
plt.tight_layout()
plt.savefig("salary_vs_retention.png")
plt.show()

plt.figure(figsize=(9,5))
dept_retention = df.groupby('Department')['left'].mean().sort_values(ascending=False)
sns.barplot(x=dept_retention.index, y=dept_retention.values*100)
plt.title("Department vs Employee Retention (%)")
plt.ylabel("Percentage of Employees Who Left")
plt.xlabel("Department")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("department_vs_retention.png")
plt.show()

features = [
    'satisfaction_level',
    'last_evaluation',
    'number_project',
    'average_montly_hours',
    'time_spend_company',
    'Work_accident',
    'promotion_last_5years',
    'salary',
    'Department'
]
X = df[features]
y = df['left']

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
])

model = Pipeline([
    ('pre', preprocessor),
    ('logreg', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("\n Analysis complete. Plots saved as:")
print(" - salary_vs_retention.png")
print(" - department_vs_retention.png")
print(" - confusion_matrix.png")
