# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("dataset.csv")

# Separate features and target
X = df.drop(columns=["target"], axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Linear Regression
# -----------------------------
lm = LinearRegression()
lm.fit(X_train, y_train)

print("Linear Regression R2 Score:", lm.score(X_test, y_test))
print("Linear Regression Coefficients:\n", lm.coef_)

# -----------------------------
# Lasso Regression (L1)
# -----------------------------
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)

print("Lasso R2 Score:", lasso_model.score(X_test, y_test))
print("Lasso Coefficients:\n", lasso_model.coef_)

# -----------------------------
# Ridge Regression (L2)
# -----------------------------
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

print("Ridge R2 Score:", ridge_model.score(X_test, y_test))
print("Ridge Coefficients:\n", ridge_model.coef_)
