import pandas as pd

# CSV file read
df = pd.read_csv("home_prices.csv")
print(df)

# One Hot Encoding (all categories)
df_dummies = pd.get_dummies(df, columns=["locality"])
print(df_dummies)

# One Hot Encoding (drop first to avoid dummy trap)
df_encoded = pd.get_dummies(df, columns=["locality"], drop_first=True)
print(df_encoded)

# Train-test split and Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df_encoded.drop("price_lakhs", axis=1)
y = df_encoded["price_lakhs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Model accuracy
print(model.score(X_test, y_test))

# Testing with new data
test = pd.DataFrame([
    {
        "area_sqr_ft": 1600,
        "bedrooms": 2,
        "locality_Kollur": False,
        "locality_Mankhal": False
    },
    {
        "area_sqr_ft": 1600,
        "bedrooms": 2,
        "locality_Kollur": False,
        "locality_Mankhal": True
    }
])

print(model.predict(test))