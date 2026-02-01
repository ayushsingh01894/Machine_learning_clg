import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

column_names = [
    'mpg', 'cylinders', 'displacement', 'horsepower',
    'weight', 'acceleration', 'model_year', 'origin', 'car_name'
]

df = pd.read_csv(
    url,
    sep=r'\s+',
    names=column_names,
    na_values='?'
)

# 1. Data cleaning
df.dropna(inplace=True)

# 2. Feature selection
X = df[['cylinders', 'displacement', 'horsepower', 'weight',
        'acceleration', 'model_year', 'origin']]
y = df['mpg']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Prediction
y_pred = model.predict(X_test)

# 6. Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 7. Predict mileage for a new car
sample_car = [[
    4,     # cylinders
    140,   # displacement
    90,    # horsepower
    2400,  # weight
    15.0,  # acceleration
    82,    # model_year
    1      # origin
]]

predicted_mpg = model.predict(sample_car)
print("Predicted Mileage (MPG):", predicted_mpg[0])
