import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Read CSV file
df = pd.read_csv("home_prices.csv")
print(df.head())

# Categorical columns
cat_cols = ['locality']

# One Hot Encoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit and transform
encoded_array = encoder.fit_transform(df[cat_cols])

# Create DataFrame of encoded values
encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoder.get_feature_names_out(cat_cols)
)

# Match index
encoded_df.index = df.index

# Drop original categorical column
df = df.drop(cat_cols, axis=1)

# Concatenate encoded columns
df = pd.concat([df, encoded_df], axis=1)

print(df.head())

# Drop one column to avoid dummy variable trap
df.drop(columns=["locality_Banjara Hills"], inplace=True)

print(df.head())