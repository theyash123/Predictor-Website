# laptop_price_predictor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("laptopPrice.csv")

# Display column names and preview for debugging
print("Columns:", df.columns.tolist())
print("First 3 rows:\n", df.head(3))

# Drop unwanted columns if they exist
for col in ['rating', 'Number of Ratings', 'Number of Reviews']:
    if col in df.columns:
        df = df.drop(columns=[col])

# Clean and convert data
def clean_memory(x):
    try:
        return int(str(x).split()[0])
    except:
        return 0

def convert_weight(x):
    try:
        return float(str(x).split()[0])
    except:
        return np.nan

# Apply conversions
for col in ['ram_gb', 'ssd', 'hdd', 'graphic_card_gb']:
    if col in df.columns:
        df[col] = df[col].apply(clean_memory)

if 'weight' in df.columns:
    df['weight'] = df['weight'].apply(convert_weight)
    mean_weight = df['weight'].mean()
    df['weight'] = df['weight'].fillna(mean_weight)

# Fill missing categorical values with mode
cat_cols = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn',
            'ram_type', 'os', 'os_bit', 'warranty', 'Touchscreen', 'msoffice']

for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# Print final shape
print("Rows after cleaning/filling:", df.shape[0])

# Label encode categorical columns
label_encoders = {}

for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Ensure target column exists
if 'Price' not in df.columns:
    raise ValueError("'Price' column not found in dataset.")

X = df.drop(columns=['Price'])
y = df['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model and encoders
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Model trained and saved successfully.")