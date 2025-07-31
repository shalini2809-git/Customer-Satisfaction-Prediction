# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("customer_support_tickets (3).csv")

# Drop irrelevant columns
df = df.drop(columns=["Ticket ID", "Customer Name", "Customer Email", "Ticket Description", "Resolution", "Ticket Subject"])

# Drop rows with missing satisfaction rating (target variable)
df = df.dropna(subset=["Customer Satisfaction Rating"])

# Fill or drop missing values in features
df["Time to Resolution"] = pd.to_datetime(df["Time to Resolution"], errors='coerce')
df["First Response Time"] = pd.to_datetime(df["First Response Time"], errors='coerce')
df["Response Time (hrs)"] = (df["Time to Resolution"] - df["First Response Time"]).dt.total_seconds() / 3600
df["Response Time (hrs)"] = df["Response Time (hrs)"].fillna(df["Response Time (hrs)"].median())
df = df.drop(columns=["Time to Resolution", "First Response Time"])

# Encode categorical columns
categorical_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop("Customer Satisfaction Rating", axis=1)
y = df["Customer Satisfaction Rating"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind="barh", figsize=(10, 6))
plt.title("Feature Importance")
plt.show()
