import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the preprocessed dataset
df = pd.read_csv("preprocessed_crunchbase_companies.csv")

# Define features and target variable (Assuming 'lead_score' as target)
X = df.drop(columns=['lead_score'])  # Features
y = df['lead_score']  # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the Model
joblib.dump(model, "lead_scoring_model.pkl")