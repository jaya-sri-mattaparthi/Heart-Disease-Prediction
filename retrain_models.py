import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('Dataset/heart.csv')

# Prepare features and target
x = df.drop('target', axis=1)
y = df['target']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(x_train_scaled, y_train)

# Save models with current scikit-learn version
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
pickle.dump(rf, open('models/rf.pkl', 'wb'))

print("Models retrained and saved successfully with scikit-learn 1.7.2!")
print(f"Random Forest Accuracy: {rf.score(x_test_scaled, y_test):.4f}")
