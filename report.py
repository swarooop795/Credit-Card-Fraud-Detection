import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
# Replace 'path_to_file.csv' with the actual path to your dataset
data = pd.read_csv('creditcard_2023.csv')

# Ensure the dataset contains the required columns
if 'Class' not in data.columns:
    raise ValueError("The dataset must contain a 'Class' column for labels.")

# Splitting features and target variable
X = data.drop('Class', axis=1)  # Features (all columns except 'Class')
y = data['Class']              # Target (fraudulent or genuine)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Generate and display the classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
