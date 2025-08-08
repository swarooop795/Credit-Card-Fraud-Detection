import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE  # pip install imbalanced-learn

# Load dataset
data = pd.read_csv('creditcard_2023.csv')

# Check if Class column exists
if 'Class' not in data.columns:
    raise ValueError("The dataset must contain a 'Class' column for labels.")

# Separate features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Check class distribution
print("Original class distribution:")
print(y.value_counts())

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE only to training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nResampled class distribution (training data):")
print(pd.Series(y_train_resampled).value_counts())

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train_resampled)

# Predictions
y_pred = model.predict(X_test_scaled)

# Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
