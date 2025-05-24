import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load dataset with your path
data = pd.read_csv(r'C:\Users\PUP-CITE-CAD17\Documents\Project\API_BackEnd-1\csv\student_habits_performance.csv')

# Map extracurricular participation to numeric
data['extracurricular_participation'] = data['extracurricular_participation'].map({'Yes': 1, 'No': 0})

# Create multi-class target
def performance_class(score):
    if score < 50:
        return "Fail"
    elif score < 60:
        return "Borderline"
    elif score < 80:
        return "Pass"
    else:
        return "High Performer"

data['performance_label'] = data['exam_score'].apply(performance_class)

# Features and target
features = data[['study_hours_per_day', 'social_media_hours', 'attendance_percentage', 'extracurricular_participation']]
target = data['performance_label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Save artifacts
os.makedirs("trained_data", exist_ok=True)
joblib.dump(model, "trained_data/random_forest_multiclass.joblib")
joblib.dump(scaler, "trained_data/scaler.joblib")

# Save SHAP background data
background_data = X_train_scaled[:100]
np.save("trained_data/background_data.npy", background_data)

# Print classification report
from sklearn.metrics import classification_report
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
