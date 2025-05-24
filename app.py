import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS  # Optional, enable if you have cross-origin requests

app = Flask(__name__)
CORS(app)  # Enable CORS if needed

# Load the pre-trained model and scaler
model = joblib.load("trained_data/random_forest_multiclass.joblib")
scaler = joblib.load("trained_data/scaler.joblib")

feature_names = ['study_hours_per_day', 'social_media_hours', 'attendance_percentage', 'extracurricular_participation']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Validate numeric inputs
        try:
            study_hours = float(data.get("study_hours_per_day"))
            social_media_hours = float(data.get("social_media_hours"))
            attendance = float(data.get("attendance_percentage"))
        except (TypeError, ValueError):
            return jsonify({"error": "study_hours_per_day, social_media_hours, and attendance_percentage must be numeric."}), 400

        # Validate extracurricular participation input
        extracurricular = data.get("extracurricular_participation", "").strip().lower()
        if extracurricular not in ("yes", "no"):
            return jsonify({"error": "extracurricular_participation must be 'Yes' or 'No'."}), 400
        extracurricular_numeric = 1 if extracurricular == "yes" else 0

        # Prepare features and scale
        features_df = pd.DataFrame(
            [[study_hours, social_media_hours, attendance, extracurricular_numeric]], 
            columns=feature_names
        )
        features_scaled = scaler.transform(features_df)

        # Prediction & confidence
        prediction = model.predict(features_scaled)[0]
        confidence = round(max(model.predict_proba(features_scaled)[0]) * 100, 2)

        # Improved, nuanced suggestions
        suggestions = []

        # Prediction-based advice
        if prediction == "Fail":
            suggestions.append("Your current habits indicate a high risk of failing. Consider creating a structured study schedule and seeking academic support.")
        elif prediction == "Borderline":
            suggestions.append("You're on the edge. Small improvements can help secure a pass — try focusing on weak subjects and consistent revision.")
        elif prediction == "Pass":
            suggestions.append("Good job maintaining a passing level! Keep reinforcing your strengths and aim to improve further.")
        elif prediction == "High Performer":
            suggestions.append("Excellent performance! To keep excelling, explore advanced topics and leadership opportunities.")

        # Study hours advice
        if study_hours < 2:
            suggestions.append("Increase daily study hours to at least 2 for better comprehension and retention.")
        elif study_hours > 5:
            suggestions.append("Great dedication! Ensure to balance study with breaks to avoid burnout.")

        # Social media advice
        if social_media_hours > 3:
            suggestions.append("Reducing social media time to under 2 hours can help improve focus and productivity.")
        elif social_media_hours < 1:
            suggestions.append("Good job minimizing distractions from social media.")

        # Attendance advice
        if attendance < 75:
            suggestions.append(f"Your attendance is {attendance}%, which may impact learning. Aim for above 90% for best results.")
        elif attendance >= 95:
            suggestions.append("Excellent attendance! This greatly supports academic success.")

        # Extracurricular advice
        if extracurricular_numeric == 0:
            suggestions.append("Consider joining extracurricular activities to build soft skills and improve your overall profile.")
        else:
            suggestions.append("Participation in extracurriculars is great for developing teamwork and leadership.")

        # Motivation
        suggestions.append("Remember, consistent effort and a positive mindset can significantly improve your performance!")

        # Return JSON response
        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "suggestions": suggestions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
