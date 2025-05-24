from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes (customize origins if needed)

# Load model and scaler
model = joblib.load("trained_data/random_forest_multiclass.joblib")
scaler = joblib.load("trained_data/scaler.joblib")

feature_names = ['study_hours_per_day', 'social_media_hours', 'attendance_percentage', 'extracurricular_participation']

def get_suggestions(prediction, study_hours, social_media_hours, attendance, extracurricular):
    suggestions = []

    # Motivational behavior suggestions based on class
    if prediction == "Fail":
        suggestions.append("Don't be discouraged—every expert was once a beginner. Focus on fundamentals and seek help from a mentor.")
        suggestions.append("Remember, consistent effort beats talent when talent doesn’t work hard. Increase your study hours and reduce distractions.")
        suggestions.append("Your progress matters. Small daily improvements lead to big success!")
    elif prediction == "Borderline":
        suggestions.append("You're on the right track! A little more consistency and dedication will push you over the edge.")
        suggestions.append("Believe in yourself—you've got the potential to improve significantly.")
        suggestions.append("Celebrate your progress so far, and keep building on it.")
    elif prediction == "Pass":
        suggestions.append("Good job! Your hard work is paying off—keep this momentum going.")
        suggestions.append("Balance your studies with activities you enjoy to maintain motivation.")
        suggestions.append("Remember, every step forward is progress toward your goals.")
    elif prediction == "High Performer":
        suggestions.append("Outstanding work! You’re setting an example for others—keep challenging yourself.")
        suggestions.append("Consider mentoring peers or tackling advanced topics to grow even more.")
        suggestions.append("Your dedication and passion will take you far—keep shining!")

    # Personalized behavior suggestions with encouragement
    if study_hours < 2:
        suggestions.append("Try to dedicate at least 2 hours daily to studying—you have what it takes to make it count!")
    if social_media_hours > 3:
        suggestions.append("Limiting social media to under 2 hours a day can help you stay focused and achieve your goals.")
    if attendance < 75:
        suggestions.append("Regular attendance boosts your learning—show up and show off your potential!")
    if extracurricular == 0:
        suggestions.append("Joining extracurricular activities can help you build valuable skills and friendships. You’re worth it!")

    return suggestions

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Validate input fields
        required_fields = ['study_hours_per_day', 'social_media_hours', 'attendance_percentage', 'extracurricular_participation']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Extract inputs
        study_hours = float(data['study_hours_per_day'])
        social_media_hours = float(data['social_media_hours'])
        attendance = float(data['attendance_percentage'])
        extracurricular_raw = data['extracurricular_participation']

        if isinstance(extracurricular_raw, str):
            extracurricular = 1 if extracurricular_raw.lower() == 'yes' else 0
        else:
            extracurricular = int(extracurricular_raw)

        # Prepare features and scale
        features = np.array([[study_hours, social_media_hours, attendance, extracurricular]])
        features_scaled = scaler.transform(features)

        # Predict class and confidence
        prediction = model.predict(features_scaled)[0]
        confidence = round(np.max(model.predict_proba(features_scaled)) * 100, 2)

        # Get suggestions
        suggestions = get_suggestions(prediction, study_hours, social_media_hours, attendance, extracurricular)

        return jsonify({
            "prediction": prediction,
            "confidence": f"{confidence}%",
            "suggestions": suggestions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
