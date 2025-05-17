from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Rule-based response function
def get_response(user_input):
    user_input = user_input.lower().strip()

    if user_input in ['hi', 'hello', 'hey']:
        return "Hello! How can I assist you?"
    elif "your name" in user_input or "who are you" in user_input:
        return "I am J.A.R.V.I.S., your virtual assistant."
    elif "time" in user_input:
        return f"The current system time is {datetime.now().strftime('%I:%M %p')}."
    elif "date" in user_input:
        return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."
    elif "how are you" in user_input:
        return "Functioning optimally, thank you."
    elif user_input in ["creator","created","develop","developer"]:
        return "I was developed by Lander Bombasi."
    elif "joke" in user_input:
        return "Why don’t robots get scared? Because they have nerves of steel."
    else:
        return "I'm sorry, I don't have a response for that yet."

# Chat endpoint
@app.route('/api/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    reply = get_response(user_input)
    return jsonify({'reply': reply})

# Optional reset (for future expansion)
@app.route('/api/reset', methods=['POST'])
def reset_chat():
    return jsonify({'status': 'Chat reset complete.'})

if __name__ == '__main__':
    app.run(debug=True)
