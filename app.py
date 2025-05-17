from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip().lower()

    # Simple rule-based reply
    if user_message == 'hi':
        reply = 'hello'
    else:
        reply = f"You said: {user_message}"

    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True)