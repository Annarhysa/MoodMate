from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import MentalHealthChatbot

app = Flask(__name__)
CORS(app)  # Enable CORS
chatbot = MentalHealthChatbot('data/data.csv')

@app.route('/api/get_answer', methods=['POST'])
def get_answer():
    data = request.json
    print(f"Received question: {data}")  # Debugging line
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    answer = chatbot.get_answer(question)
    print(f"Answer: {answer}")  # Debugging line
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
