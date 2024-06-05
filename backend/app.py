from flask import Flask, request, jsonify
from chatbot import MentalHealthChatbot

app = Flask(__name__)
chatbot = MentalHealthChatbot('data/data.csv')

@app.route('/api/get_answer', methods=['POST'])
def get_answer():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    answer = chatbot.get_answer(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
