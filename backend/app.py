from flask import Flask, request, jsonify
from flask_cors import CORS
from model import MentalHealthModel

app = Flask(__name__)
CORS(app)  # Enable CORS
model = MentalHealthModel('data/data.csv')

@app.route('/api/get_answer', methods=['POST'])
def get_answer():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    answer = model.get_answer(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
