import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MentalHealthModel:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.questions = self.df['Questions'].values
        self.answers = self.df['Answers'].values
        self.question_vectors = self.vectorizer.fit_transform(self.questions)

    def get_answer(self, user_question):
        user_vector = self.vectorizer.transform([user_question])
        similarity_scores = cosine_similarity(user_vector, self.question_vectors)
        best_match_idx = np.argmax(similarity_scores)
        best_match_score = similarity_scores[0][best_match_idx]
        if best_match_score > 0.2:  # You can adjust the threshold
            return self.answers[best_match_idx]
        else:
            return "I'm sorry, I don't have an answer for that question. Please contact a mental health professional for help."

# Example usage
# model = MentalHealthModel('data/faq.csv')
# print(model.get_answer('What does it mean to have a mental illness?'))
