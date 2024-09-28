import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np

class MentalHealthModel:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.questions = self.df['Questions'].values
        self.answers = self.df['Answers'].values
        self.categories = self.df['Category'].values
        self.question_vectors = self.vectorizer.fit_transform(self.questions)
        
        # Fit K-Means again to predict categories for new questions
        self.num_clusters = len(self.df['Category'].unique())
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
        self.kmeans.fit(self.question_vectors)

        # General responses per category
        self.general_responses = self._get_general_responses()

    def _get_general_responses(self):
        general_responses = {}
        for category in set(self.categories):
            general_responses[category] = self.df[self.df['Category'] == category]['Answers'].values
        return general_responses

    def get_answer(self, user_question):
        user_vector = self.vectorizer.transform([user_question])
        similarity_scores = cosine_similarity(user_vector, self.question_vectors)
        best_match_idx = np.argmax(similarity_scores)
        best_match_score = similarity_scores[0][best_match_idx]
        if best_match_score > 0.2:  # Adjust the threshold as needed
            return self.answers[best_match_idx]
        else:
            return self.get_general_response(user_vector)

    def get_general_response(self, user_vector):
        cluster = self.kmeans.predict(user_vector)[0]
        category = 'Category_' + str(cluster)
        return np.random.choice(self.general_responses.get(category, ["I'm sorry, I don't have an answer for that. Please contact a mental health professional."]))

# Example usage
# model = MentalHealthModel('data/data.csv')
# print(model.get_answer("I'm feeling really sad and hopeless."))

