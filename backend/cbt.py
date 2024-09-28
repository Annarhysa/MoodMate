import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import pipeline

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

        # Initialize sentiment analysis pipeline
        self.sentiment_analyzer = pipeline('sentiment-analysis')


    def analyze_sentiment(self, user_input):
        sentiment = self.sentiment_analyzer(user_input)[0]
        return sentiment['label'], sentiment['score']

    def get_cbt_response(self, user_input):
        sentiment_label, sentiment_score = self.analyze_sentiment(user_input)
        print(f"Sentiment: {sentiment_label}, Score: {sentiment_score}")  # Debugging statement
        
        if sentiment_label == 'NEGATIVE':
            return "It sounds like you're feeling down. Remember, it's okay to feel this way. Try to focus on positive aspects and take small steps to improve your mood."
        elif sentiment_label == 'POSITIVE':
            return "I'm glad to hear you're feeling good! Keep up the positive mindset and continue doing what makes you happy."
        else:
            return "I'm here to help. Can you tell me more about how you're feeling?"

# Example usage
if __name__ == "__main__":
    model = MentalHealthModel('data/data.csv')
    user_input = "I'm feeling really sad and hopeless."
    print("CBT Response:", model.get_cbt_response(user_input))
    #print("FAQ Response:", model.get_answer(user_input))