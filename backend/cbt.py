import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from transformers import pipeline

class MentalHealthModelCBT:
    def __init__(self, data_path):
        # Load the dataset
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
        # Perform sentiment analysis
        sentiment = self.sentiment_analyzer(user_input)[0]
        return sentiment['label'], sentiment['score']

    def find_faq_answer(self, user_question):
        # Find the closest question from the dataset using cosine similarity
        user_question_vector = self.vectorizer.transform([user_question])
        similarities = cosine_similarity(user_question_vector, self.question_vectors).flatten()
        best_match_index = similarities.argmax()  # Find the index of the closest match
        return self.answers[best_match_index]

    def get_cbt_response(self, user_input):
        # Analyze the sentiment of the user input
        sentiment_label, sentiment_score = self.analyze_sentiment(user_input)
        print(f"Sentiment: {sentiment_label}, Score: {sentiment_score}")  # Debugging statement
        
        # Generate a supportive response based on sentiment
        if sentiment_label == 'NEGATIVE':
            cbt_response = "It sounds like you're feeling down. Remember, it's okay to feel this way. Try to focus on positive aspects and take small steps to improve your mood."
        elif sentiment_label == 'POSITIVE':
            cbt_response = "I'm glad to hear you're feeling good! Keep up the positive mindset and continue doing what makes you happy."
        else:
            cbt_response = "I'm here to help. Can you tell me more about how you're feeling?"

        # Get the most relevant answer from the FAQ dataset
        faq_answer = self.find_faq_answer(user_input)

        # Combine the CBT response with the FAQ-based answer
        final_response = f"{cbt_response}\n\nAlso, based on what you said, here is some additional information:\n{faq_answer}"
        return final_response

# Example usage
if __name__ == "__main__":
    model = MentalHealthModelCBT('data/data.csv')
    user_input = "I'm feeling really sad and hopeless."
    print("CBT + FAQ Response:", model.get_cbt_response(user_input))
