import pandas as pd

class MentalHealthChatbot:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)

    def get_answer(self, question):
        # Simple keyword-based matching
        for index, row in self.df.iterrows():
            if question.lower() in row['Questions'].lower():
                return row['Answers']
        return "I'm sorry, I don't have an answer for that question. Please contact a mental health professional for help."

# Example usage
# chatbot = MentalHealthChatbot('data/faq.csv')
# print(chatbot.get_answer('What does it mean to have a mental illness?'))
