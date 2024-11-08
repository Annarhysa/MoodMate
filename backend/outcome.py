import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import MentalHealthModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Mock dataset
data = pd.read_csv("./data/response.csv")

# Convert to DataFrame
test_df = pd.DataFrame(data)

# Initialize the chatbot model with the path to your dataset
model = MentalHealthModel('data/data.csv')

# Predict responses for the test dataset
test_df['Predicted_Response'] = test_df['User_Question'].apply(model.get_answer)

# Evaluate accuracy
accuracy = accuracy_score(test_df['Expected_Response'], test_df['Predicted_Response'])
precision = precision_score(test_df['Expected_Response'], test_df['Predicted_Response'], average='weighted')
recall = recall_score(test_df['Expected_Response'], test_df['Predicted_Response'], average='weighted')
f1 = f1_score(test_df['Expected_Response'], test_df['Predicted_Response'], average='weighted')

# Print evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Visualize the evaluation metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
scores = [accuracy, precision, recall, f1]

plt.figure(figsize=(10, 6))
plt.bar(metrics, scores, color='skyblue')
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Chatbot Model Performance')
plt.show()