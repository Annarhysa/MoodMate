import pandas as pd
from model import MentalHealthModel
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import os

def calculate_accuracy_and_rouge(model, test_data_path):
    # Load the labeled dataset
    test_data = pd.read_csv(test_data_path)
    
    correct_predictions = 0
    total_predictions = 0
    rouge_scores = []

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for index, row in test_data.iterrows():
        user_question = row['User_Question']
        actual_answer = row['Expected_Response']
        
        # Predict the answer using the model
        predicted_answer = model.get_answer(user_question)
        
        # Compare the predicted answer with the actual answer
        if predicted_answer == actual_answer:
            correct_predictions += 1
        
        total_predictions += 1
        
        # Calculate ROUGE score
        scores = scorer.score(actual_answer, predicted_answer)
        rouge_scores.append(scores)
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    
    # Calculate average ROUGE scores
    avg_rouge_scores = {
        'rouge1': sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores),
        'rouge2': sum(score['rouge2'].fmeasure for score in rouge_scores) / len(rouge_scores),
        'rougeL': sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores),
    }
    
    return accuracy, avg_rouge_scores

def visualize_performance(avg_rouge_scores, save_path='./plots/performance_plot.png'):
    # Visualize the performance metrics
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    scores = [avg_rouge_scores['rouge1'], avg_rouge_scores['rouge2'], avg_rouge_scores['rougeL']]

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, scores, color='skyblue')
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Chatbot Model Performance')
    
    # Save the plot to the specified path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

# Example usage
if __name__ == "__main__":
    model = MentalHealthModel('data/data.csv')
    accuracy, avg_rouge_scores = calculate_accuracy_and_rouge(model, 'data/response.csv')
    print(f"Model Accuracy: {accuracy:.2f}")
    print(f"Average ROUGE Scores: {avg_rouge_scores}")
    
    visualize_performance(avg_rouge_scores)
