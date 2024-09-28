import pandas as pd
from sklearn.model_selection import train_test_split

# Load the original data
data = pd.read_csv('data/data.csv')

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the test data to a new CSV file
test_data.to_csv('data/test_data.csv', index=False)

print("Test data created successfully.")