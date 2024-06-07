import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# Load data
df = pd.read_csv('data/data.csv')

# Vectorize questions
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Questions'])

# Perform K-Means clustering
num_clusters = 5  # Adjust based on your data
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
df['Cluster'] = kmeans.fit_predict(X)

# Assign cluster numbers as categories
df['Category'] = 'Category_' + df['Cluster'].astype(str)

# Save the clustered data
df.to_csv('data/data.csv', index=False)
print("Clustered data saved to 'data/data.csv'")
