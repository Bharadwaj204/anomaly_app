# model.py
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import re
import os

# Load your dataset (make sure to replace with the actual path if needed)
data = pd.read_csv("models/app_reviews.csv", encoding='latin-1')

# Clean the reviews (lowercase, remove punctuation, etc.)
def clean_text(text):
    """Function to clean the text data."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Apply the cleaning function to the reviews in the dataset
data['clean_review'] = data['review'].apply(clean_text)

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the cleaned reviews into feature vectors
X = vectorizer.fit_transform(data['clean_review'])

# Initialize and train the Isolation Forest model
model = IsolationForest(contamination=0.1)  # Adjust the contamination as per your dataset
model.fit(X)

# Ensure that the models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# Save the trained model and vectorizer
joblib.dump(model, 'models/model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("Model and vectorizer saved successfully!")
