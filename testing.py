import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import re

# Create a sample dataset with college reviews
sample_data = {
    "College Name": [
        "ABC University", "XYZ Institute", "LMN College", 
        "PQR Academy", "EFG University", "HIJ Institute",
        "ABC University", "XYZ Institute", "LMN College",
        "PQR Academy", "EFG University", "HIJ Institute"
    ],
    "Review Text": [
        "The faculty is very supportive and the campus is beautiful. I had a great experience!",
        "The infrastructure is good but placements are not as expected.",
        "I love the library and research facilities, but the food in the canteen could be better.",
        "Very poor management and outdated curriculum. Would not recommend.",
        "The college has excellent labs and industry tie-ups. Great for practical learning.",
        "Professors are knowledgeable, but the administration is slow and unresponsive.",
        "Amazing research opportunities and state-of-the-art facilities.",
        "Decent academics but the campus life is quite boring.",
        "Great location and friendly atmosphere, but tuition is too expensive.",
        "Terrible experience with student housing and campus security.",
        "The career services department is phenomenal at helping students find jobs.",
        "Classes are too large and it's hard to get individual attention from professors."
    ],
    "Manual_Score": [9.0, 6.5, 7.5, 3.0, 8.5, 5.5, 8.0, 5.0, 6.0, 2.5, 9.0, 4.0]  # Sample scores (0-10 scale)
}

# Convert to DataFrame
df = pd.DataFrame(sample_data)

# Simple text preprocessing without NLTK
def simple_preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply simple preprocessing
df['Processed_Text'] = df['Review Text'].apply(simple_preprocess_text)

# 1. Simple TextBlob sentiment analysis
df['TextBlob_Score'] = df['Review Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
# Scale to 0-10 range
df['TextBlob_Score_Scaled'] = (df['TextBlob_Score'] + 1) * 5

# 2. Custom sentiment model using TF-IDF and Linear Regression
# Split the data
X = df['Processed_Text']
y = df['Manual_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer(max_features=100)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_tfidf, y_train)

# Make predictions
df['Predicted_Score'] = np.nan
test_indices = X_test.index
df.loc[test_indices, 'Predicted_Score'] = model.predict(X_test_tfidf)

# Function to predict score for new reviews
def predict_sentiment_score(review_text):
    # Preprocess the text
    processed = simple_preprocess_text(review_text)
    
    # TextBlob sentiment
    textblob_score = TextBlob(review_text).sentiment.polarity
    textblob_scaled = (textblob_score + 1) * 5
    
    # Model prediction
    tfidf_vector = tfidf_vectorizer.transform([processed])
    model_prediction = float(model.predict(tfidf_vector)[0])  # Convert numpy.float64 to Python float
    
    # Combine scores (simple average)
    final_score = (textblob_scaled + model_prediction) / 2
    
    return {
        'TextBlob_Score': round(textblob_score, 2),
        'TextBlob_Score_Scaled': round(textblob_scaled, 2),
        'Model_Prediction': round(model_prediction, 2),
        'Final_Score': round(final_score, 2)
    }

# Function to analyze a new college with multiple reviews
def analyze_college(college_name, reviews):
    """
    Analyze a new college based on a list of reviews
    
    Parameters:
    college_name (str): Name of the college
    reviews (list): List of review texts
    
    Returns:
    dict: Dictionary containing sentiment analysis results
    """
    results = []
    for review in reviews:
        score_data = predict_sentiment_score(review)
        results.append(score_data['Final_Score'])
    
    average_score = sum(results) / len(results)
    
    return {
        'College Name': college_name,
        'Number of Reviews': len(reviews),
        'Average Score': round(average_score, 2),
        'Individual Scores': results
    }

# Example: Analyze a new college
new_college_reviews = [
    "The professors are excellent but the facilities need improvement.",
    "I love the campus environment, but the classes are too crowded.",
    "The curriculum is outdated and doesn't prepare students for real-world jobs."
]

new_college_result = analyze_college("New University", new_college_reviews)
print(f"\nNew College Analysis:")
print(f"College Name: {new_college_result['College Name']}")
print(f"Number of Reviews: {new_college_result['Number of Reviews']}")
print(f"Average Score: {new_college_result['Average Score']}")
print(f"Individual Review Scores: {new_college_result['Individual Scores']}")

# Example: Analyze a single review
new_review = "This college has great professors and a beautiful campus, but the administrative staff is not helpful."
result = predict_sentiment_score(new_review)
print(f"\nSingle Review Analysis:")
print(f"Review: {new_review}")
print(f"Sentiment Analysis Results: {result}")

# Create a complete college ranking system
def create_college_ranking(colleges_data):
    """
    Create a ranking of colleges based on their reviews
    
    Parameters:
    colleges_data (dict): Dictionary with college names as keys and lists of reviews as values
    
    Returns:
    DataFrame: Ranked colleges with their scores
    """
    rankings = []
    
    for college_name, reviews in colleges_data.items():
        analysis = analyze_college(college_name, reviews)
        rankings.append({
            'College Name': college_name,
            'Average Score': analysis['Average Score'],
            'Number of Reviews': analysis['Number of Reviews']
        })
    
    # Convert to DataFrame and sort
    rankings_df = pd.DataFrame(rankings)
    rankings_df = rankings_df.sort_values('Average Score', ascending=False)
    
    return rankings_df

# Example: Create a ranking of multiple colleges
colleges_data = {
    "Stanford University": [
        "Excellent research opportunities but very competitive.",
        "Beautiful campus and great professors, but high tuition.",
        "The academic rigor is intense but rewarding."
    ],
    "MIT": [
        "Strong focus on STEM with world-class facilities.",
        "Challenging coursework but excellent job prospects.",
        "The campus culture is very innovation-driven."
    ],
    "Harvard University": [
        "The prestige is unmatched but so is the pressure.",
        "Excellent networking opportunities and faculty.",
        "The campus is historic and beautiful."
    ],
    "Community College": [
        "Affordable education with dedicated teachers.",
        "Classes are small and professors are accessible.",
        "Limited resources but good value for money."
    ]
}

print("\nCollege Rankings:")
rankings = create_college_ranking(colleges_data)
print(rankings)

# Visualize the rankings
plt.figure(figsize=(10, 6))
sns.barplot(x='Average Score', y='College Name', data=rankings)
plt.title('College Rankings by Sentiment Score')
plt.tight_layout()
plt.show()