import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os
from flask import Flask, render_template, request, jsonify
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob

app = Flask(__name__)

# Download VADER if not already installed
nltk.download('vader_lexicon')

# Load the dataset with better error handling
file_path = "collegereviews_merged.csv"  # Changed file name
try:
    print(f"Looking for file in: {os.getcwd()}")
    print(f"Full path: {os.path.join(os.getcwd(), file_path)}")
    
    df = pd.read_csv(file_path)
    print("File loaded successfully!")
    print("Available columns:", df.columns.tolist())
except FileNotFoundError:
    print(f"ERROR: File '{file_path}' not found!")
    print("Please make sure the CSV file is in:", os.getcwd())
    exit()
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Preprocessing: Drop unnamed columns, trim spaces
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df.columns = df.columns.str.strip()

# Update column mapping and clean data
column_mapping = {
    'college': 'College Name',
    'review': 'Review Text',
    'rating': 'Rating'
}

# Add debug print statements
print("Original columns:", df.columns.tolist())
df.rename(columns=column_mapping, inplace=True)
print("Columns after mapping:", df.columns.tolist())

# Convert rating to numeric and handle missing values
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df = df.dropna(subset=['Rating'])

# Check if columns exist
if 'Review Text' not in df.columns:
    print("ERROR: 'Review Text' column not found!")
    print("Available columns:", df.columns.tolist())
    exit()

# Check rating scale
max_rating = df['Rating'].max()
min_rating = df['Rating'].min()
print(f"Rating scale: Min={min_rating}, Max={max_rating}")

# Initialize Sentiment Analyzer once
sia = SentimentIntensityAnalyzer()

# Sentiment analysis function
# Add these imports at the top
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# After Flask app initialization, add BERT setup
# Load pre-trained BERT model and tokenizer
import time

# Add timing for model loading
print("Loading BERT model and tokenizer...")
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
print(f"BERT model loaded in {time.time() - start_time:.2f} seconds")

# Modify sentiment analysis function to include timing
def get_sentiment_score(text):
    try:
        start_time = time.time()
        
        # Tokenize and prepare text for BERT
        tokenize_start = time.time()
        inputs = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=512)
        tokenize_time = time.time() - tokenize_start
        
        # Get prediction
        predict_start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predict_time = time.time() - predict_start
        
        # Convert score
        score = predictions.numpy()[0]
        weighted_score = sum((i + 1) * score[i] for i in range(5)) / 5
        normalized_score = (weighted_score - 3) / 2
        
        total_time = time.time() - start_time
        if total_time > 1.0:  # Only log slow operations
            print(f"Sentiment analysis timing:")
            print(f"  Tokenization: {tokenize_time:.3f}s")
            print(f"  Prediction: {predict_time:.3f}s")
            print(f"  Total: {total_time:.3f}s")
        
        return normalized_score
        
    except Exception as e:
        print(f"Error processing text: {e}")
        return 0

# Update sentiment categorization for BERT
def categorize_sentiment(score):
    if score >= 0.2:
        return "Positive"
    elif score <= -0.2:
        return "Negative"
    else:
        return "Neutral"

# Remove BERT imports and keep TextBlob
from textblob import TextBlob

# Replace sentiment analysis function with TextBlob version
def get_sentiment_score(text):
    try:
        # TextBlob returns polarity between -1 and 1
        analysis = TextBlob(str(text))
        return analysis.sentiment.polarity
    except Exception as e:
        print(f"Error processing text: {e}")
        return 0

# Update sentiment categorization thresholds for TextBlob
def categorize_sentiment(score):
    if score >= 0.1:
        return "Positive"
    elif score <= -0.1:
        return "Negative"
    else:
        return "Neutral"

# Add these imports at the top
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Add after data preprocessing, before sentiment analysis
def prepare_train_test_data():
    # Split data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Apply sentiment analysis to training data
    train_df["Sentiment Score"] = train_df["Review Text"].apply(get_sentiment_score)
    train_df["Sentiment Category"] = train_df["Sentiment Score"].apply(categorize_sentiment)
    
    # Apply sentiment analysis to test data
    test_df["Sentiment Score"] = test_df["Review Text"].apply(get_sentiment_score)
    test_df["Sentiment Category"] = test_df["Sentiment Score"].apply(categorize_sentiment)
    
    return train_df, test_df

# Modify evaluate_sentiment_analysis function
def evaluate_sentiment_analysis():
    # Get train/test split
    train_df, test_df = prepare_train_test_data()
    
    # Create ground truth based on ratings for test set
    ground_truth = []
    predictions = []
    
    for _, row in test_df.iterrows():
        if row['Rating'] >= 7:
            ground_truth.append('Positive')
        elif row['Rating'] < 4:
            ground_truth.append('Negative')
        else:
            ground_truth.append('Neutral')
        predictions.append(row['Sentiment Category'])
    
    # Calculate metrics using test data
    metrics = {}
    for category in ['Positive', 'Negative', 'Neutral']:
        y_true = [1 if gt == category else 0 for gt in ground_truth]
        y_pred = [1 if pred == category else 0 for pred in predictions]
        
        metrics[category.lower()] = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1-score': f1_score(y_true, y_pred, zero_division=0)
        }
    
    # Calculate overall metrics
    accuracy = accuracy_score(ground_truth, predictions)
    correlation = np.corrcoef(test_df['Sentiment Score'], test_df['Rating'])[0,1]
    
    # Calculate distribution statistics
    total_reviews = len(test_df)
    category_counts = test_df['Sentiment Category'].value_counts()
    
    return {
        'accuracy': accuracy,
        'correlation': correlation,
        'metrics': metrics,
        'total_reviews': total_reviews,
        'positive_count': category_counts.get('Positive', 0),
        'negative_count': category_counts.get('Negative', 0),
        'neutral_count': category_counts.get('Neutral', 0),
        'positive_percent': (category_counts.get('Positive', 0) / total_reviews) * 100,
        'negative_percent': (category_counts.get('Negative', 0) / total_reviews) * 100,
        'neutral_percent': (category_counts.get('Neutral', 0) / total_reviews) * 100
    }

# Apply sentiment analysis to reviews
df["Sentiment Score"] = df["Review Text"].apply(get_sentiment_score)

# Categorize sentiment
def categorize_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment Category"] = df["Sentiment Score"].apply(categorize_sentiment)

# Generate wordcloud
text = " ".join(df["Review Text"].dropna())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

# Flask routes
@app.route('/')
def index():
    colleges = df["College Name"].unique().tolist()
    return render_template('index.html', colleges=colleges)

@app.route('/get_college_data', methods=['POST'])
def get_college_data():
    college_name = request.form.get('college')
    college_data = df[df["College Name"] == college_name]
    
    # Convert sentiment score to 1-10 scale
    raw_sentiment = college_data["Sentiment Score"].mean()
    sentiment_score = round(((raw_sentiment + 1) / 2) * 9 + 1, 1)
    
    # Just round the rating to one decimal place, no scaling needed
    rating = round(college_data["Rating"].mean(), 1)
    
    # Ensure scores are within bounds
    sentiment_score = max(1, min(10, sentiment_score))
    rating = max(1, min(10, rating))
    
    # Calculate final score (70% sentiment, 30% rating)
    weighted_score = (sentiment_score * 0.7) + (rating * 0.3)
    final_score = round(weighted_score, 1)
    
    return jsonify({
        'sentiment_score': sentiment_score,
        'rating': rating,
        'final_score': final_score
    })

@app.route('/college_scores')
def show_college_scores():
    # Create empty dataframe for college scores
    college_scores = pd.DataFrame()
    
    # Group by college name
    grouped = df.groupby("College Name")
    
    # For each college, calculate scores
    for college_name, group in grouped:
        raw_sentiment = group["Sentiment Score"].mean()
        raw_rating = group["Rating"].mean()
        
        # Convert sentiment to 1-10 scale
        sentiment_score = ((raw_sentiment + 1) / 2) * 9 + 1
        # Just round the rating, no scaling needed
        rating_score = round(raw_rating, 1)
        
        # Ensure scores are within 1-10 range
        sentiment_score = max(1, min(10, sentiment_score))
        rating_score = max(1, min(10, rating_score))
        
        # Calculate weighted and final scores
        weighted_score = (sentiment_score * 0.7) + (rating_score * 0.3)
        final_score = weighted_score / 2
        
        # Add to dataframe
        college_scores = pd.concat([college_scores, pd.DataFrame({
            'College Name': [college_name],
            'Sentiment Score (1-10)': [round(sentiment_score, 1)],
            'Rating (1-10)': [round(rating_score, 1)],
            'Weighted Score': [round(weighted_score, 1)],
            'Final Score': [round(final_score, 1)]
        })])
    
    # Sort by final score descending
    college_scores = college_scores.sort_values('Final Score', ascending=False).reset_index(drop=True)
    
    return render_template('college_scores.html', colleges=college_scores.to_dict('records'))

@app.route('/metrics')
def show_metrics():
    # Get all metrics
    metrics_data = evaluate_sentiment_analysis()
    
    return render_template('metrics.html', **metrics_data)

if __name__ == '__main__':
    app.run(debug=True)
    
    # After loading the dataset, add these diagnostic prints
    print("\nRating Distribution Analysis:")
    print("Unique rating values:", sorted(df['Rating'].unique()))
    print("Rating value counts:")
    print(df['Rating'].value_counts().sort_index())
    print("\nSample of raw data:")
    print(df[['College Name', 'Rating']].head())


# After sentiment analysis is applied to reviews, add evaluation code
from imblearn.over_sampling import SMOTE
from collections import Counter

# After loading and preprocessing the data, before sentiment analysis
def balance_dataset():
    # Convert sentiment categories to numerical values for SMOTE
    sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    y = df['Sentiment Category'].map(sentiment_mapping)
    X = df['Review Text']
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print("Original dataset distribution:", Counter(y))
    print("Resampled dataset distribution:", Counter(y_resampled))
    
    return X_resampled, y_resampled

# Update the evaluate_sentiment_analysis function
def evaluate_sentiment_analysis():
    metrics = {}
    
    # Calculate class-wise metrics
    for category in ['Positive', 'Negative', 'Neutral']:
        y_true = (df['Rating'] >= 7) if category == 'Positive' else \
                 (df['Rating'] < 4) if category == 'Negative' else \
                 ((df['Rating'] >= 4) & (df['Rating'] < 7))
        y_pred = df['Sentiment Category'] == category
        
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        metrics[category.lower()] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1
        }
    
    # Calculate overall metrics
    accuracy = accuracy_score(ground_truth, predictions)
    correlation = np.corrcoef(df['Sentiment Score'], df['Rating'])[0,1]
    
    # Calculate distribution statistics
    total_reviews = len(df)
    category_counts = df['Sentiment Category'].value_counts()
    
    return {
        'accuracy': accuracy,
        'correlation': correlation,
        'metrics': metrics,
        'total_reviews': total_reviews,
        'positive_count': category_counts.get('Positive', 0),
        'negative_count': category_counts.get('Negative', 0),
        'neutral_count': category_counts.get('Neutral', 0),
        'positive_percent': (category_counts.get('Positive', 0) / total_reviews) * 100,
        'negative_percent': (category_counts.get('Negative', 0) / total_reviews) * 100,
        'neutral_percent': (category_counts.get('Neutral', 0) / total_reviews) * 100
    }
    
    # Generate detailed report
    report = classification_report(ground_truth, predictions)
    
    print("\nSentiment Analysis Evaluation:")
    print(f"Overall Accuracy: {accuracy:.2%}")
    print("\nDetailed Performance Metrics:")
    print(report)
    
    # Calculate correlation between sentiment scores and ratings
    correlation = np.corrcoef(df['Sentiment Score'], df['Rating'])[0,1]
    print(f"\nCorrelation between Sentiment Scores and Ratings: {correlation:.3f}")

# Add this call after sentiment analysis is complete
evaluate_sentiment_analysis() 