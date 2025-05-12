# Keep all imports at the top
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os
from flask import Flask, render_template, request, jsonify
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import time

# ... keep data loading and preprocessing code ...

# Remove VADER initialization since we're using BERT
# sia = SentimentIntensityAnalyzer()

# Keep BERT model loading
print("Loading BERT model and tokenizer...")
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
print(f"BERT model loaded in {time.time() - start_time:.2f} seconds")

# Keep only BERT sentiment analysis function
def get_sentiment_score(text):
    try:
        start_time = time.time()
        inputs = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        score = predictions.numpy()[0]
        weighted_score = sum((i + 1) * score[i] for i in range(5)) / 5
        normalized_score = (weighted_score - 3) / 2
        return normalized_score
    except Exception as e:
        print(f"Error processing text: {e}")
        return 0

# Keep only one version of categorize_sentiment
def categorize_sentiment(score):
    if score >= 0.2:
        return "Positive"
    elif score <= -0.2:
        return "Negative"
    else:
        return "Neutral"

# Keep the train/test split version of evaluate_sentiment_analysis
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
    
    accuracy = accuracy_score(ground_truth, predictions)
    correlation = np.corrcoef(test_df['Sentiment Score'], test_df['Rating'])[0,1]
    
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

# Flask routes
app = Flask(__name__)

@app.route('/')
def index():
    colleges = df["College Name"].unique().tolist()
    return render_template('index.html', colleges=colleges)

@app.route('/get_college_data', methods=['POST'])
def get_college_data():
    college_name = request.form.get('college')
    college_data = df[df["College Name"] == college_name]
    
    raw_sentiment = college_data["Sentiment Score"].mean()
    sentiment_score = round(((raw_sentiment + 1) / 2) * 9 + 1, 1)
    rating = round(college_data["Rating"].mean(), 1)
    
    sentiment_score = max(1, min(10, sentiment_score))
    rating = max(1, min(10, rating))
    
    weighted_score = (sentiment_score * 0.7) + (rating * 0.3)
    final_score = round(weighted_score, 1)
    
    return jsonify({
        'sentiment_score': sentiment_score,
        'rating': rating,
        'final_score': final_score
    })

@app.route('/college_scores')
def show_college_scores():
    college_scores = pd.DataFrame()
    grouped = df.groupby("College Name")
    
    for college_name, group in grouped:
        raw_sentiment = group["Sentiment Score"].mean()
        raw_rating = group["Rating"].mean()
        
        sentiment_score = ((raw_sentiment + 1) / 2) * 9 + 1
        rating_score = round(raw_rating, 1)
        
        sentiment_score = max(1, min(10, sentiment_score))
        rating_score = max(1, min(10, rating_score))
        
        weighted_score = (sentiment_score * 0.7) + (rating_score * 0.3)
        final_score = weighted_score / 2
        
        college_scores = pd.concat([college_scores, pd.DataFrame({
            'College Name': [college_name],
            'Sentiment Score (1-10)': [round(sentiment_score, 1)],
            'Rating (1-10)': [round(rating_score, 1)],
            'Weighted Score': [round(weighted_score, 1)],
            'Final Score': [round(final_score, 1)]
        })])
    
    college_scores = college_scores.sort_values('Final Score', ascending=False).reset_index(drop=True)
    return render_template('college_scores.html', colleges=college_scores.to_dict('records'))

@app.route('/metrics')
def show_metrics():
    metrics_data = evaluate_sentiment_analysis()
    return render_template('metrics.html', **metrics_data)

# Add after BERT model loading
def train_bert_model(train_df, epochs=2, batch_size=32):  # Increased batch size, reduced epochs
    # Add GPU support if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Add progress tracking
    from tqdm import tqdm
    
    print(f"Training on: {device}")
    print("\nStarting BERT model training...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Prepare training data
    train_texts = train_df['Review Text'].tolist()
    train_ratings = train_df['Rating'].tolist()
    
    # Convert ratings to labels (1-5)
    train_labels = [min(5, max(1, int(rating))) - 1 for rating in train_ratings]
    
    total_batches = len(train_texts) // batch_size
    for epoch in range(epochs):
        total_loss = 0
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        for i in range(0, len(train_texts), batch_size):
            batch_texts = train_texts[i:i + batch_size]
            batch_labels = torch.tensor(train_labels[i:i + batch_size])
            
            # Tokenize and prepare batch
            inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                             max_length=512, return_tensors="pt")
            
            # Forward pass
            outputs = model(**inputs, labels=batch_labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Progress update
            batch_num = i // batch_size + 1
            if batch_num % 10 == 0:
                print(f"Batch {batch_num}/{total_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / total_batches
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
    
    print("\nTraining completed!")
    model.eval()  # Set model back to evaluation mode

# Add after sentiment analysis
from validate_scores import validate_scores

# In your main block, after calculating scores
if __name__ == '__main__':
    # Load and preprocess data
    file_path = "collegereviews_merged.csv"
    try:
        df = pd.read_csv(file_path)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df.columns = df.columns.str.strip()
        
        column_mapping = {
            'college': 'College Name',
            'review': 'Review Text',
            'rating': 'Rating'
        }
        df.rename(columns=column_mapping, inplace=True)
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        df = df.dropna(subset=['Rating'])
        
        # Apply sentiment analysis
        df["Sentiment Score"] = df["Review Text"].apply(get_sentiment_score)
        df["Sentiment Category"] = df["Sentiment Score"].apply(categorize_sentiment)
        
        # Generate wordcloud
        text = " ".join(df["Review Text"].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        
        # After calculating sentiment scores and final scores
        validation_results = validate_scores(df)
        
        # Add validation results to metrics
        # Add this route after your other routes
        @app.route('/validation')
        def show_validation():
            validation_results = {
                'rmse': calculate_rmse(df),
                'r2': calculate_r2(df),
                'correlation': calculate_correlation(df)
            }
            sample_comparisons = df.sample(n=5)
            return render_template('validation.html', results=validation_results, sample_comparisons=sample_comparisons)
        
        app.run(debug=True)
        
    except Exception as e:
        print(f"Error: {e}")
        exit()