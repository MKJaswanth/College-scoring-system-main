import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def validate_scores(df):
    # Calculate actual vs predicted scores
    actual_ratings = df['Rating'].values
    sentiment_scores = df['Sentiment Score'].values
    
    # Convert sentiment scores to same scale as ratings
    normalized_sentiment = ((sentiment_scores + 1) / 2) * 9 + 1
    
    # Calculate weighted final scores (70% sentiment, 30% rating)
    final_scores = (normalized_sentiment * 0.7) + (actual_ratings * 0.3)
    
    # Calculate metrics
    mse = mean_squared_error(actual_ratings, final_scores)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_ratings, final_scores)
    
    # Calculate correlation
    correlation = np.corrcoef(actual_ratings, final_scores)[0,1]
    
    # Print validation results
    print("\nScore Validation Results:")
    print(f"Root Mean Square Error: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"Correlation with actual ratings: {correlation:.2f}")
    
    # Sample comparison
    sample_size = min(5, len(df))
    print("\nSample Comparison:")
    print("College Name | Actual Rating | Predicted Score | Difference")
    print("-" * 60)
    
    sample = df.sample(n=sample_size)
    for _, row in sample.iterrows():
        college = row['College Name']
        actual = row['Rating']
        predicted = (((row['Sentiment Score'] + 1) / 2) * 9 + 1) * 0.7 + actual * 0.3
        diff = abs(actual - predicted)
        print(f"{college[:20]:<20} | {actual:^13.1f} | {predicted:^14.1f} | {diff:^10.1f}")

    return {
        'rmse': rmse,
        'r2': r2,
        'correlation': correlation
    }