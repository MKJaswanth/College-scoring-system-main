import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Download required NLTK data
nltk.download('vader_lexicon')

# Load the dataset
df = pd.read_csv('collegereview2023_cleaned.csv')

# Clean NaN values
df = df.dropna(subset=['Rating', 'Review Text'])

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to get sentiment scores
def analyze_sentiment(text):
    return sia.polarity_scores(str(text))['compound']

# Add sentiment scores to dataframe
df['sentiment_score'] = df['Review Text'].apply(analyze_sentiment)

# Calculate average sentiment and rating by college
college_analysis = df.groupby('College Name').agg({
    'sentiment_score': 'mean',
    'Rating': 'mean',
    'Review Text': 'count'
}).reset_index()

# Rename columns
college_analysis.columns = ['College Name', 'Avg Sentiment', 'Avg Rating', 'Review Count']

# Sort by average rating
college_analysis = college_analysis.sort_values('Avg Rating', ascending=False)

# Create visualizations
plt.figure(figsize=(12, 6))
plt.scatter(college_analysis['Avg Sentiment'], college_analysis['Avg Rating'])
plt.xlabel('Average Sentiment Score')
plt.ylabel('Average Rating')
plt.title('Correlation between Sentiment and Rating')

# Save results
college_analysis.to_csv('college_analysis_results.csv', index=False)
plt.savefig('sentiment_rating_correlation.png')

# Print summary statistics
print("\nTop 10 Colleges by Average Rating:")
print(college_analysis.head(10))

print("\nCorrelation between Sentiment and Rating:", 
      df['sentiment_score'].corr(df['Rating']))

# Prepare data for modeling
X = df[['sentiment_score']].fillna(0)  # Fill any NaN in sentiment scores with 0
y = df['Rating']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Visualize predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Ratings')
plt.savefig('prediction_accuracy.png')