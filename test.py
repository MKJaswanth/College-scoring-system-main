import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER if not already installed
nltk.download('vader_lexicon')

# Load the dataset
file_path = r"C:\Users\jaswa\OneDrive\Desktop\College scoring system\collegereview2023.csv"
df = pd.read_csv(file_path)

# Preprocessing: Drop unnamed columns, trim spaces
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df.columns = df.columns.str.strip()

# Rename columns correctly
column_mapping = {
    "college Name": "College Name",
    "Review Text": "Review Text",
    "Rating": "Rating"
}
df.rename(columns=column_mapping, inplace=True)

# Convert Rating to numeric (some might be missing or text)
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Function to get sentiment score
def get_sentiment_score(text):
    sentiment = sia.polarity_scores(str(text))  # Convert NaN to string
    return sentiment['compound']  # Compound score ranges from -1 to 1

# Apply sentiment analysis to reviews
df["Sentiment Score"] = df["Review Text"].apply(get_sentiment_score)

# Normalize scores
df["Sentiment Score"] = ((df["Sentiment Score"] + 1) / 2) * 100  # Convert -1 to 1 range → 0 to 100
df["Rating"] = df["Rating"].fillna(df["Rating"].mean()) * 10  # Convert 0-10 scale to 0-100

# Weighted score formula (70% sentiment, 30% rating)
df["Weighted Score"] = (df["Sentiment Score"] * 0.7) + (df["Rating"] * 0.3)

# Aggregate scores by college
college_scores = df.groupby("College Name")["Weighted Score"].mean().reset_index()

# Sort colleges by score (highest to lowest)
college_scores = college_scores.sort_values(by="Weighted Score", ascending=False)

# Display results
#print(college_scores)

# Save results to CSV
college_scores.to_csv("college_weighted_scores.csv", index=False)

import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(df['Sentiment Score'], bins=20, kde=True)  # Changed from 'sentiment_score' to 'Sentiment Score'
plt.title("Sentiment Score Distribution")
plt.show()

