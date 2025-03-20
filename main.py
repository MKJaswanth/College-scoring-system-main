import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Load the CSV file
file_path = r"C:\Users\jaswa\OneDrive\Desktop\College scoring system\collegereview2023.csv"
df = pd.read_csv(file_path)

# Drop the Unnamed column if it exists
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Trim spaces from column names
df.columns = df.columns.str.strip()

# Rename columns properly
column_mapping = {
    "college Name": "College Name",
    "Review Text": "Review Text",
    "Rating": "Rating"
}

df.rename(columns=column_mapping, inplace=True)

# Print updated column names
print("Fixed Columns:", df.columns.tolist())

# Print updated column names
print("Updated columns:", df.columns)



# Drop rows with missing reviews
df.dropna(subset=["Review Text"], inplace=True)

# Function to analyze sentiment
def get_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
df["Sentiment"] = df["Review Text"].apply(get_sentiment)

# Count Sentiments
sentiment_counts = df["Sentiment"].value_counts()

# Plot Sentiment Distribution
plt.figure(figsize=(7,5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=["green", "blue", "red"])
plt.title("Sentiment Distribution of College Reviews")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.show()

# Save updated data
output_file = r"C:\Users\jaswa\OneDrive\Desktop\College scoring system\college_reviews_with_sentiment.csv"
df.to_csv(output_file, index=False)
print(f"✅ Sentiment Analysis Completed! Data saved at: {output_file}")

# Show first few rows
df.head()
