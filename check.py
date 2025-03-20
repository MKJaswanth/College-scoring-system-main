import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import os
from flask import Flask, render_template, request, jsonify
from wordcloud import WordCloud
import matplotlib.pyplot as plt

app = Flask(__name__)

# Download VADER if not already installed
nltk.download('vader_lexicon')

# Load the dataset with better error handling
file_path = "collegereview2023_cleaned.csv"
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
df.rename(columns=column_mapping, inplace=True)
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df = df.dropna(subset=['Rating'])

# Initialize Sentiment Analyzer once (remove duplicate)
sia = SentimentIntensityAnalyzer()

# Single definition of sentiment analysis function
def get_sentiment_score(text):
    try:
        sentiment = sia.polarity_scores(str(text))
        return sentiment['compound']
    except Exception as e:
        print(f"Error processing text: {e}")
        return 0

# Apply sentiment analysis to reviews
df["Sentiment Score"] = df["Review Text"].apply(get_sentiment_score)

# Aggregate scores by college (Average Score)
college_scores = df.groupby("College Name")["Sentiment Score"].mean().reset_index()

# Normalize to a 100-scale for better readability
college_scores["College Score"] = ((college_scores["Sentiment Score"] + 1) / 2) * 100

# Sort colleges by score (highest to lowest)
college_scores = college_scores.sort_values(by="College Score", ascending=False)

# Display results
#print(college_scores)

#college_scores.to_csv("college_scores.csv", index=False)


# trying to predict is it positive or negative or neutral:
def categorize_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment Category"] = df["Sentiment Score"].apply(categorize_sentiment)
#worldcount   .. 
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = " ".join(df["Review Text"].dropna())
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
#plt.show()

# Replace Streamlit code with Flask routes
@app.route('/')
def index():
    colleges = df["College Name"].unique().tolist()
    return render_template('index.html', colleges=colleges)

@app.route('/get_college_data', methods=['POST'])
def get_college_data():
    college_name = request.form.get('college')
    college_data = df[df["College Name"] == college_name]
    
    sentiment_score = round(college_data["Sentiment Score"].mean(), 2)
    rating = round(college_data["Rating"].mean(), 2)
    final_score = round((sentiment_score + rating) / 2, 2)
    
    return jsonify({
        'sentiment_score': sentiment_score,
        'rating': rating,
        'final_score': final_score
    })

if __name__ == '__main__':
    app.run(debug=True)

# Remove all Streamlit related code





