import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
# Create a sample dataset with college reviews
sample_data = {
    "College Name": [
        "ABC University", "XYZ Institute", "LMN College", 
        "PQR Academy", "EFG University", "HIJ Institute"
    ],
    "Review Text": [
        "The faculty is very supportive and the campus is beautiful. I had a great experience!",
        "The infrastructure is good but placements are not as expected.",
        "I love the library and research facilities, but the food in the canteen could be better.",
        "Very poor management and outdated curriculum. Would not recommend.",
        "The college has excellent labs and industry tie-ups. Great for practical learning.",
        "Professors are knowledgeable, but the administration is slow and unresponsive."
    ],
    "Sentiment Score": [9.0, 6.5, 7.5, 3.0, 8.5, 5.5]  # Sample scores (0-10 scale)
}
# Convert to DataFrame
sample_df = pd.DataFrame(sample_data)
college_ranking = sample_df.groupby('College Name')['Sentiment Score'].mean().sort_values(ascending=False)  
sample_df['TextBlob_Score'] = sample_df['Review Text'].apply(lambda x: TextBlob(x).sentiment.polarity)  
college_ranking = sample_df.groupby('College Name')['Sentiment Score'].mean().sort_values(ascending=False)  
print(college_ranking)  
college_ranking.plot(kind='bar', title='College Ranking by Sentiment Score')  
plt.show()  
# Display the sample dataset
print(sample_df)

