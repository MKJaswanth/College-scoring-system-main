import pandas as pd

print("=" * 80)
print("🎓 College Review Sentiment Analysis and Rating System")
print("Data Preprocessing Module")
print("=" * 80)

# Read the CSV file
df = pd.read_csv('collegereview2023.csv')

# Reset index to create proper ID column
df = df.reset_index(drop=True)
df.index = df.index + 1

# Rename columns
df.columns = ['ID', 'Name', 'College Name', 'Review Text', 'Rating']

# Save the cleaned CSV
df.to_csv('collegereview2023_cleaned.csv', index=True)

print("\nProcessing Summary:")
print(f"Total records processed: {len(df)}")
print(f"Columns standardized: {', '.join(df.columns)}")
print("\nCSV file has been cleaned and saved as 'collegereview2023_cleaned.csv'")