import pandas as pd
import random

# New reviews with varied sentiments
new_reviews = [
    {
        "Name": "Anonymous Student",
        "college Name": "Generic Engineering College",
        "Review Text": "The infrastructure is outdated and needs immediate attention. Labs have old equipment, many computers don't work. Teachers are often absent and administration is unresponsive to complaints. Placement cell makes false promises.",
        "Rating": 3.0
    },
    {
        "Name": "Student",
        "college Name": "Standard Technical Institute",
        "Review Text": "Average experience. Classes happen regularly but quality varies. Some professors are good, others just read from slides. Placements are limited to IT companies. Canteen food is basic. Library needs more books.",
        "Rating": 5.0
    },
    {
        "Name": "Anonymous",
        "college Name": "Metropolitan College of Engineering",
        "Review Text": "Poor management and infrastructure. Frequent power cuts disrupt classes. Lab equipment is broken. Teachers change frequently. Placement statistics are inflated. Hostel conditions are terrible. High fees for poor facilities.",
        "Rating": 2.5
    },
    {
        "Name": "Recent Graduate",
        "college Name": "City Engineering Institute",
        "Review Text": "Neither good nor bad. Basic facilities are there but nothing exceptional. Some teachers are helpful, others not so much. Placements are okay for CS/IT but weak for other branches. Campus is clean but small.",
        "Rating": 5.5
    },
    {
        "Name": "Final Year Student",
        "college Name": "Regional Technical College",
        "Review Text": "Terrible experience. No proper labs, unqualified teachers, poor placement record. Administration only cares about fees. No extracurricular activities. Waste of time and money.",
        "Rating": 2.0
    }
]

# Read existing CSV
df = pd.read_csv(r"c:\Users\jaswa\Downloads\christ-main\christ-main\College scoring system\collegereview2023.csv")

# Create DataFrame from new reviews
new_df = pd.DataFrame(new_reviews)

# Combine existing and new reviews
combined_df = pd.concat([df, new_df], ignore_index=True)

# Save the updated dataset
combined_df.to_csv(r"c:\Users\jaswa\Downloads\christ-main\christ-main\College scoring system\collegereview2023_balanced.csv", index=False)

print("Added balanced reviews to the dataset. New file created: collegereview2023_balanced.csv")