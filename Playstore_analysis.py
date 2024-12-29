import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the datasets
file_path_store = 'C:/Users/Lenovo/Downloads/googleplaystore.csv'
file_path_reviews = 'C:/Users/Lenovo/Downloads/googleplaystore_user_reviews.csv'
playstore_data = pd.read_csv(file_path_store)
user_reviews = pd.read_csv(file_path_reviews)

# Basic Data Cleaning
# Removing duplicates and handling missing values
cleaned_data = playstore_data.drop_duplicates()
cleaned_data = cleaned_data.dropna(subset=['App', 'Category', 'Rating', 'Installs'])

# Handle non-numeric values in "Installs" by replacing them with NaN
cleaned_data['Installs'] = cleaned_data['Installs'].str.replace(',', '').str.replace('+', '')

# Remove rows where "Installs" is not a valid number (e.g., "Free")
cleaned_data = cleaned_data[cleaned_data['Installs'].str.isnumeric()]

# Convert "Installs" to integer
cleaned_data['Installs'] = cleaned_data['Installs'].astype(int)

# Convert Ratings to numeric
cleaned_data['Rating'] = pd.to_numeric(cleaned_data['Rating'], errors='coerce')

# Handling missing ratings
cleaned_data = cleaned_data.dropna(subset=['Rating'])

# -------------------------------------
# Visualization 1: Histogram of App Ratings
# -------------------------------------
plt.figure(figsize=(10, 6))
plt.hist(cleaned_data['Rating'], bins=20, color='skyblue')
plt.title('Distribution of App Ratings')
plt.xlabel('Rating')
plt.ylabel('Number of Apps')
plt.show()

# -------------------------------------
# Visualization 2: Bar Chart of Apps by Category
# -------------------------------------
category_counts = cleaned_data['Category'].value_counts()
plt.figure(figsize=(12, 8))
sns.barplot(y=category_counts.index, x=category_counts.values, palette='viridis')
plt.title('Count of Apps by Category')
plt.xlabel('Count')
plt.ylabel('Category')
plt.show()

# -------------------------------------
# Visualization 3: Scatter Plot of Installs vs. Ratings by Category
# -------------------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Installs', y='Rating', hue='Category', data=cleaned_data, palette='Set1')
plt.title('Relationship between Installs and Ratings by Category')
plt.xlabel('Number of Installs')
plt.ylabel('Rating')
plt.xscale('log')
plt.show()

# -------------------------------------
# Feature 1: List of Average Ratings by App Category
# -------------------------------------
avg_ratings_by_category = cleaned_data.groupby('Category')['Rating'].mean().sort_values(ascending=False)
print("Average Ratings by Category:")
print(avg_ratings_by_category)

# -------------------------------------
# Feature 2: Top 10 Most Installed Apps
# -------------------------------------
top_10_installed_apps = cleaned_data[['App', 'Installs']].sort_values(by='Installs', ascending=False).head(10)
print("Top 10 Most Installed Apps:")
print(top_10_installed_apps)

# -------------------------------------
# Feature 3: Top 5 Most Common Genres
# -------------------------------------
cleaned_data['Genres'] = cleaned_data['Genres'].fillna('Unknown')
top_5_genres = cleaned_data['Genres'].value_counts().head(5)
print("Top 5 Most Common Genres:")
print(top_5_genres)

# -------------------------------------
# Sentiment Analysis of User Reviews (Optional)
# -------------------------------------
from textblob import TextBlob

# Clean user reviews and calculate sentiment polarity
user_reviews['Sentiment_Polarity'] = user_reviews['Translated_Review'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Display average sentiment polarity for each app
avg_sentiment = user_reviews.groupby('App')['Sentiment_Polarity'].mean().sort_values(ascending=False)
print("Average Sentiment Polarity by App:")
print(avg_sentiment)