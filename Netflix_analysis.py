import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
from wordcloud import WordCloud

# Load the Netflix dataset
file_path = 'C:/Users/Lenovo/Downloads/netflix1.csv'  # Update with the path to your dataset
netflix_data = pd.read_csv(file_path)

# Cleaning the data by removing duplicates and handling missing values
cleaned_data = netflix_data.drop_duplicates()
cleaned_data = cleaned_data.dropna(subset=['title', 'director', 'country', 'release_year', 'listed_in', 'duration'])

# Converting 'date_added' to datetime format and extracting the year
cleaned_data['date_added'] = pd.to_datetime(cleaned_data['date_added'])
cleaned_data['added_year'] = cleaned_data['date_added'].dt.year

# Add Feature 1: Number of genres per movie or show
cleaned_data['genre_count'] = cleaned_data['listed_in'].apply(lambda x: len(x.split(',')))

# Add Feature 2: Extracting duration in minutes for movies, handling TV Show seasons separately
cleaned_data['duration_minutes'] = cleaned_data['duration'].apply(lambda x: int(x.split(' ')[0]) if 'min' in x else None)

# Create a new feature combining title, director, and listed genres for content-based filtering
cleaned_data['content_features'] = cleaned_data['title'] + ' ' + cleaned_data['director'] + ' ' + cleaned_data['listed_in']

# Use TF-IDF to vectorize the content features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(cleaned_data['content_features'])

# Calculate cosine similarity between the content features
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations based on the title
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = cleaned_data[cleaned_data['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get the top 10 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return cleaned_data['title'].iloc[movie_indices]

# Example: Get recommendations for a specific title
print("Recommendations for 'Inception':")
print(get_recommendations('Inception'))  # Replace with any movie title in your dataset

# Trend Prediction Model (Linear Regression)
# Group the data by release_year and count the number of titles per year
yearly_data = cleaned_data.groupby('release_year')['title'].count().reset_index()

# Now X will be the release_year, and y will be the number of titles per year
X = yearly_data[['release_year']]  # Feature: Release Year
y = yearly_data['title']  # Target: Number of titles released

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple linear regression model for trend prediction
model = LinearRegression()
model.fit(X_train, y_train)

# Predict trends
y_pred = model.predict(X_test)

# Show the model's performance
print("Model Coefficient:", model.coef_)
print("Model Intercept:", model.intercept_)

# Plot the predicted vs actual values for trend prediction
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.title('Trend Prediction: Actual vs Predicted')
plt.xlabel('Release Year')
plt.ylabel('Number of Titles Released')
plt.legend()
plt.show()

# -------------------------------------
# Additional Visualizations
# -------------------------------------

# Pie Chart: Genre Distribution
genre_distribution = cleaned_data['listed_in'].str.split(',').explode().value_counts()
plt.figure(figsize=(10, 7))
genre_distribution.head(10).plot(kind='pie', autopct='%1.1f%%', startangle=140, colormap='Set3')
plt.title("Top 10 Genres Distribution")
plt.ylabel('')  # Hide the y-label for better visualization
plt.tight_layout()
plt.show()

# Line Chart: Yearly Content Trends
movies = cleaned_data[cleaned_data['type'] == 'Movie']
tv_shows = cleaned_data[cleaned_data['type'] == 'TV Show']
movies_trend = movies.groupby('release_year')['title'].count()
tv_shows_trend = tv_shows.groupby('release_year')['title'].count()

plt.figure(figsize=(12, 6))
plt.plot(movies_trend, label='Movies', marker='o')
plt.plot(tv_shows_trend, label='TV Shows', marker='o')
plt.title('Content Trends Over the Years')
plt.xlabel('Year')
plt.ylabel('Number of Titles')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Word Cloud: Most Frequent Words in Titles
title_text = " ".join(cleaned_data['title'].tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(title_text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Titles")
plt.show()

# -------------------------------------
# Interactive Dashboard using Dash
# -------------------------------------
app = dash.Dash(_name_)

# Top directors by content count
top_directors_data = cleaned_data['director'].value_counts().head(10).reset_index()
top_directors_data.columns = ['director', 'count']

app.layout = html.Div([
    html.H1("Netflix Data Analysis Dashboard"),
    
    # Title count over time
    dcc.Graph(
        id='content-trends',
        figure=px.bar(cleaned_data.groupby('release_year')['title'].count().reset_index(),
                      x='release_year', y='title',
                      title='Number of Titles Released Over Time')
    ),
    
    # Top directors bar chart
    dcc.Graph(
        id='top-directors',
        figure=px.bar(top_directors_data, x='director', y='count', 
                      title='Top 10 Directors by Number of Shows/Movies')
    )
])

if _name_ == '_main_':
    app.run_server(debug=True)