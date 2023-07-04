import nltk
from bs4 import BeautifulSoup
from mastodon import Mastodon
from dotenv import dotenv_values
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

env = dotenv_values()

# Create an instance of the Mastodon class and authenticate
mastodon = Mastodon(
    access_token=env['ACCESS_TOKEN'],
    api_base_url='https://mastodon.uno'
)

# Get the home timeline posts
timeline = mastodon.timeline_home()

# Create an instance of the SentimentIntensityAnalyzer class
analyzer = SentimentIntensityAnalyzer()

# Loop through the timeline statuses
for status in timeline:
    content = status['content']
    author = status['account']['username']
    timestamp = status['created_at']

    # Use BeautifulSoup to remove HTML tags from content
    soup = BeautifulSoup(content, 'html.parser')
    filtered_content = soup.get_text()

    # Perform sentiment analysis on the filtered content
    sentiment_scores = analyzer.polarity_scores(filtered_content)
    sentiment_score = sentiment_scores['compound']

    # Classify sentiment based on the sentiment score
    if sentiment_score >= 0.05:
        sentiment_label = 'Positive'
    elif sentiment_score <= -0.05:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'

    # Print the post information and sentiment analysis result
    print(f"Author: {author}")
    print(f"Content: {filtered_content}")
    print(f"Sentiment: {sentiment_label} ({sentiment_score})")
    print("---")
