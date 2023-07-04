import nltk
from mastodon import Mastodon
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# Create an instance of the Mastodon class and authenticate
mastodon = Mastodon(
    access_token='',
    api_base_url='https://mastodon.uno'
)

# Get the home timeline posts
timeline = mastodon.timeline_home()

# Create an instance of the SentimentIntensityAnalyzer class
analyzer = SentimentIntensityAnalyzer()

# Iterate over the timeline posts and perform sentiment analysis
for status in timeline:
    content = status['content']
    author = status['account']['username']
    timestamp = status['created_at']

    # Perform sentiment analysis on the post content
    sentiment_scores = analyzer.polarity_scores(content)
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
    print(f"Content: {content}")
    print(f"Sentiment: {sentiment_label} ({sentiment_score})")
    print("---")
