import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def analyze_sentiment(review):
    scores = sia.polarity_scores(review)
    positive = scores['pos']
    negative = scores['neg']
    compound = scores['compound']
    
    sentiment = 'positive' if compound >= 0 else 'negative'
    
    return sentiment, positive, negative

st.title('Sentiment Analysis App')
st.write('This app analyzes the sentiment of a given review and shows the percentage of positive and negative sentiment.')

user_input = st.text_area('Enter a review:', '')
if st.button('Determine'):
    if user_input:
        sentiment, positive, negative = analyze_sentiment(user_input)
        st.write(f'Sentiment: {sentiment.capitalize()}')
        st.write(f'Positive Sentiment: {positive * 100:.2f}%')
        st.write(f'Negative Sentiment: {negative * 100:.2f}%')
    else:
        st.write('Please enter a review.')