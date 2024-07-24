import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# Sample dataset
data = {
    'review': [
        'I love this product, it is amazing!',
        'Terrible experience, I hate it.',
        'This is the best thing I have ever bought.',
        'Worst purchase ever, not happy with it.',
        'Absolutely fantastic, I highly recommend it!',
        'Very disappointed, will not buy again.',
        'Exceeded my expectations, great value for money.',
        'Not what I expected, very poor quality.',
        'I am extremely satisfied, will buy again.',
        'It is okay, not the best but not the worst.',
        'I am not satisfied with the product.',
        'This is amazing, very good quality.',
        'Do not buy this product, waste of money.',
        'I am very happy with this purchase.',
        'Poor performance, not worth the price.',
        'Highly recommend this, very satisfied.',
        'Not happy, product broke after one use.',
        'Incredible value, very impressed.',
        'Bad quality, do not recommend.',
        'Very pleased with this product.',
        'Terrible, it stopped working after a week.',
        'I am in love with this product, best purchase ever.',
        'Completely dissatisfied, not worth it.',
        'Fantastic product, very high quality.',
        'Very bad experience, not happy at all.',
        'Extremely happy with my purchase, five stars!',
        'Not worth the money, very disappointing.',
        'Absolutely love it, will recommend to friends.',
        'Horrible product, it broke immediately.',
        'Superb quality, very happy with it.',
        'Worst product ever, do not buy it.',
        'Great product, exceeded my expectations.',
        'Not good, very poor performance.',
        'Highly satisfied, will buy again.',
        'Very poor, not recommended.',
        'Best purchase I have made this year.',
        'Not what I expected, very unhappy.',
        'Amazing product, very happy with it.',
        'Terrible, complete waste of money.',
        'Excellent product, very high quality.'
    ],
    'sentiment': [
        'positive', 'negative', 'positive', 'negative', 'positive', 'negative', 
        'positive', 'negative', 'positive', 'neutral', 'negative', 'positive', 
        'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 
        'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 
        'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 
        'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 
        'negative', 'positive', 'negative', 'positive'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(review):
    score = sia.polarity_scores(review)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Streamlit app
st.title('Sentiment Analysis App')
st.write('This app analyzes the sentiment of a given review.')

# Display sample data
st.subheader('Sample Reviews')
st.write(df.head(10))

# User input
user_input = st.text_area('Enter a review:', '')
if st.button('Generate'):
    if user_input:
        # Predict sentiment
        sentiment = analyze_sentiment(user_input)
        st.write(f'Sentiment: {sentiment}')
    else:
        st.write('Please enter a review.')

# Analyze sample data
df['predicted_sentiment'] = df['review'].apply(analyze_sentiment)

# Display the results
st.subheader('Sentiment Analysis Results')
st.write(df)