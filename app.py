import streamlit as st
import pickle
import re
from nltk.stem.porter import PorterStemmer

# Load the trained model and vectorizer

model = pickle.load(open("model.pkl", "rb"))
bow_vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stemmer = PorterStemmer()

# Function to remove unwanted patterns from the tweet

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)

    for word in r:
        input_txt = re.sub(word, "", input_txt)

    return input_txt


# Clean the tweet in the same way as we did during training

def clean_tweet(tweet):

    # Remove Twitter handles
    tweet = remove_pattern(tweet, r"@[\w]*")

    # Remove special characters, numbers and punctuation
    tweet = re.sub("[^a-zA-Z#]", " ", tweet)

    # Remove short words
    tweet = " ".join([word for word in tweet.split() if len(word) > 3])

    # Split the tweet into individual words
    words = tweet.split()

    # Apply stemming
    words = [stemmer.stem(word) for word in words]

    # Join the words back into a single string
    tweet = " ".join(words)

    return tweet


# Page title

st.title("Twitter Sentiment Analysis")

st.write("Enter a tweet")

# Text box for entering a tweet

tweet = st.text_area("Enter your tweet:", height=120)

# Predict button

if st.button("Predict"):

    if tweet.strip() == "":
        st.warning("Please enter a tweet.")

    else:
        # Clean the entered tweet
        cleaned_tweet = clean_tweet(tweet)

        # Show the cleaned tweet
        st.write("Cleaned tweet:", cleaned_tweet)

        # Convert the tweet into TF-IDF features
        tweet_vector = bow_vectorizer.transform([cleaned_tweet])

        # Get prediction
        prediction = model.predict(tweet_vector)[0]

        # Display the result
        if prediction == 4:
            st.success("Positive Sentiment")
        else:
            st.error("Negative Sentiment")
