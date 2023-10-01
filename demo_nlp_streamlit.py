# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 15:03:58 2023

@author: pkarm
"""

import streamlit as st
from transformers import pipeline
st.set_page_config(layout="wide")



tabs_font_css = """
<style>
div[class*="stTextArea"] label p {
  font-size: 26px;
  color: red;
}

div[class*="stTextInput"] label p {
  font-size: 20px;
  color: black;
}

div[class*="stNumberInput"] label {
  font-size: 26px;
  color: black;
}

div[class*="stWrite"] label p {
  font-size: 126px;
  color: green;
}
</style>
"""

st.write(tabs_font_css, unsafe_allow_html=True)

#st.title("Demo_NLP")

def twitter_sentiment(data):
    sentiment_pipeline = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")
    tw_senti = sentiment_pipeline(data)
    senti_label_id = tw_senti[0]['label']
    senti_score = tw_senti[0]['score']

    if senti_label_id == 'LABEL_0':
        senti_label = 'Negative'
    elif senti_label_id == 'LABEL_1':
        senti_label = 'Neutral'
    elif senti_label_id == 'LABEL_2':
        senti_label = 'Positive'

    print(tw_senti)
    return senti_label, senti_score

def product_review(data):
  sentiment_pipeline = pipeline(model="nlptown/bert-base-multilingual-uncased-sentiment")
  pro_rev = sentiment_pipeline(data)
  label = pro_rev[0]['label']
  score = pro_rev[0]['score']
  print(pro_rev)
  return label, score

def emotion_sentiment(data):
  sentiment_pipeline = pipeline(model="bhadresh-savani/distilbert-base-uncased-emotion")
  emotion = sentiment_pipeline(data)
  label = emotion[0]['label']
  score = emotion[0]['score']
  print(emotion)
  return label, score



# Text input box
user_input = st.text_input(" Enter a text to analyse:")

keys = ['Sentiment Analysis', 'Emotion Analysis', 'Product Review']

# Display the dropdown
selected_key = st.selectbox("Select a Task:", keys,  index=None)

if selected_key:
    

    # Button to display the entered text
    if selected_key == "Sentiment Analysis":
        senti_label_t, senti_score_t = twitter_sentiment(user_input)
        st.write("<span style='font-size:24px;'> <b>Given Text:</b>", user_input, unsafe_allow_html=True)
        st.write("<span style='font-size:24px;'> <b>Predicted Result: </b>", senti_label_t, unsafe_allow_html=True)
        st.write("<span style='font-size:24px;'> <b>Prediction Confidence: </b>", senti_score_t, unsafe_allow_html=True)
        
        
    if selected_key == "Emotion Analysis":
        label_e, score_e = emotion_sentiment(user_input)
        st.write("<span style='font-size:24px;'> <b>Given Text:</b>", user_input, unsafe_allow_html=True)
        st.write("<span style='font-size:24px;'> <b>Predicted Result: </b>", label_e, unsafe_allow_html=True)
        st.write("<span style='font-size:24px;'> <b>Prediction Confidence: </b>", score_e, unsafe_allow_html=True)
        
        
        
    if selected_key == "Product Review":
        label_pr, score_pr = product_review(user_input)
        st.write("<span style='font-size:24px;'> <b>Given Text:</b>", user_input, unsafe_allow_html=True)
        st.write("<span style='font-size:24px;'> <b>Predicted Result: </b>", label_pr, unsafe_allow_html=True)
        st.write("<span style='font-size:24px;'> <b>Prediction Confidence: </b>", score_pr, unsafe_allow_html=True)
        
        
        
        
