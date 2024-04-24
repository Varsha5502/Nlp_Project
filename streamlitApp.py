import os

from utils.b2 import B2
from dotenv import load_dotenv
import streamlit as st

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from itertools import islice

from nltk.tokenize import RegexpTokenizer
import string

import matplotlib.pyplot as plt
import pandas as pd

from utils import sentiment_modelling as md

nltk.download('stopwords')
REMOTE_DATA = 'Twitter_sample.csv'

load_dotenv()

# load Backblaze connection
b2 = B2(endpoint=os.environ['B2_ENDPOINT'],
        key_id=os.environ['B2_keyID'],
        secret_key=os.environ['B2_applicationKey'])

def get_data():
    # collect data frame of reviews and their sentiment
    b2.set_bucket(os.environ['B2_BUCKETNAME'])
    df = b2.get_df(REMOTE_DATA)
    
    return df

def page_style():
    gradient = """
    <style>
    [data-testid="stAppViewBlockContainer"]{
    background: linear-gradient(to top, #dad4ec 0%, #dad4ec 1%, #f3e7e9 100%);
    }
    </style>
    """
    st.markdown(gradient, unsafe_allow_html=True)
    st.markdown("<Center><H1>Twitter Data Analysis</H1></Center>", unsafe_allow_html = True)

page_style()

# df = pd.read_csv("Twitter_data.csv", encoding="ISO-8859-1")
df = get_data()
st.markdown("<h3>10 most frequently used words by users on twitter</h3>", unsafe_allow_html=True)

st.markdown("Examining the top 10 most frequently used words by users on Twitter, we examine how these terms are associated with user engagement and activity levels.")

data = df.copy()
data.columns=["target","ids","date","flag","user","text"]

data['text']=data['text'].str.lower()
stopwords_list = stopwords.words('english')

# STOPWORDS = set(stopwords_list)
# def cleaning_stopwords(text):
#     return " ".join([word for word in str(text).split() if word not in STOPWORDS])

# data['text'] = data['text'].apply(lambda text: cleaning_stopwords(text))

def cleaning_email(data):
    return re.sub('@[^\s]+', ' ', data)
data['text']= data['text'].apply(lambda x: cleaning_email(x))

def cleaning_URLs(data):
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',data)
data['text'] = data['text'].apply(lambda x: cleaning_URLs(x))

english_punctuations = string.punctuation
def cleaning_punctuations(text):
    translator = str.maketrans('', '', english_punctuations)
    return text.translate(translator)
data['text'] = data['text'].apply(lambda text: cleaning_punctuations(text))

def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
data['text'] = data['text'].apply(lambda x: cleaning_numbers(x))

data_1 = data.copy()

tokenizer = RegexpTokenizer(r'\w+')
data['text'] = data['text'].apply(tokenizer.tokenize)

word_list = data["text"].values

freq_dict = {}

for i in word_list:
    for j in i:
        if j in freq_dict:
            freq_dict[j] += 1
        else:
            freq_dict[j] = 1

sorted_dict = dict(sorted(freq_dict.items(), key=lambda x: x[1], reverse=True))
top_10 = dict(islice(sorted_dict.items(), 10))

fig = plt.figure(figsize=(10, 6))
plt.bar(top_10.keys(), top_10.values(), color='orange')
plt.title('Overall Top 10 Most Frequent Words')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

st.markdown("From the graph provided, it's clear that internet users prefer content with a personal touch, as shown by the high number of first-person pronouns like 'I'm'.<br>", unsafe_allow_html=True)

st.markdown("<h3> Dataset </h3>", unsafe_allow_html=True)
st.dataframe(df.head(25))

model, acc, cnt_vct = md.modelling().model(data_1)

st.markdown("<h3> Sentiment Prediction </h3>", unsafe_allow_html=True)

text_input = st.text_input('Enter the sentence you want to know the sentiment for: ')

submit_button = st.button('Get Sentiment')

if submit_button:
    pred = md.modelling().pred(model, text_input, cnt_vct)

    res = ""
    if pred[0] == 0:
        res = "Negative"
    else:
        res = "Positive"

    st.write(f'Sentiment: {res}')
