import pandas as pd

import pyarrow.feather as feather

import string
from nltk.corpus import stopwords

STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]
    
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    # Now just remove any stopwords
    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])

def get_headlines(fileName : str):
    news = pd.read_csv("kaggle/input/million-headlines/abcnews-date-text.csv")
    news["publish_date"] = pd.to_datetime(news["publish_date"].astype(str),infer_datetime_format=True)
    
    news["headline_text"] = news.headline_text.apply(text_process)
    news = news.groupby("publish_date")["headline_text"].agg(' '.join).to_frame()
    news.index.names = ["date"]
    feather.write_feather(news,"kaggle/input/news.feather")
    return news

