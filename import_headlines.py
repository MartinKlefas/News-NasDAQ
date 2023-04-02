import pandas as pd
from tqdm import tqdm

import pyarrow.feather as feather

import string
from nltk.corpus import stopwords
from collections import Counter

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

def remove_words(mess : str ,wordList : list):
    reduced_mess = ' '.join([word for word in mess.split() if word.lower() not in wordList])
    return reduced_mess

def get_headlines(fileName : str):
    news = pd.read_csv(fileName)
    news["publish_date"] = pd.to_datetime(news["publish_date"].astype(str),infer_datetime_format=True)
    
    news["headline_text"] = news.headline_text.apply(text_process)
    news = news.groupby("publish_date")["headline_text"].agg(' '.join).to_frame()
    news.index.names = ["date"]
    feather.write_feather(news,"kaggle/input/news.feather")
    return news

def reduce_vocabulary(df : pd.DataFrame, columnName : str):
    words = df[columnName].apply(lambda x: [word.lower() for word in x.split()])

    word_counts = Counter()

    for line in words:
        word_counts.update(line)

    discard_words = list()
    print("finding low utility words")
    for word in word_counts:
        if word_counts[word] <= 3:
            discard_words.append(word)

    print(f"found {len(discard_words)} low utility words, out of {len(word_counts)}")

    print("removing them")
    tqdm.pandas()
    df[columnName] = df[columnName].progress_apply((lambda x : remove_words(x, discard_words)))
    
    return df
