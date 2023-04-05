from nltk.sentiment import SentimentIntensityAnalyzer

import pandas as pd
from tqdm import tqdm

import pyarrow.feather as feather

sia = SentimentIntensityAnalyzer()

#we'll look at each the sentiment in each headline and then create a daily aggregate score.
#this means we need to create a new headline importer:

news = pd.read_csv("kaggle/input/million-headlines/abcnews-date-text.csv",nrows=1000)
news["publish_date"] = pd.to_datetime(news["publish_date"].astype(str),infer_datetime_format=True)

news["scores"]= news["headline_text"].apply((lambda x :sia.polarity_scores(x)))


news = pd.concat([news,news["scores"].apply(pd.Series)],axis=1)

news = news[[c for c in news.columns if c in {"publish_date","neg","neu","pos","compound"}]]

news = news.groupby("publish_date").mean()

news.index.names = ["date"]

stockdata = feather.read_feather("kaggle/input/stocks.feather")


final_data = news.merge(stockdata,how="inner",on="date").copy()

