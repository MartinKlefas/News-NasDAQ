import pandas as pd
import numpy as np
import os

import pyarrow.feather as feather

import gc

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

import import_headlines, import_stocks

print("Importing News")
if os.path.getmtime("kaggle/input/million-headlines/abcnews-date-text.csv") > os.path.getmtime("kaggle/input/news.feather"):
    news = import_headlines.get_headlines("kaggle/input/million-headlines/abcnews-date-text.csv")
else:
    news = feather.read_feather("kaggle/input/news.feather")

print("Importing Stocks")
if import_stocks.getModifiedDate("kaggle/input/nasdaq-daily-stock-prices/") > os.path.getmtime("kaggle/input/stocks.feather"):
    stockdata = import_stocks.do_folder("kaggle/input/nasdaq-daily-stock-prices/")
else:
    stockdata = feather.read_feather("kaggle/input/stocks.feather")


final_data = news.merge(stockdata,how="inner",on="date").copy()

del news
del stockdata
gc.collect()

final_data = final_data[[c for c in final_data.columns if c in {"date","headline_text","Mean"}]]

print("Vectorizing")
# instantiate the vectorizer
vect = CountVectorizer()
vect.fit(final_data["headline_text"])


# fit and transform
news_vector = vect.fit_transform(final_data["headline_text"])


tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(news_vector)
tfidf_transformer.transform(news_vector)

movements = final_data.Average


#split out test and train subsets:
X_train, X_test, Y_train, Y_test = train_test_split(news_vector, movements, random_state=1)

clf = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10000,5000,100,), random_state=1,verbose=True, max_iter=20)

clf = clf.fit(X_train,Y_train)