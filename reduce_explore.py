import pandas as pd
import numpy as np
import os

import pyarrow.feather as feather

import pickle

import gc

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

import import_headlines, import_stocks

print("Importing News")
if not os.path.isfile("kaggle/input/news.feather") or os.path.getmtime("kaggle/input/million-headlines/abcnews-date-text.csv") > os.path.getmtime("kaggle/input/news.feather"):
    news = import_headlines.get_headlines("kaggle/input/million-headlines/abcnews-date-text.csv")
    news = import_headlines.reduce_vocabulary(news, "headline_text")
    feather.write_feather(news,"kaggle/input/news.feather")
else:
    news = feather.read_feather("kaggle/input/news.feather")

print("Importing Stocks")
#if not os.path.isfile("kaggle/input/stocks.feather") or import_stocks.getModifiedDate("kaggle/input/nasdaq-daily-stock-prices/") > os.path.getmtime("kaggle/input/stocks.feather"):
stockdata = import_stocks.do_folder("kaggle/input/nasdaq-daily-stock-prices/")
stockdata = import_stocks.reduce_stocks(stockdata,5)
feather.write_feather(stockdata,"kaggle/input/stocks.feather")

#else:
#    stockdata = feather.read_feather("kaggle/input/stocks.feather")


final_data = news.merge(stockdata,how="inner",on="date").copy()

del news
del stockdata
gc.collect()

final_data = final_data[[c for c in final_data.columns if c in {"date","headline_text","Average"}]]

print("Vectorizing")
# instantiate the vectorizer
vect = CountVectorizer()
news_vector = vect.fit(final_data["headline_text"])


# fit and transform
news_vector = vect.fit_transform(final_data["headline_text"])

print("Transforming")
tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(news_vector)
news_vector = tfidf_transformer.transform(news_vector)

movements = final_data.Average

print("splitting data")
#split out test and train subsets:
X_train, X_test, Y_train, Y_test = train_test_split(news_vector, movements, random_state=1)

pickle.dump(Y_test,open("models/reduced_y_test.pkl","wb"))
pickle.dump(X_test,open("models/reduced_x_test.pkl","wb"))

for modelName in ["lbfgs", "sgd", "adam"]:
    for nMax in range(1,30):

        print(f"Model {modelName}, Max Iterations {nMax*10}")
        clf = MLPRegressor(solver=modelName, alpha=1e-5,hidden_layer_sizes=(500,20), random_state=1,verbose=True, max_iter=nMax*10)

        print("starting regression")
        clf = clf.fit(X_train,Y_train)

        pickle.dump(clf, open(f"models/reduced - {modelName} - {nMax}.pkl","wb"))

