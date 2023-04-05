from nltk.sentiment import SentimentIntensityAnalyzer

import pandas as pd
from tqdm import tqdm

import pyarrow.feather as feather
import pickle


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

print("Gathering sentiment analysis")

sia = SentimentIntensityAnalyzer()

#we'll look at each the sentiment in each headline and then create a daily aggregate score.
#this means we need to create a new headline importer:

news = pd.read_csv("kaggle/input/million-headlines/abcnews-date-text.csv")
news["publish_date"] = pd.to_datetime(news["publish_date"].astype(str),infer_datetime_format=True)


tqdm.pandas()
news["scores"]= news["headline_text"].progress_apply((lambda x :sia.polarity_scores(x)))

print("Adding to dataframe")
news = pd.concat([news,news["scores"].progress_apply(pd.Series)],axis=1)

news = news[[c for c in news.columns if c in {"publish_date","neg","neu","pos","compound"}]]

print("creating daily scores")
news = news.groupby("publish_date").mean()

news.index.names = ["date"]

print("Loading Stocks")
stockdata = feather.read_feather("kaggle/input/stocks.feather")


final_data = news.merge(stockdata,how="inner",on="date").copy()

print("Converting data")
news_sentiments = final_data[["neg","neu","pos","compound"]].to_numpy()
movements = final_data.Average.to_numpy()

print("splitting data")
#split out test and train subsets:
X_train, X_test, Y_train, Y_test = train_test_split(news_sentiments, movements, random_state=1)

pickle.dump(Y_test,open("models/sentiment_y_test.pkl","wb"))
pickle.dump(X_test,open("models/sentiment_x_test.pkl","wb"))

for modelName in ["lbfgs", "sgd"]:
    for nMax in range(1,20):

        print(f"Model {modelName}, Max Iterations {nMax*10}")
        clf = MLPRegressor(solver=modelName, alpha=1e-5,hidden_layer_sizes=(500,20), random_state=1,verbose=True, max_iter=nMax*10)

        print("starting regression")
        clf = clf.fit(X_train,Y_train)

        pickle.dump(clf, open(f"models/sentiment - {modelName} - {nMax}.pkl","wb"))
