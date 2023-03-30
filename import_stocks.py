import pandas as pd
import os
from tqdm import tqdm

def do_folder(folderName: str):
    bigStockMerge = pd.DataFrame()
    for dirname, _, filenames in os.walk(folderName):
        for filename in tqdm(filenames):
            
            thisFile = os.path.join(dirname, filename)
            bigStockMerge = import_one(thisFile, bigStockMerge)
            
    
    bigStockMerge["Average"] = bigStockMerge.mean(axis=1)
    bigStockMerge.index = pd.to_datetime(bigStockMerge.index)
    bigStockMerge = bigStockMerge[[c for c in bigStockMerge.columns if c in {"date","Average"}]]
    return bigStockMerge


def import_one(fileName : str, targetFrame : pd.DataFrame):
    stock_data = pd.read_csv(fileName,index_col='date')
    ticker = stock_data["ticker"][0]
    stock_data[ticker] = (stock_data["close"] - stock_data["open"])/stock_data["open"]
    
    trimmed_stock = stock_data[[c for c in stock_data.columns if c in {"date",ticker}]]
    if targetFrame.size > 0:
        return targetFrame.merge(trimmed_stock,how="outer", on="date").copy()
    else:
        return trimmed_stock
    

def testInputs(folderName: str):

    for dirname, _, filenames in os.walk(folderName):
        for filename in tqdm(filenames):
            thisFile = os.path.join(dirname, filename)
            try:
                temp = import_one(thisFile,pd.DataFrame())
            except Exception as ex:
                print(f"\n\nProblem with {thisFile}, {ex}")

    

