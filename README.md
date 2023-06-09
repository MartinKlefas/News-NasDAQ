# Exploring the effects of Headlines
This project is a self-guided first major project in Natural Language Processing. The main goal of the project is to understand the ways in which text can be used as a modelling tool, and correlated with more standard metrics.

In this project I will be using two of the more popular Kaggle datasets, and combining them to form the inputs and target outputs of a multi-layered perceptron. The datasets are:
 - [A Million News Headlines](https://www.kaggle.com/datasets/therohk/million-headlines)
	- This contains data of news headlines published over a period of nineteen years.
	- Sourced from the reputable Australian news source ABC (Australian Broadcasting Corporation)
- [NASDAQ daily stock prices](https://www.kaggle.com/datasets/svaningelgem/nasdaq-daily-stock-prices)
	- OHLC prices between 1970-01-02 till 2023-03-29.

The goal of the perceptron will be to predict how the stock market will move based on the day's headlines.

The approach is as follows:
# Part 1
Understand the datasets, how to import them, and how to extract only the useful information from them into useful data structures.
This was undertaken on a Jupyter Notebook, predominantly [on Kaggle](https://www.kaggle.com/code/martinklefasstennett/exploring-news-headlines-nasdaq-movement/notebook) for ease of working with the data at source. The notebook is also included in [this repository](exploring-news-headlines-nasdaq-movement.ipynb).

# Part 2
The lessons learned in the notebook are then implemented in code, with a number of refinements:
- Some stock datasets ticker names automatically parse into in-built python functions - breaking pandas.
- Repeatedly merging as we add new stock information was leading to fragmentation and poor performance
- Once loaded the data can be frozen into dictionaries until the source data gets an update.
	- This saves loading time, but still means that the most up-to-date information is used for all runs.
	- To save the most possible time we're using 'feather' [as shown in this test](https://towardsdatascience.com/the-best-format-to-save-pandas-data-414dca023e0d)

# Part 3
The models were analysed on [this notebook](Inspecting-models.ipynb), and unfortunately were unable to fit the data. Data cleaning was then implemented on the two input datasets to remove:
- Massive daily movements - these are potentially skewing the model, and may not be reflected using particularly unique language in headlines. It may also be that some were extreme enough movements to skew all model steering efforts away from a measurable pattern.
- Words that only appeared less than 3 times in the headlines dataset, as these are incredibly unlikely to have been the drivers of global stock change - otherwise surely they'd appear in the headlines the next day too "x causes global stock drop" etc. These were simply bloating the vocabulary in the model and potentially masking true inputs.

With this done, the model training iterations were rerun in [reduce_explore.py](reduce_explore.py), and [a new model inspection notebook](Inspecting-reduced-models.ipynb) was created.

# Part 4
The predictions made here were marginally better than throwing darts at a set of random targets, but only just. At this point we'll try two last options:
- sentiment analysis on the original headlines to try to gauge if "worse" vs "good" days have a noticeable effect on the stock markets.
- time-shifting the headlines - as presumably people can buy and sell stocks faster than they can write news stories for publication.

Analysis of the resulting data is provided in [Sentiment's effect on regression](Inspecting-Sentiment.ipynb) & [Time stamp manipulation](Inspecting-time.ipynb)

# Conclusion
Unfortunately it seems that these two datasets might not be connected in a meaningful sense. There are a number of things that could be behind this:
- Poor modelling choices. Assumptions & guesses were made, and there's also limited computing power. 
- Too many irrelevant stories - local burglaries have no effect on global markets, but global wars should do.
- Missing key time-based information
- Investors only look at hyper-specific information to make choices
- Headlines and stocks are just completely separate with no causal relationships

> Written with [StackEdit](https://stackedit.io/).
