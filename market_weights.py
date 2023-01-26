import os
import pandas as pd
from data_querier import SharesOutStandingQuerier
import logging 
from functools import reduce
from tqdm import tqdm
import numpy as np












class MarketWeighted():


    def __init__(self, cfg):  
        self.cfg = cfg  
        self.data = self.load_data()
        self.sharesout = SharesOutStandingQuerier(self.data.columns, ("2012-01-01", "2022-12-31"), username="ghandri")
        


    def load_data(self):
        files = filter(lambda x: x.endswith("parquet"), os.listdir(self.cfg.data_dir))
        return self.preprocess(pd.concat([pd.read_parquet(os.path.join(self.cfg.data_dir, f))["Close"] for f in files], join="inner").sort_values(by="Date", axis=0))

    def preprocess(self, x, percent0=0.7, percent1=0.5):
        x = x.drop(columns=["AMZN", "GOOGL", "GOOG"])
        tmp = x.dropna(thresh=int(percent0*x.shape[1])).dropna(axis=1, thresh=int(percent1*x.shape[0])).fillna(method="ffill")
        dropped = set(x.columns) - set(tmp.columns) 
        logging.info("Preprocessing dropped the following stocks %s ".format(["-".join(list(dropped))]))
        return tmp

    def __call__(self):
        weights = self.compute_weights()
        returns = ((self.data.diff()/self.data) + 1)[weights.columns].fillna(1) * weights
        return returns.sum(axis=1).cumprod()

    def compute_weights(self):
        mapper = list(map(lambda x: x[1].drop(columns=["ticker"]).drop_duplicates("date").set_index("date").rename(columns={"shrout":x[0]}),
               filter(lambda x: len(x[1]) > 120, # some are missing certain dates 
               self.sharesout.sharesout.merge(self.sharesout.permcos, on="permco").drop(columns=["permco"]).groupby(["ticker"]))))


        tt = mapper[0]
        for elem in tqdm(mapper[1:]):
            tt = tt.merge(elem, left_index=True, right_index=True, how="outer")
        tt = tt.fillna(method="ffill")
        dates = set(self.data.index.map(lambda x: x.date()))
        dates_2_add = list(dates - dates.intersection(tt.index))
        for date in tqdm(dates_2_add):
            tt.loc[date] = np.nan
        tt = tt.sort_index()
        tt = tt.loc[self.data.index, :].fillna(method="ffill") * self.data.loc[:, tt.columns]
        return tt.div(tt.sum(axis=1), axis=0)
        reducer = reduce(lambda x, y: x.merge(y, left_index=True, right_index=True, how="outer"), tqdm(mapper)) 
        return reducer.div(reducer.sum(axis=1), axis=0) 
        




     

