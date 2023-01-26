import pandas as pd
import yfinance as yf
import os






class EquallyWeighted():


    def __init__(self, cfg, data=None):  
        self.cfg = cfg  
        self.data = self.load_data() if data is None else data


    def load_data(self):
        files = filter(lambda x: x.endswith("parquet"), os.listdir(self.cfg.data_dir))
        stocks = pd.concat([pd.read_parquet(os.path.join(self.cfg.data_dir, f))["Close"] for f in files], join="inner").sort_values(by="Date", axis=0)
        return self.preprocess(stocks)

    

    def preprocess(self, x, percent0=0.7, percent1=0.2):
        print("Preprocessing", int(percent0*x.shape[1]))
        x = x.drop(columns=["AMZN", "GOOGL", "GOOG"])
        tmp = x.dropna(axis=0, thresh=int(percent0*x.shape[1])).dropna(axis=1, thresh=int(percent1*x.shape[0])).fillna(method="ffill")
        print(len(tmp.columns))
        #tmp = tmp.drop(columns=["AMZN", "GOOGL", "GOOG"])
        dropped = set(x.columns) - set(tmp.columns) 
        print("Preprocessing dropped the following stocks" + "-".join(list(dropped)))
        return tmp
        #return x

    def __call__(self):
        #self.data = dataframe.from_pandas(self.data, npartitions=os.cpu_count())
        return ((self.data.diff()/self.data) + 1).fillna(1).mean(axis=1).cumprod()


    
             
