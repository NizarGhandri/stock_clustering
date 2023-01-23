import pandas as pd
import numpy as np
import os
from utils import compute_clean_correlation_matrix
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from numba import jit, typeof
import logging

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.DEBUG)


@jit(parallel = True, nopython=True, nogil=True)
def optimal_weights(average_returns, correlation_matrix, risk_aversion):
    inv_corr = np.linalg.inv(correlation_matrix)
    ones = np.ones(average_returns.shape)
    lambda_constraint = ((ones.T@inv_corr@average_returns - risk_aversion)/(ones.T@inv_corr@ones))
    return (1/risk_aversion)*inv_corr@(average_returns - lambda_constraint*ones)



class Markowitz():


    def __init__(self, cfg, risk_aversion=2, window_size=507, stride=1):  
        self.cfg = cfg 
        self.risk_aversion = risk_aversion
        self.data = self.load_data()
        self.window_size = len(self.data) if window_size is None else window_size
        self.returns = (self.data.diff()/self.data).fillna(0)
        self.stride = stride




    def load_data(self):
        files = filter(lambda x: x.endswith("parquet"), os.listdir(self.cfg.data_dir))
        return self.preprocess(pd.concat([pd.read_parquet(os.path.join(self.cfg.data_dir, f))["Close"] for f in files], join="inner").sort_values(by="Date", axis=0))

    def preprocess(self, x, percent0=0.7, percent1=0.5):
        print("Preprocessing", int(percent0*x.shape[1]))
        tmp = x.dropna(axis=0, thresh=int(percent0*x.shape[1])).dropna(axis=1, thresh=int(percent1*x.shape[0])).fillna(method="ffill")
        dropped = set(x.columns) - set(tmp.columns) 
        print("Preprocessing dropped the following stocks" + "-".join(list(dropped)))
        return tmp
        

    def __call__(self):
        
        return (self.returns + 1).mean(axis=1).cumprod()
        
    

    
    def rolling_optimal_weights(self, i, risk_aversion):
        windowed_return = self.returns.iloc[i-self.window_size:i]
        _, clean_corr = compute_clean_correlation_matrix(windowed_return)
        return_estimators = windowed_return.mean(axis=0).to_numpy()[:, np.newaxis]
        optimal_weights = self.optimal_weights(return_estimators, clean_corr, risk_aversion)
        return np.array(optimal_weights).reshape(-1)

  
 
    def compiled_rolling(self, i):
        #for i in tqdm(windows):
        windowed_return = self.returns.iloc[i-self.window_size:i].to_numpy()
        corr = np.corrcoef(windowed_return.T)
        corr[np.isnan(corr)] = 0.0
        np.fill_diagonal(corr, 1)
        clean_corr = compute_clean_correlation_matrix(corr, self.returns.shape[1], self.returns.shape[0])
        return_estimators = np.mean(windowed_return, axis=0).T
        op = optimal_weights(return_estimators, clean_corr, self.risk_aversion)
        return np.array(op).reshape(-1)
        #return [self.rolling_optimal_weights(i, risk_aversion=risk_aversion) for i in windows]
            
        

    def get_rolling_cumulative_return(self, parallel=False):
        windows = list(range(self.window_size, len(self.returns), self.stride))
        if parallel:
            weights = process_map(self.compiled_rolling, windows, max_workers=os.cpu_count()//3, chunksize = 10)
        else:
            weights = [self.rolling_optimal_weights(i) for i in tqdm(windows, position=0, leave=True)]

        self.weights = [weights[0]]*(self.window_size + 1) + [weight for weight in weights[1:] for _ in range(self.stride)]
        print(len(self.weights))
        self.weights = np.array(self.weights)
        
        return pd.DataFrame(((self.returns.to_numpy() + 1)*self.weights).sum(axis=1), index=self.returns.index).cumprod()

    



    
             
