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


@jit(nopython=True, nogil=True)
def optimal_weights(average_returns, correlation_matrix, risk_aversion):
    inv_corr = np.linalg.inv(correlation_matrix)
    ones = np.ones(average_returns.shape)
    lambda_constraint = ((ones.T@inv_corr@average_returns - risk_aversion)/(ones.T@inv_corr@ones))
    return (1/risk_aversion)*inv_corr@(average_returns - lambda_constraint*ones)


@jit(nopython=True, nogil=True)
def optimal_weights_constrained(average_returns, correlation_matrix, risk_aversion, C):
    beta = np.linalg.inv(risk_aversion*correlation_matrix + C*np.eye(correlation_matrix.shape[0]))
    ones = np.ones(average_returns.shape)
    lambda_constraint = ((ones.T@beta@average_returns - 1)/(ones.T@beta@ones))
    return beta@(average_returns - lambda_constraint*ones)

class Markowitz():


    def __init__(self, cfg, risk_aversion=2, window_size=367, stride=10, C = 0.1, clustering=None):  
        self.cfg = cfg 
        self.risk_aversion = risk_aversion
        self.data = self.load_data()
        self.window_size = len(self.data) if window_size is None else window_size
        self.returns = (self.data.diff()/self.data).fillna(0)
        self.stride = stride
        self.C = C
        self.clustering = clustering




    def load_data(self):
        files = filter(lambda x: x.endswith("parquet"), os.listdir(self.cfg.data_dir))
        return self.preprocess(pd.concat([pd.read_parquet(os.path.join(self.cfg.data_dir, f))["Close"] for f in files], join="inner").sort_values(by="Date", axis=0))

    def preprocess(self, x, percent0=0.7, percent1=0.5):
        print("Preprocessing", int(percent0*x.shape[1]))
        print(x.shape[1])
        x = x.drop(columns=["AMZN", "GOOGL", "GOOG", "FTNT", "DXCM", "TECH", "NDAQ"])
        print(x.shape[1])
        tmp = x.dropna(axis=0, thresh=int(percent0*x.shape[1])).dropna(axis=1, thresh=int(percent1*x.shape[0])).fillna(method="ffill")
        print(tmp.shape)
        dropped = set(x.columns) - set(tmp.columns) 
        print("Preprocessing dropped the following stocks" + "-".join(list(dropped)))
        return tmp
        

        
    

    
    def rolling_optimal_weights(self, i, risk_aversion):
        windowed_return = self.returns.iloc[i-self.window_size:i]
        _, clean_corr = compute_clean_correlation_matrix(windowed_return)
        return_estimators = windowed_return.mean(axis=0).to_numpy()[:, np.newaxis]
        optimal_weights = self.optimal_weights(return_estimators, clean_corr, risk_aversion)
        return np.array(optimal_weights).reshape(-1)

  
    
    def clustered_rolling(self, i):

        print("cluster")
        clusters = self.clustering[i]
        return_clusters = []
        for cluster in clusters:
            windowed_return = self.returns.iloc[i-self.window_size:i, cluster].to_numpy()
            corr = np.cov(windowed_return.T)
            clean_corr = compute_clean_correlation_matrix(corr, corr.shape[1], windowed_return.shape[0]) 
            clean_corr = np.array(clean_corr) 
            return_estimators = np.mean(windowed_return, axis=0).T
            op = optimal_weights(return_estimators, clean_corr)
            return_clusters.append(windowed_return@op)

        return_cluster = np.concatenate(return_clusters, axis=1)
        corr = np.cov(windowed_return.T)
        clean_corr = compute_clean_correlation_matrix(corr, corr.shape[1], windowed_return.shape[0]) 
        clean_corr = np.array(clean_corr) 
        op = optimal_weights(return_cluster.mean(axis=0)[:, np.newaxis], clean_corr)
        return op


    
 
    def compiled_rolling(self, i):
        #for i in tqdm(windows):
        windowed_return = self.returns.iloc[i-self.window_size:i].to_numpy()
        #is_var = windowed_return.std(axis=0) > 0.0
        corr = np.cov(windowed_return.T)#[is_var, :][:, is_var]
        #corr[np.isnan(corr)] = 0.0
        #np.fill_diagonal(corr, 1)
        #clean_corr = np.zeros((self.returns.shape[1], self.returns.shape[1]))
        clean_corr = compute_clean_correlation_matrix(corr, corr.shape[1], windowed_return.shape[0]) 
        clean_corr = np.array(clean_corr) #* (np.array(clean_corr) >= 0.0001).astype(float)
        # print((a > 0.0001).sum())
        # j = 0
        # if  (1 - is_var).sum():
        #     for i in range(len(clean_corr)):
        #         if is_var[i]:
        #             clean_corr[i, is_var] = a[j, :]
        #             j = j + 1
        #     np.fill_diagonal(clean_corr, 1)
        # else:
        # clean_corr = a

        # print(clean_corr, (clean_corr >= 0.0001).astype(float))
        #clean_corr[is_var, :][:, is_var] = compute_clean_correlation_matrix(corr, corr.shape[1], windowed_return.shape[0])
        #np.fill_diagonal(clean_corr, 1)
        return_estimators = np.mean(windowed_return, axis=0).T
        print(np.linalg.norm(clean_corr))
        op = optimal_weights_constrained(return_estimators, clean_corr, self.risk_aversion, self.C)
        return np.array(op).reshape(-1)
        #return [self.rolling_optimal_weights(i, risk_aversion=risk_aversion) for i in windows]
            
        

    def get_rolling_cumulative_return(self, parallel=False, cluster=False):
        windows = list(range(self.window_size, len(self.returns)-self.stride, self.stride))
        rolling_comp = self.clustered_rolling if cluster else self.compiled_rolling
        print(windows[-1], rolling_comp)
        if parallel:
            weights = process_map(rolling_comp, windows, max_workers=os.cpu_count()//4, chunksize = 25)
        else:
            weights = [rolling_comp(i) for i in tqdm(windows, position=0, leave=True)]

        self.weights = [weights[0]]*(self.window_size + 1 + self.stride) + [weight for weight in weights[1:] for _ in range(self.stride)]
        print(len(self.weights))
        self.weights = np.array(self.weights)[:self.returns.shape[0]]
        
        return pd.DataFrame(((self.returns.to_numpy() + 1)*self.weights).sum(axis=1), index=self.returns.index).cumprod()

    



    
             
