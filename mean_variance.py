import pandas as pd
import numpy as np
import os
from utils import compute_clean_correlation_matrix
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from numba import jit
from clustering import LouvainClustering
from dtaidistance import dtw


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


    def __init__(self, cfg, risk_aversion=2, window_size=367, stride=10, C = 0.1, clustering=None, type_dist="corr"):  
        self.cfg = cfg 
        self.risk_aversion = risk_aversion
        self.data = self.load_data()
        self.window_size = len(self.data) if window_size is None else window_size
        self.returns = (self.data.diff()/self.data).fillna(0)
        self.stride = stride
        self.C = C
        self.clustering = clustering
        self.__generated_clusters = []
        self.type = type_dist




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
        op = optimal_weights(return_estimators, clean_corr, risk_aversion)
        return np.array(op).reshape(-1)

    
    def get_clustering_corr(self, i):

        if self.clustering is None:
            windowed_return = self.returns.iloc[i-self.window_size:i].to_numpy()
            is_var = windowed_return.std(axis=0) > 0.0
            corr = np.corrcoef(windowed_return[:, is_var].T)
            np.fill_diagonal(corr, 1)
            n = self.returns.shape[1]
            clean_corr = np.zeros((n , n))
            a = compute_clean_correlation_matrix(corr, corr.shape[1], windowed_return.shape[0])
            j = 0
            if  (1 - is_var).sum():
                for i in range(len(clean_corr)):
                    if is_var[i]:
                        clean_corr[i, is_var] = a[j, :]
                        j = j + 1
                np.fill_diagonal(clean_corr, 1)
            else:
                clean_corr = a
            return LouvainClustering(clean_corr).cluster()
        return self.clustering[i]


    def get_clustering_dtw(self, i):

        if self.clustering is None:
            windowed_return = self.returns.iloc[i-self.window_size:i].to_numpy()
            clean_corr = self.pairwise_dtw(windowed_return)
            return LouvainClustering(clean_corr).cluster()
        return self.clustering[i]

    
    def clustered_rolling(self, i):

        print(self.type, self.type == "corr")

        cluster = self.get_clustering_corr if self.type == "corr" else self.get_clustering_dtw
        clusters = list(map(list, cluster(i)))
        return_clusters = []
        aggregation_weights = []
        for cluster in clusters:
            windowed_return = self.returns.iloc[i-self.window_size:i, cluster].to_numpy()
            if len(cluster) > 1:
                corr = np.cov(windowed_return.T)
                clean_corr = compute_clean_correlation_matrix(corr, corr.shape[1], windowed_return.shape[0]) 
                clean_corr = np.array(clean_corr) 
                return_estimators = np.mean(windowed_return, axis=0).T
                op = optimal_weights_constrained(return_estimators, clean_corr, self.risk_aversion, 0.1)
                aggregation_weights.append(op)
                return_clusters.append((windowed_return@op)[:, np.newaxis])
            else:
                aggregation_weights.append(np.ones((1,1)))
                return_clusters.append(windowed_return)

        
        return_clusters = np.concatenate(return_clusters, axis=1)
        corr = np.cov(return_clusters.T)
        clean_corr = compute_clean_correlation_matrix(corr, corr.shape[1], windowed_return.shape[0]) 
        op = optimal_weights_constrained(return_clusters.mean(axis=0)[:, np.newaxis], clean_corr, self.risk_aversion, 0.01)
        return op, aggregation_weights, clusters


    
 
    def compiled_rolling(self, i):
        windowed_return = self.returns.iloc[i-self.window_size:i].to_numpy()
        corr = np.cov(windowed_return.T)
        clean_corr = compute_clean_correlation_matrix(corr, corr.shape[1], windowed_return.shape[0]) 
        clean_corr = np.array(clean_corr) 
        return_estimators = np.mean(windowed_return, axis=0).T
        print(np.linalg.norm(clean_corr))
        op = optimal_weights_constrained(return_estimators, clean_corr, self.risk_aversion, self.C)
        return np.array(op).reshape(-1)
       
            
        

    def get_rolling_cumulative_return(self, parallel=False):
        windows = list(range(self.window_size, len(self.returns)-self.stride+1, self.stride))
        #rolling_comp = self.clustered_rolling if cluster else self.compiled_rolling
        #print(windows[-1], self.compiled_rolling)
        if parallel:
            weights = process_map(self.compiled_rolling, windows, max_workers=os.cpu_count()//4, chunksize = 25)
        else:
            weights = [self.compiled_rolling(i) for i in tqdm(windows, position=0, leave=True)]

        self.weights = [weight for weight in weights for _ in range(self.stride)] #[weights[0]]*(self.window_size + self.stride) + 
        #print(len(self.weights))
        self.weights = np.array(self.weights)[:self.returns.shape[0]]
        ret = (self.returns.to_numpy() + 1)[self.window_size:]    
        return pd.DataFrame((ret*self.weights).sum(axis=1), index=self.returns.index[self.window_size:]).cumprod()


    def get_clustered_cumulative_return(self, parallel=False):
        windows = list(range(self.window_size, len(self.returns)-self.stride+1, self.stride))
        if parallel:
            weights = process_map(self.clustered_rolling, windows, max_workers=os.cpu_count()//4, chunksize = 25)
        else:
            weights = [self.clustered_rolling(i) for i in tqdm(windows, position=0, leave=True)]

        self.weights, self.aggregation_weights, self.__generated_clusters = list(zip(*weights))
        cum_ret = self.returns.to_numpy() + 1
        cum_return = []
        print(len(self.__generated_clusters), len(self.aggregation_weights), len(self.weights), len(windows))
        for window, time_clusters, intra_cluster_weights, inter_cluster_weights in zip(windows, self.__generated_clusters, self.aggregation_weights, self.weights):
            
            X = cum_ret[window: window+self.stride]
            cum_return.append(sum([((X[:, cluster] @ w_inter) * w_intra).reshape(-1, 1) for cluster, w_inter, w_intra in zip(time_clusters, intra_cluster_weights, inter_cluster_weights)]))
            
        

        print(len(cum_return))
        print(cum_return[-1])
        print(cum_return[-1].shape)

        cum_return = np.concatenate(cum_return) #[_ for elem in cum_return for _ in elem]
        print(np.array(cum_return).shape)
        return pd.DataFrame(np.array(cum_return), index=self.returns.index[self.window_size:]).cumprod()


    def pairwise_dtw(self, time_period):
        is_var = time_period.std(axis=0) > 0.0
        ds = dtw.distance_matrix(time_period[:, is_var].T, only_triu=True, use_c=True, parallel=True)
        np.fill_diagonal(ds, 1)
        ds[ds == np.inf] = 0
        #ds_ = np.copy(ds)
        ds = ds + ds.T
        ds = 1/ds
        np.fill_diagonal(ds, 0)
        #np.corrcoef(.T)[is_var, :][:, is_var]
        #corr[np.isnan(corr)] = 0.0
        n = self.returns.shape[1]
        clean_corr = np.zeros((n, n))
        a = ds
        j = 0
        if  (1 - is_var).sum():
            for i in range(len(clean_corr)):
                if is_var[i]:
                    clean_corr[i, is_var] = a[j, :]
                    j = j + 1
            np.fill_diagonal(clean_corr, 1)
        else:
            clean_corr = a

        return clean_corr

    



    
             
