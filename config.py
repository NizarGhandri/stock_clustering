""" Contains configuration attributes used during preprocessing and training. """

import os






class Config():
    def __init__(self):
        super(Config, self).__init__()


def load_config():
    cfg = Config()
    
    """ 
    **************************************** Paths **************************************** 
    """
    cfg.data_path = os.path.join("data", "data_new.parquet")
    cfg.preprocessed_path = os.path.join("data", "preprocessed")
    cfg.save_path = os.path.join("models", "saved_models")
    cfg.data_dir = "data"
    """ 
    ************************************************************************************************
    """ 

    cfg.device = "cpu"

    cfg.companies = ['DOW', 'PEP', 'CVS', 'UNH', 'CL', 'DD', 'EXC', 'SLB', 'FDX', 'AA',
       'NKE', 'JNJ', 'VZ', 'BAX', 'GD', 'WMT', 'RTN', 'WY', 'LMT', 'BHI',
       'WMB', 'MCD', 'MDT', 'KFT', 'BMY', 'ETR', 'GE', 'COF', 'OXY',
       'ALL', 'BA', 'T', 'ABT', 'F', 'HNZ', 'FCX', 'HAL', 'UTX', 'KO',
       'LOW', 'EMR', 'MMM', 'SO', 'WFC', 'USB', 'HPQ', 'IBM', 'COP',
       'UNP', 'C', 'TWX', 'EMC', 'MET', 'GS', 'PG', 'DIS', 'CAT', 'WAG',
       'HD', 'BK', 'HON', 'CVX', 'NSC', 'MON', 'TXN', 'AEP', 'AVP', 'JPM',
       'BAC', 'MRK', 'XRX', 'TGT', 'XOM', 'PFE', 'UPS', 'AXP', 'APA']

    cfg.dates = ("2021-01-01", "2023-01-01")
    cfg.test_size = 100

    cfg.link = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    cfg.params = {"start": cfg.dates[0], "end": cfg.dates[1], "group_by" : 'column', "auto_adjust" : True,  "prepost" : True, "threads" : True, "proxy" : None}

    
    return cfg