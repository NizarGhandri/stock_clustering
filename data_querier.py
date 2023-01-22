
import pandas as pd
import yfinance as yf
import logging
import wrds


TABLE_NUM = 0
STOCK_COLUMN = "Symbol"



class DataQuerierYF: 

    def __init__ (self, cfg, load_on_init=True, save=True, **kwargs):
        self.cfg = cfg
        self.company_list = pd.read_html(self.cfg.link)[TABLE_NUM][STOCK_COLUMN].tolist()
        logging.info("Attempting load for %d stocks from %s".format([len(self.company_list), self.cfg.link]))
        self.from_params = len(kwargs)
        self.save_data = save
        self.kwargs = kwargs
        if (load_on_init):
            self()
            


    
    def __call__(self):
        self.__get_tickers(self.kwargs)
        if(self.save_data):
            self.__save()
        logging.info("Loaded %d stocks from %s".format([len(self.stock_hist.columns), self.cfg.link]))
        return self.stock_hist


    def __save(self): 
        self.stock_hist.to_parquet(self.cfg.data_path)
        
        


    def __get_tickers(self, kwargs):
        if (self.from_params):
            self.stock_hist = yf.download(  
                                        tickers = self.company_list,
                                        **kwargs
                                    )
        else:
            self.stock_hist = yf.download(  
                                        tickers = self.company_list,
                                        **self.cfg.params
                                    )
        return self.stock_hist


        
class SharesOutStandingQuerier: 

    def __init__ (self, company_permcos_name, dates, first_connection=False, username="ghandri", **kwargs): 
        self.company_permcos_name = pd.DataFrame({"ticker": company_permcos_name})
        self.dates = dates
        self.db=wrds.Connection(wrds_username=username)
        if (first_connection):
            self.db.create_pgpass_file()
        self.__get_permcos()
        print(self.permcos)
        self.__query_company()
        self.db.close()

        
    
    def __get_permcos(self):
            query_res = self.db.raw_sql("select  permco , ticker, namedt, nameenddt,comnam "
                                "from crsp.stocknames "
                                "where namedt <'2009-01-01' and nameenddt >'2009-01-01'")
            self.permcos = query_res[["permco", "ticker"]].merge(self.company_permcos_name, on="ticker")

        
    def __call__(self):
        return 


    def __query_company(self,):
        params= {'permco': tuple(self.permcos["permco"]), 'low': self.dates[0], 'high': self.dates[1]}
        self.sharesout =self.db.raw_sql("select date,permco,shrout "
           "from crsp.dsf "
           "where permco in {permco} "
           "and date >= '{low}'"
            "and date <= '{high}'".format(**params))
        return self.sharesout