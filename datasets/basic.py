import pandas as pd 
import numpy as np
import yfinance as yf
import os
import pandas as pd
from datasets.config import ROOT

RAW_FILE_PATH = ROOT + "/data/basic.csv"
CLEAN_FILE_PATH = ROOT + "/data/basic.parquet"

class Basic:
    """
    Simple dataset of 4 stocks with 10 years of historical data from yahoo finance.
    """

    def __init__(self) -> None:
        if not os.path.exists(CLEAN_FILE_PATH):

            if not os.path.exists(RAW_FILE_PATH):
                self.download()

            self.clean()
        self.df = pd.read_parquet(CLEAN_FILE_PATH)
        self.values()
    
    def download(self):
        tickers = sorted(['AAPL', 'VZ', 'F', 'COKE'])
        start = "2010-01-01"
        end = "2019-12-31"
        raw_stocks = yf.download(tickers,start,end).stack().reset_index()
        raw_stocks.to_csv(RAW_FILE_PATH, index=False)

    def clean(self):
        df = pd.read_csv(RAW_FILE_PATH)

        # Rename olumns
        df = df.rename(columns={x:x.replace(" ","_").lower() for x in df.columns})

        # Keep columns
        keep_columns = ['date', 'ticker', 'close', 'adj_close']
        df = df[keep_columns]

        # Create ret column
        df['ret'] = df.groupby('ticker')['adj_close'].pct_change()

        # Sort dataframe
        df = df.sort_values(by=['ticker','date'])

        # Reindex
        df = df.reset_index(drop=True)

        # Save
        df.to_parquet(CLEAN_FILE_PATH)

    def values(self):
        df = pd.read_parquet(CLEAN_FILE_PATH)
        self.prices = df.groupby('ticker').agg({'close': 'last'}).to_numpy().T[0]
        self.names = df['ticker'].unique()
        self.expected_returns = df.groupby("ticker")['ret'].mean().to_numpy()
        self.covariance_matrix = df.pivot(index='date', values='ret', columns='ticker').fillna(0).cov().to_numpy()





