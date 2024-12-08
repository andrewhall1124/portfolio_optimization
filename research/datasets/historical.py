import os

import gdown
import numpy as np
import pandas as pd

from research.interfaces import AssetData

from .config import ROOT

DATA_DIR = ROOT + "/data"
RAW_FILE_PATH = DATA_DIR + "/dsf.parquet"
CLEAN_FILE_PATH = DATA_DIR + "/crsp_daily_clean.parquet"


class Historical:
    """
    Historical Dataset that can sample random days.
    """

    def __init__(self) -> None:
        if not DATA_DIR:
            print("No data directory in root!")
            return

        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        if not os.path.exists(CLEAN_FILE_PATH):
            if not os.path.exists(RAW_FILE_PATH):
                print("DOWNLOADING RAW FILE")
                self.download()

            print("CLEANING RAW FILE")
            self.clean()

        print("LOADING CLEAN FILE")
        self.df = pd.read_parquet(CLEAN_FILE_PATH)

    def download(self) -> None:
        file_id = "1Zfj5XiBnf87zvYwM-l944AFRaWKfXJbo"
        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(url, RAW_FILE_PATH, quiet=False)

    def clean(self) -> None:
        # Raw file
        df = pd.read_parquet(RAW_FILE_PATH)

        # Filters
        df = df.query("10 <= shrcd <= 11")  # Stocks
        df = df.query("1 <= exchcd <= 3")  # NYSE, AMEX, NASDAQ

        # Keep only necessary columns
        keep_columns = [
            "permno",
            "date",
            "shrcd",
            "exchcd",
            "ticker",
            "shrout",
            "vol",
            "prc",
            "ret",
        ]
        df = df[keep_columns]

        # Fix ret and prc variables
        df["prc"] = abs(df["prc"])  # Stocks with unavailable prc data are negated (bid-ask spread)

        # Cast types
        df["ret"] = pd.to_numeric(df["ret"])
        df["date"] = pd.to_datetime(df["date"])

        # Sort values
        df = df.sort_values(by=["permno", "date"])

        # Drop duplicates
        df = df.drop_duplicates(subset=["permno", "date"])

        # Reset index
        df = df.reset_index(drop=True)

        df.to_parquet(CLEAN_FILE_PATH)

    def sample(
        self, n_assets: int, random_state: int = 47, constant_covar: bool = True
    ) -> AssetData:
        # Sample a random date
        random_date = self.df["date"].sample(1, random_state=random_state).iloc[0]
        day = self.df[self.df["date"] == random_date].copy().reset_index(drop=True)

        # Sample indices to ensure tickers and prices align
        sampled_indices = day.sample(n=n_assets, random_state=random_state).index
        tickers = day.loc[sampled_indices, "ticker"].to_numpy()
        prices = day.loc[sampled_indices, "prc"].to_numpy()

        # Create AssetData instance
        asset_data = AssetData(
            tickers,
            prices,
            self._expected_returns(n_assets, 47 if constant_covar else random_state),
            self._covariance_matrix(n_assets, 47 if constant_covar else random_state),
        )
        return asset_data

    def _expected_returns(self, n_assets: int, random_state: int = 47) -> np.ndarray:
        np.random.seed(random_state)
        return np.random.rand(n_assets)  # Random values between 0 and 1

    def _covariance_matrix(self, n_assets: int, random_state: int = 47) -> np.ndarray:
        np.random.seed(random_state)
        random_matrix = np.random.rand(n_assets, n_assets)
        cov_matrix = np.dot(random_matrix, random_matrix.T)
        cov_matrix = cov_matrix / np.max(cov_matrix)  # Normalize
        return cov_matrix
