import pandas as pd
import numpy as np
import pandas as pd

from research.interfaces import AssetData


class Synthetic:

    def __init__(self, price_mean, price_std, n_assets) -> None:
        self.price_mean = price_mean
        self.price_std = price_std
        self.n_assets = n_assets

        self.generate()


    def generate(self):

        # Generate synthetic stock prices
        prices = np.round(np.random.normal(self.price_mean, self.price_std, self.n_assets),2)

        # Create a DataFrame with synthetic stock prices
        stock_data = pd.DataFrame({
            'Stock': [f'Stock_{i+1}' for i in range(self.n_assets)],
            'Price': prices
        })

        # Display the first few rows
        print(stock_data)


    def clean(self):
        pass

Synthetic(100,10000,20)
