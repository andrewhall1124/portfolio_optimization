import numpy as np

from research.interfaces import AssetData


class Synthetic:

    def __init__(self, price_mean: float, price_std: float, n_assets: int) -> None:
        self.price_mean = price_mean
        self.price_std = price_std
        self.n_assets = n_assets

        self.generate()

    def generate(self) -> None:

        prices = np.round(abs(np.random.normal(self.price_mean, self.price_std, self.n_assets)), 2)

        names = np.array([f"stock_{i+1}" for i in range(self.n_assets)])

        expected_returns = np.ones(self.n_assets) / 10

        covariance_matrix = np.full((self.n_assets, self.n_assets), 0.1)
        np.fill_diagonal(covariance_matrix, 0.2)

        self.asset_data = AssetData(
            names=names,
            prices=prices,
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
        )
