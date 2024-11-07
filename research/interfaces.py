from dataclasses import dataclass
import numpy as np


@dataclass
class AssetData:
    """
    A dataclass to represent asset data, including:
    - Names of the assets
    - Prices of the assets
    - Expected returns for the assets
    - Covariance matrix of the asset returns
    """
    names: np.ndarray[str]
    prices: np.ndarray[float]
    expected_returns: np.ndarray[float]
    covariance_matrix: np.ndarray[np.ndarray[float]]