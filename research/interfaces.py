from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class AssetData:
    """
    A dataclass to represent asset data, including:
    - Names of the assets
    - Prices of the assets
    - Expected returns for the assets
    - Covariance matrix of the asset returns
    """

    names: NDArray[np.str_]  # Array of strings (asset names)
    prices: NDArray[np.float64]  # Array of floats (asset prices)
    expected_returns: NDArray[np.float64]  # Array of floats (expected returns)
    covariance_matrix: NDArray[np.float64]  # 2D array of floats (covariance matrix)
