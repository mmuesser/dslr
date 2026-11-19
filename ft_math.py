import math
import numpy as np
import pandas as pd


def mean(obj: pd.Series | np.ndarray) -> float:
    total: float = 0
    for elem in obj:
        total += elem
    return total / len(obj)


def std(obj: pd.Series, means: float) -> float:
    if len(obj > 1):
        sum_square = sum((obj - means) ** 2)
        variance = sum_square / (len(obj) - 1)
        return math.sqrt(variance)
    return 1


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))
