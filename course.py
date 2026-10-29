import pandas as pd
import math
from collections.abc import Callable


class Course:
    def __init__(self, marks: pd.DataFrame):
        self.marks = marks.dropna()
        self.marks.sort_values(inplace=True)  # type: ignore[arg-type]
        self.marks.reset_index(inplace=True, drop=True)

        self.count = math.nan
        self.mean = math.nan
        self.sum_square = math.nan
        self.variance = math.nan
        self.std = math.nan
        self.min = math.nan
        self.max = math.nan
        self.q1 = math.nan
        self.median = math.nan
        self.q3 = math.nan
        self.kurtosis = math.nan
        self.skewness = math.nan

    def analyse(self):
        self.count = len(self.marks)
        self.mean = self.cumulate(lambda x: x) / self.count
        self.sum_square = self.cumulate(lambda x: (x - self.mean) ** 2)
        self.variance = self.sum_square / (self.count - 1)
        self.std = math.sqrt(self.variance)
        self.min = self.marks.iloc[0]
        self.max = self.marks.iloc[-1]
        self.q1 = self.quantile(0.25)
        self.median = self.quantile(0.5)
        self.q3 = self.quantile(0.75)
        self.computeKurtosisAndSkewness()

    def cumulate(self, l: Callable[[float], float]) -> float:
        total = 0
        for x in (l(x) for x in self.marks):
            total += x
        return total

    def computeKurtosisAndSkewness(self):
        sum_quad = self.cumulate(lambda x: ((x - self.mean) / self.std) ** 4)
        sum_cub = self.cumulate(lambda x: ((x - self.mean) / self.std) ** 3)
        n = self.count
        self.kurtosis = ((n * (n + 1) / (n - 1)) * sum_quad - 3 * (n - 1) ** 2) / (
            (n - 2) * (n - 3)
        )

        self.skewness = n / ((n - 1) * (n - 2)) * sum_cub

    def quantile(self, q: float) -> float:
        if self.count == 0:
            return math.nan
        index = (self.count - 1) * q
        lower_index = int(index)
        upper_index = lower_index + 1
        if upper_index >= self.count:
            return self.marks.iloc[lower_index]  # type: ignore[arg-type]
        weight = index - lower_index
        return (
            self.marks.iloc[lower_index] * (1 - weight)
            + self.marks.iloc[upper_index] * weight
        )

    def getStats(self):
        return [
            self.count,
            self.mean,
            self.std,
            self.min,
            self.q1,
            self.median,
            self.q3,
            self.max,
            self.kurtosis,
            self.skewness,
        ]
