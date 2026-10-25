import pandas as pd
import math
from collections.abc import Callable


class Subject:
    def __init__(self, marks: pd.DataFrame):
        self.marks = marks.dropna()
        self.marks.sort_values(inplace=True)  # type: ignore[arg-type]
        self.marks.reset_index(inplace=True, drop=True)

    def analyse(self):
        self.count = len(self.marks)
        self.mean = self.cumulate(lambda x: x) / self.count
        sum_square = self.cumulate(lambda x: (x - self.mean) ** 2)
        self.std = math.sqrt(sum_square / (self.count - 1))
        self.min = self.marks.iloc[0]
        self.max = self.marks.iloc[-1]
        self.q1 = self.quantile(0.25)
        self.median = self.quantile(0.5)
        self.q3 = self.quantile(0.75)

    def cumulate(self, l: Callable[[float], float]) -> float:
        total = 0
        for x in (l(x) for x in self.marks):
            total += x
        return total

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
        ]
