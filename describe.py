#!/usr/bin/env python3

import pandas as pd

from config import *


from subject import Subject


class Describe:
    def __init__(self):

        datas = pd.read_csv(CSV_DIR + TRAIN_FILE)
        self.num_data = datas.select_dtypes(include=[float, int])
        self.num_data.drop(columns=["Index"], inplace=True)

    def describe(self):
        self.analyzed: dict[str, list] = {}

        for col in self.num_data.columns:
            subject = Subject(self.num_data[col])  # type: ignore[arg-type]
            subject.analyse()
            self.analyzed[col] = subject.getStats()

    def print(self):

        df = pd.DataFrame(
            self.analyzed,
            index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        )
        print(df)


if __name__ == "__main__":
    desc = Describe()

    print(desc.num_data.describe())

    desc.describe()
    desc.print()
