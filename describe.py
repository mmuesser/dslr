#!/usr/bin/env python3

# num_data qui contient les colonnes numériques
# pour chaque colonne calculer count, mean, std, min, 25%, 50%, 75%, max
# pour chaque element d'index X dans une ligne, calculer sa taille et comparer avec le reste des index X
# print premier element de chaque ligne normalement
# puis pour chaque élément à partir du deuxième, print taille_max - taille_elem + 2 espaces puis afficher elem


import pandas, math
from collections.abc import Callable

from config import *


datas = pandas.read_csv(CSV_DIR + TRAIN_FILE)

num_data = datas.select_dtypes(include=[float, int])
num_data.drop(columns=["Index"], inplace=True)


def cumulate_data(l: Callable[[float], float], col: list) -> float:
    it = (l(x) for x in col if x is not None and not math.isnan(x))
    total = 0
    for x in it:
        total += x

    return total


def count_data(col: list) -> float:
    return cumulate_data(lambda x: 1, col)


def mean_data(col: list) -> float:
    return cumulate_data(lambda x: x, col) / count_data(col)


def std_data(col: list) -> float:
    count = count_data(col)
    mean = mean_data(col)
    somme = cumulate_data(lambda x: (x - mean) ** 2, col)

    return math.sqrt(somme / (count - 1))


def min_data(col: list) -> float:
    min_value = math.nan
    it = (x for x in col if x is not None and not math.isnan(x))

    for x in it:
        if math.isnan(min_value) or x < min_value:
            min_value = x
    return min_value


def quantile_data(col: list, q: float) -> float:
    values = [x for x in col if x is not None and not math.isnan(x)]
    values.sort()
    n = len(values)
    if n == 0:
        return math.nan
    index = (n - 1) * q
    lower_index = int(index)
    upper_index = lower_index + 1
    if upper_index >= n:
        return values[lower_index]
    weight = index - lower_index
    return values[lower_index] * (1 - weight) + values[upper_index] * weight


def describe_data(num_data) -> dict[str, list]:
    analyzed: dict[str, list] = {}

    for col in num_data.columns:
        analyzed[col] = [
            count_data(num_data[col]),
            mean_data(num_data[col]),
            std_data(num_data[col]),
            quantile_data(num_data[col], 0),
            quantile_data(num_data[col], 0.25),
            quantile_data(num_data[col], 0.5),
            quantile_data(num_data[col], 0.75),
            quantile_data(num_data[col], 1),
        ]
    return analyzed


def print_data(data: dict):

    df = pandas.DataFrame(
        data, index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    )
    print(df)


if __name__ == "__main__":
    analyzed = describe_data(num_data)
    print(num_data.describe())
    print_data(analyzed)
