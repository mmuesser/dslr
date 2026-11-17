#!usr/bin/env python3

import sys
import math
import numpy as np
import pandas as pd
from config import TRAIN_FILE, CSV_DIR, WEIGHTS_FILE

def mean(obj):
    total : float = 0
    for elem in obj:
        total += elem
    return total / len(obj)

def std(obj, means):
    sum_square = sum((obj - means) ** 2)
    variance = sum_square / len(obj)
    return math.sqrt(variance)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def cost_function(y: float, p: float) -> float:
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)

    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def gradient_descent(
    weights: np.ndarray,
    bias: float,
    learning_rate: float,
    errors: np.ndarray,
    notes: np.ndarray
) -> tuple[np.ndarray, float]:

    gradients = mean(errors[:, np.newaxis] * notes)
    bias_gradient = mean(errors)

    weights -= learning_rate * gradients
    bias -= learning_rate * bias_gradient

    return weights, bias


def log_reg(
    weights: np.ndarray,
    notes: np.ndarray,
    house: str,
    students_houses: np.ndarray,
    bias: float = 0.0
) -> tuple[np.ndarray, float]:

    learning_rate = 0.1

    y = (students_houses == house).astype(float)

    for _ in range(1000):
        predictions = sigmoid(notes @ weights + bias)

        loss = mean(cost_function(y, predictions))

        errors = predictions - y

        weights, bias = gradient_descent(
            weights,
            bias,
            learning_rate,
            errors,
            notes
        )

    return weights, bias



def main() -> None:
    data = pd.read_csv(CSV_DIR + TRAIN_FILE)

    numeric_data = (
        data.drop(columns=["Index", "Hogwarts House"])
        .select_dtypes(include="number")
    )

    means = numeric_data.apply(lambda column: mean(column.dropna()), axis=0)
    numeric_data = numeric_data.fillna(means)

    stds = numeric_data.apply(lambda column : std(column, means[column.name]), axis=0)

    scales = stds.mask(stds == 0, 1.0)

    notes = ((numeric_data - means) / scales).to_numpy(dtype=float)

    students_houses = data["Hogwarts House"].to_numpy()

    houses = [
        "Gryffindor",
        "Hufflepuff",
        "Ravenclaw",
        "Slytherin"
    ]

    features = numeric_data.columns.tolist()
    nb_features = len(features)

    all_weights = {}

    for house in houses:
        weights = np.zeros(nb_features)

        weights, bias = log_reg(
            weights,
            notes,
            house,
            students_houses
        )

        all_weights[house] = np.append(weights, bias)

        print(house, all_weights[house])

    model_data = pd.DataFrame.from_dict(
        all_weights,
        orient="index",
        columns=features + ["Bias"]
    )

    for feature in features:
        model_data[f"mean::{feature}"] = means[feature]
        model_data[f"scale::{feature}"] = scales[feature]

    model_data.index.name = "Hogwarts House"

    model_data.to_csv(CSV_DIR + WEIGHTS_FILE)


if __name__ == "__main__":
    main()
