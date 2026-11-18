#!usr/bin/env python3

import math
import random
import numpy as np
import pandas as pd
from config import TRAIN_FILE, CSV_DIR, WEIGHTS_FILE
from loss_plot import TrainingStats


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


def cost_function(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)

    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def gradient_descent(
    weights: np.ndarray,
    bias: float,
    learning_rate: float,
    errors: np.ndarray,
    notes: np.ndarray,
) -> tuple[np.ndarray, float]:

    gradients = mean(errors[:, np.newaxis] * notes)
    bias_gradient = mean(errors)

    weights -= learning_rate * gradients
    bias -= learning_rate * bias_gradient

    return weights, bias

def inverse_time_decay(initial_lr, decay_rate, iteration):
    return initial_lr / (1 + decay_rate * iteration)


def log_reg(
    weights: np.ndarray,
    notes: np.ndarray,
    house: str,
    students_houses: np.ndarray,
    bias: float = 0.0,
) -> tuple[np.ndarray, float]:

    learning_rate = 0.1
    decay_rate = 0.5
    LOOPS = 10000
    stats = TrainingStats(LOOPS)
    y = (students_houses == house).astype(float)

    max = notes.shape[0]
    print("shape:", max)

    BATCH_SIZE = 1599

    #Nombre de batchs pour parcourir autant de valeurs qu'il y a dans notre set de donnees
    EPOCH = max / BATCH_SIZE

    print(EPOCH)
    for i in range(LOOPS):
        index_low = random.randrange(max - BATCH_SIZE)
        batch = notes[index_low : index_low + BATCH_SIZE, :]
        y_batch = y[index_low : index_low + BATCH_SIZE]
        predictions = sigmoid(batch @ weights + bias)

        loss = mean(cost_function(y_batch, predictions))

        stats.records[i] = loss
        errors = predictions - y_batch

        weights, bias = gradient_descent(
            weights, bias, learning_rate, errors, batch
        )
        # if BATCH_SIZE == 1 :
        #     learning_rate = inverse_time_decay(
        #     0.1,
        #     decay_rate,
        #     i)
        # else :
        #Toutes les 30 epoques on met le learning rate a jour
        if i % (EPOCH * 30) == 0:
            learning_rate = learning_rate * decay_rate

        # print(learning_rate)

    predictions = sigmoid(notes @ weights + bias)
    loss = mean(cost_function(y, predictions))
    print(loss)

    stats.plot()

    return weights, bias


def main() -> None:
    data = pd.read_csv(CSV_DIR + TRAIN_FILE)

    numeric_data = data.drop(columns=["Index", "Hogwarts House"]).select_dtypes(
        include="number"
    )

    means = numeric_data.apply(lambda column: mean(column.dropna()), axis=0)
    numeric_data = numeric_data.fillna(means)

    stds = numeric_data.apply(
        lambda column: std(column, means[column.name]), axis=0
    )

    scales = stds.mask(stds == 0, 1.0)

    notes = ((numeric_data - means) / scales).to_numpy(dtype=float)

    students_houses = data["Hogwarts House"].to_numpy()

    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

    features = numeric_data.columns.tolist()
    nb_features = len(features)

    all_weights = {}

    for house in houses:
        weights = np.zeros(nb_features)

        weights, bias = log_reg(weights, notes, house, students_houses)

        all_weights[house] = np.append(weights, bias)

        print(house, all_weights[house])

    model_data = pd.DataFrame.from_dict(
        all_weights, orient="index", columns=features + ["Bias"]
    )

    for feature in features:
        model_data[f"mean::{feature}"] = means[feature]
        model_data[f"scale::{feature}"] = scales[feature]

    model_data.index.name = "Hogwarts House"

    model_data.to_csv(CSV_DIR + WEIGHTS_FILE)


if __name__ == "__main__":
    main()
