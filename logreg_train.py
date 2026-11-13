#!usr/bin/env python3


import sys
import math
import numpy as np
import pandas as pd


def normalize_notes(notes: np.ndarray) -> np.ndarray:
    mean = notes.mean()
    std = notes.std()

    if std == 0:
        return notes

    return (notes - mean) / std


def classifier(weights: np.ndarray, notes: np.ndarray, bias: float) -> float:
    return np.dot(weights, notes) + bias


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

    gradients = np.mean(errors[:, np.newaxis] * notes, axis=0)
    bias_gradient = errors.mean()

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
    nb_students = len(notes)

    y = (students_houses == house).astype(float)

    for _ in range(1000):
        predictions = sigmoid(notes @ weights + bias)

        # Entropie croisée binaire (moyenne)
        loss = np.mean(cost_function(y, predictions))

        # Erreurs de prédiction
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
    data = pd.read_csv(sys.argv[1])

    numeric_data = (
        data.drop(columns=["Index", "Hogwarts House"])
        .select_dtypes(include="number")
    )

    # Calcul des paramètres de normalisation sur le dataset d'entraînement
    means = numeric_data.mean()
    numeric_data = numeric_data.fillna(means)

    stds = numeric_data.std(ddof=0)

    # Évite la division par zéro pour les matières constantes
    scales = stds.mask(stds == 0, 1.0)

    # Normalisation
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

    # Construction du DataFrame des poids et des biais
    model_data = pd.DataFrame.from_dict(
        all_weights,
        orient="index",
        columns=features + ["Bias"]
    )

    # Ajout des paramètres de normalisation
    for feature in features:
        model_data[f"mean::{feature}"] = means[feature]
        model_data[f"scale::{feature}"] = scales[feature]

    model_data.index.name = "Hogwarts House"

    model_data.to_csv("houses_weights.csv")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main()
