#!usr/bin/env python3

import sys
import numpy as np
import pandas as pd

from new_logreg_train import sigmoid


def main() -> None:
    # Chargement des données à prédire
    data = pd.read_csv(sys.argv[1])

    indices = data["Index"]

    # Chargement du modèle
    model = pd.read_csv(sys.argv[2], index_col=0)

    houses = [
        "Gryffindor",
        "Hufflepuff",
        "Ravenclaw",
        "Slytherin"
    ]

    # Identification des colonnes correspondant aux matières
    features = [
        column for column in model.columns
        if column != "Bias"
        and not column.startswith(("mean::", "scale::"))
    ]

    # Récupération des paramètres de normalisation
    means = model.iloc[0][
        [f"mean::{feature}" for feature in features]
    ].to_numpy(dtype=float)

    scales = model.iloc[0][
        [f"scale::{feature}" for feature in features]
    ].to_numpy(dtype=float)

    # Récupération des poids et des biais
    weights = model.loc[houses, features].to_numpy(dtype=float)
    biases = model.loc[houses, "Bias"].to_numpy(dtype=float)

    # Extraction des notes
    numeric_data = data[features]

    # Remplacement des valeurs manquantes avec les moyennes d'entraînement
    numeric_data = numeric_data.fillna(
        pd.Series(means, index=features)
    )

    notes = numeric_data.to_numpy(dtype=float)

    # Normalisation avec les paramètres d'entraînement
    notes = (notes - means) / scales

    # Prédictions vectorisées
    probabilities = sigmoid(notes @ weights.T + biases)

    predictions = np.array(houses)[
        np.argmax(probabilities, axis=1)
    ]

    # Export des prédictions
    df = pd.DataFrame({
        "Index": indices,
        "Hogwarts House": predictions
    })

    df.to_csv("houses.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main()
