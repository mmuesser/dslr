#!usr/bin/env python3

import sys
import numpy as np
import pandas as pd

from logreg_train import sigmoid


def main() -> None:
    data = pd.read_csv(sys.argv[1])

    indices = data["Index"]

    model = pd.read_csv(sys.argv[2], index_col=0)

    houses = [
        "Gryffindor",
        "Hufflepuff",
        "Ravenclaw",
        "Slytherin"
    ]

    features = [
        column for column in model.columns
        if column != "Bias"
        and not column.startswith(("mean::", "scale::"))
    ]

    means = model.iloc[0][
        [f"mean::{feature}" for feature in features]
    ].to_numpy(dtype=float)

    scales = model.iloc[0][
        [f"scale::{feature}" for feature in features]
    ].to_numpy(dtype=float)

    weights = model.loc[houses, features].to_numpy(dtype=float)
    biases = model.loc[houses, "Bias"].to_numpy(dtype=float)

    numeric_data = data[features]

    numeric_data = numeric_data.fillna(
        pd.Series(means, index=features)
    )

    notes = numeric_data.to_numpy(dtype=float)

    notes = (notes - means) / scales

    probabilities = sigmoid(notes @ weights.T + biases)

    predictions = np.array(houses)[
        np.argmax(probabilities, axis=1)
    ]

    df = pd.DataFrame({
        "Index": indices,
        "Hogwarts House": predictions
    })

    df.to_csv("houses.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main()
