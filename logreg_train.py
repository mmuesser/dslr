#!usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from config import TRAIN_FILE, CSV_DIR, WEIGHTS_FILE
from school import School
from logreg import Batch, MiniBatch, Stochastic
import ft_math as fm


def main(method: str) -> None:
    data = pd.read_csv(CSV_DIR + TRAIN_FILE)

    numeric_data = data.drop(columns=["Index", "Hogwarts House"]).select_dtypes(
        include="number"
    )

    means = numeric_data.apply(lambda column: fm.mean(column.dropna()), axis=0)
    numeric_data = numeric_data.fillna(means)

    stds = numeric_data.apply(
        lambda column: fm.std(column, means[column.name]), axis=0
    )

    scales = stds.mask(stds == 0, 1.0)

    notes = ((numeric_data - means) / scales).to_numpy(dtype=float)
    students_houses = data["Hogwarts House"].to_numpy()

    hogwarts = School()
    hogwarts.setCourses(numeric_data)

    all_weights = {}

    for house in hogwarts.houses:
        logreg = None
        if method == "sgd":
            logreg = Stochastic(notes, house, students_houses)
        elif method == "minibatch":
            logreg = MiniBatch(notes, house, students_houses)
        elif method == "batch":
            logreg = Batch(notes, house, students_houses)
        else:
            print(
                "Invalid Method name. Expected: batch or minibatch or sgd",
                file=sys.stderr,
            )
            exit(1)

        weights, bias = logreg.predict()

        all_weights[house] = np.append(weights, bias)

        print(house, all_weights[house])

    model_data = pd.DataFrame.from_dict(
        all_weights, orient="index", columns=hogwarts.courses + ["Bias"]
    )

    for feature in hogwarts.courses:
        model_data[f"mean::{feature}"] = means[feature]
        model_data[f"scale::{feature}"] = scales[feature]

    model_data.index.name = "Hogwarts House"

    model_data.to_csv(CSV_DIR + WEIGHTS_FILE)


if __name__ == "__main__":
    method = "batch"
    if len(sys.argv) > 1:
        method = sys.argv[1]
    raise SystemExit(main(method))
