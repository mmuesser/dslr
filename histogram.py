#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
from config import CSV_DIR, TRAIN_FILE

from school import School


def main():

    hogwarts = School()
    hogwarts.insertData(CSV_DIR + TRAIN_FILE)

    fig, axes = plt.subplots(3, 5, figsize=(20, 5))
    axes = axes.flatten()

    for i, course in enumerate(hogwarts.courses):
        ax = axes[i]
        bins = np.linspace(hogwarts.min_values[course], hogwarts.max_values[course], 42)

        for house in hogwarts.houses.values():
            ax.hist(
                house.markbook[course],
                bins=bins,
                alpha=0.55,
                facecolor=house.color,
                label=house.name,
                density=True,
            )

        ax.set_title(course, fontsize=8)
        ax.tick_params(axis="both", labelsize=7)

        if i == 0:
            ax.legend(fontsize=7)

    for i in range(len(hogwarts.courses), len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
