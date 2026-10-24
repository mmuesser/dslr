#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
from config import CSV_DIR, TRAIN_FILE

from school import School


def main():

    hogwarts = School()
    hogwarts.insertData(CSV_DIR + TRAIN_FILE)

    fig, axes = plt.subplots(4, 4, figsize=(16, 8))
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
            )

        ax.set_title(course, fontsize=8)
        ax.tick_params(axis="both", labelsize=7)

        legend_ax = axes[15]
        legend_ax.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc="center", frameon=False)

    for i in range(len(hogwarts.courses), len(axes) - 1):
        fig.delaxes(axes[i])

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
