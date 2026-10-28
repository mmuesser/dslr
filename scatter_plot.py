#!usr/bin/env python3

import matplotlib.pyplot as plt
from school import School
from config import *


hogwarts = School()
hogwarts.insertData(CSV_DIR + TRAIN_FILE)


def setAxe(ax, course_1, course_2, school):
    for house in school.houses.values():
        ax.scatter(
            house.markbook[course_1],
            house.markbook[course_2],
            color=house.color,
            alpha=0.3,
            label=house.name,
        )

        ax.set_xlabel(course_1)
        ax.set_ylabel(course_2)
        # ax.set_title(f"{x} vs {y}")


for course_1 in hogwarts.courses:
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(course_1, size="xx-large")
    axes = axes.flatten()
    for ax, course_2 in zip(axes, hogwarts.courses):
        setAxe(ax, course_1, course_2, hogwarts)

        legend_ax = axes[15]
        legend_ax.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        legend_ax.legend(handles, labels, loc="center", frameon=False)

    for i in range(len(hogwarts.courses), len(axes) - 1):
        fig.delaxes(axes[i])

    fig.tight_layout()
    plt.show()
