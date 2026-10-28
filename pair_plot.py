#!usr/bin/env python3


import matplotlib.pyplot as plt
from school import School
from config import *


def crop_label(text: str) -> str:
    return (
        text
        if len(text) <= MAX_LEGEND_LENGTH
        else text[: MAX_LEGEND_LENGTH - 3] + "..."
    )


def make_histogram(ax, house):
    ax.hist(
        house.markbook[course_1],
        bins=42,
        alpha=0.45,
        label=house.name,
        facecolor=house.color,
    )


def make_scatter(ax, house):
    ax.scatter(
        house.markbook[course_1],
        house.markbook[course_2],
        color=house.color,
        alpha=0.5,
        s=4,
    )


def make_plot(ax, school):
    for house in school.houses.values():
        if i == j:
            make_histogram(ax, house)
        else:
            make_scatter(ax, house)


hogwarts = School()
hogwarts.insertData(CSV_DIR + TRAIN_FILE)
num_courses = len(hogwarts.courses)
fig, axes = plt.subplots(num_courses, num_courses, figsize=(30, 30))


for i, course_1 in enumerate(hogwarts.courses):
    for j, course_2 in enumerate(hogwarts.courses):
        ax = axes[i, j]
        make_plot(ax, hogwarts)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, i].set_title(crop_label(course_1), fontsize=12, pad=12)
    axes[i, 0].set_ylabel(
        crop_label(course_1), fontsize=12, rotation=0, labelpad=60, va="center"
    )

plt.show()
