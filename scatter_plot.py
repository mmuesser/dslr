#!usr/bin/env python3

import pandas, math
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from describe import describe_data, print_data


datas = pandas.read_csv("dataset_train.csv")


def split_data(datas) -> dict:
	features : dict[str, dict[str, list]] = {}
	num_data = datas.select_dtypes(include=[float, int])

	for col in num_data.columns[1:]:
		features[col] = {"Gryffindor" : [], "Hufflepuff" : [], "Ravenclaw" : [], "Slytherin" : []}
		for _, raw in datas.iterrows():
			if raw["Hogwarts House"] == "Gryffindor":
				features[col][raw["Hogwarts House"]].append(raw.get(col))
			elif raw["Hogwarts House"] == "Hufflepuff":
						features[col][raw["Hogwarts House"]].append(raw.get(col))
			elif raw["Hogwarts House"] == "Ravenclaw":
						features[col][raw["Hogwarts House"]].append(raw.get(col))
			elif raw["Hogwarts House"] == "Slytherin":
						features[col][raw["Hogwarts House"]].append(raw.get(col))
	return features

features = split_data(datas)
pairs = list(combinations(features, 2))

graphs_per_figure = 13

for start in range(0, len(pairs), graphs_per_figure):
	batch = pairs[start:start + graphs_per_figure]
	fig, axes = plt.subplots(4, 4, figsize=(16, 12))
	axes = axes.flatten()

	for ax, (x, y) in zip(axes, batch):
		ax.scatter(features[x]["Gryffindor"], features[y]["Gryffindor"], color="red", alpha=0.5)
		ax.scatter(features[x]["Hufflepuff"], features[y]["Hufflepuff"], color="orange", alpha=0.5)
		ax.scatter(features[x]["Ravenclaw"], features[y]["Ravenclaw"], color="blue", alpha=0.5)
		ax.scatter(features[x]["Slytherin"], features[y]["Slytherin"], color="green", alpha=0.5)
		ax.set_xlabel(x)
		ax.set_ylabel(y)
		ax.set_title(f"{x} vs {y}")

	for ax in axes[len(batch):]:
		ax.set_visible(False)

# 	fig.tight_layout()

# # plt.scatter(features["Astronomy"]["Gryffindor"], features["Defense Against the Dark Arts"]["Gryffindor"], color="red", alpha=0.5)
# # plt.scatter(features["Astronomy"]["Hufflepuff"], features["Defense Against the Dark Arts"]["Hufflepuff"], color="orange", alpha=0.2)
# # plt.scatter(features["Astronomy"]["Ravenclaw"], features["Defense Against the Dark Arts"]["Ravenclaw"], color="blue", alpha=0.5)
# # plt.scatter(features["Astronomy"]["Slytherin"], features["Defense Against the Dark Arts"]["Slytherin"], color="green", alpha=0.2)
# plt.show()