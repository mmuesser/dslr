#!usr/bin/env python3

#faire une matrice des courses avec dans la diagonale un histogram du course
#et dans les autres cases le pair plot des deux courses du croisement


import matplotlib.pyplot as plt
import pandas, math
import numpy as np
from itertools import combinations
from scatter_plot import split_data

datas = pandas.read_csv("dataset_train.csv")

features = split_data(datas)

num_features = len(features.keys())

pairs = list(combinations(features, 2))

fig, axes = plt.subplots(num_features, num_features, figsize=(30, 30))
# axes = axes.flatten()

colors = ["red", "orange", "blue", "green"]
# bins = np.linspace(min(all_values), max(all_values), 31)


for i in range(num_features):
	for j in range(num_features):
		ax = axes[i, j]

		if i == j:
			for house, values, color in zip(list(features[list(features.keys())[i]]), list(features[list(features.keys())[i]].values()), colors):
				ax.hist(
					values,
					bins=31,
					alpha=0.45,
					label=house,
					facecolor=color,
					density=True,
				)
		else:
			ax.scatter(features[list(features.keys())[i]]["Gryffindor"], features[list(features.keys())[j]]["Gryffindor"], color="red", alpha=0.5)
			ax.scatter(features[list(features.keys())[i]]["Hufflepuff"], features[list(features.keys())[j]]["Hufflepuff"], color="orange", alpha=0.5)
			ax.scatter(features[list(features.keys())[i]]["Ravenclaw"], features[list(features.keys())[j]]["Ravenclaw"], color="blue", alpha=0.5)
			ax.scatter(features[list(features.keys())[i]]["Slytherin"], features[list(features.keys())[j]]["Slytherin"], color="green", alpha=0.5)
	
plt.xticks([])
plt.yticks([])
fig.tight_layout()
plt.show()