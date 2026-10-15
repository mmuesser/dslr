# !/usr/bin/env python3

#faire 4 liste pour chaque maison avec les valeurs des etudiants
#calculer ecart type pour chaque matière pour chaque maison
#faire histogramme qui affiche somme ecart type de chaque matière pour chaque maison

#calculer z score de chaques notes de chaque matière
#faire hist pour chaque matière du nombre de la valeur x du z score


import numpy as np
import pandas, math
import matplotlib.pyplot as plt
from describe import describe_data, print_data

datas = pandas.read_csv("dataset_train.csv")

num_data = datas.select_dtypes(include=[float, int])

analyzed = describe_data(num_data)

def ratio_all(house : dict, analyzed : dict):
	ratio_house : dict[str, list] = {}
	for key, lst in house.items():
		ratio_house[key] = []
		for value in lst:
			if value is not None and not math.isnan(value):
				ratio_house[key].append((value - float(analyzed[key][1])) / float(analyzed[key][2]))
	return ratio_house


test : dict[str, list[list]] = {}

gryffindor: dict[str, list] = {}
hufflepuff: dict[str, list] = {}
ravenclaw: dict[str, list] = {}
slytherin: dict[str, list] = {}

for col in num_data.columns[1:]:
	gryffindor[col] = []
	hufflepuff[col] = []
	ravenclaw[col] = []
	slytherin[col] = []
	test[col] = []
	for _, raw in datas.iterrows():
		if raw["Hogwarts House"] == "Gryffindor": #and raw.get(col) is not None and not math.isnan(raw.get(col)):
			gryffindor[col].append(raw.get(col))
		elif raw["Hogwarts House"] == "Hufflepuff": #and raw.get(col) is not None and not math.isnan(raw.get(col)):
			hufflepuff[col].append(raw.get(col))
		elif raw["Hogwarts House"] == "Ravenclaw": #and raw.get(col) is not None and not math.isnan(raw.get(col)):
			ravenclaw[col].append(raw.get(col))
		elif raw["Hogwarts House"] == "Slytherin": #and raw.get(col) is not None and not math.isnan(raw.get(col)):
			slytherin[col].append(raw.get(col))
	test[col].append(gryffindor[col])
	test[col].append(hufflepuff[col])
	test[col].append(ravenclaw[col])
	test[col].append(slytherin[col])

# for key in test.keys():
# 	test[key].append(ratio_all(gryffindor, analyzed)[key])
# 	test[key].append(ratio_all(hufflepuff, analyzed)[key])
# 	test[key].append(ratio_all(ravenclaw, analyzed)[key])
# 	test[key].append(ratio_all(slytherin, analyzed)[key])


# print(list(num_data.columns))

courses = list(num_data.columns)[1:]
houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

fig, axes = plt.subplots(3, 5, figsize=(20, 5))
axes = axes.flatten()

for i, course in enumerate(courses):
	ax = axes[i]

	all_values = [
	value
	for values in test[course]
	for value in values
	]
	
	bins = np.linspace(min(all_values), max(all_values), 31)

	barcolor = ["red", "orange", "blue", "green"]

	for house, values, color in zip(houses, test[course], barcolor):
		ax.hist(
			values,
			bins=bins,
			alpha=0.45,
			facecolor=color,
			label=house,
			density=True,
			#log=True,
		)

	ax.set_title(course, fontsize=8)
	
	ax.tick_params(axis="both", labelsize=7)

	if i == 0:
		ax.legend(fontsize=7)
	
for i in range(len(courses), len(axes)):
	fig.delaxes(axes[i])

fig.tight_layout()
plt.show()