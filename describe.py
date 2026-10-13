#num_data qui contient les colonnes numériques
#pour chaque colonne calculer count, mean, std, min, 25%, 50%, 75%, max
#pour chaque element d'index X dans une ligne, calculer sa taille et comparer avec le reste des index X
#print premier element de chaque ligne normalement
#puis pour chaque élément à partir du deuxième, print taille_max - taille_elem + 2 espaces puis afficher elem

# !/usr/bin/env python3

from numpy import NaN
import pandas, math


datas = pandas.read_csv("dataset_train.csv")

num_data = datas.select_dtypes(include=[float, int])

def count_data(col : list):
	count = 0
	for x in col:
		if x is not None and not math.isnan(x):
			count += 1
	return f"{count:.6f}"

def mean_data(col : list):
	count = 0
	somme = 0
	for x in col:
		if x is not None and not math.isnan(x):
			count += 1
			somme += x
	return f"{somme / count:.6f}"

def std_data(col : list):
	count = 0
	somme = 0
	mean = float(mean_data(col))
	for x in col:
		if x is not None and not math.isnan(x):
			count += 1
			somme += (x - mean) ** 2
	return f"{math.sqrt(somme / count):.6f}"

def min_data(col : list):
	min_value = NaN
	for x in col:
		if x is not None and not math.isnan(x):
			if min_value is None or x < min_value:
				min_value = x
	return f"{min_value:.6f}"

def quantile_data(col : list, q : float):
	values = [x for x in col if x is not None and not math.isnan(x)]
	values.sort()
	n = len(values)
	if n == 0:
		return NaN
	index = (n - 1) * q
	lower_index = int(index)
	upper_index = lower_index + 1
	if upper_index >= n:
		return f"{values[lower_index]:.6f}"
	weight = index - lower_index
	return f"{values[lower_index] * (1 - weight) + values[upper_index] * weight:.6f}"


def describe_data(num_data) -> dict[str, list]:
	analyzed : dict[str, list] = {}
	analyzed["ref"] = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

	for col in num_data.columns:
		analyzed[col] = [
			count_data(num_data[col]),
			mean_data(num_data[col]),
			std_data(num_data[col]),
			quantile_data(num_data[col], 0),
			quantile_data(num_data[col], 0.25),
			quantile_data(num_data[col], 0.5),
			quantile_data(num_data[col], 0.75),
			quantile_data(num_data[col], 1),
		]
	return analyzed

def print_data(data : dict):
	# for i in range(0, len(data["ref"])):
	# 	test : str = ''
	# 	if i == 0:
	# 		test += ' '.join(list(data.keys()))
	# 	else:
	# 		for value in data.values():
	# 			test += value[i] + '  '
	# 	print(test)
	# for key in data.keys():
	# 	lenghts = max(list(len(val) for val in data[key]))
	# 	for ligne in zip(*data.values()):
	# 		# print(ligne)
	# 		print((" ").join(ligne))
			# print((" " * (int(quantile_data(ligne, 1)) - len(val))).join(str(val) for val in ligne))
			# print(data.values())
	df = pandas.DataFrame(data)
	print(df.columns)

if __name__ == "__main__":
	analyzed = describe_data(num_data)
	print_data(analyzed)
