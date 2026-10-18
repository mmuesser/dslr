#!usr/bin/env python3

import pandas, sys
from logreg_train import classifier, sigmoid, norm_notes

def main():
	datas = pandas.read_csv(sys.argv[1])
	datas.drop(["Index", "Hogwarts House"], axis=1, inplace=True)
	num_data = datas.select_dtypes(include=[float, int])
	num_data = num_data.fillna(num_data.mean())

	data_weights = pandas.read_csv(sys.argv[2])
	lst_weights = []
	data_biasis = data_weights.tail(1)
	lst_biasis = []
	data_weights.drop(data_weights.tail(1).index, inplace=True)
	for col in data_weights.columns:
		lst_weights.append(data_weights[col].tolist())
		lst_biasis.append(data_biasis[col].get(13))

	nb_eleves = num_data.shape[0]
	nb_notes = len(num_data.columns)
	lst_notes = num_data.to_numpy(dtype=float).tolist()
	lst_houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
	for j in range(nb_notes):
		notes_matiere = [eleve[j] for eleve in lst_notes]
		norm_notes(notes_matiere)
		for i in range(nb_eleves):
			lst_notes[i][j] = notes_matiere[i]

	prediction : dict[str, list[str]] = {
		"Index" : [],
		"Hogwarts House" : []
	}
	for i in range(nb_eleves):
		p = 0
		prediction["Hogwarts House"].append("")
		for j in range(len(lst_houses)):
			z = classifier(lst_weights[j], lst_notes[i], lst_biasis[j])
			if sigmoid(z) > p:
				p = sigmoid(z)
				prediction["Hogwarts House"][i] = lst_houses[j]
		prediction["Index"].append(str(i))
	df = pandas.DataFrame(prediction)
	df.to_csv("houses.csv", index=False)

if __name__ == "__main__":
	argc = len(sys.argv)
	if argc != 3:
		pass
	else:
		main()
