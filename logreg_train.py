#!usr/bin/env python3

#créer 4 classifiers
#pour chaque classifiers, appliquer algo L : regression logistique
	#faire une fonction linéaire de type : s = somme (poids_matière_x * note_matière_x) + b
	#appliquer sigmoid : 1 / 1 + e**-s pour avoir résultat strictement compris entre 0 et 1
	# utile uniquement pour suivre evolution de la regression : 
		#appliquer entropie croisée binaire (- [yln(p) + (1 - y)ln(1 - p)]) sachant que y = 1 si maison du classifier, 0 sinon
	#descente du gradient pour modifier les poids et le biais

import sys, pandas, math

def std_data(lst_notes : list, count : int, mean : float):
	somme = 0
	for i in range(count):
		if lst_notes[i] is not None and not math.isnan(lst_notes[i]):
			somme += (lst_notes[i] - mean) ** 2
	return (math.sqrt(somme / count))

def norm_notes(lst_notes: list[float]):
    count = len(lst_notes)
    mean = sum(lst_notes) / count
    std = std_data(lst_notes, count, mean)

    if std == 0:
        return

    for i in range(count):
        lst_notes[i] = (lst_notes[i] - mean) / std


def	classifier(lst_weights : list[float], lst_notes : list[float], biais : float) -> float:
	s = 0
	for x in range(len(lst_weights)):
		s += lst_weights[x] * lst_notes[x]
	s += biais
	return s

def sigmoid(z : float) -> float:
	p = 1 / (1 + math.exp(-z))
	return p

def cost_ft(y : float, p : float) -> float:
	eps = 1e-15
	p = max(eps, min(1 - eps, p))
	x = - (y * math.log(p) + (1 - y) * math.log(1 - p))
	return x

def gradient_descent(lst_weights : list[float], biais : float, learning_rate : float, nb_notes : int, lst_error : list[float], somme : list[float], nb_eleves : int):
	for i in range(nb_notes):
		new_poids = learning_rate * (somme[i] / nb_eleves)
		lst_weights[i] = lst_weights[i] - new_poids
	new_biais = learning_rate * (sum(lst_error) / nb_eleves)
	biais = biais - new_biais
	return biais


def log_reg(lst_weights : list[float], lst_notes : list[list[float]] , house : str, nb_notes : int, nb_eleves : int, biais : float, lst_houses : list[str]) -> float:

	learning_rate = 0.1
	for count in range (1000):
		somme : list[float] = [0] * nb_notes
		lst_error : list[float] = []
		x = 0
		for i in range (nb_eleves):
			#y = 1 si classe eleve == classifier, 0 sinon
			y = 1 if lst_houses[i] == house else 0
			# z = classifier(eleves[i])
			z = classifier(lst_weights, lst_notes[i], biais)
			#p = sigmoid(z)
			p = sigmoid(z)
			#x = entropie croisée binaire(y, p)
			x += cost_ft(y, p)
			#pour chaque poids
				#poids[j] += (z - y) * notes[i]
			lst_error.append((p - y))
			for j in range(nb_notes):
				somme[j] += lst_error[i] * lst_notes[i][j]
		#moyenne de x
		x_mean = x / nb_eleves
		# if count % 100 == 0:
		# 	print(f"perte moyenne : {x_mean}")

		biais = gradient_descent(lst_weights, biais, learning_rate, nb_notes, lst_error, somme, nb_eleves)
	return biais
		#descente du gradient :
			#pour chaque poids 
				#new_poids = moyenne_x (?) * somme[j] / nb_eleves
				#poids = poids - new_poids
			#new_biais = moyenne_x (?) * sum(error) / nb_eleves
			#biais = biais - new_biais


def main() -> None:
	#log_reg Gryffindor -> set lst_weights[gryffindor_w]
	#log_reg Hufflepuf -> set lst_weights[hufflepuf_w]
	#log_reg Ravenclaw -> set lst_weights[ravenclaw_w]
	#log_reg Slytherin -> set lst_weights[slytherin_w]
	#créer fichier avec poids pour chaque classifier/maison
	datas = pandas.read_csv(sys.argv[1])
	datas.drop("Index", axis=1, inplace=True)
	num_data = datas.select_dtypes(include=[float, int])
	num_data = num_data.fillna(num_data.mean())
	nb_eleves = num_data.shape[0]
	nb_notes = len(num_data.columns)
	lst_houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
	lst_all_weights : dict[str, list] = {
			lst_houses[0] : [0] * nb_notes,
			lst_houses[1] : [0] * nb_notes,
			lst_houses[2] : [0] * nb_notes,
			lst_houses[3] : [0] * nb_notes,
			}
	lst_students_houses = list(datas["Hogwarts House"])
	print(lst_houses)
	lst_notes = num_data.to_numpy(dtype=float).tolist()
	for j in range(nb_notes):
		notes_matiere = [eleve[j] for eleve in lst_notes]
		norm_notes(notes_matiere)
		for i in range(nb_eleves):
			lst_notes[i][j] = notes_matiere[i]
	i = 0
	for key in lst_houses:
		lst_all_weights[key].append(log_reg(lst_all_weights[key], lst_notes, key, nb_notes, nb_eleves, 0, lst_students_houses))
		i += 1
		print(lst_all_weights[key])
	#creer fichier .cvs type : Hogwarts House, Features1_w, ..., FeaturesX_w, Biais
	df = pandas.DataFrame(lst_all_weights)
	df.to_csv("houses_weights.csv", index=False)
	

if __name__ == "__main__":
	argc = len(sys.argv)
	if argc != 2:
		pass
	else:
		main()
