#!usr/bin/env python3


#créer 4 classifiers
#pour chaque classifiers, appliquer algo L : regression logistique
	#faire une fonction linéaire de type : s = somme (poids_matière_x * note_matière_x) + b
	#appliquer sigmoid : 1 / 1 + e**-s pour avoir résultat strictement compris entre 0 et 1
	#appliquer entropie croisée binaire (- [yln(p) + (1 - y)ln(1 - p)]) sachant que y = 1 si maison du classifier, 0 sinon
	#descente du gradient pour modifier les poids et le biais

import sys, pandas, math


def norm_notes(lst_notes : list[float]): 
	pass

def	classifier(lst_poids : list[float], lst_notes : list[float], biais : float) -> float:
	# s = sum(lst_poids[x] * lst_notes[x] for x in range(len(lst_poids))) + biais
	s = 0
	# print(len(lst_poids))
	# print(len(lst_notes))
	for x in range(len(lst_poids)):
		s += lst_poids[x] * lst_notes[x]
	s += biais
	return s

def sigmoid(z : float) -> float:
	p = 1 / (1 + math.exp(-z))
	return p

def cost_ft(y : float, p : float) -> float:
	x = - (y * math.log(p) + (1 - y) * math.log(1 - p))
	return x

def gradient_descent(lst_poids : list[float], biais : float, learning_rate : float, nb_notes : int, lst_error : list[float], somme : list[float], nb_eleves : int):
	# new_poids : list[float] = [0] * nb_notes
	for i in range(nb_notes):
		new_poids = learning_rate * (somme[i] / nb_eleves)
		lst_poids[i] = lst_poids[i] - new_poids
	new_biais = learning_rate * (sum(lst_error) / nb_eleves)
	biais = biais - new_biais
	# print(f"learning_rate : {learning_rate}")
	# print(f"lst_error : {lst_error}")
	# print(f"somme : {somme}")

def log_reg(lst_poids : list[float], lst_notes : list[list[float]] , house : str, nb_notes : int, nb_eleves : int, biais : float, lst_houses : list[str]) -> None:
	# print(lst_poids)
	# print(biais)
	for _ in range (1000):
		somme : list[float] = [0] * nb_notes
		lst_error : list[float] = []
		x = 0
		for i in range (nb_eleves):
			#y = 1 si classe eleve == classifier, 0 sinon
			y = 1 if lst_houses[i] == house else 0
			# print(y)
			#z = classifier(eleves[i])
			z = classifier(lst_poids, lst_notes[i], biais)
			print(z)
			#p = sigmoid(z)
			p = sigmoid(z)
			print(f"i : {i}, p : {p}")
			#x = entropie croisée binaire(y, p)
			x += cost_ft(y, p)
			# print(x)
			#pour chaque poids
				#poids[j] += (z - y) * notes[i]
			lst_error.append((z - y))
			for j in range(nb_notes):
				somme[j] += lst_error[i] * lst_notes[i][j]
		#moyenne de x
		learning_rate = x / nb_eleves
		gradient_descent(lst_poids, biais, learning_rate, nb_notes, lst_error, somme, nb_eleves)
		# print(lst_poids)
		# print(biais)
		#descente du gradient :
			#pour chaque poids 
				#new_poids = moyenne_x (?) * somme[j] / nb_eleves
				#poids = poids - new_poids
			#new_biais = moyenne_x (?) * sum(error) / nb_eleves
			#biais = biais - new_biais


def main() -> None:
	#log_reg Gryffindor -> set lst_poids[gryffindor_w]
	#log_reg Hufflepuf -> set lst_poids[hufflepuf_w]
	#log_reg Ravenclaw -> set lst_poids[ravenclaw_w]
	#log_reg Slytherin -> set lst_poids[slytherin_w]
	#créer fichier avec poids pour chaque classifier/maison
	datas = pandas.read_csv(sys.argv[1])
	num_data = datas.select_dtypes(include=[float, int])
	nb_eleves = num_data.shape[0]
	nb_notes = len(num_data.columns[1:])
	# print(nb_notes)
	list_all_poids : dict[str, list[float]] = {
			"Gryffindor" : [0] * nb_notes,
			"HUfflepuf" : [0] * nb_notes,
			"Ravenclaw" : [0] * nb_notes,
			"Slytherin" : [0] * nb_notes
			}
	lst_houses = list(datas["Hogwarts House"])
	list_biais : list[float] = [0, 0, 0, 0]
	lst_notes : list[list] = [[]] * nb_eleves
	for col in num_data.columns[1:]:
		i = 0
		for _, raw in num_data.iterrows():
			lst_notes[i].append(raw.get(col))
			norm_notes(lst_notes[i])
			i += 1
	i = 0
	for key in list_all_poids.keys():
		log_reg(list_all_poids[key], lst_notes, key, nb_notes, nb_eleves, list_biais[i], lst_houses)
		# print("\n=============================================================================\n")
		i += 1
		print(list_all_poids[key])
	

if __name__ == "__main__":
	argc = len(sys.argv)
	if argc != 2:
		pass
	else:
		main()
