#!usr/bin/env python3


#créer 4 classifiers
#pour chaque classifiers, appliquer algo L : regression logistique
	#faire une fonction linéaire de type : s = somme (poids_matière_x * note_matière_x) + b
	#appliquer sigmoid : 1 / 1 + e**-s pour avoir résultat strictement compris entre 0 et 1
	#appliquer entropie croisée binaire (- [yln(p) + (1 - y)ln(1 - p)]) sachant que y = 1 si maison du classifier, 0 sinon
	#descente du gradient pour modifier les poids

biais = 0

def	classifier(list_poids : list[float], list_notes : list[float]) -> float:
	global biais
	s = sum(list_poids[x] * list_notes[x] for x in range(len(list_poids))) + biais
	return s


def log_reg() -> None:
	list_notes : list[list] = []
	list_poids : dict[str, list[float]] = {
		"gryffindor_w" : [0] * nb_notes,
		"hufflepuf_w" : [0] * nb_notes,
		"ravenclaw_w" : [0] * nb_notes,
		"slytherin_w" : [0] * nb_notes
		}
	for _ in range (1000):
		for i in range (nb_eleves):
			#classifier
			#sigmoid
			#entropie croisée binaire
		#descente du gradient
			pass


if __name__ == "__main__":
	#log_reg Gryffindor -> set lst_poids[gryffindor_w]
	#log_reg Hufflepuf -> set lst_poids[hufflepuf_w]
	#log_reg Ravenclaw -> set lst_poids[ravenclaw_w]
	#log_reg Slytherin -> set lst_poids[slytherin_w]
	#créer fichier avec poids pour chaque classifier/maison
	pass