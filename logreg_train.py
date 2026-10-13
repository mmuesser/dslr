#!usr/bin/env python3


#créer 4 classifiers
#pour chaque classifiers, appliquer algo L : regression logistique
	#faire une fonction linéaire de type : s = somme (poids_matière_x * note_matière_x) + b
	#appliquer sigmoid : 1 / 1 + e**-s pour avoir résultat strictement compris entre 0 et 1
	#appliquer entropie croisée binaire (- [yln(p) + (1 - y)ln(1 - p)]) sachant que y = 1 si maison du classifier, 0 sinon
	#descente du gradient pour modifier les poids


list_poids : dict[str, list[float]] = {
	"gryffindor_w" : [],
	"hufflepuf_w" : [],
	"ravenclaw_w" : [],
	"slytherin_w" : []}

def	classifier(list_poids : list[float], list_notes : list[float], ) -> float:
	pass