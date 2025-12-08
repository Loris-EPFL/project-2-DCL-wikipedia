# Meetings

## 08.12.2025
Essayer de faire la comparaison directement avec le mot tous seul et pas avec une phrase pour valider l'hypothèse que la phrase brouille les pistes.
vérifier que ce n'est pas un bug (double check le code qui calcul nos métrics)

Si ça ne marche pas on peut essayer de construire une phrase autour du mot principal 
On peut essayer de ne pas comparer avec l'entier de l'article mais par example seulement avec le titre.



## 03.11.2025
Premier Step, analyser la db, comprendre le problème -> objectif : relier des mots avec d'autres textes de la db
En premier on relie des chunk de document entre eux, ensuite la partie intéressante du projet sera de voir quelle partie du chunk est importante
En premier on prend un subset (uel articles fortement liés et d'autres qui n'ont pas de rapport), faire des expérience avec la taille des chunk et la taille de l'intersect
Le nombre de chunk d'un même article en relation avec un autre article est une métrique intéressante aussi 

* Parsing -> deal avec les balises
* Créer les chunk -> expérimentation avec les tailles
* Créer les embeddings
* Stocker les embeddings dans une db
