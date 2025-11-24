# Meetings

## 24.11.2025
On voit que la similarité entre les articles n'est pas directement pertinente, c'est là qu'on voit que les chunks ont leurs importance. Il faut essayer de faire de la compairaison entre chunk et comparaison entre chunk et article. Dans une phrase où il y a un lien on fait la similarité entre cette phrase et les articles. On veut plus que de la similarité on veut le lien direct, peutêtre que la similarité n'est pas complète pour faire ça et il faudra être créatif dans d'autre approche. Plus tard on pourra essayer de mettre le mot qui contient le lien comme sujet principale d'une nouvelle phrase. 

## 17.11.2025
On a besoin de stoquer les liens et la position dans la db pour faire la comparaison avec les 2 métriques. On fait du bon travail si les 2 bonnes, 1 seul n'est pas suffisant.


## 10.11.2025
Si c'est en français, les llm seront peutêtre moins bon, peutêtre qu'une db anglais serait meilleure

L'intuition nous dit que le référencement d'auteurs par exemple ne sera pas bon i.e. pas le premier, une fois que c'est confirmer, essayer de voir comment choisir parmis les top choice 
On pourra éventuellement faire de la classification sur les top choice (On donne les 4 choix et le modèle choisit le meilleure)

On peut commencer par faire uniquement sur les parties qui contiennent déjà des liens et ignorer le reste

## 03.11.2025
Premier Step, analyser la db, comprendre le problème -> objectif : relier des mots avec d'autres textes de la db
En premier on relie des chunk de document entre eux, ensuite la partie intéressante du projet sera de voir quelle partie du chunk est importante
En premier on prend un subset (uel articles fortement liés et d'autres qui n'ont pas de rapport), faire des expérience avec la taille des chunk et la taille de l'intersect
Le nombre de chunk d'un même article en relation avec un autre article est une métrique intéressante aussi 

* Parsing -> deal avec les balises
* Créer les chunk -> expérimentation avec les tailles
* Créer les embeddings
* Stocker les embeddings dans une db
