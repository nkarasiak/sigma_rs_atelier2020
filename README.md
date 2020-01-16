# Atelier Museo ToolBox et découverte des réseaux de neurones

## Museo ToolBox

### Filtres spatiaux

L'idée est d'incorporer ce qu'on a vu en cours (un filtre spatial de type médiane, moyenne) dans la bibliothèque Museo ToolBox.

Le plus simple sera de créer une function spatial, en plus des fonctions aspatiales de Museo ToolBox.

```python3
RasterMath.add_spatial_function()
```

La nouvelle fonction devra donc prendre en entrée chaque block défini par RasterMath (il n'est plus question comme en cours de charger toute l'image en mémoire), mais avec une largeur/hauteur plus grande que le block de sortie car il s'agit d'un filtre spatial.

Une fois la mise en place du filtre dans Museo ToolBox, on veillera à améliorer la rapidité de traitement.

Pour cela il conviendra d'utiliser la fonction [generic_filter de scipy](https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.ndimage.generic_filter.html).


### Travail avec plusieurs coeurs (processeur)

RasterMath ne travaille pas en parallèle. C'est-à-dire que la classe ne peut générer plusieurs blocks simultanément pour améliorer la vitesse de calcul d'une image.

Comme sur nombre de bibliothèques python, il serait bien d'ajouter un paramètre de type `n_jobs = 4` pour dispoer ici de 4 traitements en parallèle.

```python3
from joblib import Parallel, delayed
```

Après plusieurs tests, nous vous conseillons d'utiliser la bibliothèque joblib qui a déjà fait ses preuves dans nos travaux et qui est implémenté avec succès dans scikit-learn.

### Intégration continue et github

Lors de cette partie, vous allez devoir vous familiariser avec github pour versionner, mettre à jour, et sauvegarer votre code.

Pour cela créer un compte sur gitub.com puis : 
    - Faites un fork du dépôt Museo ToolBox
    - Créer une branche 'spatial_filter'
    - Basculez sur cette branche pour commencer à développer

Une fois la fonction écrite, créer un exemple utilisant un filtre spatial à cet endroit : 
`museotoolbox/examples/processing/spatial_filtering.py`

Pour tester le code et la bonne exécution et du filtre spatial, je vous montrerai comment valider votre code.

### Résumé de la partie Museo ToolBox

1. Prise en main du code et compréhension de Museo ToolBox.
2. Adaptation dans RasterMath de la gestion des filtres spatiaux (taille de fenêtre en entrée plus grande que la taille de la fenêtre en écriture). Fonction future : `RasterMath.add_spatial_function()`
3. Amélioration du temps de traitement des filtres spatiaux à l'aide de la fonction *generic_filter* de *scipy*
4. Ajout de la possibilité de la parallélisation à RasterMath


## Réseaux de neurones

Découverte des réseaux de neurones.
Cela est nouveau pour nous aussi, alors nous allons découvrir plein de choses ensemble.

À la différence de l'apprentissage automatique, les réseaux de neurones fonctionnent plutôt avec des vignettes (j'entends par là des images où chaque petit block à un label).
Bien qu'il existe des réseaux de neurones capables de ne traiter que l'information spectrale (comme on le fait en apprentissage automatique), le véritable gain des réseaux de neurones est sa capacité à reconnaitre des objets, des formes, des textures.


### Jeu de données de test

Outre l'amélioration du code déjà existant de Museo ToolBox, nous voulons développer un premier code fonctionnel utilisant des réseaux de neurones pour prédire des essences forestières. 

Pour cela nous vous sollicitons pour nous aider dans cette aventure en réalisant :

- Un comparatif (avantage/faiblesse) des différentes bibliothèques (Keras, TensorFlow, PlaidML...)
- Des premiers essais pour vous permettre de vous familiariser avec les réseaux de neurones en python (cf [towards data science](https://towardsdatascience.com/neural-network-for-satellite-data-classification-using-tensorflow-in-python-a13bcf38f3e1) ou [unilnet](https://unilnet.github.io/)).
- Un premier essai de production sur la **détection de peupliers** à partir d'**imagettes temporelles Sentinel-2**

Et surtout, vous êtes force de proposition, **toute bonne idée sera la bienvenue**. Par contre, toute mauvaise idée sera sanctionnée par 10 pompes.

### Résumé de la partie réseaux de neurones

1. Découverte des réseaux de neurones (lecture de la bibliographie et comparatif des bibliothèques).
2. Réflexions autour de la construction d'un réseau (nombre de couches et utilité des douches).
3. [Découverte du deep learning dans la télédétection avec un guide en ligne sur github](https://github.com/robmarkcole/satellite-image-deep-learning/blob/master/README.md).
4. Premiers essais avec les bases fournis comme unilnet.
5. Étude de l'implémentation des réseaux de neurones avec une dimension temporel. cf [Temporal Convolutional Neural Network for the Classification of Satellite Image Time Series](https://github.com/charlotte-pel/temporalCNN) ou [Satellite Image Time Series Classification with Pixel-Set Encoders and Temporal Self-Attention](https://github.com/VSainteuf/pytorch-psetae).
6. Application d'un modèle de réseaux de neurones pour détecter les peupleuraies à partir de série temporelles d'images satellites Sentinel-2.
