# Réarrangement d'un Entrepôt Automatisé avec Navettes et Double Profondeur

Ce dépôt contient le code et les heuristiques développés dans le cadre d'un stage de recherche opérationnelle portant sur le problème de réarrangement des stocks au sein d'un entrepôt automatisé (SBS/RS) à double profondeur.

## Contexte Scientifique

Une décision importante pour améliorer les opérations au sein des entrepôts consiste à déterminer l'affectation optimale des articles aux emplacements de stockage (Storage Location Assignment Problem - SLAP). Ces affectations, souvent basées sur les profils de demande (saisonnalité, nouveaux produits), deviennent sous-optimales au fil du temps. Le **problème de réarrangement** (Warehouse Reshuffling Problem - WRP) consiste à déterminer la séquence optimale de déplacements pour atteindre un arrangement cible depuis un arrangement initial donné en entrée.

Le système étudié est un système automatisé de stockage et de déstockage avec navettes (**SBS/RS**). Il est composé :
- D'une unique étagère de stockage haute de plusieurs étages et longue de plusieurs mètres.
- De **doubles profondeurs** par emplacement : chaque emplacement peut contenir jusqu'à deux bacs l'un devant l'autre. Le bac de second rang nécessite un cycle de relocation pour être extrait si un bac bloquant est présent au premier rang.
- De navettes horizontales dédiées (*tier-captive*) se déplaçant à chaque niveau pour stocker et récupérer les bacs.
- De zones tampons de bout d'allée et d'un ascenseur de capacité 4 pour assurer les mouvements verticaux entre étages.

Dans le cadre du problème de réarrangement, l'un des défis algorithmiques majeurs est la présence de **cycles** (permutations circulaires des emplacements entre les bacs). La résolution de ces cycles impose de libérer temporairement un emplacement pour amorcer le déplacement, introduisant une gestion complexe des manipulations intermédiaires.

Contrairement à la majorité de la littérature qui cherche classiquement à minimiser le nombre de mouvements, ce travail propose un critère plus adapté aux enjeux de durabilité et d'automatisation : la **minimisation de la distance totale parcourue** par les navettes et l'ascenseur, afin de réduire la consommation énergétique et l'usure mécanique des équipements.

### Informations du Stage
- **Laboratoire** : DISP (Décision & Information pour les Systèmes de Production – UR4570), INSA Lyon
- **Encadrement** : Anne-Laure LADIER (Maître de conférences HDR), Brice CENCI
- **Collaboration** : Marwane BOUZNIF
- **Mots-clés** : Entrepôt automatisé avec navettes, SBS/RS, Réarrangement, Double Profondeur, Heuristiques, Recherche opérationnelle

## Structure du Projet

Le projet est divisé en plusieurs modules complémentaires :

- **`instance_generator.py`** : Générateur d'instances stochastiques pour modéliser le problème SBS/RS, permettant de paramétrer le taux de remplissage, la probabilité de maintien de la profondeur, et la taille de l'entrepôt.
- **`H1.py`** : Heuristique Constructive (approche Gloutonne couplée à une Liste Tabou). Utilise un graphe de dépendances recalculé à chaque étape pour prioriser les bacs non bloqués et casser les interblocages via des pénalités empiriques.
- **`H2.py`** : Heuristique Lookahead (approche Beam Search). Navigue dynamiquement dans l'arbre des configurations futures pour résoudre les cycles et interblocages complexes en profondeur, via une évaluation topologique polynomiale.
- **`benchmark_visualization.py`** : Suite d'analyses et de tests générant de multiples graphiques pour évaluer les performances globales, les facteurs d'échecs et la scalabilité des algorithmes.

## Installation

Pour exécuter le projet en local, assurez-vous de disposer de Python 3.10 ou supérieur, puis installez les dépendances requises :

```bash
pip install -r requirements.txt
```

## Utilisation et Commandes

Le projet propose une interface en ligne de commande détaillée pour chaque module.

### 1. Générateur d'instances (`instance_generator.py`)

Ce script propose deux modes : la génération d'une instance unique (`une`), ou la génération massive par lots (`lot`).

- **Générer une instance unique** :
  ```bash
  python3 instance_generator.py une -c <colonnes> -t <taux_remplissage> -p <prob_maintien> -e <chemin_export.csv>
  ```
  *Exemple :* `python3 instance_generator.py une -c 10 -t 0.8 -p 0.8 -e instances/test.csv`

- **Générer un plan d'expérience complet (lot)** :
  ```bash
  python3 instance_generator.py lot -c <col_1> <col_2> -t <taux_1> <taux_2> -r <repetitions> -d <dossier_export>
  ```
  *Exemple :* `python3 instance_generator.py lot -c 10 20 -t 0.8 0.9 -r 5 -d instances`

### 2. Heuristiques de Résolution (`H1.py` et `H2.py`)

Les deux heuristiques possèdent la même interface, permettant la résolution sur une seule instance (`une`) ou sur tout un répertoire (`lot`).

- **Exécuter sur une instance précise** :
  ```bash
  python3 H1.py une -f <chemin_fichier.csv>
  python3 H2.py une -f <chemin_fichier.csv>
  ```

- **Exécuter sur l'intégralité d'un répertoire** :
  ```bash
  python3 H1.py lot -d <dossier_instances> -o <fichier_resultats.csv>
  python3 H2.py lot -d <dossier_instances> -o <fichier_resultats.csv>
  ```

### 3. Analyse et Visualisation (`benchmark_visualization.py`)

Une fois les résultats générés et sauvegardés dans des fichiers CSV, lancez l'outil de visualisation pour générer les figures comparatives de performances.

```bash
python3 benchmark_visualization.py
```
Les graphiques seront automatiquement exportés au format PNG dans le répertoire de destination défini dans le script.
