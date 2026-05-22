# Réarrangement d'un Entrepôt Automatisé avec Navettes et Double Profondeur

Ce dépôt contient le code et les heuristiques développés dans le cadre d'un stage de recherche opérationnelle portant sur le problème de réarrangement des stocks au sein d'un entrepôt automatisé (SBS/RS) à double profondeur.

## 📋 Contexte Scientifique

Une décision cruciale pour améliorer les opérations des entrepôts consiste à déterminer l'affectation optimale des articles aux emplacements de stockage. Ces affectations, souvent basées sur les profils de demande (saisonnalité, nouveaux produits), deviennent sous-optimales au fil du temps. Le **problème de réarrangement** consiste à calculer la séquence optimale de déplacements pour atteindre un arrangement cible depuis un arrangement initial.

Le système étudié est un système automatisé de stockage et déstockage avec navettes (SBS/RS). Il est composé :
- D'une étagère de plusieurs étages et colonnes.
- De **doubles profondeurs** par emplacement (pouvant contenir jusqu'à deux bacs l'un derrière l'autre).
- De navettes horizontales par étage et d'un élévateur.

Contrairement à la littérature qui minimise souvent le *nombre* de déplacements, ce projet s'attache à **minimiser la distance totale parcourue** par les navettes et l'ascenseur pour réduire la consommation énergétique et l'usure mécanique.

### 📅 Informations du Stage
- **Dates** : 20/04/2026 - 29/05/2026
- **Laboratoire** : DISP (Décision & Information pour les Systèmes de Production – UR4570), INSA Lyon
- **Encadrement** : Anne-Laure LADIER (Maître de conférences HDR), Brice CENCI
- **Collaboration** : Marwane BOUZNIF
- **Mots-clés** : Entrepôt automatisé avec navettes, Réarrangement, Heuristiques, Recherche opérationnelle

## 🛠️ Structure du Projet

Le projet est divisé en plusieurs modules complémentaires :

- **`instance_generator.py`** : Générateur d'instances stochastiques pour modéliser le problème SBS/RS en une ou plusieurs itérations de tests.
- **`H1.py`** : Heuristique Constructive (approche Greedy couplée à une Liste Tabou). Utilise un graphe de dépendances à chaque étape pour prioriser les bacs non bloqués.
- **`H2.py`** : Heuristique Lookahead (approche Beam Search). Navigue dynamiquement dans l'arbre des configurations futures pour résoudre les *deadlocks* complexes et libérer les cycles via une évaluation topologique polynomiale.
- **`benchmark_visualization.py`** : Suite de tests automatisés générant de multiples graphiques (`matplotlib`, `seaborn`) d'analyse de performance (Scalabilité, Temps de calcul, Pareto, etc.) entre les algorithmes.

## ⚙️ Installation

Pour exécuter le projet en local, assurez-vous d'avoir Python 3.10+ installé, puis installez les dépendances nécessaires :

```bash
pip install -r requirements.txt
```

## 🚀 Utilisation

Chaque module peut être exécuté en ligne de commande.

### 1. Génération d'instances
```bash
# Générer un lot de fichiers CSV de test (instances)
python instance_generator.py lot -c 10 20 -t 0.8 0.9 -r 5
```

### 2. Résolution (Heuristiques)
```bash
# Exécuter H1 sur une instance précise
python H1.py une -f instances/mon_instance.csv

# Exécuter H2 sur tout un dossier en mode lot
python H2.py lot -d instances -o resultats/resultat_H2.csv
```

### 3. Analyse des Benchmarks
Une fois les résultats générés dans le dossier `resultats/`, lancez l'outil de visualisation pour générer les graphiques comparatifs :
```bash
python benchmark_visualization.py
```
Les graphiques (png) seront automatiquement exportés dans le dossier `images/`.
