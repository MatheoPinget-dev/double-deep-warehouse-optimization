"""
Métaheuristiques et Grid Search pour le réarrangement SBS/RS.
═══════════════════════════════════════════════════════════════════════════
Ce module contient :
1. Les Métaheuristiques (Recuit Simulé, Algorithme Génétique, Recherche Tabou)
2. Le moteur de Grid Search parallélisé (Multiprocessing) pour évaluer
   rapidement des milliers de combinaisons sur plusieurs cœurs CPU.
"""

from __future__ import annotations

import csv
import math
import time
import random
import itertools
from pathlib import Path
import concurrent.futures

from solver import Solveur, ResultatHeuristique
from instance_generator import Configuration, importer_instance

# ═══════════════════════════════════════════════════════════════════════════════
#  Fonction d'Évaluation Commune (La "Fitness")
# ═══════════════════════════════════════════════════════════════════════════════

def evaluer_chromosome(config_initiale: Configuration, config_cible: Configuration, chromosome: list[int]) -> tuple[float, ResultatHeuristique]:
    """Évalue un chromosome (liste de priorité) via le Solveur.
    Retourne la fitness (à minimiser) et le résultat complet."""
    solveur = Solveur(config_initiale, config_cible)
    
    # L'exécution avec la priorité custom remplace la logique gloutonne classique
    res = solveur._resoudre_generique("Meta", custom_priority_list=chromosome)
    
    mal_places_fin = sum(1 for b, e in solveur.affectation.items() if e != config_cible.affectation[b])
    
    # Si le solveur a tout rangé, la fitness est le nombre de mouvements (à minimiser).
    if res.complet:
        fitness = len(res.mouvements)
    else:
        # Grosse pénalité si l'entrepôt finit bloqué
        fitness = 5000 + (mal_places_fin * 100)
        
    return fitness, res

# ═══════════════════════════════════════════════════════════════════════════════
#  1. Recuit Simulé (Simulated Annealing)
# ═══════════════════════════════════════════════════════════════════════════════

def simulated_annealing(config_initiale: Configuration, config_cible: Configuration, T0: float, alpha: float, iter_per_T: int, max_iter: int=100) -> ResultatHeuristique:
    start_time = time.time()
    bacs = list(config_initiale.affectation.keys())
    current_chromosome = random.sample(bacs, len(bacs))
    current_fitness, current_res = evaluer_chromosome(config_initiale, config_cible, current_chromosome)
    
    best_chromosome = list(current_chromosome)
    best_fitness = current_fitness
    best_res = current_res
    
    T = T0
    
    for _ in range(max_iter):
        for _ in range(iter_per_T):
            # Voisin : swap 2 bacs au hasard
            voisin = list(current_chromosome)
            idx1, idx2 = random.sample(range(len(voisin)), 2)
            voisin[idx1], voisin[idx2] = voisin[idx2], voisin[idx1]
            
            fitness_voisin, res_voisin = evaluer_chromosome(config_initiale, config_cible, voisin)
            
            delta = fitness_voisin - current_fitness
            # Acceptation stochastique
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_chromosome = voisin
                current_fitness = fitness_voisin
                current_res = res_voisin
                
                if current_fitness < best_fitness:
                    best_fitness = current_fitness
                    best_chromosome = list(current_chromosome)
                    best_res = current_res
                    
        T *= alpha
        if best_res.complet and best_fitness < len(bacs): 
            # Condition d'arrêt précoce si on trouve un parcours parfait
            break
            
    best_res.nom = "SA"
    best_res.temps_execution = time.time() - start_time
    return best_res

# ═══════════════════════════════════════════════════════════════════════════════
#  2. Algorithme Génétique (Genetic Algorithm)
# ═══════════════════════════════════════════════════════════════════════════════

def crossover_ox(parent1: list[int], parent2: list[int]) -> list[int]:
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    enfant = [-1] * size
    enfant[a:b] = parent1[a:b]
    p2_filtre = [x for x in parent2 if x not in enfant]
    idx = 0
    for i in range(size):
        if enfant[i] == -1:
            enfant[i] = p2_filtre[idx]
            idx += 1
    return enfant

def mutate_swap(chromosome: list[int], mutation_rate: float):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(chromosome)-1)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

def genetic_algorithm(config_initiale: Configuration, config_cible: Configuration, pop_size: int, generations: int, mutation_rate: float) -> ResultatHeuristique:
    start_time = time.time()
    bacs = list(config_initiale.affectation.keys())
    
    population = [random.sample(bacs, len(bacs)) for _ in range(pop_size)]
    
    best_res = None
    best_fitness = float('inf')
    
    for gen in range(generations):
        evals = []
        for chrom in population:
            fit, res = evaluer_chromosome(config_initiale, config_cible, chrom)
            evals.append((fit, chrom, res))
            if fit < best_fitness:
                best_fitness = fit
                best_res = res
                
        if best_res is not None and best_res.complet and best_fitness < len(bacs):
            break
            
        evals.sort(key=lambda x: x[0])
        
        # Élitisme (garde les 20% meilleurs)
        elite_size = max(2, int(pop_size * 0.2))
        nouvelle_pop = [list(x[1]) for x in evals[:elite_size]]
        
        while len(nouvelle_pop) < pop_size:
            p1 = random.choice(evals[:elite_size])[1]
            p2 = random.choice(evals[:elite_size])[1]
            enfant = crossover_ox(p1, p2)
            mutate_swap(enfant, mutation_rate)
            nouvelle_pop.append(enfant)
            
        population = nouvelle_pop
        
    if best_res is None:
        fit, best_res = evaluer_chromosome(config_initiale, config_cible, population[0])
        
    best_res.nom = "GA"
    best_res.temps_execution = time.time() - start_time
    return best_res

# ═══════════════════════════════════════════════════════════════════════════════
#  3. Recherche Tabou (Tabu Search)
# ═══════════════════════════════════════════════════════════════════════════════

def tabu_search(config_initiale: Configuration, config_cible: Configuration, tabu_tenure: int, max_iter: int, neighbors_size: int) -> ResultatHeuristique:
    start_time = time.time()
    bacs = list(config_initiale.affectation.keys())
    
    current_chromosome = random.sample(bacs, len(bacs))
    best_chromosome = list(current_chromosome)
    
    current_fitness, current_res = evaluer_chromosome(config_initiale, config_cible, current_chromosome)
    best_fitness = current_fitness
    best_res = current_res
    
    # Dictionnaire Tabou: { (bacA, bacB): iteration_expiration }
    tabu_list = {}
    
    for iteration in range(max_iter):
        voisinage = []
        
        # 1. Générer le voisinage
        for _ in range(neighbors_size):
            voisin = list(current_chromosome)
            i, j = random.sample(range(len(voisin)), 2)
            bacA, bacB = voisin[i], voisin[j]
            voisin[i], voisin[j] = voisin[j], voisin[i] # Swap
            voisinage.append((voisin, bacA, bacB))
            
        meilleur_voisin_fit = float('inf')
        meilleur_voisin_chrom = None
        meilleur_voisin_res = None
        mouvement_choisi = None
        
        # 2. Évaluer le voisinage
        for voisin, bacA, bacB in voisinage:
            fit, res = evaluer_chromosome(config_initiale, config_cible, voisin)
            
            # Paire identifiant le mouvement
            mouvement = (min(bacA, bacB), max(bacA, bacB))
            est_tabou = tabu_list.get(mouvement, 0) > iteration
            
            # Critère d'aspiration : si c'est tabou mais que c'est le MEILLEUR score global jamais vu, on l'accepte
            if est_tabou and fit < best_fitness:
                est_tabou = False 
                
            if not est_tabou and fit < meilleur_voisin_fit:
                meilleur_voisin_fit = fit
                meilleur_voisin_chrom = voisin
                meilleur_voisin_res = res
                mouvement_choisi = mouvement
                
        # 3. Mise à jour
        if meilleur_voisin_chrom is not None:
            current_chromosome = meilleur_voisin_chrom
            current_fitness = meilleur_voisin_fit
            
            # On ajoute le mouvement inverse dans la liste tabou
            tabu_list[mouvement_choisi] = iteration + tabu_tenure
            
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_chromosome = list(current_chromosome)
                best_res = meilleur_voisin_res
                
        if best_res.complet and best_fitness < len(bacs):
            break

    best_res.nom = "TS"
    best_res.temps_execution = time.time() - start_time
    return best_res

# ═══════════════════════════════════════════════════════════════════════════════
#  Grid Search Parallélisé (Le Moteur d'Évaluation)
# ═══════════════════════════════════════════════════════════════════════════════

def _worker_eval_instance(args):
    """Fonction Helper pour le Multiprocessing. Évalue une instance avec une config donnée."""
    fichier, filepath, meta_name, p1, p2, p3 = args
    init, cible = importer_instance(filepath)
    
    if meta_name == "SA":
        res = simulated_annealing(init, cible, T0=p1, alpha=p2, iter_per_T=p3, max_iter=30)
    elif meta_name == "GA":
        res = genetic_algorithm(init, cible, pop_size=p1, generations=p2, mutation_rate=p3)
    elif meta_name == "TS":
        res = tabu_search(init, cible, tabu_tenure=p1, max_iter=p2, neighbors_size=p3)
        
    return {
        "fichier": fichier,
        "metaheuristique": meta_name,
        "param_1": p1, "param_2": p2, "param_3": p3,
        "complet": 1 if res.complet else 0,
        "distance_totale": res.distance_totale,
        "nb_mouvements": len(res.mouvements),
        "temps_execution": round(res.temps_execution, 4),
        "nb_bacs_bien_places": res.nb_bacs_bien_places,
        "total_bacs": res.total_bacs
    }

def run_grid_search(dossier_instances="instances", fichier_sortie="resultats/resultats_metaheuristiques.csv"):
    dossier = Path(dossier_instances)
    
    # On sélectionne les instances pour le grid search
    fichiers_path = [f for f in dossier.glob("*.csv") if "recapitulatif" not in f.name]
    
    if not fichiers_path:
        print("Aucune instance trouvée pour le Grid Search.")
        return
        
    print(f"Lancement du Grid Search sur {len(fichiers_path)} instances.")
    
    # --- Définition des Hyperparamètres ---
    sa_params = list(itertools.product(
        [100, 500],   # p1: T0
        [0.90, 0.95], # p2: alpha
        [10, 20]      # p3: iter_per_T
    ))
    
    ga_params = list(itertools.product(
        [10, 20],     # p1: pop_size
        [10, 20],     # p2: generations
        [0.10, 0.20]  # p3: mutation_rate
    ))
    
    ts_params = list(itertools.product(
        [5, 10],      # p1: tabu_tenure
        [20, 50],     # p2: max_iter
        [10, 20]      # p3: neighbors_size
    ))
    
    # --- Préparation des tâches pour le Multiprocessing ---
    taches = []
    for f in fichiers_path:
        for p1, p2, p3 in sa_params:
            taches.append((f.name, f, "SA", p1, p2, p3))
        for p1, p2, p3 in ga_params:
            taches.append((f.name, f, "GA", p1, p2, p3))
        for p1, p2, p3 in ts_params:
            taches.append((f.name, f, "TS", p1, p2, p3))
            
    print(f"Total des exécutions à réaliser : {len(taches)}")
    print("Calcul en parallèle en cours (cela peut prendre quelques minutes)...")
    
    resultats = []
    
    # Exécution sur tous les cœurs du processeur
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # map permet de distribuer les tâches et tqdm pourrait être ajouté pour une barre de progression
        for i, resultat in enumerate(executor.map(_worker_eval_instance, taches), 1):
            resultats.append(resultat)
            if i % 50 == 0 or i == len(taches):
                print(f"  Progression : {i} / {len(taches)} configurations évaluées...")

    # --- Export ---
    chemin_sortie = Path(fichier_sortie)
    entetes = ["fichier", "metaheuristique", "param_1", "param_2", "param_3", "complet", "distance_totale", "nb_mouvements", "temps_execution", "nb_bacs_bien_places", "total_bacs"]
    with chemin_sortie.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=entetes)
        writer.writeheader()
        writer.writerows(resultats)
        
    print(f"\nGrid Search terminé ! Résultats sauvegardés dans {fichier_sortie}")

if __name__ == "__main__":
    run_grid_search()