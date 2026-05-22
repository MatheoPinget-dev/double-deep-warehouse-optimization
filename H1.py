# =============================================================================
# H1.py  –  Heuristique Constructive (Greedy + Liste Tabou)
# Auteur : Projet SBS/RS
#
# Description :
#   Résout le problème de réarrangement d'un entrepôt automatique SBS/RS
#   à double profondeur en un seul étage.
#
# STRATÉGIE ("Eligible-First") :
#   1. Construction d'un graphe de dépendances logiques à chaque itération.
#   2. Priorisation des bacs "libres" (Happy Path) via un système de poids.
#   3. Gestion des cycles (Deadlocks) par mise en buffer stratégique.
#   4. Utilisation d'une Liste Tabou pour éviter les oscillations (ping-pong).
#
# NOTE STRUCTURELLE — Absence de mécanisme "Ouvreur" :
#   Le générateur d'instances impose l'invariant Back-First : un bac ne peut
#   occuper la position AVANT d'une demi-colonne que si la position ARRIERE
#   est aussi occupée. Il en découle que tout emplacement ARRIERE vide a
#   nécessairement son AVANT vide, donc est directement accessible par la
#   navette. L'ancien mécanisme "Ouvreur" (pousser un bac AVANT vers un
#   ARRIERE vide pour "libérer" l'AVANT) était donc redondant : sa cible
#   était déjà dans vides_accessibles avant toute intervention.
# =============================================================================

from enum import Enum, auto
from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import time
import copy

from instance_generator import (
    Configuration, EmplacementStock, Cote, Profondeur,
    distance, importer_instance
)

# =============================================================================
# 0. Exceptions Personnalisées
# =============================================================================

class DeadlockInsoluble(Exception):
    """Levée quand un cycle ne peut être cassé (aucun buffer disponible)."""
    pass

class TimeoutHeuristique(Exception):
    """Levée quand max_iterations est atteint sans convergence."""
    pass

# =============================================================================
# 1. Structures de Graphe et Dépendances
# =============================================================================

class TypeDependance(Enum):
    """Définit la nature du blocage entre deux bacs."""
    DUR = auto()          # Blocage physique LIFO (AVANT bloque ARRIERE)
    OCCUPATION = auto()   # Blocage logique (Un bac occupe la cible d'un autre)
    SEQUENCEMENT = auto() # Blocage d'ordre (ARRIERE doit entrer avant l'AVANT)

@dataclass
class Arc:
    """Représente un arc orienté A -> B : A bloque B."""
    source: int       # ID du bac bloquant
    cible: int        # ID du bac bloqué
    type_dep: TypeDependance

class GrapheDependance:
    """
    Graphe orienté représentant tous les blocages logiques et physiques
    entre les bacs à l'itération courante.

    Chaque nœud est un `bac_id`. Un arc A -> B signifie :
    "Le bac A doit bouger AVANT que le bac B puisse atteindre sa cible."

    Ce graphe est recalculé intégralement à chaque itération.
    
    Attributes:
        config_courante (Configuration): L'état actuel de l'entrepôt.
        config_cible (Configuration): L'état cible à atteindre.
        noeuds (set): L'ensemble des identifiants des bacs présents.
        arcs (list[Arc]): La liste des dépendances (arêtes) orientées.
    """
    def __init__(self, config_courante: Configuration, config_cible: Configuration):
        self.config_courante = config_courante
        self.config_cible = config_cible
        self.noeuds = set(config_courante.affectation.keys())
        self.arcs: list[Arc] = []
        self._construire_graphe()

    def _construire_graphe(self):
        """
        Construit le graphe orienté des dépendances.
        
        Identifie et instancie les trois types de blocages (DUR, OCCUPATION, 
        SÉQUENCEMENT) en fonction de l'état physique et logique des bacs.
        """
        occupes = {emp: bac_id for bac_id, emp in self.config_courante.affectation.items()}
        cibles_inverse = {emp: bac_id for bac_id, emp in self.config_cible.affectation.items()}

        # --- 1. DÉPENDANCES PHYSIQUES (DUR) ---
        # Contrainte LIFO : impossible de sortir le bac ARRIERE si l'AVANT est occupé.
        for c, s in self.config_courante.etage.demi_colonnes():
            emp_avant = EmplacementStock(c, s, Profondeur.AVANT)
            emp_arriere = EmplacementStock(c, s, Profondeur.ARRIERE)
            if emp_avant in occupes and emp_arriere in occupes:
                self.arcs.append(Arc(
                    source=occupes[emp_avant],
                    cible=occupes[emp_arriere],
                    type_dep=TypeDependance.DUR
                ))

        # --- 2. DÉPENDANCES D'OCCUPATION ---
        # Un bac bloque un autre s'il occupe sa place cible (ou l'AVANT de sa cible ARRIERE).
        for bac_j, cible_j in self.config_cible.affectation.items():
            if self.config_courante.affectation[bac_j] == cible_j:
                continue  # Déjà en place

            if cible_j in occupes:
                bac_k = occupes[cible_j]
                if bac_k != bac_j:
                    self.arcs.append(Arc(source=bac_k, cible=bac_j, type_dep=TypeDependance.OCCUPATION))

            # Blocage indirect : si la cible est ARRIERE, l'AVANT doit aussi être libre
            if cible_j.profondeur == Profondeur.ARRIERE:
                cible_j_avant = EmplacementStock(cible_j.colonne, cible_j.cote, Profondeur.AVANT)
                if cible_j_avant in occupes:
                    bac_k = occupes[cible_j_avant]
                    if bac_k != bac_j:
                        self.arcs.append(Arc(source=bac_k, cible=bac_j, type_dep=TypeDependance.OCCUPATION))

        # --- 3. DÉPENDANCES DE SÉQUENCEMENT ---
        # L'ARRIERE doit être inséré avant l'AVANT de la même demi-colonne cible.
        for bac_j, cible_j in self.config_cible.affectation.items():
            if cible_j.profondeur == Profondeur.ARRIERE:
                if self.config_courante.affectation[bac_j] != cible_j:
                    cible_j_avant = EmplacementStock(cible_j.colonne, cible_j.cote, Profondeur.AVANT)
                    bac_k = cibles_inverse.get(cible_j_avant)
                    if bac_k is not None:
                        self.arcs.append(Arc(source=bac_j, cible=bac_k, type_dep=TypeDependance.SEQUENCEMENT))

    def trouver_bacs_sans_blocage(self) -> list[int]:
        """
        Identifie les bacs qui n'ont aucune contrainte préalable.
        
        Returns:
            list[int]: La liste des identifiants de bacs n'ayant aucun blocage entrant.
        """
        bacs_bloques = {arc.cible for arc in self.arcs}
        return [bac for bac in self.noeuds if bac not in bacs_bloques]

    def detecter_cycle(self) -> list[Arc] | None:
        """
        Exécute un parcours en profondeur (DFS) pour détecter un cycle.
        
        Un cycle (deadlock) correspond à une séquence bloquante circulaire 
        du type A bloque B qui bloque ... qui bloque A.
        
        Returns:
            list[Arc] | None: La liste des arcs formant le premier cycle détecté,
            ou None si le graphe est acyclique.
        """
        visite = set()
        pile_appel = []
        pile_arcs = []

        adj = {n: [] for n in self.noeuds}
        for arc in self.arcs:
            adj[arc.source].append(arc)

        def dfs(noeud: int) -> list[Arc] | None:
            visite.add(noeud)
            pile_appel.append(noeud)
            for arc in adj[noeud]:
                voisin = arc.cible
                pile_arcs.append(arc)
                if voisin in pile_appel:  # Cycle détecté
                    idx = pile_appel.index(voisin)
                    return pile_arcs[idx:]
                if voisin not in visite:
                    cycle = dfs(voisin)
                    if cycle:
                        return cycle
                pile_arcs.pop()
            pile_appel.pop()
            return None

        for noeud in self.noeuds:
            if noeud not in visite:
                cycle = dfs(noeud)
                if cycle:
                    return cycle
        return None

    def analyser_graphe(self) -> dict:
        """
        Extrait les métriques topologiques globales du graphe.
        
        Utilisé pour le tracking itératif et l'analyse post-mortem des échecs.

        Returns:
            dict: Un dictionnaire contenant les clés :
                - somme_degres_entrants (int): Proxy de la complexité de blocage global.
                - degre_max (int): Le degré entrant le plus élevé parmi tous les bacs.
                - nb_bacs_bloques (int): Le nombre total de bacs subissant un blocage.
                - bacs_bloqueurs (list[int]): Les bacs causant un blocage.
                - nb_noeuds_en_cycle (int): Le nombre de bacs piégés dans un interblocage.
        """
        degres_entrants = {n: 0 for n in self.noeuds}
        degres_sortants = {n: 0 for n in self.noeuds}
        for arc in self.arcs:
            degres_entrants[arc.cible] += 1
            degres_sortants[arc.source] += 1

        somme_degres = sum(degres_entrants.values())
        degre_max = max(degres_entrants.values()) if degres_entrants else 0
        bacs_bloqueurs = [n for n in self.noeuds if degres_sortants[n] > 0]

        noeuds_en_cycle = set()
        cycle = self.detecter_cycle()
        if cycle:
            for arc in cycle:
                noeuds_en_cycle.add(arc.source)
                noeuds_en_cycle.add(arc.cible)

        return {
            "somme_degres_entrants": somme_degres,
            "degre_max": degre_max,
            "nb_bacs_bloques": len({arc.cible for arc in self.arcs}),
            "bacs_bloqueurs": bacs_bloqueurs,
            "nb_noeuds_en_cycle": len(noeuds_en_cycle),
        }

    def analyser_cycles_et_buffers(self) -> dict:
        """
        Exécute l'algorithme de Tarjan pour trouver toutes les composantes fortement connexes.
        
        Identifie les cycles complexes et calcule la disponibilité des buffers de dégagement
        par rapport aux bacs impliqués dans ces cycles.
        
        Returns:
            dict: Les métriques de survie des cycles, avec la structure détaillée 
            pour les buffers extra-cycles.
        """
        index = 0
        indices = {}
        lowlinks = {}
        pile = []
        en_pile = set()
        sccs = []
        
        adj = {n: [] for n in self.noeuds}
        for arc in self.arcs:
            adj[arc.source].append(arc.cible)

        def strongconnect(v):
            nonlocal index
            indices[v] = index
            lowlinks[v] = index
            index += 1
            pile.append(v)
            en_pile.add(v)
            
            for w in adj[v]:
                if w not in indices:
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif w in en_pile:
                    lowlinks[v] = min(lowlinks[v], indices[w])
                    
            if lowlinks[v] == indices[v]:
                scc = set()
                while True:
                    w = pile.pop()
                    en_pile.remove(w)
                    scc.add(w)
                    if w == v:
                        break
                if len(scc) > 1:
                    sccs.append(scc)
                    
        for v in self.noeuds:
            if v not in indices:
                strongconnect(v)
                
        colonnes_tous_cycles = set()
        for scc in sccs:
            for bac_id in scc:
                emp = self.config_courante.affectation[bac_id]
                colonnes_tous_cycles.add((emp.colonne, emp.cote))
                
        vides_accessibles = [
            emp for emp in self.config_courante.emplacements_vides_stock()
            if self.config_courante.est_accessible(emp)
        ]
        
        resultats_cycles = []
        for scc in sccs:
            colonnes_ce_cycle = set()
            for bac_id in scc:
                emp = self.config_courante.affectation[bac_id]
                colonnes_ce_cycle.add((emp.colonne, emp.cote))
                
            buffers_hors_ce_cycle = [
                emp for emp in vides_accessibles 
                if (emp.colonne, emp.cote) not in colonnes_ce_cycle
            ]
            
            buffers_hors_tous_cycles = [
                emp for emp in vides_accessibles 
                if (emp.colonne, emp.cote) not in colonnes_tous_cycles
            ]
            
            resultats_cycles.append({
                "taille": len(scc),
                "bacs": sorted(list(scc)),
                "nb_buffers_hors_ce_cycle": len(buffers_hors_ce_cycle),
                "nb_buffers_hors_tous_cycles": len(buffers_hors_tous_cycles)
            })
            
        return {
            "nb_cycles": len(sccs),
            "details_cycles": resultats_cycles,
            "nb_buffers_hors_tous_cycles": len([emp for emp in vides_accessibles if (emp.colonne, emp.cote) not in colonnes_tous_cycles]) if sccs else len(vides_accessibles)
        }

# =============================================================================
# 2. Heuristique H1 (Greedy Eligible-First)
# =============================================================================

class HeuristiqueConstructive:
    """
    Implémentation de l'heuristique H1.
    
    Elle privilégie les bacs éligibles (Happy Path) selon des priorités empiriques
    (poids statiques), et casse les deadlocks en évaluant les buffers par score
    de pénalité (méthode du Grand-M).
    
    Attributes:
        courante (Configuration): État courant de l'entrepôt.
        cible (Configuration): État cible final.
        colonne_navette (int): Position actuelle de la navette.
        distance_totale_x (int): Distance cumulée parcourue.
        sequence_mouvements (list[str]): Historique des actions effectuées.
        nb_double_handling (int): Compteur des dépôts en zone buffer.
    """
    def __init__(self, config_init: Configuration, config_cible: Configuration, p_lib=1, p_vul=2, p_ino=3, pen_avant=1, cap_tabou=20):
        self.courante = config_init
        self.cible = config_cible
        self.colonne_navette = 0
        self.distance_totale_x = 0
        self.sequence_mouvements = []
        self.nb_double_handling = 0   # Mouvements de buffer non-productifs
        self.derniers_buffers = {}    # bac_id -> itération du dernier buffer
        self.tabu_mouvements = {}     # (bac_id, emp) -> itération_expiration
        self.iteration_actuelle = 0
        self.historique: list[dict] = []  # Tracking : métriques graphiques par itération

        n_bacs = len(config_init.affectation)
        # Plafond sécurisé : n² plafonné à 20×n pour éviter timeout sur grandes instances
        self.max_iterations = min(n_bacs ** 2, n_bacs * 20)
        self.p_lib = p_lib
        self.p_vul = p_vul
        self.p_ino = p_ino
        self.pen_avant = pen_avant
        self.cap_tabou = cap_tabou

        # Détection d'échec précoce avant de lancer la résolution
        self._verifier_instance()

    def _bac_est_place(self, bac_id: int) -> bool:
        """
        Vérifie si le bac a atteint sa destination finale.
        
        Args:
            bac_id (int): L'identifiant du bac.
            
        Returns:
            bool: True si le bac est sur sa cible.
        """
        return self.courante.affectation[bac_id] == self.cible.affectation[bac_id]

    def _est_cible_pour_un_bac(self, emp: EmplacementStock) -> bool:
        """
        Vérifie si un emplacement est une destination finale pour n'importe quel bac.
        
        Args:
            emp (EmplacementStock): L'emplacement à tester.
            
        Returns:
            bool: True si l'emplacement est ciblé par au moins un bac.
        """
        return emp in self.cible.affectation.values()

    def _verifier_instance(self):
        """
        Détection d'échec précoce : vérifie la cohérence structurelle avant résolution.
        
        Raises:
            ValueError: Si l'instance est physiquement ou logiquement invalide 
            (bacs manquants, cibles doublons, cibles impossibles).
        """
        aff_init = self.courante.affectation
        aff_cible = self.cible.affectation

        # 1. Même ensemble de bacs dans les deux configurations
        if set(aff_init.keys()) != set(aff_cible.keys()):
            raise ValueError("Les bacs de la config initiale et cible ne correspondent pas.")

        # 2. Unicité des cibles (pas deux bacs vers le même emplacement final)
        cibles = list(aff_cible.values())
        if len(cibles) != len(set(cibles)):
            raise ValueError("Deux bacs ou plus partagent la même destination finale.")

        # 3. Cohérence AVANT/ARRIÈRE : tout bac ciblant un AVANT doit avoir un
        #    bac compagnon ciblant l'ARRIÈRE de la même demi-colonne.
        cibles_set = set(aff_cible.values())
        for bac_id, cible_emp in aff_cible.items():
            if cible_emp.profondeur == Profondeur.AVANT:
                emp_arr = EmplacementStock(cible_emp.colonne, cible_emp.cote, Profondeur.ARRIERE)
                if emp_arr not in cibles_set:
                    raise ValueError(
                        f"Bac {bac_id} cible un AVANT ({cible_emp}) sans bac ciblant l'ARRIÈRE correspondant."
                    )

    def trouver_buffer(self, bac_id: int) -> tuple[EmplacementStock | None, float]:
        """
        Recherche le meilleur emplacement vide accessible comme buffer temporaire.

        Le score combine les distances de déplacement et des pénalités dissuasives 
        (méthode Big-M) pour éviter les oscillations (tabou), le vol de cibles
        et la création de nouveaux blocages.

        Args:
            bac_id (int): L'identifiant du bac à déplacer.
            
        Returns:
            tuple[EmplacementStock | None, float]: Le meilleur buffer trouvé et son score.
            Si aucun buffer n'est accessible, retourne (None, inf).
        """
        emp_actuel = self.courante.affectation[bac_id]
        col_actuelle = emp_actuel.colonne
        col_cible = self.cible.affectation[bac_id].colonne

        vides_accessibles = [
            emp for emp in self.courante.emplacements_vides_stock()
            if self.courante.est_accessible(emp)
        ]

        meilleur_buffer = None
        meilleur_score = float('inf')

        for emp in vides_accessibles:
            score = abs(col_actuelle - emp.colonne) + abs(emp.colonne - col_cible)

            # Légère préférence pour l'ARRIERE (laisse l'AVANT disponible)
            if emp.profondeur == Profondeur.AVANT:
                score += self.pen_avant

            # PÉNALITÉ TABOU : interdit de revenir sur la position source récente
            expiration = self.tabu_mouvements.get((bac_id, emp), -1)
            if self.iteration_actuelle <= expiration:
                score += 1000 * self.courante.etage.n_colonnes  # Exclusion quasi-totale dynamique

            # PÉNALITÉ CIBLE : évite de voler la cible finale d'un autre bac
            if self._est_cible_pour_un_bac(emp):
                score += 100 * self.courante.etage.n_colonnes  # Big-M dynamique

            # PÉNALITÉ PRISON : évite de bloquer un bac mal placé à l'ARRIERE
            if emp.profondeur == Profondeur.AVANT:
                emp_arr = EmplacementStock(emp.colonne, emp.cote, Profondeur.ARRIERE)
                bac_arr = self.courante.bac_a(emp_arr)
                if bac_arr is not None and self.cible.affectation[bac_arr] != emp_arr:
                    score += 500 * self.courante.etage.n_colonnes  # Big-M dynamique prison

            if score < meilleur_score:
                meilleur_score = score
                meilleur_buffer = emp

        return meilleur_buffer, meilleur_score

    def _deplacer_bac(self, bac_id: int, dest: EmplacementStock, est_buffer: bool = False):
        """
        Exécute le déplacement d'un bac et met à jour toutes les métriques.
        
        Actualise la distance de la navette, la configuration courante de l'entrepôt, 
        la liste tabou pour éviter les retours en arrière immédiats, et le journal.
        
        Args:
            bac_id (int): L'identifiant du bac.
            dest (EmplacementStock): L'emplacement cible du mouvement.
            est_buffer (bool): True si le déplacement est vers un buffer.
        """
        source = self.courante.affectation[bac_id]

        self.distance_totale_x += (
            abs(self.colonne_navette - source.colonne) +
            abs(source.colonne - dest.colonne)
        )
        self.colonne_navette = dest.colonne
        self.courante.affectation[bac_id] = dest

        # Durée tabou proportionnelle à la taille de l'entrepôt
        # [PARAMÈTRE OPTIMISÉ] cap = 20 pour H1
        duree_tabou = min(self.cap_tabou, len(self.courante.affectation) // 2)
        self.tabu_mouvements[(bac_id, source)] = self.iteration_actuelle + duree_tabou

        action = (
            f"Bac {bac_id:2d} : "
            f"{source.colonne:2d}-{source.cote.value}-{source.profondeur.value} -> "
            f"{dest.colonne:2d}-{dest.cote.value}-{dest.profondeur.value}"
        )
        if est_buffer:
            action += " (BUFFER)"
            self.nb_double_handling += 1
        self.sequence_mouvements.append(action)

    def _calculer_poids_strategique(self, bac_id: int) -> int:
        """
        Détermine la priorité d'un bac éligible pour le Happy Path.

        Catégorise le bac en trois profils (du plus au moins prioritaire) :
          1. Libérateur : En position AVANT et bloque un bac ARRIERE mal placé.
          2. Vulnérable : En position ARRIERE, risque d'être bloqué.
          3. Inoffensif : Pas d'impact direct immédiat.

        Args:
            bac_id (int): L'identifiant du bac à évaluer.
            
        Returns:
            int: Le poids statique attribué (plus faible = plus prioritaire).
        """
        # Guard : si le bac est déjà à sa cible (ne devrait pas arriver), profil neutre
        if self._bac_est_place(bac_id):
            return self.p_ino

        emp = self.courante.affectation[bac_id]

        if emp.profondeur == Profondeur.ARRIERE:
            return self.p_vul  # Vulnérable

        emp_arriere = EmplacementStock(emp.colonne, emp.cote, Profondeur.ARRIERE)
        bac_arriere = self.courante.bac_a(emp_arriere)

        if bac_arriere is None:
            return self.p_ino  # Inoffensif (pas de bac derrière)
        elif self._bac_est_place(bac_arriere):
            return self.p_ino  # Inoffensif (le bac derrière est déjà bien placé)
        else:
            return self.p_lib  # Libérateur

    def planifier_mouvements(self):
        """
        Exécute la boucle principale de résolution itérative.

        Sépare le flux en deux phases :
          - PHASE 1 (Happy Path) : Sélectionne et place un bac sans contrainte
            vers sa cible (heuristique gloutonne).
          - PHASE 2 (Deadlock) : En l'absence de bac libre, identifie le cycle,
            choisit une racine physique et la déplace en buffer temporaire.
            
        Raises:
            TimeoutHeuristique: Si le nombre maximal d'itérations est dépassé.
            DeadlockInsoluble: Si aucun buffer n'est accessible pour débloquer.
        """
        while True:
            self.iteration_actuelle += 1
            if self.iteration_actuelle > self.max_iterations:
                raise TimeoutHeuristique(
                    f"Limite de {self.max_iterations} itérations atteinte sans convergence."
                )

            bacs_a_placer = [
                bac_id for bac_id, emp in self.courante.affectation.items()
                if emp != self.cible.affectation[bac_id]
            ]
            if not bacs_a_placer:
                break  # Tous les bacs sont à leur cible → terminé

            vides_accessibles = [
                emp for emp in self.courante.emplacements_vides_stock()
                if self.courante.est_accessible(emp)
            ]
            if not vides_accessibles:
                raise DeadlockInsoluble("Aucun emplacement vide accessible (instance invalide ?).")

            # --- TRACKING ITÉRATIF : métriques graphiques de l'état courant ---
            graphe = GrapheDependance(self.courante, self.cible)
            metriques = graphe.analyser_graphe()
            self.historique.append({
                "iteration": self.iteration_actuelle,
                "bacs_mal_places": len(bacs_a_placer),
                "somme_degres_entrants": metriques["somme_degres_entrants"],
                "degre_max": metriques["degre_max"],
                "nb_noeuds_en_cycle": metriques["nb_noeuds_en_cycle"],
            })

            eligibles = [b for b in graphe.trouver_bacs_sans_blocage() if b in bacs_a_placer]

            # --- PHASE 1 : HAPPY PATH (Greedy pur : poids statique + distance) ---
            if eligibles:
                eligibles.sort(key=lambda b: (
                    self._calculer_poids_strategique(b),
                    abs(self.courante.affectation[b].colonne - self.colonne_navette)
                ))
                bac_choisi = eligibles[0]
                self._deplacer_bac(bac_choisi, self.cible.affectation[bac_choisi], est_buffer=False)
                continue

            # --- PHASE 2 : DEADLOCK ---
            arcs_cycle = graphe.detecter_cycle()
            if not arcs_cycle:
                raise DeadlockInsoluble("Blocage insoluble sans cycle détecté (incohérence logique).")

            # Identifier les racines physiques du cycle (bacs AVANT accessibles)
            candidats = set()
            for arc in arcs_cycle:
                bac = arc.source
                emp_bac = self.courante.affectation[bac]
                if emp_bac.profondeur == Profondeur.AVANT:
                    candidats.add(bac)
                else:
                    emp_avant = EmplacementStock(emp_bac.colonne, emp_bac.cote, Profondeur.AVANT)
                    bac_avant = self.courante.bac_a(emp_avant)
                    candidats.add(bac_avant if bac_avant is not None else bac)

            if not candidats:
                raise DeadlockInsoluble("Cycle insoluble : aucune racine physique accessible.")

            evaluations = [
                (b, buf, score)
                for b in candidats
                for buf, score in [self.trouver_buffer(b)]
                if buf is not None
            ]

            if not evaluations:
                raise DeadlockInsoluble("Aucun buffer accessible pour casser le cycle.")

            evaluations.sort(key=lambda x: (x[2], self.derniers_buffers.get(x[0], -1)))
            bac_choisi, buffer_emp, _ = evaluations[0]

            self._deplacer_bac(bac_choisi, buffer_emp, est_buffer=True)
            self.derniers_buffers[bac_choisi] = self.iteration_actuelle

# =============================================================================
# 3. Exécution et Interface
# =============================================================================

def resoudre_instance(fichier: str, verbose: bool = True) -> dict:
    """
    Point d'entrée pour la résolution complète d'une instance avec H1.
    
    Args:
        fichier (str): Le chemin vers le fichier CSV de l'instance.
        verbose (bool): Si True, affiche les logs de la résolution.
        
    Returns:
        dict: Les métriques de la résolution (statut, distance, temps, etc.).
    """
    start_time = time.time()
    heuristique = None
    echec_degre_max = echec_somme_degres = echec_nb_cycles = echec_iteration = 0
    try:
        init, cible = importer_instance(fichier)
        if verbose:
            print(init.afficher("ETAT INITIAL"))
            print(cible.afficher("ETAT CIBLE"))

            graphe_init = GrapheDependance(init, cible)
            analyse = graphe_init.analyser_cycles_et_buffers()
            print(f"\n--- Analyse Initiale des Cycles ---")
            print(f"Nombre de cycles (SCC > 1) détectés : {analyse['nb_cycles']}")
            for i, c in enumerate(analyse['details_cycles'], 1):
                print(f"  Cycle {i} (Taille: {c['taille']} bacs) :")
                print(f"    - Buffers disponibles hors de CE cycle : {c['nb_buffers_hors_ce_cycle']}")
                print(f"    - Buffers disponibles hors de TOUS les cycles : {c['nb_buffers_hors_tous_cycles']}")
            print("-" * 33)

        heuristique = HeuristiqueConstructive(init, cible)
        heuristique.planifier_mouvements()
        distance_val = heuristique.distance_totale_x
        mouvements = len(heuristique.sequence_mouvements)
        statut = "Succès"
        if verbose:
            print(f"\n[Terminé] Distance Totale X : {distance_val}")
            for i, m in enumerate(heuristique.sequence_mouvements, 1):
                print(f"  {i:3d}. {m}")
    except (DeadlockInsoluble, TimeoutHeuristique) as e:
        statut = "Deadlock" if isinstance(e, DeadlockInsoluble) else "Timeout"
        distance_val = heuristique.distance_totale_x if heuristique else -1
        mouvements = len(heuristique.sequence_mouvements) if heuristique else -1
        # Capture les métriques graphiques au moment de l'échec
        if heuristique and heuristique.historique:
            dernier = heuristique.historique[-1]
            echec_degre_max = dernier["degre_max"]
            echec_somme_degres = dernier["somme_degres_entrants"]
            echec_nb_cycles = dernier["nb_noeuds_en_cycle"]
            echec_iteration = dernier["iteration"]
        else:
            echec_degre_max = echec_somme_degres = echec_nb_cycles = echec_iteration = -1
        if verbose:
            print(f"\n[Échec - {statut}] {e}")
    except ValueError as e:
        statut = f"Instance invalide"
        distance_val, mouvements = -1, -1
        echec_degre_max = echec_somme_degres = echec_nb_cycles = echec_iteration = -1
        if verbose:
            print(f"\n[Échec - Validation] {e}")
    except Exception as e:
        statut = f"Erreur: {str(e)}"
        distance_val, mouvements = -1, -1
        echec_degre_max = echec_somme_degres = echec_nb_cycles = echec_iteration = -1
        if verbose:
            print(f"\n[Échec] {statut}")

    temps_exe = round(time.time() - start_time, 3)
    return {
        "Fichier": Path(fichier).name,
        "Statut": statut,
        "Distance_X": distance_val,
        "Mouvements": mouvements,
        "Double_Handling": heuristique.nb_double_handling if heuristique else 0,
        "Temps_s": temps_exe,
        # Métriques graphiques au moment de l'échec (0 si succès)
        "Echec_Degre_Max": echec_degre_max,
        "Echec_Somme_Degres": echec_somme_degres,
        "Echec_Nb_Cycles": echec_nb_cycles,
        "Echec_Iteration": echec_iteration,
        # Historique pour tracking itératif (liste de dict)
        "_historique": heuristique.historique if heuristique else [],
    }

def main():
    """
    Interface en ligne de commande (CLI) du module H1.
    
    Permet de lancer la résolution soit sur une instance unique (`une`), 
    soit en traitement par lot sur un dossier complet (`lot`).
    """
    parser = argparse.ArgumentParser(description="Heuristique H1 - SBS/RS")
    subparsers = parser.add_subparsers(dest="commande", required=True)

    p_une = subparsers.add_parser("une", help="Instance unique")
    p_une.add_argument("fichier", type=str)

    p_lot = subparsers.add_parser("lot", help="Dossier complet")
    p_lot.add_argument("-d", "--dossier", type=str, default="instances")
    p_lot.add_argument("-o", "--output", type=str, default="resultats/resultat_H1.csv")

    args = parser.parse_args()

    if args.commande == "une":
        res = resoudre_instance(args.fichier, verbose=True)
        print(f"Résultat : {res['Statut']} (Distance: {res['Distance_X']})")

    elif args.commande == "lot":
        dossier = Path(args.dossier)
        fichiers = list(dossier.glob("*.csv"))
        resultats = []
        for i, f in enumerate(fichiers, 1):
            print(f"[{i}/{len(fichiers)}] {f.name}...", end=" ", flush=True)
            res = resoudre_instance(str(f), verbose=False)
            print(f"-> {res['Statut']}")
            resultats.append(res)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f_out:
            writer = csv.DictWriter(
                f_out,
                fieldnames=[
                        "Fichier", "Statut", "Distance_X", "Mouvements",
                        "Double_Handling", "Temps_s",
                        "Echec_Degre_Max", "Echec_Somme_Degres",
                        "Echec_Nb_Cycles", "Echec_Iteration"
                    ]
            )
            writer.writeheader()
            writer.writerows({k: v for k, v in r.items() if k != "_historique"} for r in resultats)
        print(f"\nRésultats sauvegardés dans '{args.output}'.")

if __name__ == "__main__":
    main()