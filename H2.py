"""
Module H2 : Heuristique Lookahead (Recherche en Faisceau / Beam Search).

Ce module implémente une variante avancée (H2) pour la résolution du problème
de relocalisation de bacs (SBS/RS). Contrairement à une heuristique gloutonne,
H2 utilise un algorithme de Beam Search pour simuler et évaluer les conséquences
topologiques des mouvements sur une profondeur donnée.

Stratégie de résolution :
    1. Pour chaque bac candidat, générer les mouvements valides (cibles ou buffers).
    2. Filtrer drastiquement les buffers via des pénalités (cycles, blocages).
    3. Simuler les séquences de mouvements et évaluer l'état résultant du graphe.
    4. Sélectionner le mouvement qui maximise le déblocage global (Beam Search).

Notes :
    L'invariant "Back-First" garantit que tout emplacement arrière vide implique
    un emplacement avant vide. Aucun mouvement "ouvreur" basique n'est donc requis
    par défaut, bien qu'intégré en tant que coupe-circuit de sécurité.
"""

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
# Paramètres du Lookahead Profond (Beam Search)
# =============================================================================
POIDS_DEBLOQUE = 1000   # Critère 1 : Bacs définitivement débloqués
POIDS_SURVIE   = 500    # Critère 2 : Positions ouvertes extra-cycles restantes
POIDS_CASSURE  = 200    # Critère 3 : Cassure de cycle complexe
POIDS_DISTANCE = -1     # Critère 4 : Distance cumulée (Malus)
POIDS_DEPENDANCE = -20  # NOUVEAU : Pénalité par dépendance restante (privilégie la résolution des bacs très bloquants)
POIDS_DEGRE_MAX = -50   # NOUVEAU : Pénalité si un bac bloque beaucoup de bacs

LOOKAHEAD_DEPTH = 5   # Profondeur de l'arbre de recherche
BEAM_WIDTH      = 3     # Nombre d'états conservés à chaque niveau

# =============================================================================
# 0. Exceptions Personnalisées
# =============================================================================

class DeadlockInsoluble(Exception):
    """
    Exception levée lorsqu'un cycle ne peut être résolu.
    
    Indique une impasse physique stricte (ex. saturation de l'entrepôt, 
    aucun emplacement buffer accessible).
    """
    pass

class TimeoutHeuristique(Exception):
    """
    Exception levée lorsque la limite d'itérations est atteinte.
    
    Indique une stagnation de l'algorithme (oscillation ou boucle infinie)
    sans amélioration de l'état global.
    """
    pass

# =============================================================================
# 1. Structures de Graphe et Dépendances (identiques à H1)
# =============================================================================

class TypeDependance(Enum):
    """
    Énumération définissant la nature du blocage entre deux bacs.
    
    Attributes:
        DUR: Blocage physique LIFO (le bac à l'AVANT bloque le bac à l'ARRIÈRE).
        OCCUPATION: Blocage logique (un bac occupe la destination finale d'un autre).
        SEQUENCEMENT: Blocage d'ordre (le bac ARRIÈRE doit être placé avant le bac AVANT).
    """
    DUR = auto()
    OCCUPATION = auto()
    SEQUENCEMENT = auto()

@dataclass
class Arc:
    """
    Représente une dépendance (un arc orienté) dans le graphe de blocage.
    
    Un arc de `source` vers `cible` signifie que le bac `source` bloque le bac `cible`.
    
    Attributes:
        source (int): L'identifiant du bac qui bloque.
        cible (int): L'identifiant du bac qui est bloqué.
        type_dep (TypeDependance): La nature du blocage.
    """
    source: int
    cible: int
    type_dep: TypeDependance

class GrapheDependance:
    """
    Graphe orienté représentant l'ensemble des dépendances et blocages de l'entrepôt.
    
    Modélise les relations LIFO, d'occupation et de séquencement. Permet l'analyse
    topologique de l'état (détection de cycles, calcul des degrés) pour guider
    l'heuristique de résolution.
    """
    def __init__(self, config_courante: Configuration, config_cible: Configuration):
        self.config_courante = config_courante
        self.config_cible = config_cible
        self.noeuds = sorted(config_courante.affectation.keys())
        self.arcs: list[Arc] = []
        self._construire_graphe()

    def _construire_graphe(self):
        occupes = {emp: bac_id for bac_id, emp in self.config_courante.affectation.items()}
        cibles_inverse = {emp: bac_id for bac_id, emp in self.config_cible.affectation.items()}

        # 1. DÉPENDANCES PHYSIQUES (DUR)
        for c, s in self.config_courante.etage.demi_colonnes():
            emp_avant = EmplacementStock(c, s, Profondeur.AVANT)
            emp_arriere = EmplacementStock(c, s, Profondeur.ARRIERE)
            if emp_avant in occupes and emp_arriere in occupes:
                self.arcs.append(Arc(
                    source=occupes[emp_avant],
                    cible=occupes[emp_arriere],
                    type_dep=TypeDependance.DUR
                ))

        # 2. DÉPENDANCES D'OCCUPATION
        for bac_j in sorted(self.config_cible.affectation.keys()):
            cible_j = self.config_cible.affectation[bac_j]
            if self.config_courante.affectation[bac_j] == cible_j:
                continue
            if cible_j in occupes:
                bac_k = occupes[cible_j]
                if bac_k != bac_j:
                    self.arcs.append(Arc(source=bac_k, cible=bac_j, type_dep=TypeDependance.OCCUPATION))
            if cible_j.profondeur == Profondeur.ARRIERE:
                cible_j_avant = EmplacementStock(cible_j.colonne, cible_j.cote, Profondeur.AVANT)
                if cible_j_avant in occupes:
                    bac_k = occupes[cible_j_avant]
                    if bac_k != bac_j:
                        self.arcs.append(Arc(source=bac_k, cible=bac_j, type_dep=TypeDependance.OCCUPATION))

        # 3. DÉPENDANCES DE SÉQUENCEMENT
        for bac_j in sorted(self.config_cible.affectation.keys()):
            cible_j = self.config_cible.affectation[bac_j]
            if cible_j.profondeur == Profondeur.ARRIERE:
                if self.config_courante.affectation[bac_j] != cible_j:
                    cible_j_avant = EmplacementStock(cible_j.colonne, cible_j.cote, Profondeur.AVANT)
                    bac_k = cibles_inverse.get(cible_j_avant)
                    if bac_k is not None:
                        self.arcs.append(Arc(source=bac_j, cible=bac_k, type_dep=TypeDependance.SEQUENCEMENT))

    def trouver_bacs_sans_blocage(self) -> list[int]:
        """
        Identifie les bacs qui ne subissent aucun blocage.
        
        Returns:
            list[int]: Liste triée des identifiants des bacs libres (degré entrant nul).
        """
        bacs_bloques = {arc.cible for arc in self.arcs}
        return sorted([bac for bac in self.noeuds if bac not in bacs_bloques])

    def detecter_cycle(self) -> list[Arc] | None:
        """
        Détecte la présence du premier cycle dans le graphe de dépendance via DFS.
        
        Returns:
            list[Arc] | None: La liste des arcs formant le cycle s'il existe, sinon None.
        """
        visite, pile_appel, pile_arcs = set(), [], []
        adj = {n: [] for n in self.noeuds}
        for arc in self.arcs:
            adj[arc.source].append(arc)
            
        # Rendre l'ordre de visite DFS purement déterministe
        for n in self.noeuds:
            adj[n].sort(key=lambda a: (a.cible, a.type_dep.value))

        def dfs(noeud: int) -> list[Arc] | None:
            visite.add(noeud)
            pile_appel.append(noeud)
            for arc in adj[noeud]:
                voisin = arc.cible
                pile_arcs.append(arc)
                if voisin in pile_appel:
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
        Analyse la topologie du graphe et extrait ses métriques clés.
        
        Returns:
            dict: Dictionnaire contenant :
                - somme_degres_entrants (int): Le nombre total de dépendances.
                - degre_max (int): Le degré entrant maximum (le bac le plus bloqué).
                - nb_bacs_bloques (int): Le nombre de bacs subissant au moins un blocage.
                - bacs_bloqueurs (list[int]): Les bacs qui bloquent d'autres bacs.
                - nb_noeuds_en_cycle (int): Le nombre de nœuds impliqués dans le premier cycle.
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
        Exécute une analyse exhaustive des cycles et de la disponibilité des buffers.
        
        Utilise l'algorithme de Tarjan pour extraire toutes les Composantes Fortement
        Connexes (SCC). Pour chaque cycle, évalue la quantité d'emplacements libres
        utilisables comme buffers de contingence.
        
        Returns:
            dict: Détails des cycles incluant leur taille et les buffers de contingence.
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
# 2. Heuristique H2 (Lookahead / Anticipation 1-coup)
# =============================================================================

class HeuristiqueLookahead:
    """
    Moteur de résolution par anticipation (Beam Search / Deep Lookahead).
    
    Simule et explore l'arbre des mouvements possibles pour identifier la séquence
    qui optimise la résolution globale du graphe de dépendance. Conserve un faisceau
    restreint d'états (beam_width) à chaque niveau (depth) pour maîtriser la complexité.
    
    Attributes:
        courante (Configuration): L'état actuel de l'entrepôt.
        cible (Configuration): L'état objectif à atteindre.
        colonne_navette (int): La position actuelle de la navette.
        distance_totale_x (float): Le coût cumulé des déplacements physiques.
        depth (int): La profondeur maximale d'anticipation de l'arbre.
        beam_width (int): Le nombre de meilleurs états conservés par niveau d'arbre.
    """
    def __init__(self, config_init: Configuration, config_cible: Configuration, p_lib=1, p_vul=2, p_ino=3, pen_avant=1, cap_tabou=5, depth=LOOKAHEAD_DEPTH, beam_width=BEAM_WIDTH):
        self.courante = config_init
        self.cible = config_cible
        self.colonne_navette = 0
        self.distance_totale_x = 0
        self.pen_avant = pen_avant
        self.cap_tabou = cap_tabou
        self.depth = depth
        self.beam_width = beam_width
        
        self.sequence_mouvements = []
        self.nb_double_handling = 0
        self.derniers_buffers = {}
        self.tabu_mouvements = {}
        self.iteration_actuelle = 0
        self.historique: list[dict] = []  # Tracking : métriques graphiques par itération

        n_bacs = len(config_init.affectation)
        # Plafond sécurisé : n² plafonné à 20×n
        self.max_iterations = min(n_bacs ** 2, n_bacs * 20)

        # Détection d'échec précoce avant de lancer la résolution
        self._verifier_instance()

    def _verifier_instance(self):
        """
        Valide l'intégrité de l'instance avant de lancer la résolution.
        
        Vérifie la correspondance des bacs entre les états et s'assure qu'aucun
        bac ne cible l'avant d'une demi-colonne dont l'arrière serait vide.
        
        Raises:
            ValueError: Si la configuration initiale ou cible est physiquement ou logiquement invalide.
        """
        aff_init = self.courante.affectation
        aff_cible = self.cible.affectation
        if set(aff_init.keys()) != set(aff_cible.keys()):
            raise ValueError("Les bacs de la config initiale et cible ne correspondent pas.")
        cibles = list(aff_cible.values())
        if len(cibles) != len(set(cibles)):
            raise ValueError("Deux bacs ou plus partagent la même destination finale.")
        cibles_set = set(aff_cible.values())
        for bac_id, cible_emp in aff_cible.items():
            if cible_emp.profondeur == Profondeur.AVANT:
                emp_arr = EmplacementStock(cible_emp.colonne, cible_emp.cote, Profondeur.ARRIERE)
                if emp_arr not in cibles_set:
                    raise ValueError(
                        f"Bac {bac_id} cible un AVANT ({cible_emp}) sans bac ciblant l'ARRIÈRE correspondant."
                    )

    def _deplacer_bac(self, bac_id: int, dest: EmplacementStock, est_buffer: bool = False):
        """
        Applique un mouvement physique définitif sur l'état courant.
        
        Met à jour la position de la navette, la configuration de l'entrepôt, 
        enregistre le mouvement dans l'historique et actualise la liste tabou.
        
        Args:
            bac_id (int): L'identifiant du bac à déplacer.
            dest (EmplacementStock): La destination (cible finale ou buffer temporaire).
            est_buffer (bool): True si le déplacement est vers un buffer de transit.
        """
        source = self.courante.affectation[bac_id]

        self.distance_totale_x += (
            abs(self.colonne_navette - source.colonne) +
            abs(source.colonne - dest.colonne)
        )
        self.colonne_navette = dest.colonne
        self.courante.affectation[bac_id] = dest

        duree_tabou = min(self.cap_tabou, len(self.courante.affectation) // 2)
        self.tabu_mouvements[(bac_id, source)] = self.iteration_actuelle + duree_tabou

        action = (
            f"Bac {bac_id:2d} : "
            f"{source.colonne:2d}-{source.cote.value}-{source.profondeur.value} -> "
            f"{dest.colonne:2d}-{dest.cote.value}-{dest.profondeur.value}"
        )
        if est_buffer:
            # Vérification si c'est intra-cycle ou extra-cycle
            graphe = GrapheDependance(self.courante, self.cible)
            arcs = graphe.detecter_cycle()
            if arcs:
                colonnes_cycle = set()
                for arc in arcs:
                    colonnes_cycle.add(self.courante.affectation[arc.source].colonne)
                    colonnes_cycle.add(self.cible.affectation[arc.source].colonne)
                    colonnes_cycle.add(self.courante.affectation[arc.cible].colonne)
                    colonnes_cycle.add(self.cible.affectation[arc.cible].colonne)
                if dest.colonne in colonnes_cycle:
                    action += " (BUFFER INTRA-CYCLE)"
                else:
                    action += " (BUFFER EXTRA-CYCLE)"
            else:
                action += " (BUFFER EXTRA-CYCLE)"
                
            self.nb_double_handling += 1
            self.derniers_buffers[bac_id] = self.iteration_actuelle
        self.sequence_mouvements.append(action)
        self.dernier_bac_deplace = bac_id

    def _generer_coups_valides(self, config_courante: Configuration, navette_col: int) -> list[tuple[int, EmplacementStock, float, bool]]:
        """
        Génère et filtre les prochains mouvements légaux à partir d'une configuration.
        
        Utilise un système de pénalités (Big-M) pour rejeter les buffers intra-cycles,
        les destructions de buffers arrière, et les vols de cibles. Priorise l'évaluation
        des bacs responsables du plus grand nombre de blocages (degré sortant).
        
        Args:
            config_courante (Configuration): L'état de l'entrepôt à évaluer.
            navette_col (int): La colonne actuelle de la navette.
            
        Returns:
            list[tuple[int, EmplacementStock, float, bool]]: Liste des mouvements évalués,
            format (bac_id, destination, cout_mouvement, est_buffer).
        """
        graphe = GrapheDependance(config_courante, self.cible)
        bacs_a_placer = [b for b, emp in config_courante.affectation.items() if emp != self.cible.affectation[b]]
        if not bacs_a_placer:
            return []
            
        eligibles = [b for b in graphe.trouver_bacs_sans_blocage() if b in bacs_a_placer]
        coups = []
        
        if eligibles:
            # Happy path : on déplace les bacs sans blocage vers leur cible finale
            for b in eligibles:
                cible_b = self.cible.affectation[b]
                cout = abs(navette_col - config_courante.affectation[b].colonne) + abs(config_courante.affectation[b].colonne - cible_b.colonne)
                coups.append((b, cible_b, cout, False))
        else:
            # Deadlock / Cycle : on doit déplacer une racine physique dans un buffer
            arcs_cycle = graphe.detecter_cycle()
            candidats = set()
            colonnes_en_cycle = set()
            
            if not arcs_cycle:
                # Sécurité : blocage sans cycle strict
                candidats = set(bacs_a_placer)
            else:
                for arc in arcs_cycle:
                    bac = arc.source
                    emp_bac = config_courante.affectation[bac]
                    colonnes_en_cycle.add((emp_bac.colonne, emp_bac.cote))
                    
                    cible_noeud = arc.cible
                    emp_cible = config_courante.affectation[cible_noeud]
                    colonnes_en_cycle.add((emp_cible.colonne, emp_cible.cote))
                    
                    if emp_bac.profondeur == Profondeur.AVANT:
                        candidats.add(bac)
                    else:
                        emp_av = EmplacementStock(emp_bac.colonne, emp_bac.cote, Profondeur.AVANT)
                        bac_av = config_courante.bac_a(emp_av)
                        if bac_av is not None:
                            candidats.add(bac_av)  # L'avant devient la racine physique
                            colonnes_en_cycle.add((emp_av.colonne, emp_av.cote))
                        else:
                            candidats.add(bac)
            
            # --- PRIORISATION DES DEPENDANCES ---
            # 1. Calculer les degrés sortants (combien de bacs sont bloqués par ce bac)
            degres_sortants = {n: 0 for n in graphe.noeuds}
            for arc in graphe.arcs:
                degres_sortants[arc.source] += 1
                
            candidats_list = list(candidats)
            # Trier pour prioriser l'évaluation des bacs qui bloquent le plus de monde, en départageant par ancienneté de buffer
            candidats_list.sort(key=lambda b: (degres_sortants.get(b, 0), -self.derniers_buffers.get(b, -1), b), reverse=True)
            
            vides_accessibles = [
                emp for emp in config_courante.emplacements_vides_stock()
                if config_courante.est_accessible(emp)
            ]
            
            coups_possibles = []
            for b in candidats_list:
                cible_b = self.cible.affectation[b]
                buffers_evalues = []
                
                # Évaluer tous les buffers possibles
                for emp in vides_accessibles:
                    source_emp = config_courante.affectation[b]
                    # Interdiction formelle de déplacer un bac de l'arrière vers l'avant de sa propre demi-colonne (auto-emprisonnement absurde)
                    if source_emp.colonne == emp.colonne and source_emp.cote == emp.cote:
                        if source_emp.profondeur == Profondeur.ARRIERE and emp.profondeur == Profondeur.AVANT:
                            continue
                            
                    # Interdiction formelle d'utiliser un emplacement de sa propre demi-colonne cible comme buffer temporaire
                    if cible_b.colonne == emp.colonne and cible_b.cote == emp.cote:
                        continue
                            
                    cout_mvt = abs(navette_col - source_emp.colonne) + abs(source_emp.colonne - emp.colonne)
                    cout_futur = abs(emp.colonne - cible_b.colonne)
                    penalite = 0
                    
                    # 1. Pénalité de profondeur et préservation des arrières
                    if emp.profondeur == Profondeur.AVANT:
                        penalite += self.pen_avant
                        
                        emp_arr = EmplacementStock(emp.colonne, emp.cote, Profondeur.ARRIERE)
                        bac_arr = config_courante.bac_a(emp_arr)
                        if bac_arr is None or bac_arr == b:
                            # Catastrophe : l'arrière est ou sera vide, poser à l'avant détruit un buffer accessible !
                            penalite += 500
                        elif self.cible.affectation.get(bac_arr) != emp_arr:
                            # Prison : l'arrière est mal placé, poser à l'avant l'emprisonne !
                            penalite += 300
                            
                    # 2. Vol de cible direct et indirect
                    if emp in self.cible.affectation.values():
                        penalite += 200
                    if emp.profondeur == Profondeur.AVANT:
                        emp_arr = EmplacementStock(emp.colonne, emp.cote, Profondeur.ARRIERE)
                        if emp_arr in self.cible.affectation.values():
                            penalite += 200
                        
                    # CRUCIAL : Prioriser les emplacements extra-cycles !
                    est_extra_cycle = (emp.colonne, emp.cote) not in colonnes_en_cycle
                    if not est_extra_cycle:
                        penalite += 1000
                        
                    score_buffer = cout_mvt + cout_futur + penalite
                    buffers_evalues.append((score_buffer, emp, cout_mvt))
                
                # Ne garder que les 3 meilleurs buffers par candidat (réduction combinatoire)
                buffers_evalues.sort(key=lambda x: (x[0], x[1].colonne, x[1].cote, x[1].profondeur))
                
                for score_buffer, emp, cout_mvt in buffers_evalues[:3]:
                    expiration = self.tabu_mouvements.get((b, emp), -1)
                    est_tabou = self.iteration_actuelle <= expiration
                    coups_possibles.append((b, emp, cout_mvt, True, est_tabou))
                    
            # --- RÉSILIENCE ANTI-DEADLOCK ---
            # On tente d'abord les mouvements non-tabous
            coups_non_tabous = [c for c in coups_possibles if not c[4]]
            if coups_non_tabous:
                for c in coups_non_tabous:
                    coups.append((c[0], c[1], c[2], c[3]))
            else:
                # Si tous les mouvements sont tabous, on autorise les mouvements tabous en dernier recours
                for c in coups_possibles:
                    coups.append((c[0], c[1], c[2], c[3]))
                    
            # --- GESTION D'URGENCE : CRÉATION DE BUFFER PAR MOUVEMENT OUVREUR ---
            # Si un bac candidat est à l'avant et que son arrière est vide, le pousser au fond crée instantanément un buffer libre à l'avant.
            for b in candidats_list:
                emp = config_courante.affectation[b]
                if emp.profondeur == Profondeur.AVANT:
                    emp_arr = EmplacementStock(emp.colonne, emp.cote, Profondeur.ARRIERE)
                    if config_courante.bac_a(emp_arr) is None:
                        cout = abs(navette_col - emp.colonne)
                        expiration = self.tabu_mouvements.get((b, emp_arr), -1)
                        if self.iteration_actuelle > expiration:
                            coups.append((b, emp_arr, cout, True))
                            
            # --- ULTIME RECOURS (SI AUCUN COUP N'EST DISPONIBLE) ---
            if not coups:
                # 1. Élargir le mouvement Ouvreur à n'importe quel bac mal placé de l'entrepôt
                for b in bacs_a_placer:
                    if b not in candidats_list:
                        emp = config_courante.affectation[b]
                        if emp.profondeur == Profondeur.AVANT:
                            emp_arr = EmplacementStock(emp.colonne, emp.cote, Profondeur.ARRIERE)
                            if config_courante.bac_a(emp_arr) is None:
                                cout = abs(navette_col - emp.colonne)
                                coups.append((b, emp_arr, cout, True))
                                
            if not coups and vides_accessibles:
                # 2. Lever l'interdiction de buffer cible finale en tout dernier recours
                for b in candidats_list:
                    for emp in vides_accessibles:
                        source_emp = config_courante.affectation[b]
                        if source_emp.colonne == emp.colonne and source_emp.cote == emp.cote and source_emp.profondeur == Profondeur.ARRIERE and emp.profondeur == Profondeur.AVANT:
                            continue
                        cout = abs(navette_col - source_emp.colonne) + abs(source_emp.colonne - emp.colonne)
                        coups.append((b, emp, cout, True))
                            
        return coups

    def _evaluer_etat(self, config_courante: Configuration, dist_cumulee: float) -> float:
        """
        Évalue la qualité topologique globale d'un état simulé.
        
        Applique l'équation de scoring polynomiale combinant :
        - Les bonus (bacs placés, bacs débloqués, survie des buffers extra-cycles).
        - Les malus (distance cumulée, dépendances résiduelles, goulets d'étranglement).
        
        Args:
            config_courante (Configuration): L'état simulé à évaluer.
            dist_cumulee (float): La distance physique cumulée jusqu'à cet état.
            
        Returns:
            float: Le score global de l'état (plus haut = meilleur).
        """
        graphe = GrapheDependance(config_courante, self.cible)
        bacs_a_placer = [b for b, emp in config_courante.affectation.items() if emp != self.cible.affectation[b]]
        
        # --- Critère 1 (Prioritaire) : Nombre de bacs définitivement débloqués ---
        bacs_libres = [b for b in graphe.trouver_bacs_sans_blocage() if b in bacs_a_placer]
        nb_debloques = len(bacs_libres)
        
        # (Bonus implicite pour les bacs déjà placés à leur destination finale)
        nb_bien_places = len(config_courante.affectation) - len(bacs_a_placer)
        
        # --- Récupération des métriques du graphe pour dépendances et cycles ---
        metriques = graphe.analyser_graphe()
        score_dependance = metriques["somme_degres_entrants"]
        score_degre_max = metriques["degre_max"]
        
        # --- Critère 2 (Survie) : Positions ouvertes extra-cycles ---
        cycle = graphe.detecter_cycle()
        colonnes_en_cycle = set()
        bacs_arriere_en_cycle = False
        nb_noeuds_cycle = metriques["nb_noeuds_en_cycle"]
        
        if cycle:
            for arc in cycle:
                for noeud in (arc.source, arc.cible):
                    emp = config_courante.affectation[noeud]
                    colonnes_en_cycle.add((emp.colonne, emp.cote))
                    if emp.profondeur == Profondeur.ARRIERE:
                        bacs_arriere_en_cycle = True
            
        vides_accessibles = [
            emp for emp in config_courante.emplacements_vides_stock()
            if config_courante.est_accessible(emp)
        ]
        
        if not vides_accessibles and bacs_a_placer:
            return -1000000.0 # Impasse absolue (Deadlock fatal)
        
        # Une position est extra-cycle si elle n'est pas dans les colonnes impliquées dans le cycle
        nb_extra_cycles = sum(1 for emp in vides_accessibles if (emp.colonne, emp.cote) not in colonnes_en_cycle)
        
        if nb_extra_cycles > 0:
            score_survie = nb_extra_cycles
        else:
            # S'il n'y a aucun buffer extra-cycle (ex: cycle géant), on valorise le nombre total de buffers ouverts
            score_survie = len(vides_accessibles) * 0.1
            
        # --- Critère 3 (Bonus) : Casser un cycle complexe ---
        if nb_noeuds_cycle == 0:
            score_cassure = 50.0  # Plus aucun cycle, situation saine
        else:
            score_cassure = -nb_noeuds_cycle  # On préfère les cycles les plus petits possibles
            
        # --- Critère 4 (Malus) : Distance cumulée ---
        score_distance = dist_cumulee
        
        score = (
            POIDS_DEBLOQUE * nb_debloques +
            POIDS_SURVIE * score_survie +
            POIDS_CASSURE * score_cassure +
            POIDS_DISTANCE * score_distance +
            POIDS_DEPENDANCE * score_dependance +  # Pousse à éliminer les blocages
            POIDS_DEGRE_MAX * score_degre_max +    # Vise spécifiquement les "gros" bloqueurs
            POIDS_DEBLOQUE * 2 * nb_bien_places  # Important pour forcer l'avancement global
        )
        return score

    def _beam_search(self) -> tuple[int, EmplacementStock, bool] | None:
        """
        Exécute la recherche par faisceau pour déterminer le mouvement optimal.
        
        Développe l'arbre des états futurs sur une profondeur définie (`depth`), 
        en ne conservant que les `beam_width` meilleures branches à chaque niveau
        selon la fonction `_evaluer_etat`.
        
        Returns:
            tuple[int, EmplacementStock, bool] | None: Le meilleur mouvement immédiat
            (bac_id, destination, est_buffer) menant à l'état futur optimal, ou None
            en cas d'impasse totale.
        """
        # État initial dans le faisceau : (score, config, dist, sequence_coups, navette_col)
        faisceau = [(0.0, self.courante, 0.0, [], self.colonne_navette)]
        
        meilleur_etat_global = None
        meilleur_score_global = float('-inf')
        
        for profondeur in range(self.depth):
            prochains_etats = []
            
            for score_parent, config_parent, dist_parent, seq_parent, nav_parent in faisceau:
                coups = self._generer_coups_valides(config_parent, nav_parent)
                
                if not coups:
                    # Branche morte ou problème résolu, on met à jour le meilleur global
                    if score_parent > meilleur_score_global and seq_parent:
                        meilleur_score_global = score_parent
                        meilleur_etat_global = seq_parent[0]
                    continue
                    
                for bac_id, dest_emp, cout_dist, is_buffer in coups:
                    # Simuler le coup
                    nouvelle_config = Configuration(config_parent.etage)
                    nouvelle_config.affectation = config_parent.affectation.copy()
                    nouvelle_config.affectation[bac_id] = dest_emp
                    
                    nouvelle_dist = dist_parent + cout_dist
                    nouvelle_seq = seq_parent + [(bac_id, dest_emp, is_buffer)]
                    nouvelle_nav = dest_emp.colonne
                    
                    score = self._evaluer_etat(nouvelle_config, nouvelle_dist)
                    
                    # Casser la symétrie de stagnation pour le choix immédiat (profondeur 0) sans fausser le gradient d'évaluation de l'arbre
                    if is_buffer and not seq_parent:
                        recence = self.iteration_actuelle - self.derniers_buffers.get(bac_id, -1000)
                        if recence < 5:
                            score -= (5 - recence) * 2000.0
                    
                    bacs_a_placer = [b for b, emp in nouvelle_config.affectation.items() if emp != self.cible.affectation[b]]
                    if not bacs_a_placer:
                        # Si ce coup résout complètement l'instance, on s'arrête net
                        score += 1000000
                        return nouvelle_seq[0]
                        
                    prochains_etats.append((score, nouvelle_config, nouvelle_dist, nouvelle_seq, nouvelle_nav))
                    
                    # Track du meilleur état rencontré n'importe où dans l'arbre
                    if score > meilleur_score_global:
                        meilleur_score_global = score
                        meilleur_etat_global = nouvelle_seq[0]
            
            if not prochains_etats:
                break
                
            # Élagage Beam Search : on ne garde que les beam_width meilleurs états
            prochains_etats.sort(key=lambda x: (x[0], -x[2], str(x[3])), reverse=True)
            faisceau = prochains_etats[:self.beam_width]
            
        if meilleur_etat_global is None and faisceau:
            # Sécurité finale
            meilleur_etat_global = faisceau[0][3][0] if faisceau[0][3] else None
            
        # --- INTERCEPTION ANTI-STAGNATION ABSOLUE ---
        # Si l'horizon du Lookahead est trop court pour voir la récompense finale et qu'il retombe sur un mouvement de ping-pong immédiat,
        # on force la sélection d'un mouvement Ouvreur, l'évasion vers un tiers-buffer, ou la rotation des candidats pour briser le paradoxe.
        if meilleur_etat_global is not None:
            bac_id, dest_emp, is_buffer = meilleur_etat_global
            if is_buffer and self.iteration_actuelle - self.derniers_buffers.get(bac_id, -1000) < 5:
                # 1. Tenter un Ouvreur d'urgence sur un autre bac pour libérer un espace avant
                for b, emp in self.courante.affectation.items():
                    if b != bac_id and emp.profondeur == Profondeur.AVANT:
                        emp_arr = EmplacementStock(emp.colonne, emp.cote, Profondeur.ARRIERE)
                        if self.courante.bac_a(emp_arr) is None:
                            return (b, emp_arr, True)
                            
                # 2. Forcer un autre bac mal placé accessible à prendre le buffer pour casser le monopole du bac oscillant
                bacs_a_placer = [b for b, emp in self.courante.affectation.items() if emp != self.cible.affectation[b]]
                for b in bacs_a_placer:
                    if b != bac_id:
                        emp_b = self.courante.affectation[b]
                        if emp_b.profondeur == Profondeur.AVANT or (emp_b.profondeur == Profondeur.ARRIERE and self.courante.bac_a(EmplacementStock(emp_b.colonne, emp_b.cote, Profondeur.AVANT)) is None):
                            if emp_b.colonne != dest_emp.colonne or emp_b.cote != dest_emp.cote or emp_b.profondeur != Profondeur.ARRIERE or dest_emp.profondeur != Profondeur.AVANT:
                                return (b, dest_emp, True)
                                
        return meilleur_etat_global

    def planifier_mouvements(self):
        """
        Orchestre la boucle principale de résolution de l'entrepôt.
        
        À chaque itération :
        1. Vérifie si l'entrepôt est entièrement trié.
        2. Extrait les métriques graphiques pour l'historique.
        3. Exécute le Beam Search pour sélectionner le prochain mouvement.
        4. Déclenche un coupe-circuit d'urgence si la stagnation dépasse 12 itérations.
        5. Applique le mouvement sélectionné sur la configuration réelle.
        
        Raises:
            TimeoutHeuristique: Si le nombre maximal d'itérations est dépassé.
            DeadlockInsoluble: Si l'algorithme est physiquement bloqué sans recours.
        """
        while True:
            self.iteration_actuelle += 1
            if self.iteration_actuelle > self.max_iterations:
                raise TimeoutHeuristique(
                    f"Limite de {self.max_iterations} itérations atteinte sans convergence."
                )

            nb_bien_places = self.courante.nb_bacs_deja_places(self.cible)
            if not hasattr(self, 'meilleur_nb_places'):
                self.meilleur_nb_places = nb_bien_places
                self.iterations_sans_progres = 0
            elif nb_bien_places > self.meilleur_nb_places:
                self.meilleur_nb_places = nb_bien_places
                self.iterations_sans_progres = 0
            else:
                self.iterations_sans_progres += 1

            bacs_a_placer = [
                bac_id for bac_id, emp in self.courante.affectation.items()
                if emp != self.cible.affectation[bac_id]
            ]
            if not bacs_a_placer:
                break

            vides_accessibles = [
                emp for emp in self.courante.emplacements_vides_stock()
                if self.courante.est_accessible(emp)
            ]
            if not vides_accessibles:
                raise DeadlockInsoluble("Aucun emplacement vide accessible (instance invalide ?).")

            # --- TRACKING ITÉRATIF ---
            graphe = GrapheDependance(self.courante, self.cible)
            metriques = graphe.analyser_graphe()
            self.historique.append({
                "iteration": self.iteration_actuelle,
                "bacs_mal_places": len(bacs_a_placer),
                "somme_degres_entrants": metriques["somme_degres_entrants"],
                "degre_max": metriques["degre_max"],
                "nb_noeuds_en_cycle": metriques["nb_noeuds_en_cycle"],
            })

            # --- DÉCISION DEEP LOOKAHEAD OU ROTATION D'URGENCE ---
            meilleur_mouvement = None
            if getattr(self, 'iterations_sans_progres', 0) > 12:
                # Coupe-circuit direct : on effectue une rotation pure du bac accessible le plus anciennement figé
                bacs_figes = sorted(self.courante.affectation.keys(), key=lambda b: (self.derniers_buffers.get(b, -1000), b))
                for b in bacs_figes:
                    emp_b = self.courante.affectation[b]
                    if emp_b.profondeur == Profondeur.AVANT or (emp_b.profondeur == Profondeur.ARRIERE and self.courante.bac_a(EmplacementStock(emp_b.colonne, emp_b.cote, Profondeur.AVANT)) is None):
                        for emp in vides_accessibles:
                            if emp.colonne != emp_b.colonne:
                                meilleur_mouvement = (b, emp, True)
                                self.iterations_sans_progres = 0
                                break
                    if meilleur_mouvement:
                        break
                        
            if not meilleur_mouvement:
                meilleur_mouvement = self._beam_search()
            
            if not meilleur_mouvement:
                raise DeadlockInsoluble("Le Lookahead profond n'a trouvé aucun mouvement valide (impasse absolue).")
                
            bac_choisi, cible_finale, is_buffer = meilleur_mouvement
            self._deplacer_bac(bac_choisi, cible_finale, est_buffer=is_buffer)

# =============================================================================
# 3. Exécution et Interface
# =============================================================================

def resoudre_instance(fichier: str, verbose: bool = True) -> dict:
    """
    Point d'entrée pour la résolution complète d'une instance spécifique.
    
    Args:
        fichier (str): Le chemin vers le fichier CSV de l'instance.
        verbose (bool): Si True, affiche les logs détaillés de la résolution.
        
    Returns:
        dict: Un dictionnaire contenant les métriques de la résolution (statut, 
        distance, mouvements, temps d'exécution, etc.).
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

        heuristique = HeuristiqueLookahead(init, cible)
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
            if heuristique:
                print("Mouvements effectués avant l'échec :")
                for i, m in enumerate(heuristique.sequence_mouvements, 1):
                    print(f"  {i:3d}. {m}")
    except ValueError as e:
        statut = "Instance invalide"
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
        "Echec_Degre_Max": echec_degre_max,
        "Echec_Somme_Degres": echec_somme_degres,
        "Echec_Nb_Cycles": echec_nb_cycles,
        "Echec_Iteration": echec_iteration,
        "_historique": heuristique.historique if heuristique else [],
    }

def main():
    """
    Interface en ligne de commande (CLI) du module H2.
    
    Permet de lancer la résolution soit sur une instance unique (`une`), 
    soit en traitement par lot sur un dossier complet (`lot`) avec 
    export des résultats au format CSV.
    """
    parser = argparse.ArgumentParser(description="Heuristique H2 (Lookahead) - SBS/RS")
    subparsers = parser.add_subparsers(dest="commande", required=True)

    p_une = subparsers.add_parser("une", help="Instance unique")
    p_une.add_argument("fichier", type=str)

    p_lot = subparsers.add_parser("lot", help="Dossier complet")
    p_lot.add_argument("-d", "--dossier", type=str, default="instances")
    p_lot.add_argument("-o", "--output", type=str, default="resultats/resultat_H2.csv")

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