# =============================================================================
# H2.py  –  Heuristique Lookahead (Anticipation 1-coup)
# Auteur : Projet SBS/RS
#
# Description :
#   Variante évoluée de H1. Au lieu de poids statiques, H2 simule chaque
#   mouvement possible et choisit celui qui débloque le plus de bacs
#   à l'itération suivante (anticipation 1-coup).
#
# STRATÉGIE ("Lookahead") :
#   1. Pour chaque bac éligible, simuler son placement à sa cible finale.
#   2. Compter les bacs qui deviendraient libres (réaction en chaîne).
#   3. Choisir le mouvement maximisant les déblocages futurs.
#   4. En cas de deadlock, simuler toutes les combinaisons (candidat × buffer)
#      pour trouver la cassure de cycle la plus prometteuse.
#
# NOTE STRUCTURELLE — Absence de mécanisme "Ouvreur" :
#   Le générateur d'instances impose l'invariant Back-First : un bac ne peut
#   occuper la position AVANT d'une demi-colonne que si la position ARRIERE
#   est aussi occupée. Il en découle que tout emplacement ARRIERE vide a
#   nécessairement son AVANT vide, donc est directement accessible par la
#   navette. L'ancien mécanisme "Ouvreur" était donc redondant : sa cible
#   était déjà dans vides_accessibles avant toute intervention.
# =============================================================================

from enum import Enum, auto
from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import time

from instance_generator import (
    Configuration, EmplacementStock, Cote, Profondeur,
    distance, importer_instance
)

# =============================================================================
# 1. Structures de Graphe et Dépendances (identiques à H1)
# =============================================================================

class TypeDependance(Enum):
    """Définit la nature du blocage entre deux bacs."""
    DUR = auto()          # Blocage physique LIFO (AVANT bloque ARRIERE)
    OCCUPATION = auto()   # Blocage logique (Un bac occupe la cible d'un autre)
    SEQUENCEMENT = auto() # Blocage d'ordre (ARRIERE doit entrer avant l'AVANT)

@dataclass
class Arc:
    """Représente un arc orienté A -> B : A bloque B."""
    source: int
    cible: int
    type_dep: TypeDependance

class GrapheDependance:
    """
    Graphe orienté des blocages entre bacs. Identique à H1 pour garantir
    la cohérence des règles métier entre les deux heuristiques.
    """
    def __init__(self, config_courante: Configuration, config_cible: Configuration):
        self.config_courante = config_courante
        self.config_cible = config_cible
        self.noeuds = set(config_courante.affectation.keys())
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
        for bac_j, cible_j in self.config_cible.affectation.items():
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
        for bac_j, cible_j in self.config_cible.affectation.items():
            if cible_j.profondeur == Profondeur.ARRIERE:
                if self.config_courante.affectation[bac_j] != cible_j:
                    cible_j_avant = EmplacementStock(cible_j.colonne, cible_j.cote, Profondeur.AVANT)
                    bac_k = cibles_inverse.get(cible_j_avant)
                    if bac_k is not None:
                        self.arcs.append(Arc(source=bac_j, cible=bac_k, type_dep=TypeDependance.SEQUENCEMENT))

    def trouver_bacs_sans_blocage(self) -> list[int]:
        """Retourne les bacs libres (degré entrant = 0)."""
        bacs_bloques = {arc.cible for arc in self.arcs}
        return [bac for bac in self.noeuds if bac not in bacs_bloques]

    def detecter_cycle(self) -> list[Arc] | None:
        """Détecte un cycle par DFS et retourne les arcs impliqués, ou None."""
        visite, pile_appel, pile_arcs = set(), [], []
        adj = {n: [] for n in self.noeuds}
        for arc in self.arcs:
            adj[arc.source].append(arc)

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

# =============================================================================
# 2. Heuristique H2 (Lookahead / Anticipation 1-coup)
# =============================================================================

class HeuristiqueLookahead:
    """
    Implémentation de l'heuristique H2.

    DIFFÉRENCE FONDAMENTALE AVEC H1 :
    H1 choisit selon des poids statiques prédéfinis (règles empiriques).
    H2 simule chaque mouvement candidat dans un clone de l'entrepôt, reconstruit
    le graphe sur ce futur virtuel, et choisit le mouvement qui maximise le
    nombre de bacs débloqués au tour suivant.
    """
    def __init__(self, config_init: Configuration, config_cible: Configuration, p_lib=1, p_vul=2, p_ino=3, pen_avant=1, cap_tabou=5):
        self.courante = config_init
        self.cible = config_cible
        self.colonne_navette = 0
        self.distance_totale_x = 0
        self.p_lib = p_lib
        self.p_vul = p_vul
        self.p_ino = p_ino
        self.pen_avant = pen_avant
        self.cap_tabou = cap_tabou
        self.sequence_mouvements = []
        self.nb_double_handling = 0
        self.derniers_buffers = {}
        self.tabu_mouvements = {}
        self.iteration_actuelle = 0

        # [PARAMÈTRE OPTIMISÉ] 10× le nombre de bacs garantit la résolution
        self.max_iterations = len(config_init.affectation) * 10

    def _bac_est_place(self, bac_id: int) -> bool:
        return self.courante.affectation[bac_id] == self.cible.affectation[bac_id]

    def _est_cible_pour_un_bac(self, emp: EmplacementStock) -> bool:
        return emp in self.cible.affectation.values()

    def _simuler_et_evaluer(self, bac_id: int, dest: EmplacementStock, bacs_a_placer: list[int]) -> int:
        """
        Cœur de la stratégie Lookahead.

        Clone la configuration courante, applique virtuellement le mouvement
        (bac_id -> dest), reconstruit le graphe de dépendances sur ce futur,
        et retourne le nombre de bacs mal placés qui deviendraient libres.

        Plus le score est élevé, plus le mouvement est stratégiquement intéressant.
        """
        config_simulee = Configuration(self.courante.etage)
        config_simulee.affectation = self.courante.affectation.copy()
        config_simulee.affectation[bac_id] = dest

        graphe_sim = GrapheDependance(config_simulee, self.cible)
        libres = graphe_sim.trouver_bacs_sans_blocage()

        return sum(1 for b in libres if b in bacs_a_placer and b != bac_id)

    def _calculer_score_buffer(self, bac_id: int, emp: EmplacementStock) -> float:
        """
        Score de pénalité d'un emplacement buffer (second critère après le lookahead).
        Score bas = bon buffer. Pénalités identiques à H1.

        Pénalités :
          +1      : position AVANT (légère préférence ARRIERE)
          +10 000 : emplacement tabou
          +600    : emplacement = cible finale d'un autre bac
          +5 000  : poser ici créerait une prison pour un bac ARRIERE mal placé
        """
        emp_actuel = self.courante.affectation[bac_id]
        col_actuelle = emp_actuel.colonne
        col_cible = self.cible.affectation[bac_id].colonne

        score = abs(col_actuelle - emp.colonne) + abs(emp.colonne - col_cible)

        if emp.profondeur == Profondeur.AVANT:
            score += self.pen_avant

        expiration = self.tabu_mouvements.get((bac_id, emp), -1)
        if self.iteration_actuelle <= expiration:
            score += 1000 * self.courante.etage.n_colonnes  # Exclusion quasi-totale dynamique

        if self._est_cible_pour_un_bac(emp):
            score += 100 * self.courante.etage.n_colonnes  # Big-M dynamique cible

        if emp.profondeur == Profondeur.AVANT:
            emp_arr = EmplacementStock(emp.colonne, emp.cote, Profondeur.ARRIERE)
            bac_arr = self.courante.bac_a(emp_arr)
            if bac_arr is not None and self.cible.affectation[bac_arr] != emp_arr:
                score += 500 * self.courante.etage.n_colonnes  # Big-M dynamique prison

        return score

    def _calculer_poids_strategique(self, bac_id: int) -> int:
        """
        Poids H1 utilisé comme troisième critère de départage (après lookahead
        et distance) pour les bacs éligibles au Happy Path.
        """
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

    def _deplacer_bac(self, bac_id: int, dest: EmplacementStock, est_buffer: bool = False):
        """
        Exécute le déplacement et met à jour distance, configuration, tabou et log.
        """
        source = self.courante.affectation[bac_id]

        self.distance_totale_x += (
            abs(self.colonne_navette - source.colonne) +
            abs(source.colonne - dest.colonne)
        )
        self.colonne_navette = dest.colonne
        self.courante.affectation[bac_id] = dest

        # [PARAMÈTRE OPTIMISÉ] cap = 5 pour H2 : l'anticipation évite les oscillations,
        # une durée courte laisse à H2 plus de liberté dans les zones denses.
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

    def planifier_mouvements(self):
        """
        Boucle principale de décision Lookahead.

        À chaque itération :
          — PHASE 1 (Happy Path) : simuler le placement de chaque bac éligible
            à sa cible, choisir celui qui maximise les déblocages suivants.
          — PHASE 2 (Deadlock) : explorer toutes les combinaisons
            (racine physique × buffer disponible), choisir celle qui maximise
            les déblocages tout en minimisant les pénalités.
        """
        while True:
            self.iteration_actuelle += 1
            if self.iteration_actuelle > self.max_iterations:
                raise RuntimeError("Boucle infinie détectée dans l'heuristique.")

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
                # Par l'invariant Back-First du générateur, cette situation
                # ne devrait pas survenir sur une instance valide.
                raise RuntimeError("Aucun emplacement vide accessible (instance invalide ?).")

            graphe = GrapheDependance(self.courante, self.cible)
            eligibles = [b for b in graphe.trouver_bacs_sans_blocage() if b in bacs_a_placer]

            # --- PHASE 1 : HAPPY PATH AVEC ANTICIPATION ---
            if eligibles:
                evaluations = []
                for b in eligibles:
                    cible_b = self.cible.affectation[b]
                    nb_liberes = self._simuler_et_evaluer(b, cible_b, bacs_a_placer)
                    poids = self._calculer_poids_strategique(b)
                    dist = abs(self.courante.affectation[b].colonne - self.colonne_navette)
                    evaluations.append((b, cible_b, nb_liberes, poids, dist))

                # Tri : MAX déblocages, puis poids H1 (départage), puis MIN distance
                evaluations.sort(key=lambda x: (-x[2], x[3], x[4]))
                bac_choisi, cible_finale, *_ = evaluations[0]
                self._deplacer_bac(bac_choisi, cible_finale, est_buffer=False)
                continue

            # --- PHASE 2 : DEADLOCK AVEC ANTICIPATION EXHAUSTIVE ---
            arcs_cycle = graphe.detecter_cycle()
            if not arcs_cycle:
                raise RuntimeError("Blocage insoluble sans cycle détecté (incohérence logique).")

            # Identifier les racines physiques du cycle (bacs AVANT accessibles)
            candidats = set()
            for arc in arcs_cycle:
                bac = arc.source
                emp_bac = self.courante.affectation[bac]
                if emp_bac.profondeur == Profondeur.AVANT:
                    candidats.add(bac)
                else:
                    emp_av = EmplacementStock(emp_bac.colonne, emp_bac.cote, Profondeur.AVANT)
                    bac_av = self.courante.bac_a(emp_av)
                    candidats.add(bac_av if bac_av is not None else bac)

            if not candidats:
                raise RuntimeError("Cycle insoluble : aucune racine physique accessible.")

            # Explorer toutes les combinaisons (candidat × buffer) avec simulation
            evaluations = []
            for b in candidats:
                for emp in vides_accessibles:
                    nb_liberes = self._simuler_et_evaluer(b, emp, bacs_a_placer)
                    score_h1 = self._calculer_score_buffer(b, emp)
                    evaluations.append((b, emp, nb_liberes, score_h1))

            if not evaluations:
                raise RuntimeError("Aucun buffer accessible pour casser le cycle.")

            # Tri : MAX déblocages, puis MIN pénalités H1, puis moins récemment bougé
            evaluations.sort(key=lambda x: (-x[2], x[3], self.derniers_buffers.get(x[0], -1)))
            bac_choisi, buffer_emp, *_ = evaluations[0]

            self._deplacer_bac(bac_choisi, buffer_emp, est_buffer=True)
            self.derniers_buffers[bac_choisi] = self.iteration_actuelle

# =============================================================================
# 3. Exécution et Interface
# =============================================================================

def resoudre_instance(fichier: str, verbose: bool = True) -> dict:
    """Lance H2 sur une instance et retourne les résultats."""
    start_time = time.time()
    heuristique = None
    try:
        init, cible = importer_instance(fichier)
        heuristique = HeuristiqueLookahead(init, cible)
        heuristique.planifier_mouvements()
        distance_val = heuristique.distance_totale_x
        mouvements = len(heuristique.sequence_mouvements)
        statut = "Succès"
        if verbose:
            print(f"\n[Terminé] Distance Totale X : {distance_val}")
            for i, m in enumerate(heuristique.sequence_mouvements, 1):
                print(f"  {i:3d}. {m}")
    except Exception as e:
        distance_val, mouvements, statut = -1, -1, f"Erreur: {str(e)}"
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
    }

def main():
    """Point d'entrée CLI : mode unitaire ou lot."""
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
                fieldnames=["Fichier", "Statut", "Distance_X", "Mouvements", "Double_Handling", "Temps_s"]
            )
            writer.writeheader()
            writer.writerows(resultats)
        print(f"\nRésultats sauvegardés dans '{args.output}'.")

if __name__ == "__main__":
    main()