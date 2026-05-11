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

from instance_generator import (
    Configuration, EmplacementStock, Cote, Profondeur,
    distance, importer_instance
)

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

    Chaque nœud est un bac_id. Un arc A -> B signifie :
    "Le bac A doit bouger AVANT que le bac B puisse atteindre sa cible."

    Ce graphe est recalculé intégralement à chaque itération.
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
        Un arc A -> B signifie que le bac A empêche le bac B de faire son mouvement.
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
        """Retourne les bacs libres (degré entrant = 0) : personne ne les bloque."""
        bacs_bloques = {arc.cible for arc in self.arcs}
        return [bac for bac in self.noeuds if bac not in bacs_bloques]

    def detecter_cycle(self) -> list[Arc] | None:
        """
        Détection de cycle par DFS (parcours en profondeur).
        Un cycle = deadlock : A bloque B qui bloque ... qui bloque A.
        Retourne la liste des arcs formant le cycle, ou None s'il n'y en a pas.
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

# =============================================================================
# 2. Heuristique H1 (Greedy Eligible-First)
# =============================================================================

class HeuristiqueConstructive:
    """
    Implémentation de l'heuristique H1.
    Elle privilégie les bacs éligibles (Happy Path) selon des priorités empiriques
    (poids statiques), et casse les deadlocks en évaluant les buffers par score
    de pénalité (méthode du Grand-M).
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

        # [PARAMÈTRE OPTIMISÉ] 10× le nombre de bacs garantit la résolution
        self.max_iterations = len(config_init.affectation) * 10
        self.p_lib = p_lib
        self.p_vul = p_vul
        self.p_ino = p_ino
        self.pen_avant = pen_avant
        self.cap_tabou = cap_tabou

    def _bac_est_place(self, bac_id: int) -> bool:
        """Vérifie si le bac a atteint sa destination finale."""
        return self.courante.affectation[bac_id] == self.cible.affectation[bac_id]

    def _est_cible_pour_un_bac(self, emp: EmplacementStock) -> bool:
        """Vérifie si un emplacement est une destination finale pour n'importe quel bac."""
        return emp in self.cible.affectation.values()

    def trouver_buffer(self, bac_id: int) -> tuple[EmplacementStock | None, float]:
        """
        Recherche le meilleur emplacement vide accessible comme buffer temporaire.

        Score = distance_navette_vers_buffer + distance_buffer_vers_cible_finale
                + pénalités (Tabou, Cible réservée, Création de prison)

        Retourne (meilleur_emplacement, score). Si aucun, retourne (None, inf).
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
        Exécute le déplacement d'un bac et met à jour toutes les métriques :
        distance navette, configuration courante, liste tabou, log des mouvements.
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

        Profils (du plus au moins prioritaire) :
          1 — Libérateur : AVANT, bloque un bac mal placé à l'ARRIERE → déplacer débloque.
          2 — Vulnérable : ARRIERE → risque d'emprisonnement si un bac se pose devant.
          3 — Inoffensif : AVANT, ARRIERE déjà à sa cible finale → basse priorité.
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

    def planifier_mouvements(self):
        """
        Boucle principale de résolution.

        À chaque itération :
          — PHASE 1 (Happy Path) : s'il existe des bacs libres (degré entrant = 0),
            choisir le plus prioritaire et le placer directement à sa cible.
          — PHASE 2 (Deadlock) : s'il n'y a aucun bac libre, détecter le cycle,
            identifier la racine physique et la mettre en buffer temporaire.
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

            # --- PHASE 1 : HAPPY PATH ---
            graphe = GrapheDependance(self.courante, self.cible)
            eligibles = [b for b in graphe.trouver_bacs_sans_blocage() if b in bacs_a_placer]

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
                raise RuntimeError("Blocage insoluble sans cycle détecté (incohérence logique).")

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
                raise RuntimeError("Cycle insoluble : aucune racine physique accessible.")

            # Évaluer le meilleur buffer pour chaque candidat
            evaluations = [
                (b, buf, score)
                for b in candidats
                for buf, score in [self.trouver_buffer(b)]
                if buf is not None
            ]

            if not evaluations:
                raise RuntimeError("Aucun buffer accessible pour casser le cycle.")

            # Choisir le candidat dont le buffer est le moins pénalisant
            evaluations.sort(key=lambda x: (x[2], self.derniers_buffers.get(x[0], -1)))
            bac_choisi, buffer_emp, _ = evaluations[0]

            self._deplacer_bac(bac_choisi, buffer_emp, est_buffer=True)
            self.derniers_buffers[bac_choisi] = self.iteration_actuelle

# =============================================================================
# 3. Exécution et Interface
# =============================================================================

def resoudre_instance(fichier: str, verbose: bool = True) -> dict:
    """Charge une instance, lance H1 et retourne les résultats."""
    start_time = time.time()
    heuristique = None
    try:
        init, cible = importer_instance(fichier)
        heuristique = HeuristiqueConstructive(init, cible)
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
                fieldnames=["Fichier", "Statut", "Distance_X", "Mouvements", "Double_Handling", "Temps_s"]
            )
            writer.writeheader()
            writer.writerows(resultats)
        print(f"\nRésultats sauvegardés dans '{args.output}'.")

if __name__ == "__main__":
    main()