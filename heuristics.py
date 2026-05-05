"""
Moteur de résolution pour le réarrangement SBS/RS double profondeur.
═══════════════════════════════════════════════════════════════════════════

Ce module contient l'intelligence décisionnelle (le Décodeur) pour le 
problème de réarrangement d'entrepôt. 

Il intègre des optimisations algorithmiques (indexation bidirectionnelle en O(1)) 
et propose 7 heuristiques différentes, inspirées de la littérature scientifique 
récente, pour résoudre les conflits de placement ("Double Handling") et 
optimiser les distances de déplacement.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

from instance_generator import (
    Cote,
    Profondeur,
    EmplacementStock,
    Configuration,
    distance,
    importer_instance,
)

# =============================================================================
#  1. Structures de données des Résultats
# =============================================================================

@dataclass
class Mouvement:
    """Représente une action unitaire de la navette (déplacement d'un bac)."""
    bac_id: int
    source: EmplacementStock
    destination: EmplacementStock
    distance: int
    est_double_handling: bool = False  # True si le mouvement ne place pas le bac à sa cible finale

@dataclass
class ResultatHeuristique:
    """Encapsule les métriques de performance d'une résolution complète."""
    nom: str
    distance_totale: int
    mouvements: list[Mouvement]
    complet: bool                      # True si l'algorithme a atteint la configuration cible
    temps_execution: float
    nb_double_handling: int
    nb_bacs_bien_places: int = 0
    total_bacs: int = 0


# =============================================================================
#  2. Moteur de Résolution (Solveur)
# =============================================================================

class Solveur:
    """
    Classe principale gérant la résolution d'une instance.
    Maintient l'état de l'entrepôt et exécute les règles métiers des heuristiques.
    """
    
    def __init__(self, config_initiale: Configuration, config_cible: Configuration):
        self.config_cible = config_cible
        self.n_colonnes = config_initiale.etage.n_colonnes
        self.etage = config_initiale.etage
        
        # Indexation bidirectionnelle : permet des recherches en O(1) au lieu de O(N)
        # affectation : bac_id -> emplacement
        # grille : emplacement -> bac_id
        self.affectation = dict(config_initiale.affectation)
        self.grille = {emp: bac_id for bac_id, emp in self.affectation.items()}
        self.grille_cible = {emp: bac_id for bac_id, emp in config_cible.affectation.items()}

        self.mouvements: list[Mouvement] = []
        
        # Gestion des cycles : Dictionnaire tabou pour empêcher les boucles infinies
        # bac_id -> (emplacement_precedent, iteration_du_mouvement)
        self.tabou: dict[int, tuple[EmplacementStock, int]] = {}
        self.iteration = 0
        
    @property
    def config_courante(self) -> Configuration:
        """Retourne un snapshot de l'état actuel sous forme d'objet Configuration."""
        return Configuration(etage=self.etage, affectation=dict(self.affectation))
        
    def _est_accessible(self, emplacement: EmplacementStock) -> bool:
        """Vérifie si un emplacement (AVANT ou ARRIERE) est libre d'accès pour la navette."""
        if emplacement.profondeur == Profondeur.AVANT:
            return True
        avant = EmplacementStock(emplacement.colonne, emplacement.cote, Profondeur.AVANT)
        return avant not in self.grille

    def _executer_mouvement(self, bac_id: int, destination: EmplacementStock, est_double_handling: bool = False) -> None:
        """Applique physiquement un mouvement, met à jour les index O(1) et trace l'historique."""
        source = self.affectation[bac_id]
        dist = distance(source, destination, self.n_colonnes)
        
        del self.grille[source]
        self.grille[destination] = bac_id
        self.affectation[bac_id] = destination
        
        self.mouvements.append(Mouvement(bac_id, source, destination, dist, est_double_handling))

    def _bac_bloquant(self, emplacement: EmplacementStock) -> int | None:
        """Identifie l'ID du bac en position AVANT qui obstrue l'accès à la position ARRIERE."""
        if emplacement.profondeur == Profondeur.AVANT:
            return None
        avant = EmplacementStock(emplacement.colonne, emplacement.cote, Profondeur.AVANT)
        return self.grille.get(avant)

    def _dependance(self, bac: int) -> int | None:
        """
        Détermine si un bac est "dépendant" d'un autre (blocage physique).
        Un bac est bloqué si :
        1. Sa cible finale "Arrière" sera obstruée par la cible finale "Avant".
        2. Sa position actuelle est bloquée par un bac devant lui.
        3. Sa cible finale est occupée par un autre bac.
        4. Sa cible finale est bloquée par un bac en position "Avant".
        """
        cible = self.config_cible.affectation[bac]
        source = self.affectation[bac]
        
        # Blocage logique par la configuration cible (respect de la règle Back-First en prévision)
        if cible.profondeur == Profondeur.AVANT:
            arriere = EmplacementStock(cible.colonne, cible.cote, Profondeur.ARRIERE)
            if arriere not in self.grille:
                bac_arriere_cible = self.grille_cible.get(arriere)
                if bac_arriere_cible is not None and bac_arriere_cible != bac:
                    return bac_arriere_cible
                        
        # Blocages physiques de l'état actuel
        bloqueur_source = self._bac_bloquant(source)
        if bloqueur_source is not None and bloqueur_source != bac:
            return bloqueur_source
            
        occupant_cible = self.grille.get(cible)
        if occupant_cible is not None and occupant_cible != bac:
            return occupant_cible
                
        bloqueur_cible = self._bac_bloquant(cible)
        if bloqueur_cible is not None and bloqueur_cible != bac:
            return bloqueur_cible
            
        return None

    def _suivre_chaine(self, bac_initial: int) -> tuple[list[int], bool]:
        """
        Analyse de graphe : Construit la chaîne de dépendance d'un bac.
        Retourne la chaîne et un booléen indiquant si la chaîne forme un cycle (deadlock).
        """
        chaine = []
        courant = bac_initial
        visites = set()
        
        while courant is not None:
            if courant in visites:
                # Cycle détecté : on ne retourne que la boucle
                idx = chaine.index(courant)
                return chaine[idx:], True 
            
            chaine.append(courant)
            visites.add(courant)
            
            # Fin de chaîne si le bac est à sa position finale
            if self.affectation[courant] == self.config_cible.affectation[courant]:
                break
                
            dep = self._dependance(courant)
            if dep is None:
                break
            courant = dep
            
        return chaine, False

    def _trouver_emplacement_vide_accessible(self, exclure: list[EmplacementStock] = None, source: EmplacementStock = None, methode: str = "H1", autoriser_blocage: bool = False) -> EmplacementStock | None:
        """
        Moteur de recherche de destination temporaire (pour le Double Handling).
        Filtre les emplacements selon les spécificités de chaque heuristique.
        """
        exclure = set(exclure) if exclure else set()
        candidats = []
        
        # Outil Look-Ahead (H5) : Indexation des cibles pour éviter de polluer une case utile
        cibles_mal_places = set()
        if methode == "H5":
            cibles_mal_places = {self.config_cible.affectation[b] for b, e in self.affectation.items() if e != self.config_cible.affectation[b]}

        for e in self.etage.stocks:
            if e in self.grille or e in exclure:
                continue
                
            if self._est_accessible(e):
                # Règle Back-First : Évite de déposer devant une case vide (sauf si forcé via autoriser_blocage)
                if e.profondeur == Profondeur.AVANT:
                    arriere = EmplacementStock(e.colonne, e.cote, Profondeur.ARRIERE)
                    if arriere not in self.grille:
                        continue 
                    else:
                        if not autoriser_blocage:
                            bac_arriere = self.grille.get(arriere)
                            # Évite de bloquer un bac qui n'est pas encore à sa cible
                            if bac_arriere is not None and self.affectation[bac_arriere] != self.config_cible.affectation[bac_arriere]:
                                continue
                            
                # Spécificité H2 : Contraint le dégagement à la même colonne
                if methode == "H2" and source and e.colonne != source.colonne:
                    continue
                    
                candidats.append(e)
                
        if not candidats:
            return None
            
        if source:
            if methode == "H5":
                # Look-Ahead : Pénalité si le candidat est une cible future
                def score_h5(x):
                    penalite = 100 if x in cibles_mal_places else 0
                    return distance(x, source, self.n_colonnes) + penalite
                candidats.sort(key=score_h5)
            elif methode == "H7":
                # Topologie Respirante : Envoie les bloqueurs le plus loin possible (colonne max)
                candidats.sort(key=lambda x: -x.colonne)
            else:
                # Classique (Nearest Neighbor) : Trie par distance minimale
                candidats.sort(key=lambda x: distance(x, source, self.n_colonnes))
            
        return candidats[0]

    def _degager(self, bac_id: int, methode: str, chaine: list[int] = None) -> bool:
        """
        Procédure d'éviction ("Double Handling").
        Déplace un bac bloquant vers un emplacement vide temporaire.
        """
        source = self.affectation[bac_id]
        exclure = [source]
        
        # Exclut les emplacements cibles de la chaîne actuelle pour ne pas créer d'autres blocages
        if chaine:
            for b in chaine:
                cible = self.config_cible.affectation[b]
                exclure.append(cible)
                if cible.profondeur == Profondeur.ARRIERE:
                    exclure.append(EmplacementStock(cible.colonne, cible.cote, Profondeur.AVANT))
        
        # Intégration Tabou : Évite le ping-pong en excluant la position précédente du bac
        if bac_id in self.tabou:
            pos_tabou, iteration_tabou = self.tabou[bac_id]
            if self.iteration - iteration_tabou <= 10:
                exclure.append(pos_tabou)
        
        # Phase 1 : Recherche stricte selon la méthode
        dest_vide = self._trouver_emplacement_vide_accessible(exclure, source, methode=methode)
        
        # Phase 2 : Fallbacks (Si l'entrepôt est trop dense)
        if dest_vide is None and methode == "H2":
            dest_vide = self._trouver_emplacement_vide_accessible(exclure, source, methode="H1")
            
        if dest_vide is None:
            # Assouplissement de la contrainte de non-blocage
            dest_vide = self._trouver_emplacement_vide_accessible(exclure, source, methode=methode, autoriser_blocage=True)
            
        if dest_vide is None:
            # Dernier recours
            dest_vide = self._trouver_emplacement_vide_accessible([source], source, methode="H1", autoriser_blocage=True)
            
        if dest_vide:
            self._executer_mouvement(bac_id, dest_vide, est_double_handling=True)
            self.tabou[bac_id] = (source, self.iteration)
            return True
            
        return False

    def _resoudre_generique(self, methode: str, custom_priority_list: list[int] = None, max_mouvements: int = 5000) -> ResultatHeuristique:
        """
        Boucle principale de résolution.
        Applique le tri spécifique à l'heuristique choisie, et itère jusqu'à la résolution
        ou l'atteinte de la limite de mouvements (deadlock complexe).
        """
        start_time = time.time()
        self.mouvements.clear()
        self.tabou.clear()
        self.iteration = 0
        
        navette_pos = 0 
        
        while self.iteration < max_mouvements:
            self.iteration += 1
            mal_places = [b for b, e in self.affectation.items() if e != self.config_cible.affectation[b]]
            
            if not mal_places:
                break
                
            # ────────────────────────────────────────────────────────────────────────
            #  APPLICATION DES POLITIQUES DE SÉLECTION (H1 à H7)
            # ────────────────────────────────────────────────────────────────────────
            
            if custom_priority_list is not None:
                if not hasattr(self, '_custom_priority_dict'):
                    self._custom_priority_dict = {b: i for i, b in enumerate(custom_priority_list)}
                mal_places.sort(key=lambda b: self._custom_priority_dict.get(b, 999999))
                
            elif methode == "H4":
                # H4 : Tri Topologique. Priorise la rupture des cycles, puis les chaînes longues.
                scores = {}
                for b in mal_places:
                    chaine, est_cycle = self._suivre_chaine(b)
                    if not est_cycle:
                        tabou_blocked = False
                        if chaine[-1] in self.tabou:
                            pt, it = self.tabou[chaine[-1]]
                            if self.config_cible.affectation[chaine[-1]] == pt and self.iteration - it <= 10:
                                tabou_blocked = True
                        # Un cycle coûte moins cher (score 2) qu'une chaîne tabou (score 3)
                        scores[b] = (3, len(chaine)) if tabou_blocked else (1, len(chaine))
                    else:
                        scores[b] = (2, len(chaine))
                mal_places.sort(key=lambda b: scores[b])
                
            elif methode == "H3":
                # H3 : Depth-First. Force le remplissage des positions ARRIERE en priorité stricte.
                mal_places.sort(key=lambda b: (
                    0 if self.config_cible.affectation[b].profondeur == Profondeur.ARRIERE else 1,
                    distance(self.affectation[b], EmplacementStock(navette_pos, Cote.GAUCHE, Profondeur.AVANT), self.n_colonnes)
                ))
                
            elif methode == "H6":
                # H6 : Macro-Structurelle (Linear Merge). Traite le rangement colonne par colonne séquentiellement.
                mal_places.sort(key=lambda b: self.config_cible.affectation[b].colonne)
                
            else:
                # H1, H2, H5, H7 : Nearest Neighbor. Trie purement basé sur la distance à la navette.
                mal_places.sort(key=lambda b: distance(self.affectation[b], EmplacementStock(navette_pos, Cote.GAUCHE, Profondeur.AVANT), self.n_colonnes))
                
            # ────────────────────────────────────────────────────────────────────────
            #  EXÉCUTION ET TRAITEMENT DES DÉPENDANCES
            # ────────────────────────────────────────────────────────────────────────
            
            action_faite = False
            bloque_par_tabou = False
            
            for candidat in mal_places:
                chaine, est_cycle = self._suivre_chaine(candidat)
                
                if est_cycle:
                    # Cas de Deadlock cyclique : nécessite un dégagement (Double Handling) pour briser le cycle
                    bac_a_degager = chaine[-1]
                    if self._degager(bac_a_degager, methode, chaine):
                        navette_pos = self.affectation[bac_a_degager].colonne
                        action_faite = True
                        break
                else:
                    # Cas standard : on déplace le dernier bac de la chaîne de dépendance
                    dernier_bac = chaine[-1]
                    cible = self.config_cible.affectation[dernier_bac]
                    
                    if self.affectation[dernier_bac] == cible:
                        # Si le dernier bac est bien placé mais obstrue la chaîne, on doit le dégager
                        if self._degager(dernier_bac, methode, chaine):
                            navette_pos = self.affectation[dernier_bac].colonne
                            action_faite = True
                            break
                    else:
                        # Validation Tabou pour empêcher le bouclage
                        if dernier_bac in self.tabou:
                            pos_tabou, it_tabou = self.tabou[dernier_bac]
                            if cible == pos_tabou and self.iteration - it_tabou <= 10:
                                bloque_par_tabou = True
                                continue 
                                
                        self._executer_mouvement(dernier_bac, cible)
                        navette_pos = cible.colonne
                        action_faite = True
                        break
                        
            if not action_faite:
                if bloque_par_tabou: continue 
                else: break # Deadlock complet (Impasse algorithmique)
                
        complet = (self.affectation == self.config_cible.affectation)
        dist_totale = sum(m.distance for m in self.mouvements)
        nb_dh = sum(1 for m in self.mouvements if m.est_double_handling)
        temps = time.time() - start_time
        
        total_bacs = len(self.config_cible.affectation)
        mal_places_fin = sum(1 for b, e in self.affectation.items() if e != self.config_cible.affectation[b])
        nb_bacs_bien_places = total_bacs - mal_places_fin
        
        return ResultatHeuristique(
            methode, dist_totale, list(self.mouvements), complet, 
            temps, nb_dh, nb_bacs_bien_places, total_bacs
        )

    def resoudre(self, methode: str) -> ResultatHeuristique:
        """Point d'entrée public pour résoudre l'instance via l'heuristique demandée."""
        return self._resoudre_generique(methode=methode)


# =============================================================================
#  3. Outil d'Évaluation Globale (Benchmarker)
# =============================================================================

def evaluer_heuristiques(dossier_instances: str | Path, fichier_sortie: str | Path = "resultats/resultats_heuristiques.csv"):
    """
    Fonction utilitaire traitant un dossier complet d'instances CSV.
    Exécute les 7 heuristiques sur chaque fichier et consolide les performances
    dans un rapport CSV global pour analyse ultérieure.
    """
    dossier = Path(dossier_instances)
    if not dossier.exists() or not dossier.is_dir():
        print(f"Dossier {dossier} introuvable.")
        return
        
    fichiers = list(dossier.glob("*.csv"))
    fichiers = [f for f in fichiers if f.name != "recapitulatif.csv" and f.name != "resultats_heuristiques.csv"]
    
    if not fichiers:
        print(f"Aucun fichier CSV trouvé dans {dossier}.")
        return
        
    chemin_sortie = Path(fichier_sortie)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    entetes = [
        "fichier", "heuristique", "complet", "distance_totale", 
        "nb_mouvements", "nb_double_handling", "temps_execution", 
        "nb_bacs_bien_places", "total_bacs"
    ]
    
    resultats = []
    
    # Benchmarking des 7 stratégies implémentées
    heuristiques_a_tester = ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]
    
    for i, f in enumerate(fichiers, 1):
        try:
            init, cible = importer_instance(f)
            
            for nom_h in heuristiques_a_tester:
                solveur = Solveur(init, cible)
                res = solveur.resoudre(nom_h)
                resultats.append({
                    "fichier": f.name,
                    "heuristique": res.nom,
                    "complet": 1 if res.complet else 0,
                    "distance_totale": res.distance_totale,
                    "nb_mouvements": len(res.mouvements),
                    "nb_double_handling": res.nb_double_handling,
                    "temps_execution": f"{res.temps_execution:.4f}",
                    "nb_bacs_bien_places": res.nb_bacs_bien_places,
                    "total_bacs": res.total_bacs
                })
            
            print(f"[{i}/{len(fichiers)}] {f.name} évalué avec succès.")
            
        except Exception as e:
            print(f"Erreur inattendue sur {f.name} : {e}")
            
    with chemin_sortie.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=entetes)
        writer.writeheader()
        writer.writerows(resultats)
        
    print(f"\nÉvaluation terminée. Résultats consolidés dans {chemin_sortie}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Évaluateur d'heuristiques avancées SBS/RS.")
    parser.add_argument("--dossier", "-d", type=str, default="instances", help="Dossier contenant les instances CSV.")
    parser.add_argument("--sortie", "-o", type=str, default="resultats/resultats_heuristiques.csv", help="Fichier de sortie.")
    args = parser.parse_args()
    
    evaluer_heuristiques(args.dossier, args.sortie)