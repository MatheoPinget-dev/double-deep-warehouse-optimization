from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import sys

from instance_generator import (
    Configuration, EmplacementStock, Cote, Profondeur,
    distance, importer_instance
)

class TypeDependance(Enum):
    DUR = auto()          # LIFO physique actuel
    OCCUPATION = auto()   # Occupation de la destination
    SEQUENCEMENT = auto() # Ordre d'insertion (ARRIERE doit être placé avant AVANT)

@dataclass
class Arc:
    source: int  # bac_id qui bloque
    cible: int   # bac_id qui est bloqué
    type_dep: TypeDependance

class GrapheDependance:
    """Graphe représentant les blocages entre les bacs de l'entrepôt."""
    def __init__(self, config_courante: Configuration, config_cible: Configuration):
        self.config_courante = config_courante
        self.config_cible = config_cible
        self.noeuds = set(config_courante.affectation.keys())
        self.arcs: list[Arc] = []
        self._construire_graphe()

    def _construire_graphe(self):
        occupes = {emp: bac_id for bac_id, emp in self.config_courante.affectation.items()}
        cibles_inverse = {emp: bac_id for bac_id, emp in self.config_cible.affectation.items()}
            
        # 1. Contrainte DUR (LIFO physique actuel)
        for c, s in self.config_courante.etage.demi_colonnes():
            emp_avant = EmplacementStock(c, s, Profondeur.AVANT)
            emp_arriere = EmplacementStock(c, s, Profondeur.ARRIERE)
            if emp_avant in occupes and emp_arriere in occupes:
                bac_avant = occupes[emp_avant]
                bac_arriere = occupes[emp_arriere]
                self.arcs.append(Arc(source=bac_avant, cible=bac_arriere, type_dep=TypeDependance.DUR))
                
        # 2. Contrainte OCCUPATION (Cible occupée)
        for bac_j, cible_j in self.config_cible.affectation.items():
            if self.config_courante.affectation[bac_j] == cible_j:
                continue
                
            # Si cible exacte est occupée
            if cible_j in occupes:
                bac_k = occupes[cible_j]
                if bac_k != bac_j:
                    self.arcs.append(Arc(source=bac_k, cible=bac_j, type_dep=TypeDependance.OCCUPATION))
                    
            # Si cible est ARRIERE et l'AVANT est occupé
            if cible_j.profondeur == Profondeur.ARRIERE:
                cible_j_avant = EmplacementStock(cible_j.colonne, cible_j.cote, Profondeur.AVANT)
                if cible_j_avant in occupes:
                    bac_k = occupes[cible_j_avant]
                    if bac_k != bac_j: 
                        self.arcs.append(Arc(source=bac_k, cible=bac_j, type_dep=TypeDependance.OCCUPATION))

        # 3. Contrainte SEQUENCEMENT (Insertion ARRIERE avant AVANT)
        for bac_j, cible_j in self.config_cible.affectation.items():
            if cible_j.profondeur == Profondeur.ARRIERE:
                if self.config_courante.affectation[bac_j] != cible_j:
                    cible_j_avant = EmplacementStock(cible_j.colonne, cible_j.cote, Profondeur.AVANT)
                    bac_k = cibles_inverse.get(cible_j_avant)
                    if bac_k is not None:
                        # j n'est pas en place, donc k ne peut pas s'y mettre
                        self.arcs.append(Arc(source=bac_j, cible=bac_k, type_dep=TypeDependance.SEQUENCEMENT))

    def trouver_bacs_sans_blocage(self) -> list[int]:
        """Retourne les bacs avec un degré entrant = 0 (feuilles)."""
        bacs_bloques = {arc.cible for arc in self.arcs}
        return [bac for bac in self.noeuds if bac not in bacs_bloques]
        
    def detecter_cycle(self) -> list[Arc] | None:
        """Détecte un cycle de dépendances et retourne la liste des arcs impliqués."""
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

class HeuristiqueConstructive:
    """Algorithme de routage glouton pour le réarrangement SBS/RS."""
    def __init__(self, config_init: Configuration, config_cible: Configuration):
        self.courante = config_init
        self.cible = config_cible
        self.colonne_navette = 0 # Point de départ de la navette (colonne d'entrée)
        self.distance_totale_x = 0
        self.sequence_mouvements = []
        self.derniers_buffers = {} # Mémorise l'itération à laquelle un bac a été mis en buffer
        self.iteration_actuelle = 0
        
    def _bac_est_place(self, bac_id: int) -> bool:
        return self.courante.affectation[bac_id] == self.cible.affectation[bac_id]
        
    def _bacs_mal_places(self) -> set[int]:
        return {bac for bac in self.courante.affectation if not self._bac_est_place(bac)}
        
    def trouver_buffer(self, bac_id: int) -> EmplacementStock:
        """Trouve le meilleur emplacement vide temporaire sur le trajet (ou au plus proche)."""
        emp_actuel = self.courante.affectation[bac_id]
        emp_cible = self.cible.affectation[bac_id]
        col_actuelle = emp_actuel.colonne
        col_cible = emp_cible.colonne
        
        vides = self.courante.emplacements_vides_stock()
        vides_accessibles = [emp for emp in vides if self.courante.est_accessible(emp)]
        
        if not vides_accessibles:
            raise RuntimeError("Aucun emplacement vide accessible pour servir de buffer !")
            
        cibles_mal_places = {self.cible.affectation[b] for b in self._bacs_mal_places()}
        
        # Fonction de score pour éviter de bloquer des bacs mal placés ou de prendre des cibles
        def score_buffer(emp: EmplacementStock) -> int:
            score_base = 0
            if emp in cibles_mal_places:
                score_base += 10  # Très mauvais : on occupe la cible d'un autre bac !
                
            if emp.profondeur == Profondeur.ARRIERE:
                return score_base + 0  # Ne bloque personne
            
            emp_arriere = EmplacementStock(emp.colonne, emp.cote, Profondeur.ARRIERE)
            bac_arriere = self.courante.bac_a(emp_arriere)
            
            if bac_arriere is None:
                return score_base + 1  # Ne bloque personne pour l'instant
            elif self._bac_est_place(bac_arriere):
                return score_base + 2  # Bloque un bac bien placé
            else:
                return score_base + 3  # Bloque un bac mal placé
                
        min_col = min(col_actuelle, col_cible)
        max_col = max(col_actuelle, col_cible)
        sur_trajet = [e for e in vides_accessibles if min_col <= e.colonne <= max_col]
        
        cibles = sur_trajet if sur_trajet else vides_accessibles
        
        # On trie d'abord par score de sécurité, puis par distance
        cibles.sort(key=lambda e: (score_buffer(e), abs(e.colonne - col_actuelle)))
        return cibles[0]

    def _deplacer_bac(self, bac_id: int, dest: EmplacementStock, est_buffer: bool = False):
        """Simule le déplacement d'un bac par la navette."""
        source = self.courante.affectation[bac_id]
        
        dist_aller = abs(self.colonne_navette - source.colonne)
        dist_retour = abs(source.colonne - dest.colonne)
        
        self.distance_totale_x += dist_aller + dist_retour
        self.colonne_navette = dest.colonne
        
        self.courante.affectation[bac_id] = dest
        action = f"Bac {bac_id:2d} : {source.colonne:2d}-{source.cote.value}-{source.profondeur.value} -> {dest.colonne:2d}-{dest.cote.value}-{dest.profondeur.value}"
        if est_buffer:
            action += " (BUFFER)"
        self.sequence_mouvements.append(action)

    def planifier_mouvements(self):
        """Boucle principale de décision du routage de la navette."""
        max_iterations = len(self.courante.affectation) * 5
        
        while True:
            self.iteration_actuelle += 1
            if self.iteration_actuelle > max_iterations:
                print("--- BOUCLE INFINIE DETECTEE ---")
                print("\n".join(self.sequence_mouvements[-30:]))
                raise RuntimeError("Boucle infinie détectée dans l'heuristique.")
                
            bacs_a_placer = self._bacs_mal_places()
            if not bacs_a_placer:
                break
                
            graphe = GrapheDependance(self.courante, self.cible)
            eligibles_graphe = graphe.trouver_bacs_sans_blocage()
            
            # Ne sélectionner que les bacs mal placés
            eligibles = [b for b in eligibles_graphe if b in bacs_a_placer]
            
            if not eligibles:
                arcs_cycle = graphe.detecter_cycle()
                if arcs_cycle:
                    bacs_bloques_dur = {arc.cible for arc in graphe.arcs if arc.type_dep == TypeDependance.DUR}
                    
                    # On cherche un arc du cycle qui peut être cassé en bougeant sa source
                    candidats = []
                    for arc in arcs_cycle:
                        if arc.type_dep in (TypeDependance.DUR, TypeDependance.OCCUPATION):
                            bac = arc.source
                            # S'il n'est pas physiquement bloqué
                            if bac not in bacs_bloques_dur:
                                candidats.append(bac)
                                
                    if not candidats:
                        raise RuntimeError("Cycle insoluble : tous les candidats sont physiquement inaccessibles ou le cycle est purement séquentiel.")
                        
                    # Choisir le candidat en pénalisant ceux qui ont été récemment mis en buffer
                    candidats.sort(key=lambda b: (self.derniers_buffers.get(b, -1), abs(self.courante.affectation[b].colonne - self.colonne_navette)))
                    bac_choisi = candidats[0]
                    buffer_emp = self.trouver_buffer(bac_choisi)
                    self._deplacer_bac(bac_choisi, buffer_emp, est_buffer=True)
                    self.derniers_buffers[bac_choisi] = self.iteration_actuelle
                    continue
                else:
                    raise RuntimeError("Blocage insoluble sans cycle détecté !")
            else:
                eligibles.sort(key=lambda b: abs(self.courante.affectation[b].colonne - self.colonne_navette))
                bac_choisi = eligibles[0]
                cible_finale = self.cible.affectation[bac_choisi]
                self._deplacer_bac(bac_choisi, cible_finale, est_buffer=False)
                
        print(f"\n[Terminé] Tous les bacs sont placés correctement.")
        print(f"Distance Totale X (Navette) : {self.distance_totale_x}")
        print("Séquence de mouvements :")
        for i, m in enumerate(self.sequence_mouvements, 1):
            print(f"  {i:3d}. {m}")

def main():
    fichier = "instance_test.csv"
    if len(sys.argv) > 1:
        fichier = sys.argv[1]
        
    if not Path(fichier).exists():
        print(f"Erreur: L'instance '{fichier}' est introuvable.")
        print("Veuillez générer une instance d'abord (ex: python instance_generator.py une -c 15 -t 0.7 -e instance_test.csv)")
        return
        
    print(f"--- Lancement de l'Heuristique sur : {fichier} ---")
    init, cible = importer_instance(fichier)
    heuristique = HeuristiqueConstructive(init, cible)
    heuristique.planifier_mouvements()

if __name__ == "__main__":
    main()
