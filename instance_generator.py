"""
Générateur d'instances — Réarrangement SBS/RS double profondeur (1 étage).
═══════════════════════════════════════════════════════════════════════════

Ce module génère des instances de test pour le problème de réarrangement 
d'un entrepôt automatisé (SBS/RS) à double profondeur. 

Il produit des paires de configurations (État Initial -> État Cible) composées 
du même ensemble de bacs, permettant d'évaluer les algorithmes de résolution.
"""

from __future__ import annotations
import csv
import math
import random
import argparse
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

# =============================================================================
#  1. Structures de base (Énumérations)
# =============================================================================

class Cote(StrEnum):
    """Définit le côté de l'allée centrale où se situe l'étagère."""
    GAUCHE = "G"
    DROITE = "D"

class Profondeur(StrEnum):
    """
    Définit la position dans la double profondeur de l'étagère.
    - AVANT : Accès direct depuis l'allée.
    - ARRIERE : Accessible uniquement si la position AVANT est vide.
    """
    AVANT = "F"    
    ARRIERE = "B"  

class StrategieDistribution(StrEnum):
    """
    Définit la loi de probabilité pour la répartition spatiale des bacs.
    - UNIFORME : Répartition purement aléatoire.
    - NORMALE : Concentration plus forte près de l'entrée (colonne 0).
    """
    UNIFORME = "uniforme" 
    NORMALE = "normale"   

# =============================================================================
#  2. Définition des emplacements
# =============================================================================

class EmplacementStock(NamedTuple):
    """
    Modélise une coordonnée spatiale unique dans l'entrepôt.
    Défini par : (colonne, côté, profondeur).
    """
    colonne: int
    cote: Cote
    profondeur: Profondeur

    def __repr__(self) -> str:
        return f"({self.colonne},{self.cote},{self.profondeur})"

type Emplacement = EmplacementStock

def colonne_de(emplacement: Emplacement, n_colonnes: int) -> int:
    """Extrait l'indice de colonne d'un emplacement."""
    return emplacement.colonne

def distance(e1: Emplacement, e2: Emplacement, n_colonnes: int) -> int:
    """Calcule la distance de Manhattan stricte sur l'axe des colonnes."""
    return abs(colonne_de(e1, n_colonnes) - colonne_de(e2, n_colonnes))

# =============================================================================
#  3. Modélisation de l'espace (L'Étage)
# =============================================================================

@dataclass(slots=True)
class Etage:
    """
    Représente la structure physique de l'entrepôt.
    Définit la capacité et génère les emplacements disponibles.
    """
    n_colonnes: int

    def __post_init__(self) -> None:
        if self.n_colonnes < 1:
            raise ValueError("Le nombre de colonnes doit être ≥ 1.")

    @property
    def stocks(self) -> list[EmplacementStock]:
        """Génère la liste exhaustive des (4 * n_colonnes) emplacements possibles."""
        return [
            EmplacementStock(c, s, p)
            for c in range(self.n_colonnes)
            for s in Cote
            for p in Profondeur
        ]

    @property
    def tous(self) -> list[Emplacement]:
        return self.stocks

    @property
    def capacite_stock(self) -> int:
        """Capacité de stockage théorique maximale."""
        return 4 * self.n_colonnes

    @property
    def capacite_totale(self) -> int:
        return self.capacite_stock

    def demi_colonnes(self) -> list[tuple[int, Cote]]:
        """
        Génère les paires (colonne, côté).
        Une "demi-colonne" correspond à une travée offrant 2 profondeurs (AVANT/ARRIERE).
        """
        return [(c, s) for c in range(self.n_colonnes) for s in Cote]

# =============================================================================
#  4. Gestion de l'état (La Configuration)
# =============================================================================

@dataclass(slots=True)
class Configuration:
    """
    Définit un état de l'entrepôt à un instant T.
    Mappe de manière bijective les identifiants de bacs vers des emplacements.
    """
    etage: Etage
    affectation: dict[int, Emplacement] = field(default_factory=dict)

    def emplacements_occupes(self) -> set[Emplacement]:
        """Retourne l'ensemble des coordonnées actuellement occupées."""
        return set(self.affectation.values())

    def emplacements_vides_stock(self) -> list[EmplacementStock]:
        """Retourne l'ensemble des coordonnées libres."""
        occupes = self.emplacements_occupes()
        return [e for e in self.etage.stocks if e not in occupes]

    def bac_a(self, emplacement: Emplacement) -> int | None:
        """Identifie quel bac occupe un emplacement précis."""
        for bac, emp in self.affectation.items():
            if emp == emplacement:
                return bac
        return None

    @property
    def n_bacs(self) -> int:
        return len(self.affectation)

    @property
    def taux_remplissage(self) -> float:
        return self.n_bacs / self.etage.capacite_stock if self.etage.capacite_stock else 0.0

    def est_valide(self) -> bool:
        """
        Vérification de la contrainte physique "Back-First".
        Il est physiquement impossible d'avoir un bac en position AVANT 
        si la position ARRIERE de la même demi-colonne est vide.
        """
        occupes = self.emplacements_occupes()
        for c, s in self.etage.demi_colonnes():
            avant = EmplacementStock(c, s, Profondeur.AVANT)
            arriere = EmplacementStock(c, s, Profondeur.ARRIERE)
            
            # Violation de contrainte : bac à l'avant mais vide à l'arrière
            if avant in occupes and arriere not in occupes:
                return False 
        return True

    def est_accessible(self, emplacement: Emplacement) -> bool:
        """
        Vérifie si un emplacement est accessible par la navette.
        La position ARRIERE nécessite que la position AVANT soit libre.
        """
        occupes = self.emplacements_occupes()
        if emplacement.profondeur == Profondeur.AVANT:
            return True
        avant = EmplacementStock(emplacement.colonne, emplacement.cote, Profondeur.AVANT)
        return avant not in occupes

    def distance_totale_vers(self, cible: Configuration) -> int:
        """
        Calcule la borne inférieure de déplacement.
        Somme des distances de Manhattan directes entre chaque bac et sa cible.
        """
        total = 0
        for bac_id, emp_init in self.affectation.items():
            emp_fin = cible.affectation[bac_id]
            total += distance(emp_init, emp_fin, self.etage.n_colonnes)
        return total

    def nb_bacs_deja_places(self, cible: Configuration) -> int:
        """Compte le nombre de bacs dont la position initiale est déjà la position cible."""
        return sum(
            1 for bac_id, emp in self.affectation.items()
            if cible.affectation.get(bac_id) == emp
        )

    def afficher(self, titre: str = "Configuration") -> str:
        """Génère une représentation textuelle ASCII de la configuration de l'entrepôt."""
        nc = self.etage.n_colonnes
        largeur = 6 

        def _contenu(c: int, s: Cote, p: Profondeur) -> str:
            emp = EmplacementStock(c, s, p)
            bac = self.bac_a(emp)
            return f"{bac:4d}" if bac is not None else "  --"

        en_tete = "         " + "".join(f"{c:^{largeur + 1}}" for c in range(nc))
        sep_haut = "         ┌" + "┬".join("─" * largeur for _ in range(nc)) + "┐"
        sep_mid = "         ├" + "┼".join("─" * largeur for _ in range(nc)) + "┤  ← allée"
        sep_bas = "         └" + "┴".join("─" * largeur for _ in range(nc)) + "┘"

        def _ligne(s: Cote, p: Profondeur) -> str:
            label = f"  {s}-{p}    "
            cellules = "│".join(f"{_contenu(c, s, p):^{largeur}}" for c in range(nc))
            return f"{label}│{cellules}│"

        lignes = [
            f"\n  ╔═ {titre} ({self.n_bacs} bacs, remplissage {self.taux_remplissage:.0%}) ═╗\n",
            en_tete,
            sep_haut,
            _ligne(Cote.GAUCHE, Profondeur.ARRIERE),
            _ligne(Cote.GAUCHE, Profondeur.AVANT),
            sep_mid,
            _ligne(Cote.DROITE, Profondeur.AVANT),
            _ligne(Cote.DROITE, Profondeur.ARRIERE),
            sep_bas,
        ]
        return "\n".join(lignes)

    def __repr__(self) -> str:
        return self.afficher()

# =============================================================================
#  5. Moteurs de Génération Stochastique
# =============================================================================

def _repartir_bacs_uniforme(n_bacs: int, etage: Etage, rng: random.Random) -> list[int]:
    """
    Répartit le volume de bacs aléatoirement sur toutes les demi-colonnes.
    Retourne une liste contenant le nombre de bacs (0, 1 ou 2) affectés à chaque demi-colonne.
    """
    demi_cols_dispos = etage.demi_colonnes()
    occupations_dict = {dc: 0 for dc in demi_cols_dispos}
    
    bacs_restants = n_bacs
    while bacs_restants > 0:
        # Sélection équiprobable d'une demi-colonne
        choix = rng.choice(demi_cols_dispos)
        occupations_dict[choix] += 1
        bacs_restants -= 1
        
        # Retrait de la demi-colonne si sa capacité maximale (2) est atteinte
        if occupations_dict[choix] == 2:
            demi_cols_dispos.remove(choix)
            
    return [occupations_dict[dc] for dc in etage.demi_colonnes()]

def _repartir_bacs_normale(n_bacs: int, etage: Etage, rng: random.Random) -> list[int]:
    """
    Répartit le volume de bacs via une distribution de Gauss (Loi Normale).
    Cette méthode simule la méthode logistique ABC en concentrant les probabilités
    d'affectation vers la colonne 0 (entrée de l'allée).
    """
    demi_cols_dispos = etage.demi_colonnes()
    occupations_dict = {dc: 0 for dc in demi_cols_dispos}
    
    # Paramètre d'étalement pondérant les colonnes proches de l'indice 0
    sigma = max(1.0, etage.n_colonnes / 3.0)
    def get_weight(c: int) -> float:
        return math.exp(-0.5 * (c / sigma)**2)
        
    weights = [get_weight(c) for c, _ in demi_cols_dispos]
    
    bacs_restants = n_bacs
    while bacs_restants > 0:
        # Sélection pondérée favorisant l'entrée
        choix = rng.choices(demi_cols_dispos, weights=weights, k=1)[0]
        occupations_dict[choix] += 1
        bacs_restants -= 1
        
        if occupations_dict[choix] == 2:
            idx = demi_cols_dispos.index(choix)
            demi_cols_dispos.pop(idx)
            weights.pop(idx)
            
    return [occupations_dict[dc] for dc in etage.demi_colonnes()]


def _generer_emplacements(
    n_bacs: int,
    etage: Etage,
    rng: random.Random,
    strategie: StrategieDistribution
) -> list[EmplacementStock]:
    """Génère la liste des emplacements cibles respectant la contrainte Back-First."""
    demi_cols = etage.demi_colonnes()
    
    if strategie == StrategieDistribution.NORMALE:
        occupations = _repartir_bacs_normale(n_bacs, etage, rng)
    else:
        occupations = _repartir_bacs_uniforme(n_bacs, etage, rng)

    emplacements: list[EmplacementStock] = []
    for (c, s), occ in zip(demi_cols, occupations):
        if occ >= 1:
            emplacements.append(EmplacementStock(c, s, Profondeur.ARRIERE))
        if occ == 2:
            emplacements.append(EmplacementStock(c, s, Profondeur.AVANT))
            
    return emplacements

def generer_configuration(
    etage: Etage,
    bac_ids: list[int],
    rng: random.Random,
    strategie: StrategieDistribution
) -> Configuration:
    """
    Construit une configuration valide en respectant la contrainte Back-First.
    """
    emplacements = _generer_emplacements(len(bac_ids), etage, rng, strategie)

    # Mélange final des identifiants et des positions
    bacs = list(bac_ids)
    rng.shuffle(bacs)          
    rng.shuffle(emplacements)  

    affectation = dict(zip(bacs, emplacements, strict=True))
    return Configuration(etage=etage, affectation=affectation)

def generer_configuration_cible(
    config_init: Configuration,
    rng: random.Random,
    strategie: StrategieDistribution,
    prob_maintien_profondeur: float
) -> Configuration:
    """
    Construit la configuration cible en favorisant le maintien de la profondeur 
    (AVANT/ARRIERE) selon une probabilité donnée.
    """
    emplacements = _generer_emplacements(config_init.n_bacs, config_init.etage, rng, strategie)
    
    cibles_arr = [e for e in emplacements if e.profondeur == Profondeur.ARRIERE]
    cibles_av = [e for e in emplacements if e.profondeur == Profondeur.AVANT]
    
    bacs_voulant_arr = []
    bacs_voulant_av = []
    
    for bac_id, emp_init in config_init.affectation.items():
        if rng.random() < prob_maintien_profondeur:
            if emp_init.profondeur == Profondeur.ARRIERE:
                bacs_voulant_arr.append(bac_id)
            else:
                bacs_voulant_av.append(bac_id)
        else:
            if emp_init.profondeur == Profondeur.ARRIERE:
                bacs_voulant_av.append(bac_id)
            else:
                bacs_voulant_arr.append(bac_id)
                
    # Équilibrage pour correspondre exactement aux emplacements générés
    while len(bacs_voulant_arr) > len(cibles_arr):
        idx = rng.randrange(len(bacs_voulant_arr))
        bacs_voulant_av.append(bacs_voulant_arr.pop(idx))
        
    while len(bacs_voulant_av) > len(cibles_av):
        idx = rng.randrange(len(bacs_voulant_av))
        bacs_voulant_arr.append(bacs_voulant_av.pop(idx))
        
    rng.shuffle(bacs_voulant_arr)
    rng.shuffle(cibles_arr)
    rng.shuffle(bacs_voulant_av)
    rng.shuffle(cibles_av)
    
    affectation = {}
    for b, e in zip(bacs_voulant_arr, cibles_arr, strict=True):
        affectation[b] = e
    for b, e in zip(bacs_voulant_av, cibles_av, strict=True):
        affectation[b] = e
        
    return Configuration(etage=config_init.etage, affectation=affectation)


def generer_instance(
    n_colonnes: int,
    taux_remplissage: float = 0.90,
    graine: int | None = None,
    strategie: StrategieDistribution = StrategieDistribution.UNIFORME,
    prob_maintien_profondeur: float = 0.80
) -> tuple[Configuration, Configuration]:
    """
    Génère une instance de problème complète.
    Retourne un tuple (Configuration Initiale, Configuration Cible) utilisant 
    exactement le même ensemble de bacs.
    """
    if not 0.0 < taux_remplissage <= 1.0:
        raise ValueError("Le taux de remplissage doit être dans ]0, 1].")
    if not 0.0 <= prob_maintien_profondeur <= 1.0:
        raise ValueError("La probabilité de maintien de profondeur doit être dans [0, 1].")

    rng = random.Random(graine)
    etage = Etage(n_colonnes)

    # Calcul du volume de bacs à générer selon le taux demandé
    n_bacs = round(taux_remplissage * etage.capacite_stock)
    n_bacs = max(1, min(n_bacs, etage.capacite_stock))

    bac_ids = list(range(1, n_bacs + 1))

    # Génération de l'état initial, puis de l'état cible en tenant compte du paramètre
    config_initiale = generer_configuration(etage, bac_ids, rng, strategie)
    config_cible = generer_configuration_cible(config_initiale, rng, strategie, prob_maintien_profondeur)

    # Validations de sécurité de l'intégrité des données
    assert config_initiale.est_valide(), "Validation échouée: Configuration initiale invalide."
    assert config_cible.est_valide(), "Validation échouée: Configuration cible invalide."
    assert config_initiale.n_bacs == config_cible.n_bacs, "Validation échouée: Asymétrie des bacs."

    return config_initiale, config_cible

# =============================================================================
#  6. Sérialisation (Import/Export CSV)
# =============================================================================

_EN_TETES_CSV = ["config", "bac_id", "type", "colonne", "cote", "profondeur"]

def exporter_instance(config_init: Configuration, config_cible: Configuration, chemin: str | Path) -> None:
    """Exporte l'instance générée (Initiale et Cible) au format CSV standardisé."""
    chemin = Path(chemin)
    with chemin.open("w", newline="", encoding="utf-8") as f:
        # En-tête des métadonnées pour la traçabilité
        f.write(
            f"# n_colonnes={config_init.etage.n_colonnes}"
            f" n_bacs={config_init.n_bacs}"
            f" taux={config_init.taux_remplissage:.4f}\n"
        )
        writer = csv.DictWriter(f, fieldnames=_EN_TETES_CSV)
        writer.writeheader()

        # Écriture séquentielle des deux configurations
        for nom_config, config in [("initiale", config_init), ("cible", config_cible)]:
            for bac_id in sorted(config.affectation):
                emp = config.affectation[bac_id]
                writer.writerow({
                    "config": nom_config, "bac_id": bac_id, "type": "stock",
                    "colonne": emp.colonne, "cote": emp.cote, "profondeur": emp.profondeur,
                })

def importer_instance(chemin: str | Path) -> tuple[Configuration, Configuration]:
    """Recharge en mémoire une instance à partir de son fichier CSV."""
    chemin = Path(chemin)
    n_colonnes: int | None = None

    with chemin.open("r", encoding="utf-8") as f:
        premiere_ligne = f.readline().strip()
        if premiere_ligne.startswith("#"):
            for fragment in premiere_ligne.split():
                if fragment.startswith("n_colonnes="):
                    n_colonnes = int(fragment.split("=")[1])

        if n_colonnes is None:
            raise ValueError("Métadonnées manquantes : n_colonnes introuvable.")

        etage = Etage(n_colonnes)
        affectations: dict[str, dict[int, Emplacement]] = {"initiale": {}, "cible": {}}

        reader = csv.DictReader(f)
        for ligne in reader:
            bac_id = int(ligne["bac_id"])
            emp: Emplacement = EmplacementStock(
                int(ligne["colonne"]), Cote(ligne["cote"]), Profondeur(ligne["profondeur"]))
            affectations[ligne["config"]][bac_id] = emp

    return (
        Configuration(etage=etage, affectation=affectations["initiale"]),
        Configuration(etage=etage, affectation=affectations["cible"]),
    )

# =============================================================================
#  7. Mode Laboratoire (Plan d'expériences par lots)
# =============================================================================

def generer_lot(
    colonnes: list[int] | None = None,
    taux: list[float] | None = None,
    repetitions: int = 10,
    strategie: StrategieDistribution = StrategieDistribution.UNIFORME,
    prob_maintien_profondeur: float = 0.80,
    dossier: str | Path = "instances",
) -> Path:
    """
    Exécute un plan d'expériences massif pour évaluer les algorithmes.
    Génère des instances pour diverses configurations de tailles et taux de remplissage.
    """
    if colonnes is None:
        colonnes = [5, 10, 20, 50]
    if taux is None:
        taux = [0.50, 0.60, 0.70, 0.80, 0.90, 0.92]

    dossier = Path(dossier)
    dossier.mkdir(parents=True, exist_ok=True)

    # Initialisation du fichier de métriques
    recap_path = dossier / "recapitulatif.csv"
    recap_entetes = [
        "fichier", "n_colonnes", "taux_remplissage", "n_bacs", "strategie",
        "graine", "prob_maintien_profondeur", "bacs_a_deplacer", "deja_places", "borne_inferieure",
    ]

    lignes_recap: list[dict] = []
    n_total = len(colonnes) * len(taux) * repetitions
    compteur = 0

    for nc in colonnes:
        for t in taux:
            for rep in range(repetitions):
                # Calcul d'une graine déterministe pour garantir la reproductibilité
                # de l'expérience scientifique indépendamment de l'environnement d'exécution.
                graine = nc * 100_000 + int(t * 1000) + rep
                compteur += 1

                config_init, config_cible = generer_instance(
                    n_colonnes=nc, taux_remplissage=t, graine=graine, strategie=strategie, 
                    prob_maintien_profondeur=prob_maintien_profondeur
                )

                strat_str = "uni" if strategie == StrategieDistribution.UNIFORME else "norm"
                nom = f"inst_c{nc}_t{int(t*100)}_{strat_str}_r{rep:02d}.csv"
                chemin = dossier / nom
                exporter_instance(config_init, config_cible, chemin)

                # Extraction des métriques de complexité
                deja = config_init.nb_bacs_deja_places(config_cible)
                borne = config_init.distance_totale_vers(config_cible)

                lignes_recap.append({
                    "fichier": nom,
                    "n_colonnes": nc,
                    "taux_remplissage": round(t, 2),
                    "n_bacs": config_init.n_bacs,
                    "strategie": strategie.value,
                    "graine": graine,
                    "prob_maintien_profondeur": prob_maintien_profondeur,
                    "bacs_a_deplacer": config_init.n_bacs - deja,
                    "deja_places": deja,
                    "borne_inferieure": borne,
                })

                print(f"  [{compteur:3d}/{n_total}] {nom}  ({config_init.n_bacs} bacs)")

    # Sauvegarde des métriques du plan d'expériences
    with recap_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=recap_entetes)
        writer.writeheader()
        writer.writerows(lignes_recap)

    print(f"\n  ✓ {compteur} instances générées dans le dossier '{dossier}/'")
    return dossier

# =============================================================================
#  8. Interface en Ligne de Commande (CLI)
# =============================================================================

def main() -> None:
    """Point d'entrée du programme. Gère le parsing des arguments système."""
    analyseur = argparse.ArgumentParser(
        description="Générateur d'instances SBS/RS double profondeur."
    )
    sous_commandes = analyseur.add_subparsers(dest="commande", help="Sélection du mode d'exécution")

    # Mode 1 : Génération unitaire
    p_une = sous_commandes.add_parser("une", help="Génération d'une instance unique")
    p_une.add_argument("--colonnes", "-c", type=int, default=10)
    p_une.add_argument("--taux", "-t", type=float, default=0.90)
    p_une.add_argument("--graine", "-g", type=int, default=None)
    p_une.add_argument("--strategie", "-s", type=str, choices=["uniforme", "normale"], default="uniforme")
    p_une.add_argument("--prob-maintien", "-p", type=float, default=0.80, help="Probabilité qu'un bac garde la même profondeur (0 à 1)")
    p_une.add_argument("--export", "-e", type=str, default=None)

    # Mode 2 : Génération par lot (Batch)
    p_lot = sous_commandes.add_parser("lot", help="Génération d'un plan d'expériences par lots")
    p_lot.add_argument("--colonnes", "-c", type=int, nargs="+", default=[5, 10, 20, 50])
    p_lot.add_argument("--taux", "-t", type=float, nargs="+", default=[0.50, 0.60, 0.70, 0.80, 0.90, 0.92])
    p_lot.add_argument("--repetitions", "-r", type=int, default=10)
    p_lot.add_argument("--strategie", "-s", type=str, choices=["uniforme", "normale"], default="uniforme")
    p_lot.add_argument("--prob-maintien", "-p", type=float, default=0.80, help="Probabilité qu'un bac garde la même profondeur (0 à 1)")
    p_lot.add_argument("--dossier", "-d", type=str, default="instances")

    args = analyseur.parse_args()

    if args.commande == "lot":
        strat = StrategieDistribution(args.strategie)
        generer_lot(
            colonnes=args.colonnes, taux=args.taux, repetitions=args.repetitions,
            strategie=strat, prob_maintien_profondeur=args.prob_maintien, dossier=args.dossier,
        )

    elif args.commande == "une":
        strat = StrategieDistribution(args.strategie)
        config_init, config_cible = generer_instance(
            n_colonnes=args.colonnes, taux_remplissage=args.taux, graine=args.graine, strategie=strat,
            prob_maintien_profondeur=args.prob_maintien
        )

        print(config_init.afficher(f"ETAT INITIAL ({strat.value.upper()})"))
        print()
        print(config_cible.afficher(f"ETAT CIBLE ({strat.value.upper()})"))

        borne_inf = config_init.distance_totale_vers(config_cible)
        deja_places = config_init.nb_bacs_deja_places(config_cible)
        
        print(f"\n  ── Métriques ──")
        print(f"  Bacs à déplacer  : {config_init.n_bacs - deja_places} / {config_init.n_bacs}")
        print(f"  Bacs validés     : {deja_places}")
        print(f"  Borne inférieure : {borne_inf} cases")

        if args.export:
            exporter_instance(config_init, config_cible, args.export)

    else:
        analyseur.print_help()

if __name__ == "__main__":
    main()