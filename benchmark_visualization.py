import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_benchmark_graphs(dossier_resultats: str = "resultats"):
    """
    Lit les fichiers CSV générés par solver.py et metaheuristics.py
    et produit des graphiques pertinents pour le benchmark.
    """
    path_res = Path(dossier_resultats)
    fichier_heuristiques = path_res / "resultats_heuristiques.csv"
    fichier_meta = path_res / "resultats_metaheuristiques.csv"
    
    # 1. Chargement des données
    dfs = []
    
    if fichier_heuristiques.exists():
        df_h = pd.read_csv(fichier_heuristiques)
        # Harmonisation du nom de la colonne pour regrouper les algorithmes
        df_h.rename(columns={"heuristique": "algorithme"}, inplace=True)
        df_h["type"] = "Heuristique"
        dfs.append(df_h)
    else:
        print(f"⚠️ Fichier {fichier_heuristiques} introuvable.")
        
    if fichier_meta.exists():
        df_m = pd.read_csv(fichier_meta)
        df_m.rename(columns={"metaheuristique": "algorithme"}, inplace=True)
        df_m["type"] = "Métaheuristique"
        dfs.append(df_m)
    else:
        print(f"⚠️ Fichier {fichier_meta} introuvable.")
        
    if not dfs:
        print("Aucune donnée à analyser. Veuillez exécuter solver.py et metaheuristics.py d'abord.")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Extraction du taux de remplissage depuis le nom du fichier (ex: inst_c10_t70_norm_r03.csv -> 70)
    # L'expression régulière cherche '_t' suivi de chiffres, suivi de '_'
    df['taux_remplissage'] = df['fichier'].str.extract(r'_t(\d+)_').astype(float)
    
    # Extraction du nombre de colonnes (taille de l'entrepôt)
    df['n_colonnes'] = df['fichier'].str.extract(r'_c(\d+)_').astype(float)
    
    # Calcul des ratios pertinents
    df['ratio_energetique'] = df['distance_totale'] / df['nb_mouvements']
    df['ratio_double_handling'] = (df['nb_double_handling'] / df['nb_mouvements']) * 100
    
    # Configuration du style visuel
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Création du dossier pour les graphiques s'il n'existe pas
    dossier_graphes = Path("images")
    dossier_graphes.mkdir(exist_ok=True)
    
    print("Génération des graphiques en cours...")

    # =========================================================================
    # Graphique 1 : Taux de réussite (Complétion)
    # =========================================================================
    plt.figure(figsize=(10, 6))
    success_rates = df.groupby('algorithme')['complet'].mean() * 100
    success_rates = success_rates.sort_values(ascending=False).reset_index()
    
    ax = sns.barplot(x='algorithme', y='complet', data=success_rates, hue='algorithme', legend=False, palette="viridis")
    plt.title("Taux de résolution complète par algorithme (%)", fontsize=14, pad=15)
    plt.ylabel("Taux de succès (%)")
    plt.xlabel("Algorithme")
    plt.ylim(0, 110)
    
    # Ajout des valeurs sur les barres
    for i in ax.containers:
        ax.bar_label(i, fmt='%.1f%%', padding=3)
        
    plt.tight_layout()
    plt.savefig(dossier_graphes / "01_taux_reussite.png", dpi=300)
    plt.close()

    # =========================================================================
    # Filtrage : On ne garde que les instances résolues pour comparer la qualité
    # =========================================================================
    df_resolu = df[df['complet'] == 1]
    
    if not df_resolu.empty:
        # Graphique 2 : Qualité de la solution (Distance Totale)
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='algorithme', y='distance_totale', data=df_resolu, hue='algorithme', legend=False, palette="Set2")
        plt.title("Distribution de la Distance Totale Parcourue (Instances résolues)", fontsize=14, pad=15)
        plt.ylabel("Distance Totale (Unité de case)")
        plt.xlabel("Algorithme")
        plt.tight_layout()
        plt.savefig(dossier_graphes / "02_qualite_distance.png", dpi=300)
        plt.close()
        
        # Graphique 3 : Efficacité en nombre de mouvements (Mouvements totaux vs Double Handling)
        plt.figure(figsize=(12, 6))
        df_melt = df_resolu.melt(id_vars=['algorithme'], value_vars=['nb_mouvements', 'nb_double_handling'], 
                                 var_name='Métrique', value_name='Quantité')
        sns.barplot(x='algorithme', y='Quantité', hue='Métrique', data=df_melt, errorbar=None, palette="pastel")
        plt.title("Moyenne des Mouvements et Double Handling par algorithme", fontsize=14, pad=15)
        plt.ylabel("Nombre moyen d'opérations")
        plt.xlabel("Algorithme")
        plt.tight_layout()
        plt.savefig(dossier_graphes / "03_efficacite_mouvements.png", dpi=300)
        plt.close()

    # =========================================================================
    # Graphique 4 : Temps d'exécution (CPU Time)
    # =========================================================================
    plt.figure(figsize=(10, 6))
    # On utilise une échelle logarithmique si l'écart est très grand entre Heuristiques et Métaheuristiques
    ax = sns.barplot(x='algorithme', y='temps_execution', data=df, hue='type', errorbar=None)
    plt.title("Temps d'exécution moyen (en secondes)", fontsize=14, pad=15)
    plt.ylabel("Secondes (Moyenne)")
    plt.xlabel("Algorithme")
    plt.yscale("log") # Log-scale très utile car SA/GA sont beaucoup plus lents que H1-H7
    plt.tight_layout()
    plt.savefig(dossier_graphes / "04_temps_execution_log.png", dpi=300)
    plt.close()
    
    # =========================================================================
    # Graphique 5 : Taux de réussite en fonction du Taux de Remplissage
    # =========================================================================
    if 'taux_remplissage' in df.columns and not df['taux_remplissage'].isna().all():
        plt.figure(figsize=(12, 6))
        
        # Calcul du % de réussite par algorithme et par taux de remplissage
        success_by_fill = df.groupby(['algorithme', 'taux_remplissage'])['complet'].mean().reset_index()
        success_by_fill['complet'] *= 100
        
        sns.lineplot(data=success_by_fill, x='taux_remplissage', y='complet', hue='algorithme', marker='o', linewidth=2)
        plt.title("Évolution du Taux de Réussite selon la Densité de l'Entrepôt", fontsize=14, pad=15)
        plt.ylabel("Taux de succès (%)")
        plt.xlabel("Taux de Remplissage (%)")
        plt.ylim(-5, 105)
        
        # Ajout d'une grille plus fine
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend(title='Algorithme', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(dossier_graphes / "05_taux_reussite_vs_remplissage.png", dpi=300)
        plt.close()
        
    # =========================================================================
    # Graphique 6 : Évolution de la Distance Totale selon la Taille de l'Entrepôt
    # =========================================================================
    if 'n_colonnes' in df.columns and not df['n_colonnes'].isna().all() and not df_resolu.empty:
        plt.figure(figsize=(12, 6))
        
        sns.lineplot(data=df_resolu, x='n_colonnes', y='distance_totale', hue='algorithme', marker='s', linewidth=2)
        plt.title("Impact de la Taille de l'Entrepôt sur la Distance (Scalabilité)", fontsize=14, pad=15)
        plt.ylabel("Distance Totale (Énergie)")
        plt.xlabel("Longueur de l'Entrepôt (Nombre de colonnes)")
        
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend(title='Algorithme', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(dossier_graphes / "06_distance_vs_taille.png", dpi=300)
        plt.close()

    # =========================================================================
    # Graphique 7 : Poids du "Double Handling" dans les mouvements (selon la densité)
    # =========================================================================
    if 'ratio_double_handling' in df.columns and not df_resolu.empty:
        plt.figure(figsize=(12, 6))
        
        # On calcule le ratio moyen par algorithme et par taux de remplissage
        dh_by_fill = df_resolu.groupby(['algorithme', 'taux_remplissage'])['ratio_double_handling'].mean().reset_index()
        
        sns.lineplot(data=dh_by_fill, x='taux_remplissage', y='ratio_double_handling', hue='algorithme', marker='^', linewidth=2)
        plt.title("Impact de la Densité sur les Blocages (Double Profondeur)", fontsize=14, pad=15)
        plt.ylabel("% de mouvements qui sont du Double Handling")
        plt.xlabel("Taux de Remplissage (%)")
        
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend(title='Algorithme', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(dossier_graphes / "07_double_handling_vs_remplissage.png", dpi=300)
        plt.close()
        
    # =========================================================================
    # Graphique 8 : Ratio Énergétique (Distance par Mouvement)
    # =========================================================================
    if 'ratio_energetique' in df.columns and not df_resolu.empty:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='algorithme', y='ratio_energetique', data=df_resolu, hue='algorithme', legend=False, palette="Set3")
        plt.title("Profil Énergétique : Distance parcourue par opération navette", fontsize=14, pad=15)
        plt.ylabel("Distance par mouvement (Ratio)")
        plt.xlabel("Algorithme")
        plt.tight_layout()
        plt.savefig(dossier_graphes / "08_ratio_energetique.png", dpi=300)
        plt.close()

    # =========================================================================
    # Graphique 9 : Front de Pareto (Temps de Calcul vs Énergie/Distance)
    # =========================================================================
    if not df_resolu.empty:
        plt.figure(figsize=(10, 8))
        
        # Moyennes globales par algorithme pour y voir clair
        pareto_data = df_resolu.groupby('algorithme').agg({
            'temps_execution': 'mean',
            'distance_totale': 'mean',
            'type': 'first'
        }).reset_index()
        
        ax = sns.scatterplot(data=pareto_data, x='temps_execution', y='distance_totale', 
                             hue='algorithme', style='type', s=200, palette="deep")
        
        # Ajout des labels à côté des points
        for i, point in pareto_data.iterrows():
            ax.text(point['temps_execution'] * 1.1, point['distance_totale'], point['algorithme'], 
                    horizontalalignment='left', size='medium', color='black', weight='semibold')

        plt.title("Compromis Temps de Calcul / Consommation Énergétique", fontsize=14, pad=15)
        plt.ylabel("Distance Totale Moyenne (Coût Énergétique)")
        plt.xlabel("Temps d'exécution Moyen (s)")
        plt.xscale("log") # Log scale car les métaheuristiques explosent l'axe X
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(dossier_graphes / "09_front_pareto.png", dpi=300)
        plt.close()
    
    print(f"✅ Graphiques générés avec succès dans le dossier : {dossier_graphes}/")
    print("- 01_taux_reussite.png")
    print("- 02_qualite_distance.png")
    print("- 03_efficacite_mouvements.png")
    print("- 04_temps_execution_log.png")
    print("- 05_taux_reussite_vs_remplissage.png")
    print("- 06_distance_vs_taille.png")
    print("- 07_double_handling_vs_remplissage.png")
    print("- 08_ratio_energetique.png")
    print("- 09_front_pareto.png")

if __name__ == "__main__":
    generate_benchmark_graphs()
