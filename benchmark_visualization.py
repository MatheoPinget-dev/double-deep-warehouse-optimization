# =============================================================================
# benchmark_visualization.py
# Auteur : Projet SBS/RS
# Description :
#   Script d'analyse comparative de performance exhaustive (14 graphiques).
#   Parcourt automatiquement le dossier 'resultats/' pour agréger les données.
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

def generate_benchmark_graphs(dossier_resultats: str = "resultats"):
    """
    Analyse tous les fichiers CSV de résultats et génère une suite de 14 graphiques.
    Récupère les paramètres de l'instance directement depuis les noms de fichiers.
    """
    path_res = Path(dossier_resultats)
    dfs = []

    if not path_res.exists() or not path_res.is_dir():
        print(f"Erreur : Le dossier {dossier_resultats} est introuvable.")
        return

    # 1. Chargement dynamique de tous les fichiers de résultats
    for fichier in path_res.glob("*.csv"):
        # On ne traite que les fichiers contenant 'resultat' (évite le recapitulatif.csv)
        if "resultat" not in fichier.stem.lower():
            continue

        # Extraction du nom de l'algorithme (ex: resultat_H1.csv -> H1)
        nom_algo = fichier.stem.replace("resultat_", "").replace("resultats_", "").upper()
        
        try:
            df_temp = pd.read_csv(fichier)
            if df_temp.empty:
                continue

            # Harmonisation des noms de colonnes
            df_temp.rename(columns={
                "Fichier": "fichier",
                "Distance_X": "distance_totale",
                "Mouvements": "nb_mouvements",
                "Double_Handling": "nb_double_handling",
                "Temps_s": "temps_execution",
                "Statut": "statut"
            }, inplace=True)

            # Identification de l'algorithme et de son type
            df_temp['algorithme'] = nom_algo
            if any(x in nom_algo for x in ["SA", "GA", "META"]):
                df_temp['type'] = "Métaheuristique"
            else:
                df_temp['type'] = "Heuristique"

            # Indicateur de succès
            df_temp['complet'] = df_temp['statut'].apply(lambda x: 1 if "Succès" in str(x) else 0)
            
            # Gestion des colonnes manquantes pour les anciens fichiers
            if 'nb_double_handling' not in df_temp.columns:
                df_temp['nb_double_handling'] = 0

            dfs.append(df_temp)
            print(f"Chargé : {fichier.name} ({len(df_temp)} lignes)")
        except Exception as e:
            print(f"⚠️ Erreur sur {fichier.name} : {e}")

    if not dfs:
        print("Aucune donnée de résultat trouvée.")
        return

    df = pd.concat(dfs, ignore_index=True)

    # 2. Extraction des métriques d'instance depuis les noms de fichiers CSV
    # Format type : inst_c15_t85_uni_p90_r01.csv
    def extract_params(filename):
        c = re.search(r'_c(\d+)_', filename)
        t = re.search(r'_t(\d+)_', filename)
        p = re.search(r'_p(\d+)_', filename)
        return pd.Series({
            'n_colonnes': int(c.group(1)) if c else None,
            'taux_remplissage': int(t.group(1)) if t else None,
            'prob_maintien': int(p.group(1))/100.0 if p else 0.80 # 0.80 par défaut si non spécifié
        })

    df[['n_colonnes', 'taux_remplissage', 'prob_maintien']] = df['fichier'].apply(extract_params)

    # Calcul de ratios
    df['ratio_energetique'] = df['distance_totale'] / df.apply(lambda x: x['nb_mouvements'] if x['nb_mouvements'] > 0 else 1, axis=1)
    df['ratio_double_handling'] = (df['nb_double_handling'] / df.apply(lambda x: x['nb_mouvements'] if x['nb_mouvements'] > 0 else 1, axis=1)) * 100

    # Filtrage pour les calculs de qualité (on ne garde que les succès)
    df_resolu = df[df['complet'] == 1].copy()

    # Configuration Visuelle
    sns.set_theme(style="whitegrid")
    dossier_img = Path("images")
    dossier_img.mkdir(exist_ok=True)

    print(f"Génération de 14 graphiques vers {dossier_img}/...")

    # --- GRAPH 01 : Taux de réussite global ---
    plt.figure(figsize=(10, 6))
    success = df.groupby('algorithme')['complet'].mean() * 100
    success = success.sort_values(ascending=False).reset_index()
    ax = sns.barplot(x='algorithme', y='complet', data=success, palette="viridis", hue='algorithme', legend=False)
    plt.title("1. Taux de réussite global par algorithme (%)", pad=15)
    plt.ylabel("Taux de succès (%)")
    plt.ylim(0, 110)
    for i in ax.containers: ax.bar_label(i, fmt='%.1f%%', padding=3)
    plt.savefig(dossier_img / "01_taux_reussite.png", dpi=300)
    plt.close()

    # --- GRAPH 02 : Distribution Distance (Boîtes) ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='algorithme', y='distance_totale', data=df_resolu, palette="Set2", hue='algorithme', legend=False)
    plt.title("2. Distribution de la Distance Totale (Succès uniquement)")
    plt.savefig(dossier_img / "02_qualite_distance.png", dpi=300)
    plt.close()

    # --- GRAPH 03 : Mouvements vs Double Handling ---
    plt.figure(figsize=(10, 6))
    df_m = df_resolu.melt(id_vars=['algorithme'], value_vars=['nb_mouvements', 'nb_double_handling'], var_name='Type', value_name='Count')
    sns.barplot(x='algorithme', y='Count', hue='Type', data=df_m, palette="pastel", errorbar=None)
    plt.title("3. Efficacité : Mouvements totaux vs Double Handling")
    plt.savefig(dossier_img / "03_efficacite_mouvements.png", dpi=300)
    plt.close()

    # --- GRAPH 04 : Temps d'exécution (Log) ---
    plt.figure(figsize=(10, 6))
    sns.barplot(x='algorithme', y='temps_execution', data=df, hue='type', errorbar=None)
    plt.title("4. Temps d'exécution moyen (Échelle Logarithmique)")
    plt.yscale("log")
    plt.ylabel("Temps (s)")
    plt.savefig(dossier_img / "04_temps_execution_log.png", dpi=300)
    plt.close()

    # --- GRAPH 05 : Taux réussite vs Remplissage ---
    plt.figure(figsize=(10, 6))
    tr_fill = df.groupby(['algorithme', 'taux_remplissage'])['complet'].mean().reset_index()
    tr_fill['complet'] *= 100
    sns.lineplot(data=tr_fill, x='taux_remplissage', y='complet', hue='algorithme', marker='o', linewidth=2)
    plt.title("5. Taux de réussite vs Taux de remplissage")
    plt.ylim(-5, 105)
    plt.savefig(dossier_img / "05_taux_reussite_vs_remplissage.png", dpi=300)
    plt.close()

    # --- GRAPH 06 : Distance vs Taille (Scalabilité) ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_resolu, x='n_colonnes', y='distance_totale', hue='algorithme', marker='s')
    plt.title("6. Scalabilité : Distance Totale vs Nombre de colonnes")
    plt.savefig(dossier_img / "06_distance_vs_taille.png", dpi=300)
    plt.close()

    # --- GRAPH 07 : Double Handling vs Remplissage ---
    plt.figure(figsize=(10, 6))
    dh_fill = df_resolu.groupby(['algorithme', 'taux_remplissage'])['ratio_double_handling'].mean().reset_index()
    sns.lineplot(data=dh_fill, x='taux_remplissage', y='ratio_double_handling', hue='algorithme', marker='^')
    plt.title("7. Part du Double Handling vs Taux de remplissage")
    plt.savefig(dossier_img / "07_double_handling_vs_remplissage.png", dpi=300)
    plt.close()

    # --- GRAPH 08 : Ratio Énergétique ---
    plt.figure(figsize=(10, 6))
    sns.barplot(x='algorithme', y='ratio_energetique', data=df_resolu, palette="magma", hue='algorithme', legend=False)
    plt.title("8. Ratio Énergétique (Distance moyenne par mouvement)")
    plt.savefig(dossier_img / "08_ratio_energetique.png", dpi=300)
    plt.close()

    # --- GRAPH 09 : Front de Pareto ---
    plt.figure(figsize=(10, 7))
    pareto = df_resolu.groupby('algorithme').agg({'temps_execution': 'mean', 'distance_totale': 'mean', 'type': 'first'}).reset_index()
    ax = sns.scatterplot(data=pareto, x='temps_execution', y='distance_totale', hue='algorithme', style='type', s=200)
    for i, p in pareto.iterrows():
        ax.text(p['temps_execution'] * 1.1, p['distance_totale'], p['algorithme'], weight='bold')
    plt.xscale("log")
    plt.title("9. Front de Pareto : Temps de calcul vs Énergie")
    plt.savefig(dossier_img / "09_front_pareto.png", dpi=300)
    plt.close()

    # --- GRAPH 10 : Taux réussite vs Colonnes (Line) ---
    plt.figure(figsize=(10, 6))
    tr_col = df.groupby(['algorithme', 'n_colonnes'])['complet'].mean().reset_index()
    tr_col['complet'] *= 100
    sns.lineplot(data=tr_col, x='n_colonnes', y='complet', hue='algorithme', marker='o')
    plt.title("10. Taux de réussite vs Nombre de colonnes")
    plt.ylim(-5, 105)
    plt.savefig(dossier_img / "10_reussite_vs_colonnes.png", dpi=300)
    plt.close()

    # --- GRAPH 11 : Taux réussite vs Remplissage (Bar) ---
    plt.figure(figsize=(12, 6))
    sns.barplot(data=tr_fill, x='taux_remplissage', y='complet', hue='algorithme')
    plt.title("11. Comparatif de réussite par taux de remplissage")
    plt.ylabel("Taux de succès (%)")
    plt.savefig(dossier_img / "11_reussite_vs_taux_remplissage.png", dpi=300)
    plt.close()

    # --- GRAPH 12 : Taux réussite vs Probabilité -p ---
    if df['prob_maintien'].nunique() > 1:
        plt.figure(figsize=(10, 6))
        tr_p = df.groupby(['algorithme', 'prob_maintien'])['complet'].mean().reset_index()
        tr_p['complet'] *= 100
        sns.lineplot(data=tr_p, x='prob_maintien', y='complet', hue='algorithme', marker='D')
        plt.title("12. Impact du paramètre -p sur le taux de réussite")
        plt.ylim(-5, 105)
        plt.savefig(dossier_img / "12_reussite_vs_prob_maintien.png", dpi=300)
        plt.close()

    # --- GRAPH 13 : Distance vs Remplissage (Line) ---
    plt.figure(figsize=(10, 6))
    dist_fill = df_resolu.groupby(['algorithme', 'taux_remplissage'])['distance_totale'].mean().reset_index()
    sns.lineplot(data=dist_fill, x='taux_remplissage', y='distance_totale', hue='algorithme', marker='o')
    plt.title("13. Distance moyenne vs Taux de remplissage")
    plt.savefig(dossier_img / "13_distance_vs_taux_remplissage.png", dpi=300)
    plt.close()

    # --- GRAPH 14 : Distance vs Probabilité -p ---
    if df_resolu['prob_maintien'].nunique() > 1:
        plt.figure(figsize=(10, 6))
        dist_p = df_resolu.groupby(['algorithme', 'prob_maintien'])['distance_totale'].mean().reset_index()
        sns.lineplot(data=dist_p, x='prob_maintien', y='distance_totale', hue='algorithme', marker='D')
        plt.title("14. Impact du paramètre -p sur la distance parcourue")
        plt.savefig(dossier_img / "14_distance_vs_prob_maintien.png", dpi=300)
        plt.close()

    print("✅ Analyse terminée. Images disponibles dans le dossier 'images/'.")

if __name__ == "__main__":
    generate_benchmark_graphs()
