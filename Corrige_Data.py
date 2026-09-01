import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.impute import KNNImputer

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DOSSIER_PROJET = r"C:\Ba7ath_project\Tunisia-fire-detection"
FICHIER_ENTREE = os.path.join(DOSSIER_PROJET, "dataset_complet_ml_20260901.geojson")
FICHIER_SORTIE = os.path.join(DOSSIER_PROJET, "dataset_ml_impute.geojson")

def imputer_donnees_spatiales(chemin_entree: str, chemin_sortie: str):
    """
    Identifie les valeurs aberrantes (0.0 pour la végétation) et manquantes (NaN pour la météo),
    puis les impute en utilisant l'algorithme K-Nearest Neighbors (KNN).
    """
    if not os.path.exists(chemin_entree):
        logging.error(f"Le fichier d'entrée est introuvable : {chemin_entree}")
        return

    try:
        logging.info(f"Chargement du dataset : {chemin_entree}")
        gdf = gpd.read_file(chemin_entree)
        
        # 1. Traitement préalable : Remplacement des 0.0 par des vrais NaN (Not a Number)
        # L'algorithme KNN ignore les NaN mais traite les 0.0 comme des valeurs réelles. 
        # Il faut donc forcer le masquage pour les lignes où Planetary Computer a échoué.
        masque_veg = (gdf['ndvi'] == 0.0) & (gdf['ndwi'] == 0.0)
        gdf.loc[masque_veg, ['ndvi', 'ndwi']] = np.nan
        logging.info(f"{masque_veg.sum()} valeurs de végétation nulles masquées (converties en NaN).")
        
        masque_meteo = gdf['t_max'].isnull()
        logging.info(f"{masque_meteo.sum()} ligne(s) météo manquante(s) détectée(s).")

        # 2. Préparation de la matrice d'imputation
        # On inclut les coordonnées GPS pour forcer le KNN à chercher des voisins géographiquement proches
        gdf['lon'] = gdf.geometry.x
        gdf['lat'] = gdf.geometry.y
        
        colonnes_ml = ['lon', 'lat', 't_max', 'h_mean', 'wind_max', 'precip_sum', 'ndvi', 'ndwi']
        matrice_a_imputer = gdf[colonnes_ml].copy()

        # 3. Application de l'imputateur KNN
        # On utilise k=3 (les 3 feux les plus proches) pondérés par la distance
        logging.info("Exécution de l'algorithme KNNImputer...")
        imputer = KNNImputer(n_neighbors=3, weights='distance')
        matrice_imputee = imputer.fit_transform(matrice_a_imputer)
        
        # 4. Reconstruction du DataFrame nettoyé
        df_impute = pd.DataFrame(matrice_imputee, columns=colonnes_ml)
        
        # Réintégration des colonnes avec le bon formatage décimal
        gdf['t_max'] = df_impute['t_max'].round(1)
        gdf['h_mean'] = df_impute['h_mean'].round(1)
        gdf['wind_max'] = df_impute['wind_max'].round(1)
        gdf['precip_sum'] = df_impute['precip_sum'].round(1)
        gdf['ndvi'] = df_impute['ndvi'].round(3)
        gdf['ndwi'] = df_impute['ndwi'].round(3)

        # Nettoyage des colonnes temporaires
        gdf = gdf.drop(columns=['lon', 'lat'])

        # 5. Validation et Sauvegarde
        valeurs_nulles_restantes = gdf[['t_max', 'ndvi']].isnull().sum().sum()
        if valeurs_nulles_restantes == 0:
            logging.info("Toutes les données ont été imputées avec succès (0 NaN restant).")
        else:
            logging.warning("Attention, des valeurs nulles persistent après imputation.")

        gdf.to_file(chemin_sortie, driver="GeoJSON")
        logging.info(f"Fichier imputé sauvegardé dans : {chemin_sortie}")

    except Exception as e:
        logging.error(f"Une erreur est survenue lors de l'imputation : {e}")

# -----------------------------------------------------------------------------
# Exécution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    imputer_donnees_spatiales(FICHIER_ENTREE, FICHIER_SORTIE)