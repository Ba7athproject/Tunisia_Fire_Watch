import os
import logging
import time
import geopandas as gpd
import pandas as pd
from datetime import datetime, timedelta

# Bibliothèques OSINT & Géospatiales
import pystac_client
import planetary_computer
import odc.stac
import numpy as np

# -----------------------------------------------------------------------------
# Configuration OSINT & STAC
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DOSSIER_PROJET = r"C:\Ba7ath_project\Tunisia-fire-detection"
FICHIER_ENTREE = os.path.join(DOSSIER_PROJET, f"anomalies_enrichies_meteo_{datetime.now().strftime('%Y%m%d')}.geojson")
FICHIER_SORTIE = os.path.join(DOSSIER_PROJET, f"dataset_complet_ml_{datetime.now().strftime('%Y%m%d')}.geojson")

# Point d'accès public (Zero-Credential OSINT)
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

def init_stac_client():
    """
    Initialise la connexion au catalogue STAC avec signature SAS automatique.
    Aucune clé API n'est requise.
    """
    logging.info("Connexion au catalogue Planetary Computer (Accès Public)...")
    return pystac_client.Client.open(
        STAC_API_URL, 
        modifier=planetary_computer.sign_inplace
    )

def calculer_indices_vegetation_odc(catalog, bbox, date_cible, max_jours_avant=15):
    """
    Recherche l'image Sentinel-2 pré-incendie, charge les bandes via odc-stac 
    (DataCube) et calcule vectoriellement NDVI et NDWI.
    """
    try:
        # 1. Fenêtre temporelle : chercher l'état du combustible AVANT le feu
        date_fin = datetime.strptime(date_cible, "%Y-%m-%d")
        date_debut = date_fin - timedelta(days=max_jours_avant)
        time_range = f"{date_debut.strftime('%Y-%m-%d')}/{date_fin.strftime('%Y-%m-%d')}"
        
        # 2. Requête STAC (Sentinel-2 L2A, nuages < 20%)
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=time_range,
            query={"eo:cloud_cover": {"lt": 20}}
        )
        
        items = list(search.items())
        
        if not items:
            return None, None
            
        # Priorité à l'image la plus récente juste avant l'anomalie
        meilleur_item = items[0]
        
        # 3. Chargement paresseux avec odc-stac (DataCube)
        # On extrait uniquement B04 (Rouge), B08 (NIR) et B11 (SWIR) sur notre Bounding Box
        cube = odc.stac.load(
            [meilleur_item],
            bands=["B04", "B08", "B11"],
            bbox=bbox,
            resolution=10, # Résolution native Sentinel-2 (10 mètres)
            chunks={"x": 512, "y": 512}, # Optimisation RAM
            groupby="solar_day"
        )
        
        # 4. Conversion en float pour les calculs mathématiques
        cube = cube.astype(float)
        
        # 5. Calcul vectorisé (Xarray) des indices
        ndvi_array = (cube.B08 - cube.B04) / (cube.B08 + cube.B04)
        ndwi_array = (cube.B08 - cube.B11) / (cube.B08 + cube.B11)
        
        # 6. Extraction de la moyenne spatiale et temporelle (pour obtenir un seul float)
        ndvi_mean = float(ndvi_array.mean().compute().values)
        ndwi_mean = float(ndwi_array.mean().compute().values)
        
        # 7. Nettoyage mémoire explicite (Zero-Persistence)
        del cube
        del ndvi_array
        del ndwi_array
        
        # Remplacement des éventuels NaN (si division par zéro) par 0
        ndvi_val = 0.0 if np.isnan(ndvi_mean) else round(ndvi_mean, 3)
        ndwi_val = 0.0 if np.isnan(ndwi_mean) else round(ndwi_mean, 3)
        
        return ndvi_val, ndwi_val

    except Exception as e:
        logging.error(f"Erreur extraction odc-stac pour la bbox {bbox} : {e}")
        return None, None

def integrer_vegetation_aux_donnees(chemin_entree: str, chemin_sortie: str):
    """
    Parcourt les anomalies, extrait le NDVI/NDWI via le STAC et enrichit le fichier.
    """
    if not os.path.exists(chemin_entree):
        logging.error(f"Fichier introuvable : {chemin_entree}")
        return

    logging.info(f"Chargement des données depuis : {chemin_entree}")
    gdf = gpd.read_file(chemin_entree)
    
    # Validation du client STAC
    catalog = init_stac_client()
    
    ndvi_list = []
    ndwi_list = []
    
    logging.info(f"Début de l'extraction satellitaire pour {len(gdf)} cellules...")
    
    for index, row in gdf.iterrows():
        # Date d'acquisition FIRMS
        date_acq = str(row['acq_date'])[:10]
        # Bounding box exacte de la cellule géométrique
        bbox = list(row.geometry.bounds)
        
        ndvi, ndwi = calculer_indices_vegetation_odc(catalog, bbox, date_acq)
        
        ndvi_list.append(ndvi)
        ndwi_list.append(ndwi)
        
        if (index + 1) % 10 == 0:
            logging.info(f"Progression : {index + 1}/{len(gdf)} requêtes STAC traitées.")
            
        # Respect strict du rate-limiting public (environ 40-50 requêtes/min autorisées)
        time.sleep(1.5)
        
    gdf['ndvi'] = ndvi_list
    gdf['ndwi'] = ndwi_list
    
    logging.info(f"Sauvegarde du dataset ML final dans : {chemin_sortie}")
    gdf.to_file(chemin_sortie, driver="GeoJSON")
    logging.info("Végétation intégrée avec succès.")

# -----------------------------------------------------------------------------
# Exécution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Prérequis : pip install pystac-client planetary-computer odc-stac xarray
    integrer_vegetation_aux_donnees(FICHIER_ENTREE, FICHIER_SORTIE)