import os
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import geopandas as gpd
import pandas as pd
import logging
import numpy as np
from datetime import datetime

# -----------------------------------------------------------------------------
# Configuration et Bonnes Pratiques OSINT
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chemins relatifs stricts pour assurer la portabilité cloud (GitHub Actions, conteneurs Linux)
FICHIER_GRILLE = "grille_tunisie_1km.geojson"
FICHIER_SORTIE = f"grille_meteo_previsionnelle_{datetime.now().strftime('%Y%m%d')}.geojson"


def extraire_points_meteo_uniques(gdf_grille: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Module 1 : Réduction de la résolution spatiale.
    Transforme les cellules de 1km en stations météorologiques virtuelles de ~11km
    pour respecter l'éthique de collecte (rate-limiting) d'Open-Meteo.
    """
    logging.info("Calcul des centroïdes et réduction de la résolution spatiale...")
    
    # Correction géodésique : Reprojection en EPSG:32632 (Métrique, adapté à la Tunisie) 
    # pour un calcul précis du centroïde, puis retour en EPSG:4326 (GPS)
    gdf_proj = gdf_grille.to_crs(epsg=32632)
    centroides = gdf_proj.geometry.centroid.to_crs(epsg=4326)
    
    # Arrondir à 1 décimale équivaut à un maillage d'environ 11.1 km
    df_coords = pd.DataFrame({
        'lat_meteo': np.round(centroides.y, 1),
        'lon_meteo': np.round(centroides.x, 1)
    })
    
    # Suppression des doublons pour obtenir uniquement les stations virtuelles uniques
    points_uniques = df_coords.drop_duplicates().reset_index(drop=True)
    logging.info(f"Optimisation réussie : {len(points_uniques)} points météorologiques à requêter au lieu de {len(gdf_grille)}.")
    
    return points_uniques, df_coords


def requeter_open_meteo_batch(points_uniques: pd.DataFrame, batch_size=25) -> pd.DataFrame:
    """
    Module 2 : Acquisition Météorologique (Scraping éthique et légal).
    Interroge l'API par lots avec un mécanisme de résilience (Retry) en cas de surcharge.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    resultats = []
    
    logging.info(f"Début de l'ingestion API par lots réduits de {batch_size} pour soulager le serveur...")
    
    # Configuration d'une session avec stratégie de réessai automatique (Best Practice)
    session = requests.Session()
    retry_strategy = Retry(
        total=4, # Nombre maximum de tentatives
        backoff_factor=2, # Temps d'attente exponentiel entre les tentatives (2s, 4s, 8s...)
        status_forcelist=[429, 500, 502, 503, 504], # Codes HTTP déclenchant un réessai
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    for i in range(0, len(points_uniques), batch_size):
        batch = points_uniques.iloc[i:i+batch_size]
        lats = batch['lat_meteo'].astype(str).tolist()
        lons = batch['lon_meteo'].astype(str).tolist()
        
        params = {
            "latitude": ",".join(lats),
            "longitude": ",".join(lons),
            "daily": ["temperature_2m_max", "relative_humidity_2m_mean", "wind_speed_10m_max", "precipitation_sum"],
            "timezone": "Africa/Tunis",
            "forecast_days": 1 # Données prévisionnelles pour demain
        }
        
        try:
            # Augmentation du timeout à 45 secondes pour laisser le temps au serveur de calculer
            response = session.get(url, params=params, timeout=45)
            response.raise_for_status()
            data = response.json()
            
            # Standardisation de la réponse si un seul point est renvoyé
            if isinstance(data, dict) and "latitude" in data:
                data = [data]
            
            # Structuration des données
            for idx, res in enumerate(data):
                daily = res.get("daily", {})
                resultats.append({
                    "lat_meteo": batch.iloc[idx]['lat_meteo'],
                    "lon_meteo": batch.iloc[idx]['lon_meteo'],
                    "t_max": daily.get("temperature_2m_max", [None])[0],
                    "h_mean": daily.get("relative_humidity_2m_mean", [None])[0],
                    "wind_max": daily.get("wind_speed_10m_max", [None])[0],
                    "precip_sum": daily.get("precipitation_sum", [None])[0]
                })
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Échec critique sur le lot {i}-{i+batch_size} malgré les réessais : {e}")
            # En cas d'échec total du lot, on remplit avec des valeurs nulles pour ne pas bloquer le dataframe complet
            for idx in range(len(batch)):
                 resultats.append({
                    "lat_meteo": batch.iloc[idx]['lat_meteo'],
                    "lon_meteo": batch.iloc[idx]['lon_meteo'],
                    "t_max": None, "h_mean": None, "wind_max": None, "precip_sum": None
                })
        
        # Pause éthique entre chaque lot pour respecter le rate-limiting
        time.sleep(1.5)
        
    return pd.DataFrame(resultats)


def integrer_meteo_au_maillage(chemin_grille: str, chemin_sortie: str):
    """
    Module 3 : Orchestration.
    Charge la grille, extrait la météo, et propage les résultats à la géométrie complète.
    """
    if not os.path.exists(chemin_grille):
        logging.error(f"ERREUR CRITIQUE : Fichier de grille introuvable à l'emplacement relatif : '{chemin_grille}'.")
        exit(1)
        
    logging.info(f"Chargement du maillage spatial : {chemin_grille}")
    gdf_grille = gpd.read_file(chemin_grille)
    
    points_uniques, df_mapping = extraire_points_meteo_uniques(gdf_grille)
    df_meteo = requeter_open_meteo_batch(points_uniques, batch_size=25)
    
    if df_meteo.empty or df_meteo['t_max'].isna().all():
        logging.error("Aucune donnée météo exploitable récupérée. Arrêt du processus.")
        exit(1)
        
    logging.info("Propagation des données météo sur la grille fine (1km)...")
    
    # Remplacement des valeurs manquantes éventuelles par une méthode d'interpolation (forward fill) pour maintenir l'intégrité
    df_meteo.ffill(inplace=True)
    df_enrichi = df_mapping.merge(df_meteo, on=['lat_meteo', 'lon_meteo'], how='left')
    
    gdf_grille['t_max'] = df_enrichi['t_max']
    gdf_grille['h_mean'] = df_enrichi['h_mean']
    gdf_grille['wind_max'] = df_enrichi['wind_max']
    gdf_grille['precip_sum'] = df_enrichi['precip_sum']
    
    logging.info(f"Sauvegarde de la grille prédictive dans : {chemin_sortie}")
    gdf_grille.to_file(chemin_sortie, driver="GeoJSON")
    logging.info("✔ Pipeline météo prédictif exécuté avec succès.")


if __name__ == "__main__":
    integrer_meteo_au_maillage(FICHIER_GRILLE, FICHIER_SORTIE)