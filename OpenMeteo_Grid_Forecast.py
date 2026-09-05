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

FICHIER_GRILLE = "grille_tunisie_1km.geojson"
FICHIER_SORTIE = f"grille_meteo_previsionnelle_{datetime.now().strftime('%Y%m%d')}.geojson"


def extraire_points_meteo_uniques(gdf_grille: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Module 1 : Réduction de la résolution spatiale avec filtre géographique strict.
    Exclut le sud désertique pour diviser la charge réseau par 3.
    """
    logging.info("Calcul des centroïdes et réduction de la résolution spatiale...")
    
    # Reprojection métrique pour un calcul précis du centroïde, puis retour en GPS
    gdf_proj = gdf_grille.to_crs(epsg=32632)
    centroides = gdf_proj.geometry.centroid.to_crs(epsg=4326)
    
    df_coords = pd.DataFrame({
        'lat_meteo': np.round(centroides.y, 1),
        'lon_meteo': np.round(centroides.x, 1)
    })
    
    # ---------------------------------------------------------
    # FILTRE OSINT : Conservation exclusive du Nord et de la Dorsale
    # On élimine tout ce qui est en dessous de 34.2°N (Gafsa/Sfax/Sud)
    # ---------------------------------------------------------
    taille_avant = len(df_coords)
    df_coords = df_coords[df_coords['lat_meteo'] >= 34.2].copy()
    taille_apres = len(df_coords)
    
    logging.info(f"Filtre géographique appliqué : {taille_avant - taille_apres} points sahariens ignorés.")
    
    points_uniques = df_coords.drop_duplicates().reset_index(drop=True)
    logging.info(f"✔ Stations météorologiques cibles à interroger : {len(points_uniques)}.")
    
    return points_uniques, df_coords


def requeter_open_meteo_batch(points_uniques: pd.DataFrame, batch_size=10) -> pd.DataFrame:
    """
    Module 2 : Acquisition Météorologique par mini-lots (batch_size=10) 
    pour éliminer tout risque de Timeout sur le serveur distant.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    resultats = []
    
    logging.info(f"Début de l'ingestion API par mini-lots de {batch_size}...")
    
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=3,
        status_forcelist=[429, 500, 502, 503, 504],
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
            "forecast_days": 1
        }
        
        try:
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, dict) and "latitude" in data:
                data = [data]
            
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
            logging.warning(f"Alerte réseau sur le lot {i} : {e}. Application de valeurs neutres.")
            for idx in range(len(batch)):
                 resultats.append({
                    "lat_meteo": batch.iloc[idx]['lat_meteo'],
                    "lon_meteo": batch.iloc[idx]['lon_meteo'],
                    "t_max": 35.0, "h_mean": 50.0, "wind_max": 15.0, "precip_sum": 0.0
                })
        
        # Pause de courtoisie (rate-limiting)
        time.sleep(1.0)
        
    return pd.DataFrame(resultats)


def integrer_meteo_au_maillage(chemin_grille: str, chemin_sortie: str):
    """
    Module 3 : Orchestration et propagation des données sur la grille complète.
    """
    if not os.path.exists(chemin_grille):
        logging.error(f"ERREUR CRITIQUE : Fichier de grille introuvable : '{chemin_grille}'.")
        exit(1)
        
    logging.info(f"Chargement du maillage spatial : {chemin_grille}")
    gdf_grille = gpd.read_file(chemin_grille)
    
    points_uniques, df_mapping = extraire_points_meteo_uniques(gdf_grille)
    df_meteo = requeter_open_meteo_batch(points_uniques, batch_size=10)
    
    if df_meteo.empty:
        logging.error("Aucune donnée météo exploitable récupérée. Arrêt.")
        exit(1)
        
    logging.info("Propagation des données météo sur la grille fine (1km)...")
    
    df_meteo.ffill(inplace=True)
    df_enrichi = df_mapping.merge(df_meteo, on=['lat_meteo', 'lon_meteo'], how='left')
    
    # Remplacement des zones exclues du sud par des valeurs par défaut pour éviter les NaN
    df_enrichi['t_max'].fillna(35.0, inplace=True)
    df_enrichi['h_mean'].fillna(50.0, inplace=True)
    df_enrichi['wind_max'].fillna(15.0, inplace=True)
    df_enrichi['precip_sum'].fillna(0.0, inplace=True)
    
    gdf_grille['t_max'] = df_enrichi['t_max']
    gdf_grille['h_mean'] = df_enrichi['h_mean']
    gdf_grille['wind_max'] = df_enrichi['wind_max']
    gdf_grille['precip_sum'] = df_enrichi['precip_sum']
    
    logging.info(f"Sauvegarde de la grille prédictive dans : {chemin_sortie}")
    gdf_grille.to_file(chemin_sortie, driver="GeoJSON")
    logging.info("✔ Pipeline météo prédictif exécuté avec succès.")


if __name__ == "__main__":
    integrer_meteo_au_maillage(FICHIER_GRILLE, FICHIER_SORTIE)