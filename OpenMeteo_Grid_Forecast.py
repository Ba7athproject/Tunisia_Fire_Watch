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


def extraire_points_meteo_uniques(gdf_grille: gpd.GeoDataFrame) -> tuple:
    """
    Module 1 : Réduction de la résolution spatiale avec filtre géographique strict.
    Exclut le sud désertique pour diviser la charge réseau, tout en figeant l'index spatial.
    """
    logging.info("Calcul des centroïdes et réduction de la résolution spatiale...")
    
    # Reprojection métrique pour un calcul précis du centroïde, puis retour en GPS
    gdf_proj = gdf_grille.to_crs(epsg=32632)
    centroides = gdf_proj.geometry.centroid.to_crs(epsg=4326)
    
    # Conservation stricte de l'index d'origine pour éviter le désalignement spatial ultérieur
    df_coords = pd.DataFrame({
        'lat_meteo': np.round(centroides.y, 1),
        'lon_meteo': np.round(centroides.x, 1)
    }, index=gdf_grille.index)
    
    # Filtre OSINT : Conservation exclusive du Nord et de la Dorsale
    points_uniques = df_coords[df_coords['lat_meteo'] >= 34.2].drop_duplicates().reset_index(drop=True)
    
    logging.info(f"✔ Stations météorologiques et topographiques cibles : {len(points_uniques)}.")
    
    return points_uniques, df_coords


def recuperer_elevation_batch(points_uniques: pd.DataFrame, batch_size: int = 5) -> pd.DataFrame:
    """
    Module 2A : Interroge l'API Open-Meteo Elevation par ultra-petits lots (5 points).
    """
    url = "https://api.open-meteo.com/v1/elevation"
    altitudes_totales = []

    logging.info(f"Début de l'extraction topographique réelle pour {len(points_uniques)} points (lots de {batch_size})...")

    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)

    for i in range(0, len(points_uniques), batch_size):
        batch = points_uniques.iloc[i:i+batch_size]
        
        lats = ",".join(batch['lat_meteo'].astype(str).tolist())
        lons = ",".join(batch['lon_meteo'].astype(str).tolist())
        
        params = {
            "latitude": lats,
            "longitude": lons
        }
        
        try:
            response = session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Gestion robuste du parsing JSON
            if isinstance(data, dict):
                elevations = data.get('elevation', [])
                if isinstance(elevations, (int, float)):
                    elevations = [elevations]
            elif isinstance(data, list):
                elevations = [item.get('elevation', 250.0) for item in data]
            else:
                elevations = []

            if len(elevations) != len(batch):
                elevations = [250.0] * len(batch)
                
            altitudes_totales.extend(elevations)
            
        except Exception as e:
            logging.warning(f"Erreur d'élévation sur le lot {i} : {e}. Application de 250m par défaut.")
            altitudes_totales.extend([250.0] * len(batch))
            
        time.sleep(0.5)

    points_uniques = points_uniques.copy()
    points_uniques['elevation_m'] = altitudes_totales
    logging.info("✔ Extraction topographique réelle terminée.")
    return points_uniques


def requeter_open_meteo_batch(points_uniques: pd.DataFrame, batch_size: int = 10) -> pd.DataFrame:
    """
    Module 2B : Acquisition Météorologique par mini-lots (10 points).
    """
    url = "https://api.open-meteo.com/v1/forecast"
    resultats = []
    
    logging.info(f"Début de l'ingestion météo par mini-lots de {batch_size}...")
    
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
            logging.warning(f"Alerte réseau sur le lot météo {i} : {e}. Application de valeurs neutres.")
            for idx in range(len(batch)):
                 resultats.append({
                    "lat_meteo": batch.iloc[idx]['lat_meteo'],
                    "lon_meteo": batch.iloc[idx]['lon_meteo'],
                    "t_max": 35.0, "h_mean": 50.0, "wind_max": 15.0, "precip_sum": 0.0
                })
        
        time.sleep(1.0)
        
    return pd.DataFrame(resultats)


def integrer_meteo_au_maillage(chemin_grille: str, chemin_sortie: str):
    """
    Module 3 : Orchestration et propagation des données avec maintien de l'intégrité matricielle.
    """
    if not os.path.exists(chemin_grille):
        logging.error(f"ERREUR CRITIQUE : Fichier de grille introuvable : '{chemin_grille}'.")
        exit(1)
        
    logging.info(f"Chargement du maillage spatial : {chemin_grille}")
    gdf_grille = gpd.read_file(chemin_grille)
    
    # 1. Extraction avec index original verrouillé
    points_uniques, df_coords = extraire_points_meteo_uniques(gdf_grille)
    
    # 2. Récupération des données API
    points_uniques = recuperer_elevation_batch(points_uniques, batch_size=5)
    df_meteo = requeter_open_meteo_batch(points_uniques, batch_size=10)
    
    if df_meteo.empty:
        logging.error("Aucune donnée météo exploitable récupérée. Arrêt.")
        exit(1)
        
    # Fusion des points uniques
    df_complet_uniques = points_uniques.merge(df_meteo, on=['lat_meteo', 'lon_meteo'], how='left')
    
    logging.info("Propagation des données sur la grille fine avec verrouillage spatial...")
    
    # CORRECTION CRITIQUE : reset_index() et set_index() assurent que chaque valeur retourne à la bonne maille
    df_enrichi = df_coords.reset_index().merge(
        df_complet_uniques, 
        on=['lat_meteo', 'lon_meteo'], 
        how='left'
    ).set_index('index')
    
    # Nettoyage des éventuels NaN pour garantir l'exécution de XGBoost
    df_enrichi['t_max'] = df_enrichi['t_max'].fillna(35.0)
    df_enrichi['h_mean'] = df_enrichi['h_mean'].fillna(50.0)
    df_enrichi['wind_max'] = df_enrichi['wind_max'].fillna(15.0)
    df_enrichi['precip_sum'] = df_enrichi['precip_sum'].fillna(0.0)
    df_enrichi['elevation_m'] = df_enrichi['elevation_m'].fillna(250.0)
    
    # Assignation 1:1 vers la grille
    gdf_grille['t_max'] = df_enrichi['t_max']
    gdf_grille['h_mean'] = df_enrichi['h_mean']
    gdf_grille['wind_max'] = df_enrichi['wind_max']
    gdf_grille['precip_sum'] = df_enrichi['precip_sum']
    gdf_grille['elevation_m'] = df_enrichi['elevation_m']
    
    logging.info(f"Sauvegarde de la grille prédictive enrichie dans : {chemin_sortie}")
    gdf_grille.to_file(chemin_sortie, driver="GeoJSON")
    logging.info("✔ Pipeline météo et topographique exécuté avec succès.")


if __name__ == "__main__":
    integrer_meteo_au_maillage(FICHIER_GRILLE, FICHIER_SORTIE)