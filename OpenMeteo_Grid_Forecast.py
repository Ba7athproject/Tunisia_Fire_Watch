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
    logging.info(f"✔ Stations météorologiques et topographiques cibles : {len(points_uniques)}.")
    
    return points_uniques, df_coords


def recuperer_elevation_batch(points_uniques: pd.DataFrame, batch_size: int = 10) -> pd.DataFrame:
    """
    Module 2A : Interroge l'API Open-Meteo Elevation par petits lots (10 points) 
    pour éviter les erreurs de longueur d'URL (HTTP 414) ou les Timeouts.
    """
    url = "https://api.open-meteo.com/v1/elevation"
    altitudes_totales = []

    logging.info(f"Début de l'extraction topographique réelle pour {len(points_uniques)} points...")

    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
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
            response = session.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            # Gestion robuste du format de réponse de l'API Open-Meteo
            if isinstance(data, dict):
                elevations = data.get('elevation', [])
                if isinstance(elevations, (int, float)):
                    elevations = [elevations]
            elif isinstance(data, list):
                elevations = [item.get('elevation', 250.0) for item in data]
            else:
                elevations = []

            # Si le nombre d'altitudes ne correspond pas, on comble par précaution
            if len(elevations) != len(batch):
                elevations = [250.0] * len(batch)
                
            altitudes_totales.extend(elevations)
            
        except Exception as e:
            logging.warning(f"Erreur d'élévation sur le lot {i} : {e}. Application de 250m par défaut.")
            altitudes_totales.extend([250.0] * len(batch))
            
        time.sleep(0.3)

    points_uniques = points_uniques.copy()
    points_uniques['elevation_m'] = altitudes_totales
    logging.info("✔ Extraction topographique réelle terminée.")
    return points_uniques


def requeter_open_meteo_batch(points_uniques: pd.DataFrame, batch_size=10) -> pd.DataFrame:
    """
    Module 2B : Acquisition Météorologique par mini-lots (batch_size=10) 
    pour éliminer tout risque de Timeout sur le serveur distant.
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
    Module 3 : Orchestration et propagation des données météo et topographiques sur la grille complète.
    """
    if not os.path.exists(chemin_grille):
        logging.error(f"ERREUR CRITIQUE : Fichier de grille introuvable : '{chemin_grille}'.")
        exit(1)
        
    logging.info(f"Chargement du maillage spatial : {chemin_grille}")
    gdf_grille = gpd.read_file(chemin_grille)
    
    # 1. Extraction des points uniques et filtrage géographique
    points_uniques, df_mapping = extraire_points_meteo_uniques(gdf_grille)
    
    # 2. Récupération de l'élévation réelle via l'API Open-Meteo Elevation
    points_uniques = recuperer_elevation_batch(points_uniques, batch_size=100)
    
    # 3. Récupération de la météo prévisionnelle
    df_meteo = requeter_open_meteo_batch(points_uniques, batch_size=10)
    
    if df_meteo.empty:
        logging.error("Aucune donnée météo exploitable récupérée. Arrêt.")
        exit(1)
        
    # Fusion des données météo et topographiques sur les points uniques
    df_complet_uniques = points_uniques.merge(df_meteo, on=['lat_meteo', 'lon_meteo'], how='left')
    
    logging.info("Propagation des données météo et topographiques sur la grille fine (1km)...")
    
    df_complet_uniques.ffill(inplace=True)
    df_enrichi = df_mapping.merge(df_complet_uniques, on=['lat_meteo', 'lon_meteo'], how='left')
    
    # Valeurs par défaut de secours pour éviter les NaN
    df_enrichi['t_max'] = df_enrichi['t_max'].fillna(35.0)
    df_enrichi['h_mean'] = df_enrichi['h_mean'].fillna(50.0)
    df_enrichi['wind_max'] = df_enrichi['wind_max'].fillna(15.0)
    df_enrichi['precip_sum'] = df_enrichi['precip_sum'].fillna(0.0)
    df_enrichi['elevation_m'] = df_enrichi['elevation_m'].fillna(250.0)
    
    # Assignation finale au GeoDataFrame de la grille
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