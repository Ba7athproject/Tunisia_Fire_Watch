import os
import time
import requests
import geopandas as gpd
import pandas as pd
import logging
import numpy as np
from datetime import datetime

# -----------------------------------------------------------------------------
# Configuration et Bonnes Pratiques
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DOSSIER_PROJET = r"C:\Ba7ath_project\Tunisia-fire-detection"
FICHIER_GRILLE = os.path.join(DOSSIER_PROJET, "grille_tunisie_1km.geojson")
FICHIER_SORTIE = os.path.join(DOSSIER_PROJET, f"grille_meteo_previsionnelle_{datetime.now().strftime('%Y%m%d')}.geojson")

def extraire_points_meteo_uniques(gdf_grille: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Réduit les 163 000 cellules de 1km à des points météorologiques de ~11km
    pour éviter de saturer l'API Open-Meteo (limite de 10 000 requêtes/jour).
    """
    logging.info("Calcul des centroïdes et réduction de la résolution spatiale...")
    
    # 1. Récupération des centroïdes des cellules en coordonnées GPS (EPSG:4326)
    centroides = gdf_grille.geometry.centroid
    
    # 2. L'astuce Datajournalisme : Arrondir à 1 décimale équivaut à un maillage d'environ 11.1 km
    df_coords = pd.DataFrame({
        'lat_meteo': np.round(centroides.y, 1),
        'lon_meteo': np.round(centroides.x, 1)
    })
    
    # 3. Suppression des doublons pour obtenir uniquement les stations virtuelles uniques
    points_uniques = df_coords.drop_duplicates().reset_index(drop=True)
    logging.info(f"Optimisation réussie : {len(points_uniques)} points météorologiques à requêter au lieu de {len(gdf_grille)}.")
    
    return points_uniques, df_coords

def requeter_open_meteo_batch(points_uniques: pd.DataFrame, batch_size=50) -> pd.DataFrame:
    """
    Interroge l'API de prévision d'Open-Meteo par lots (batches) de coordonnées.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    resultats = []
    
    logging.info(f"Début de l'ingestion API par lots de {batch_size}...")
    
    # Découpage du DataFrame en lots pour respecter la limite de l'URL
    for i in range(0, len(points_uniques), batch_size):
        batch = points_uniques.iloc[i:i+batch_size]
        lats = batch['lat_meteo'].astype(str).tolist()
        lons = batch['lon_meteo'].astype(str).tolist()
        
        # Open-Meteo permet de passer des listes séparées par des virgules
        params = {
            "latitude": ",".join(lats),
            "longitude": ",".join(lons),
            "daily": ["temperature_2m_max", "relative_humidity_2m_mean", "wind_speed_10m_max", "precipitation_sum"],
            "timezone": "Africa/Tunis",
            "forecast_days": 1 # 1 = Données prévisionnelles d'aujourd'hui
        }
        
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            # Gestion de la réponse : un seul point renvoie un dict, plusieurs points renvoient une liste
            if isinstance(data, dict) and "latitude" in data:
                data = [data]
            
            # Extraction structurée et sécurisée (utilisation de .get pour éviter les KeyError)
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
            logging.error(f"Erreur réseau sur le lot {i}-{i+batch_size} : {e}")
        
        # Tempo strict pour respecter le rate-limiting de l'API gratuite
        time.sleep(1)
        
    return pd.DataFrame(resultats)

def integrer_meteo_au_maillage(chemin_grille: str, chemin_sortie: str):
    """
    Fonction orchestratrice : charge la grille, extrait la météo, et propage 
    les résultats à la géométrie complète.
    """
    if not os.path.exists(chemin_grille):
        logging.error(f"Fichier de grille introuvable : {chemin_grille}")
        return
        
    logging.info(f"Chargement du maillage spatial : {chemin_grille}")
    gdf_grille = gpd.read_file(chemin_grille)
    
    # 1. Extraction des points de requête uniques et du masque de mapping
    points_uniques, df_mapping = extraire_points_meteo_uniques(gdf_grille)
    
    # 2. Requête API Open-Meteo
    df_meteo = requeter_open_meteo_batch(points_uniques)
    
    if df_meteo.empty:
        logging.error("Aucune donnée météo récupérée. Arrêt du processus.")
        return
        
    # 3. Jointure (Merge) des données météo sur le mapping original (163k lignes)
    logging.info("Propagation des données météo sur la grille fine (1km)...")
    df_enrichi = df_mapping.merge(df_meteo, on=['lat_meteo', 'lon_meteo'], how='left')
    
    # 4. Intégration stricte dans le GeoDataFrame original
    gdf_grille['t_max'] = df_enrichi['t_max']
    gdf_grille['h_mean'] = df_enrichi['h_mean']
    gdf_grille['wind_max'] = df_enrichi['wind_max']
    gdf_grille['precip_sum'] = df_enrichi['precip_sum']
    
    # 5. Sauvegarde
    logging.info(f"Sauvegarde de la grille prédictive dans : {chemin_sortie}")
    gdf_grille.to_file(chemin_sortie, driver="GeoJSON")
    logging.info("Pipeline météo prédictif exécuté avec succès.")

# -----------------------------------------------------------------------------
# Point d'entrée
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    integrer_meteo_au_maillage(FICHIER_GRILLE, FICHIER_SORTIE)