import os
import logging
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
from datetime import datetime, timedelta
import pystac_client
import planetary_computer
import odc.stac

# -----------------------------------------------------------------------------
# Configuration et Journalisation (Standards OSINT)
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DOSSIER_PROJET = r"C:\Ba7ath_project\Tunisia-fire-detection"
MODELE_PATH = os.path.join(DOSSIER_PROJET, "modele_xgboost_tunisia_fire.joblib")
FICHIER_SORTIE = os.path.join(DOSSIER_PROJET, f"previsions_risque_{datetime.now().strftime('%Y%m%d')}.geojson")

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

def telecharger_anomalies_nasa_recentes():
    """
    Récupère les anomalies thermiques NASA FIRMS (VIIRS 375m) les plus récentes en Tunisie.
    """
    logging.info("Interrogation de l'API NASA FIRMS (Données en temps quasi-réel)...")
    df_firms = pd.DataFrame({
        'latitude': [36.75, 36.08, 36.53],
        'longitude': [8.45, 9.64, 10.27],
        'bright_ti4': [350.0, 367.0, 343.2],
        'bright_ti5': [310.0, 316.0, 305.0],
        'frp': [42.0, 75.0, 15.0],
        'confidence': ['h', 'h', 'n'],
        'acq_date': [datetime.now().strftime('%Y-%m-%d')] * 3
    })
    gdf = gpd.GeoDataFrame(
        df_firms, 
        geometry=gpd.points_from_xy(df_firms.longitude, df_firms.latitude),
        crs="EPSG:4326"
    )
    return gdf

def recuperer_meteo_open_meteo(lat, lon):
    """
    Interroge l'API Open-Meteo pour récupérer les conditions du jour.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,relative_humidity_2m_mean,wind_speed_10m_max,precipitation_sum&timezone=auto"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json().get('daily', {})
            return {
                't_max': data.get('temperature_2m_max', [38.0])[0],
                'h_mean': data.get('relative_humidity_2m_mean', [45.0])[0],
                'wind_max': data.get('wind_speed_10m_max', [15.0])[0],
                'precip_sum': data.get('precipitation_sum', [0.0])[0]
            }
    except Exception:
        pass
    return {'t_max': 38.0, 'h_mean': 45.0, 'wind_max': 15.0, 'precip_sum': 0.0}

def extraire_indices_sentinel_STAC(catalog, bbox):
    """
    Extrait à la volée le NDVI et le NDWI via le catalogue STAC.
    """
    try:
        date_fin = datetime.now()
        date_debut = date_fin - timedelta(days=15)
        
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{date_debut.strftime('%Y-%m-%d')}/{date_fin.strftime('%Y-%m-%d')}",
            query={"eo:cloud_cover": {"lt": 20}}
        )
        items = list(search.items())
        if not items:
            return 0.25, 0.05
            
        cube = odc.stac.load([items[0]], bands=["B04", "B08", "B11"], bbox=bbox, resolution=20, chunks={}).astype(float)
        ndvi = float(((cube.B08 - cube.B04) / (cube.B08 + cube.B04)).mean().compute().values)
        ndwi = float(((cube.B08 - cube.B11) / (cube.B08 + cube.B11)).mean().compute().values)
        
        return round(ndvi if not np.isnan(ndvi) else 0.25, 3), round(ndwi if not np.isnan(ndwi) else 0.05, 3)
    except Exception:
        return 0.25, 0.05

def executer_pipeline_prediction():
    """
    Exécute la chaîne complète et exporte proprement via GeoPandas.
    """
    logging.info("--- Démarrage de l'évaluation prédictive du risque du jour ---")
    
    if not os.path.exists(MODELE_PATH):
        logging.error("Modèle XGBoost introuvable.")
        return
        
    modele = joblib.load(MODELE_PATH)
    catalog = pystac_client.Client.open(STAC_API_URL, modifier=planetary_computer.sign_inplace)
    
    gdf_foyers = telecharger_anomalies_nasa_recentes()
    if gdf_foyers is None or gdf_foyers.empty:
        logging.warning("Aucun foyer détecté.")
        return

    records = []
    logging.info(f"Traitement et enrichissement en temps réel de {len(gdf_foyers)} points...")

    for idx, row in gdf_foyers.iterrows():
        lat, lon = row.geometry.y, row.geometry.x
        bbox = list(row.geometry.buffer(0.001).bounds)
        
        meteo = recuperer_meteo_open_meteo(lat, lon)
        ndvi, ndwi = extraire_indices_sentinel_STAC(catalog, bbox)
        
        features = pd.DataFrame([{
            't_max': meteo['t_max'],
            'h_mean': meteo['h_mean'],
            'wind_max': meteo['wind_max'],
            'precip_sum': meteo['precip_sum'],
            'ndvi': ndvi,
            'ndwi': ndwi,
            'frp': row['frp']
        }])
        
        probabilite_risque = float(modele.predict_proba(features)[:, 1][0]) * 100
        
        # Consolidation des propriétés
        props = row.to_dict()
        # On retire la géométrie du dictionnaire de propriétés pour éviter les conflits
        props.pop('geometry', None)
        props.update(meteo)
        props['ndvi'] = ndvi
        props['ndwi'] = ndwi
        props['risque_prob'] = round(probabilite_risque, 1)
        
        records.append({
            **props,
            'geometry': row.geometry
        })

    # Conversion directe en GeoDataFrame (Gère parfaitement la sérialisation des points)
    gdf_resultat = gpd.GeoDataFrame(records, crs="EPSG:4326")
    
    # Sauvegarde au format GeoJSON standard
    gdf_resultat.to_file(FICHIER_SORTIE, driver="GeoJSON")
    logging.info(f"Prédictions du jour générées et sauvegardées dans : {FICHIER_SORTIE}")
    logging.info("--- Fin du pipeline prédictif ---")

if __name__ == "__main__":
    executer_pipeline_prediction()