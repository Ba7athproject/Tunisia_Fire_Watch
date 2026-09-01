import os
import requests
import geopandas as gpd
import pandas as pd
import logging
import time
from datetime import datetime

# -----------------------------------------------------------------------------
# Configuration et Bonnes Pratiques
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chemins du projet
DOSSIER_PROJET = r"C:\Ba7ath_project\Tunisia-fire-detection"
# Utilisation du dernier fichier généré (à adapter selon le jour d'exécution)
FICHIER_ANOMALIES = os.path.join(DOSSIER_PROJET, f"anomalies_thermiques_{datetime.now().strftime('%Y%m%d')}.geojson")
FICHIER_SORTIE = os.path.join(DOSSIER_PROJET, f"anomalies_enrichies_meteo_{datetime.now().strftime('%Y%m%d')}.geojson")

def requeter_open_meteo(lat: float, lon: float, date_acquisition: str) -> dict:
    """
    Interroge l'API historique d'Open-Meteo (ERA5) pour récupérer la météo 
    au jour exact de l'anomalie thermique.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # CORRECTION : Nettoyage strict de la date au format YYYY-MM-DD requis par Open-Meteo.
    # str() garantit le type, et [:10] coupe tout ce qui dépasse (ex: " 00:00:00").
    date_propre = str(date_acquisition)[:10]

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_propre,
        "end_date": date_propre,
        "daily": ["temperature_2m_max", "relative_humidity_2m_mean", "wind_speed_10m_max", "precipitation_sum"],
        "timezone": "Africa/Tunis"
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Extraction sécurisée des données journalières
        daily = data.get("daily", {})
        return {
            "t_max": daily.get("temperature_2m_max", [None])[0],
            "h_mean": daily.get("relative_humidity_2m_mean", [None])[0],
            "wind_max": daily.get("wind_speed_10m_max", [None])[0],
            "precip_sum": daily.get("precipitation_sum", [None])[0]
        }
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur API Open-Meteo pour lat:{lat}/lon:{lon} : {e}")
        return {"t_max": None, "h_mean": None, "wind_max": None, "precip_sum": None}


def enrichir_anomalies_avec_meteo(chemin_entree: str, chemin_sortie: str):
    """
    Charge le GeoJSON des anomalies, extrait la géométrie, requête la météo 
    pour chaque point, et sauvegarde un nouveau GeoJSON enrichi.
    """
    if not os.path.exists(chemin_entree):
        logging.error(f"Fichier introuvable : {chemin_entree}")
        return

    logging.info(f"Chargement des anomalies depuis : {chemin_entree}")
    gdf = gpd.read_file(chemin_entree)
    
    if gdf.empty:
        logging.warning("Le GeoDataFrame est vide. Aucun enrichissement possible.")
        return

    # Initialisation des nouvelles colonnes
    nouvelles_colonnes = {"t_max": [], "h_mean": [], "wind_max": [], "precip_sum": []}
    
    logging.info(f"Début de l'enrichissement météorologique pour {len(gdf)} anomalies...")
    
    # Itération sur chaque anomalie pour récupérer la météo
    for index, row in gdf.iterrows():
        # Extraction des coordonnées à partir de la géométrie (Point)
        lon, lat = row.geometry.x, row.geometry.y
        # FIRMS fournit la date sous la colonne 'acq_date'
        date_acq = row['acq_date'] 
        
        # Pause de 0.2s pour respecter les limites de requêtes (Rate Limiting) de l'API gratuite
        time.sleep(0.2)
        
        meteo = requeter_open_meteo(lat, lon, date_acq)
        
        nouvelles_colonnes["t_max"].append(meteo["t_max"])
        nouvelles_colonnes["h_mean"].append(meteo["h_mean"])
        nouvelles_colonnes["wind_max"].append(meteo["wind_max"])
        nouvelles_colonnes["precip_sum"].append(meteo["precip_sum"])
        
        if (index + 1) % 50 == 0:
            logging.info(f"Progression : {index + 1}/{len(gdf)} anomalies traitées.")

    # Assignation des nouvelles données au GeoDataFrame
    for col_name, valeurs in nouvelles_colonnes.items():
        gdf[col_name] = valeurs

    # Sauvegarde du résultat
    logging.info(f"Sauvegarde des données enrichies dans : {chemin_sortie}")
    gdf.to_file(chemin_sortie, driver="GeoJSON")
    logging.info("Enrichissement terminé avec succès.")

# -----------------------------------------------------------------------------
# Point d'entrée
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    enrichir_anomalies_avec_meteo(FICHIER_ANOMALIES, FICHIER_SORTIE)