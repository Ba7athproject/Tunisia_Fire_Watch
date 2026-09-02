import os
import io
import zlib
import time
import socket
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
import requests
import pystac_client
import planetary_computer
import odc.stac
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

import urllib3.util.connection as urllib3_cn
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -----------------------------------------------------------------------------
# 1. Patch Réseau & Configuration Environnement
# -----------------------------------------------------------------------------
# Forcer l'utilisation exclusive d'IPv4 pour contourner l'incompatibilité 
# IPv6 entre les runners GitHub Actions (Azure) et les serveurs de la NASA.
def allowed_gai_family():
    return socket.AF_INET

urllib3_cn.allowed_gai_family = allowed_gai_family

load_dotenv()

# Configuration de la journalisation double (Fichier local + Console GitHub Actions)
logging.basicConfig(
    filename='fire_watch_automation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

# Chemins et clés d'accès
MODELE_PATH = "modele_xgboost_tunisia_fire.joblib"
SUPABASE_DB_URI = os.getenv("SUPABASE_DB_URI")
FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY")
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Cache mémoire pour mutualiser les requêtes météo des points géographiquement proches
METEO_CACHE = {}


# -----------------------------------------------------------------------------
# 2. Fonctions d'Acquisition de Données Satellitaires, Météo & Croisement Spatial
# -----------------------------------------------------------------------------
def fetch_recent_firms_data():
    """
    Interroge l'API NASA FIRMS via une emprise spatiale (Bounding Box) couvrant la Tunisie.
    Utilise le capteur VIIRS SNPP NRT (résolution 375m) sur les dernières 24 heures.
    """
    logging.info("Interrogation de l'API NASA FIRMS via Bounding Box spatiale...")
    
    if not FIRMS_MAP_KEY:
        logging.error("Clé API NASA FIRMS manquante dans les variables d'environnement.")
        return gpd.GeoDataFrame()
        
    # Bounding box Tunisie : Ouest, Sud, Est, Nord
    bbox_tunisie = "7.5,30.2,11.6,37.6"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/VIIRS_SNPP_NRT/{bbox_tunisie}/1"
    
    # Session avec réessais automatiques en cas de latence serveur
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Accept": "text/csv,application/csv,text/plain"
    }
    
    try:
        response = session.get(url, headers=headers, timeout=45)
        response.raise_for_status() 
        
        df_firms = pd.read_csv(io.StringIO(response.text))
        
        if df_firms.empty:
            logging.info("API NASA : Aucun foyer thermique détecté dans les dernières 24h sur l'emprise tunisienne.")
            return gpd.GeoDataFrame()
            
        # Génération d'un identifiant entier unique et reproductible basé sur les coordonnées GPS
        df_firms['cell_id'] = df_firms.apply(
            lambda row: zlib.crc32(f"{row['latitude']:.4f}_{row['longitude']:.4f}".encode()), 
            axis=1
        )
        
        if 'confidence' in df_firms.columns:
            df_firms['confidence'] = df_firms['confidence'].astype(str)
        else:
            df_firms['confidence'] = 'u'
            
        gdf = gpd.GeoDataFrame(
            df_firms, 
            geometry=gpd.points_from_xy(df_firms.longitude, df_firms.latitude),
            crs="EPSG:4326"
        )
        
        logging.info(f"✔ API NASA : {len(gdf)} foyers thermiques bruts récupérés.")
        return gdf
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur de connexion réseau à l'API NASA FIRMS : {e}")
        return gpd.GeoDataFrame()
    except Exception as e:
        logging.error(f"Erreur inattendue lors du traitement des données FIRMS : {e}")
        return gpd.GeoDataFrame()


def get_open_meteo_forecast(lat, lon):
    """
    Récupère les paramètres météorologiques avec cache spatial via WeatherAPI.
    Utilise une clé API pour éviter le bannissement IP des serveurs GitHub Actions.
    """
    cache_key = (round(lat, 1), round(lon, 1))
    if cache_key in METEO_CACHE:
        return METEO_CACHE[cache_key]

    time.sleep(1)
    
    if not WEATHER_API_KEY:
        logging.error("Clé WEATHER_API_KEY manquante. Valeurs par défaut appliquées.")
        fallback = {'t_max': 38.0, 'h_mean': 45.0, 'wind_max': 15.0, 'precip_sum': 0.0}
        METEO_CACHE[cache_key] = fallback
        return fallback

    # Appel vers WeatherAPI (forecast sur 1 jour)
    url = f"https://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={lat},{lon}&days=1"
    
    session = requests.Session()
    retry_strategy = Retry(total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    headers = {
        "User-Agent": "TunisiaFireWatch-OSINT/1.0 (ba7ath investigative project)"
    }

    try:
        response = session.get(url, headers=headers, timeout=12)
        
        if response.status_code == 200:
            data = response.json()
            # Extraction des données journalières depuis la réponse JSON
            day = data['forecast']['forecastday'][0]['day']
            
            result = {
                't_max': float(day['maxtemp_c']),
                'h_mean': float(day['avghumidity']),
                'wind_max': float(day['maxwind_kph']),
                'precip_sum': float(day['totalprecip_mm'])
            }
            METEO_CACHE[cache_key] = result
            return result
        else:
            logging.warning(f"API Météo code {response.status_code}. Valeurs de repli appliquées.")
            
    except Exception as e:
        logging.warning(f"Alerte API Météo ({e}). Valeurs de repli appliquées.")
        
    # Fallback de sécurité
    fallback = {'t_max': 38.0, 'h_mean': 45.0, 'wind_max': 15.0, 'precip_sum': 0.0}
    METEO_CACHE[cache_key] = fallback
    return fallback


def get_sentinel_indices(catalog, bbox):
    """
    Extrait la médiane des indices de végétation (NDVI) et d'humidité (NDWI) 
    à partir des images Sentinel-2 L2A via Planetary Computer.
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
            
        cube = odc.stac.load(
            [items[0]], 
            bands=["B04", "B08", "B11"], 
            bbox=bbox, 
            resolution=20, 
            chunks={}
        ).astype(float)
        
        ndvi = float(((cube.B08 - cube.B04) / (cube.B08 + cube.B04)).mean().compute().values)
        ndwi = float(((cube.B08 - cube.B11) / (cube.B08 + cube.B11)).mean().compute().values)
        
        return round(ndvi if not np.isnan(ndvi) else 0.25, 3), round(ndwi if not np.isnan(ndwi) else 0.05, 3)
    except Exception:
        return 0.25, 0.05


def enrichir_contexte_spatial(engine):
    """
    Exécute une mise à jour spatiale dans PostGIS pour lier 
    les foyers actifs aux entités administratives (gouvernorats) via ST_Intersects.
    """
    logging.info("Exécution du croisement spatial PostGIS (Gouvernorats)...")
    
    # Requête SQL spatiale s'appuyant sur shapeName du GeoJSON importé
    query_spatial_join = """
    UPDATE foyers_actifs f
    SET gouvernorat = g."shapeName"
    FROM tunisia_gouvernorats g
    WHERE ST_Intersects(f.geom, g.geometry)
    AND (f.gouvernorat IS NULL OR f.gouvernorat = '');
    """
    
    try:
        with engine.begin() as conn:
            conn.execute(text(query_spatial_join))
        logging.info("✔ Croisement spatial PostGIS effectué avec succès.")
    except Exception as e:
        logging.warning(f"Alerte lors du croisement spatial : {e}")


# -----------------------------------------------------------------------------
# 3. Pipeline Principal d'Exécution & Ingestion PostGIS
# -----------------------------------------------------------------------------
def run_automated_pipeline():
    logging.info("==================================================")
    logging.info("--- Démarrage de la synchronisation automatisée ---")
    
    config_valide = True
    
    if not SUPABASE_DB_URI:
        logging.error("❌ ERREUR CRITIQUE : La variable SUPABASE_DB_URI est vide.")
        config_valide = False
    else:
        logging.info("✔ SUPABASE_DB_URI détectée avec succès.")

    if not FIRMS_MAP_KEY:
        logging.error("❌ ERREUR CRITIQUE : La variable FIRMS_MAP_KEY est vide.")
        config_valide = False
    else:
        logging.info("✔ FIRMS_MAP_KEY détectée avec succès.")

    if not os.path.exists(MODELE_PATH):
        logging.error(f"❌ ERREUR CRITIQUE : Le fichier modèle '{MODELE_PATH}' est introuvable.")
        config_valide = False
    else:
        logging.info(f"✔ Modèle trouvé à l'emplacement : {MODELE_PATH}")

    if not config_valide:
        logging.error("Arrêt du script : Prérequis de configuration non satisfaits.")
        return

    engine = create_engine(SUPABASE_DB_URI)
    model = joblib.load(MODELE_PATH)
    catalog = pystac_client.Client.open(STAC_API_URL, modifier=planetary_computer.sign_inplace)

    gdf_firms = fetch_recent_firms_data()
    if gdf_firms.empty:
        logging.info("Aucune nouvelle anomalie détectée via NASA FIRMS. Fin du traitement.")
        return

    records = []
    for _, row in gdf_firms.iterrows():
        lat, lon = row.geometry.y, row.geometry.x
        bbox = list(row.geometry.buffer(0.001).bounds)
        cell_id = int(row['cell_id'])
        acq_date = str(row['acq_date']).strip()
        
        # Déduplication au niveau de la base Supabase / PostGIS
        with engine.connect() as conn:
            query_check = text("SELECT COUNT(*) FROM foyers_actifs WHERE cell_id = :cid AND acq_date = :adate")
            res = conn.execute(query_check, {"cid": cell_id, "adate": acq_date}).scalar()
            if res > 0:
                logging.info(f"Foyer GPS ({lat:.4f}, {lon:.4f}) du {acq_date} déjà présent. Ignoré.")
                continue

        meteo = get_open_meteo_forecast(lat, lon)
        ndvi, ndwi = get_sentinel_indices(catalog, bbox)
        frp_value = float(row.get('frp', 0.0))

        # Préparation des variables explicatives pour le modèle XGBoost
        features = pd.DataFrame([{
            't_max': meteo['t_max'],
            'h_mean': meteo['h_mean'],
            'wind_max': meteo['wind_max'],
            'precip_sum': meteo['precip_sum'],
            'ndvi': ndvi,
            'ndwi': ndwi
        }])
        # Le modèle ne s'attend plus à recevoir le FRP
        features = features[['t_max', 'h_mean', 'wind_max', 'precip_sum', 'ndvi', 'ndwi']]

        # Calcul de la probabilité de risque d'incendie (0 à 100%)
        risque_prob = float(model.predict_proba(features)[:, 1][0]) * 100

        records.append({
            'cell_id': cell_id,
            'acq_date': pd.to_datetime(acq_date),
            'latitude': lat,
            'longitude': lon,
            'frp': frp_value,
            't_max': float(meteo['t_max']),
            'h_mean': float(meteo['h_mean']),
            'wind_max': float(meteo['wind_max']),
            'ndvi': float(ndvi),
            'ndwi': float(ndwi),
            'risque_prob': round(risque_prob, 1),
            'confidence': str(row['confidence']),
            'geom': row.geometry
        })

    if records:
        # Création explicite du GeoDataFrame en liant la colonne 'geom'
        gdf_resultat = gpd.GeoDataFrame(records, geometry='geom', crs="EPSG:4326")
        
        try:
            gdf_resultat.to_postgis('foyers_actifs', engine, if_exists='append', index=False)
            logging.info(f"Succès : {len(records)} nouveaux foyers insérés dans Supabase.")
            
            # --- Lancement du croisement spatial PostGIS ---
            enrichir_contexte_spatial(engine)
            
        except Exception as e:
            logging.error(f"Erreur d'insertion PostGIS : {e}")
    else:
        logging.info("Aucun enregistrement inédit à insérer après filtrage des doublons.")
    
    logging.info("--- Fin du cycle d'automatisation ---")


if __name__ == "__main__":
    run_automated_pipeline()