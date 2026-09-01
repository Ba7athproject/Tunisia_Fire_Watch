import os
import io
import zlib
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

# -----------------------------------------------------------------------------
# Configuration & Journalisation Sécurisée
# -----------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    filename='fire_watch_automation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(console_handler)

MODELE_PATH = "modele_xgboost_tunisia_fire.joblib"
SUPABASE_DB_URI = os.getenv("SUPABASE_DB_URI")
FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY")
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

def fetch_recent_firms_data():
    """
    Récupère les véritables données d'anomalies thermiques en temps quasi-réel (NRT) 
    via l'API NASA FIRMS en utilisant une emprise spatiale (Bounding Box) pour la Tunisie.
    """
    logging.info("Interrogation de l'API NASA FIRMS via Bounding Box spatiale...")
    
    if not FIRMS_MAP_KEY:
        logging.error("Clé API NASA FIRMS manquante. Vérifiez vos variables d'environnement.")
        return gpd.GeoDataFrame()
        
    # L'endpoint /country/ étant désactivé par la NASA, on utilise /area/
    # Bounding Box de la Tunisie : West,South,East,North (7.5,30.2,11.6,37.6)
    # Source : VIIRS_SNPP_NRT (375m) / Durée : 1 jour
    bbox_tunisie = "7.5,30.2,11.6,37.6"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_MAP_KEY}/VIIRS_SNPP_NRT/{bbox_tunisie}/1"
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status() 
        
        df_firms = pd.read_csv(io.StringIO(response.text))
        
        if df_firms.empty:
            logging.info("API NASA : Aucun foyer thermique détecté dans les dernières 24h sur l'emprise tunisienne.")
            return gpd.GeoDataFrame()
            
        # Génération de l'identifiant anti-doublon
        df_firms['cell_id'] = df_firms.apply(
            lambda row: zlib.crc32(f"{row['latitude']}_{row['longitude']}".encode()), 
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
    except Exception as e:
        logging.warning(f"Alerte API Open-Meteo (Valeurs par défaut appliquées) : {e}")
    return {'t_max': 38.0, 'h_mean': 45.0, 'wind_max': 15.0, 'precip_sum': 0.0}

def get_sentinel_indices(catalog, bbox):
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
        logging.error("Arrêt du script : Échec de la validation des prérequis de configuration.")
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
        
        with engine.connect() as conn:
            query_check = text("SELECT COUNT(*) FROM foyers_actifs WHERE cell_id = :cid AND acq_date = :adate")
            res = conn.execute(query_check, {"cid": int(row['cell_id']), "adate": row['acq_date']}).scalar()
            if res > 0:
                logging.info(f"Foyer GPS ({lat:.4f}, {lon:.4f}) du {row['acq_date']} déjà présent. Ignoré.")
                continue

        meteo = get_open_meteo_forecast(lat, lon)
        ndvi, ndwi = get_sentinel_indices(catalog, bbox)

        # L'API NASA VIIRS renvoie 'frp'. Nous vérifions son existence.
        frp_value = float(row.get('frp', 0.0))

        features = pd.DataFrame([{
            't_max': meteo['t_max'],
            'h_mean': meteo['h_mean'],
            'wind_max': meteo['wind_max'],
            'precip_sum': meteo['precip_sum'],
            'ndvi': ndvi,
            'ndwi': ndwi,
            'frp': frp_value
        }])
        features = features[['t_max', 'h_mean', 'wind_max', 'precip_sum', 'ndvi', 'ndwi', 'frp']]

        risque_prob = float(model.predict_proba(features)[:, 1][0]) * 100

        records.append({
            'cell_id': int(row['cell_id']),
            'acq_date': pd.to_datetime(row['acq_date']),
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
        gdf_resultat = gpd.GeoDataFrame(records, crs="EPSG:4326")
        gdf_resultat = gdf_resultat.rename_geometry('geom')
        try:
            gdf_resultat.to_postgis('foyers_actifs', engine, if_exists='append', index=False)
            logging.info(f"Succès : {len(records)} nouveaux foyers insérés dans Supabase.")
        except Exception as e:
            logging.error(f"Erreur d'insertion PostGIS : {e}")
    else:
        logging.info("Aucun enregistrement inédit à insérer après filtrage des doublons.")
    
    logging.info("--- Fin du cycle d'automatisation ---")

if __name__ == "__main__":
    run_automated_pipeline()