import os
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from io import StringIO
import logging
from datetime import datetime
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Configuration et Bonnes Pratiques
# -----------------------------------------------------------------------------
# Configuration basique du logging pour la traçabilité (Standard OSINT / Data Pipeline)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chargement automatique des variables depuis le fichier .env local
load_dotenv()

# Récupération sécurisée de la clé. Si le .env est absent, le script lève une alerte claire.
NASA_FIRMS_KEY = os.getenv("NASA_FIRMS_KEY")
if not NASA_FIRMS_KEY:
    raise ValueError("⚠️ ERREUR CRITIQUE : La clé NASA_FIRMS_KEY est introuvable. Veuillez vérifier votre fichier .env.")

# Bounding Box englobant la Tunisie (Min_Lon, Min_Lat, Max_Lon, Max_Lat)
BBOX_TUNISIE = "7.5,30.2,11.6,37.5"
DAYS_RANGE = 5 # Récupérer les 5 derniers jours (limite maximale autorisée par l'API Area FIRMS : [1..5])
SOURCE = "VIIRS_SNPP_NRT" # Capteur VIIRS (375m de résolution, idéal pour détection précoce)

# Chemins du projet
DOSSIER_PROJET = r"C:\Ba7ath_project\Tunisia-fire-detection"
FICHIER_GRILLE = os.path.join(DOSSIER_PROJET, "grille_tunisie_1km.geojson")
FICHIER_SORTIE = os.path.join(DOSSIER_PROJET, f"anomalies_thermiques_{datetime.now().strftime('%Y%m%d')}.geojson")

# -----------------------------------------------------------------------------
# Fonctions Principales
# -----------------------------------------------------------------------------
def telecharger_donnees_firms(api_key: str, source: str, bbox: str, days: int) -> pd.DataFrame:
    """
    Interroge l'API NASA FIRMS et retourne un DataFrame Pandas contenant les anomalies.
    """
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{source}/{bbox}/{days}"
    
    logging.info(f"Interrogation de l'API NASA FIRMS ({source}) sur les {days} derniers jours...")
    
    try:
        # Ajout d'un timeout pour éviter les blocages réseau (Bonne pratique)
        response = requests.get(url, timeout=30)
        
        if not response.ok:
            logging.error(f"Erreur API FIRMS ({response.status_code}) : {response.text.strip()}")
            return pd.DataFrame()
        
        # FIRMS retourne un CSV. On le lit directement en mémoire (StringIO) sans l'écrire sur le disque.
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        
        if df.empty:
            logging.warning("L'API a répondu correctement, mais aucune anomalie n'a été détectée dans cette zone/période.")
        else:
            logging.info(f"{len(df)} anomalies thermiques brutes téléchargées.")
            
        return df

    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur réseau lors de l'accès à l'API FIRMS : {e}")
        return pd.DataFrame() # Retourne un DataFrame vide en cas d'échec
    except Exception as e:
        logging.error(f"Erreur inattendue lors du traitement des données FIRMS : {e}")
        return pd.DataFrame()


def traiter_et_croiser_donnees(df_firms: pd.DataFrame, chemin_grille: str, chemin_sortie: str):
    """
    Nettoie les données FIRMS, les transforme en géométries, et les croise avec la grille nationale.
    """
    if df_firms.empty:
        logging.info("Traitement annulé : aucune donnée FIRMS à traiter.")
        return

    try:
        # 1. Nettoyage des données : Filtrer les fausses alertes (Confiance faible)
        # Pour VIIRS, la confiance est 'l' (low), 'n' (nominal), 'h' (high)
        # On exclut les 'l' pour réduire le bruit (reflets solaires, toits industriels)
        df_filtre = df_firms[df_firms['confidence'].isin(['n', 'h'])].copy()
        logging.info(f"Après filtrage de la confiance (nominal/high) : {len(df_filtre)} anomalies retenues.")

        if df_filtre.empty:
            logging.info("Aucune anomalie de haute confiance à traiter.")
            return

        # 2. Conversion Pandas DataFrame -> GeoPandas GeoDataFrame
        logging.info("Transformation des coordonnées GPS en géométries spatiales...")
        geometrie = [Point(xy) for xy in zip(df_filtre['longitude'], df_filtre['latitude'])]
        gdf_firms = gpd.GeoDataFrame(df_filtre, geometry=geometrie, crs="EPSG:4326")

        # 3. Chargement de la grille nationale générée précédemment
        logging.info(f"Chargement de la grille spatiale depuis : {chemin_grille}")
        if not os.path.exists(chemin_grille):
            logging.error("Fichier de grille introuvable. Veuillez générer la grille d'abord avec le script précédent.")
            return
            
        gdf_grille = gpd.read_file(chemin_grille)

        # 4. Jointure Spatiale (Spatial Join)
        # Objectif : Identifier dans quelle 'cell_id' (kilomètre carré) tombe chaque anomalie thermique
        logging.info("Exécution de la jointure spatiale (Point in Polygon)...")
        # 'predicate=within' signifie : l'anomalie est à l'intérieur du polygone de la cellule
        gdf_anomalies_mappees = gpd.sjoin(gdf_firms, gdf_grille, how="inner", predicate="within")

        # 5. Export des résultats enrichis
        logging.info(f"Sauvegarde des anomalies mappées dans : {chemin_sortie}")
        gdf_anomalies_mappees.to_file(chemin_sortie, driver="GeoJSON")
        
        logging.info("Traitement terminé avec succès. Données prêtes pour le Dashboard.")
        return gdf_anomalies_mappees

    except Exception as e:
        logging.error(f"Erreur lors du traitement spatial : {e}")
        return None

# -----------------------------------------------------------------------------
# Point d'entrée de l'application
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logging.info("--- Démarrage du pipeline FIRMS Tunisia Fire Watch ---")
    
    # Étape 1 : Extraction
    df_brut = telecharger_donnees_firms(NASA_FIRMS_KEY, SOURCE, BBOX_TUNISIE, DAYS_RANGE)
    
    # Étape 2 : Traitement et Géomappage
    if not df_brut.empty:
         traiter_et_croiser_donnees(df_brut, FICHIER_GRILLE, FICHIER_SORTIE)
         
    logging.info("--- Fin du processus ---")