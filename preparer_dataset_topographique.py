import os
import pandas as pd
import logging
from sqlalchemy import create_engine
from dotenv import load_dotenv
import requests

# Import de la fonction d'extraction par lots que nous venons de valider
from extraire_topographie import enrichir_topographie_historique

# Configuration OSINT
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

SUPABASE_DB_URI = os.getenv("SUPABASE_DB_URI")
FICHIER_DATASET_SORTIE = "dataset_incendies_avec_topographie.csv"

def telecharger_historique_supabase(engine) -> pd.DataFrame:
    """Récupère l'intégralité des foyers historiques documentés."""
    logging.info("Connexion à la base de données Supabase...")
    
    # Retrait de precip_sum de la requête SQL pour éviter le crash PostGIS
    query = """
        SELECT cell_id, acq_date, latitude, longitude, frp, 
               t_max, h_mean, wind_max, ndvi, ndwi, confidence 
        FROM foyers_actifs;
    """
    df = pd.read_sql(query, engine)
    
    # Reconstruction de la variable pour maintenir la compatibilité avec XGBoost
    df['precip_sum'] = 0.0
    
    logging.info(f"{len(df)} enregistrements historiques récupérés.")
    return df
    
def generer_dataset_enrichi():
    """Orchestre la récupération, l'enrichissement topographique et la sauvegarde."""
    if not SUPABASE_DB_URI:
        logging.error("La variable d'environnement SUPABASE_DB_URI est introuvable.")
        return

    engine = create_engine(SUPABASE_DB_URI)
    
    # 1. Extraction des données brutes
    df_historique = telecharger_historique_supabase(engine)
    
    if df_historique.empty:
        logging.error("La table des foyers est vide.")
        return

    # 2. Injection de la dimension topographique (API Open-Meteo)
    df_enrichi = enrichir_topographie_historique(df_historique, batch_size=100)
    
    # 3. Sauvegarde de la preuve numérique (fichier plat) pour le ré-entraînement
    df_enrichi.to_csv(FICHIER_DATASET_SORTIE, index=False)
    logging.info(f"✔ Dataset d'apprentissage généré avec succès : {FICHIER_DATASET_SORTIE}")

if __name__ == "__main__":
    generer_dataset_enrichi()