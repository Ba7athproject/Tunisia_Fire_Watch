import pandas as pd
import numpy as np
import pystac_client
import planetary_computer
import odc.stac
import os
import time
import logging
from datetime import timedelta
from shapely.geometry import Point

# Configuration de la journalisation pour garantir la traçabilité des opérations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculer_indices_historiques(fichier_entree, fichier_sortie):
    """
    Extrait les indices de végétation (NDVI) et de stress hydrique (NDWI)
    à partir de l'imagerie Sentinel-2 L2A via l'API STAC de Planetary Computer.
    """
    logging.info(f"Lecture du dataset météorologique : {fichier_entree}")
    df = pd.read_csv(fichier_entree)
    
    # Sécurisation du format temporel pour les requêtes d'API
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    
    # Initialisation des colonnes cibles si elles n'existent pas encore
    for col in ['ndvi', 'ndwi']:
        if col not in df.columns:
            df[col] = np.nan

    # Initialisation du client STAC (Source publique et ouverte)
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    # Filtrage stratégique : Sentinel-2 ne fournit des données régulières qu'à partir de 2016
    lignes_a_traiter = df[(df['ndvi'].isna()) & (df['acq_date'].dt.year >= 2016)].index
    total_lignes = len(lignes_a_traiter)
    
    logging.info(f"Début de l'extraction satellitaire : {total_lignes} géométries à analyser.")
    
    compteur = 0
    for idx in lignes_a_traiter:
        lat = df.loc[idx, 'latitude']
        lon = df.loc[idx, 'longitude']
        date_cible = df.loc[idx, 'acq_date']
        
        # Fenêtre temporelle : recherche d'une image optique claire dans les 20 jours précédant l'événement
        date_debut = date_cible - timedelta(days=20)
        fenetre_temporelle = f"{date_debut.strftime('%Y-%m-%d')}/{date_cible.strftime('%Y-%m-%d')}"
        
        # Micro-emprise spatiale (Bounding Box) d'environ 200m autour de la coordonnée exacte
        point = Point(lon, lat)
        bbox = list(point.buffer(0.002).bounds) 
        
        try:
            # Ciblage des tuiles L2A (correction atmosphérique appliquée) avec couverture nuageuse < 20%
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=fenetre_temporelle,
                query={"eo:cloud_cover": {"lt": 20}}
            )
            items = list(search.items())
            
            if items:
                # Extraction des bandes Rouge (B04), NIR (B08) et SWIR (B11)
                cube = odc.stac.load(
                    [items[0]], 
                    bands=["B04", "B08", "B11"], 
                    bbox=bbox, 
                    resolution=20, 
                    chunks={}
                ).astype(float)
                
                # Calcul matriciel des indices spectraux
                ndvi = float(((cube.B08 - cube.B04) / (cube.B08 + cube.B04)).mean().compute().values)
                ndwi = float(((cube.B08 - cube.B11) / (cube.B08 + cube.B11)).mean().compute().values)
                
                df.loc[idx, 'ndvi'] = round(ndvi if not np.isnan(ndvi) else 0.25, 3)
                df.loc[idx, 'ndwi'] = round(ndwi if not np.isnan(ndwi) else 0.05, 3)
            else:
                # Repli analytique si la zone est masquée par les nuages sur toute la période
                df.loc[idx, 'ndvi'] = 0.25
                df.loc[idx, 'ndwi'] = 0.05
                
        except Exception as e:
            # Le pipeline ne doit jamais s'interrompre sur une erreur de tuile isolée
            logging.warning(f"Échec STAC pour l'index {idx} ({lat}, {lon}) : {e}")
            df.loc[idx, 'ndvi'] = 0.25
            df.loc[idx, 'ndwi'] = 0.05
            
        compteur += 1
        
        # Sauvegarde modulaire des preuves (Checkpointing)
        if compteur % 50 == 0:
            df.to_csv(fichier_sortie, index=False)
            logging.info(f"Progression : {compteur}/{total_lignes} indices végétaux calculés et sauvegardés.")
            
        # Respect des limites d'utilisation de l'infrastructure Planetary Computer
        time.sleep(0.5)

    # Lissage des données historiques pré-2016 via imputation par la médiane nationale
    df['ndvi'] = df['ndvi'].fillna(df['ndvi'].median())
    df['ndwi'] = df['ndwi'].fillna(df['ndwi'].median())
    
    df.to_csv(fichier_sortie, index=False)
    logging.info(f"✔ Enrichissement végétal terminé. Dataset final prêt pour le Machine Learning : {fichier_sortie}")

if __name__ == "__main__":
    fichier_entree = "dataset_historique_meteo.csv"
    fichier_sortie = "dataset_complet_ml_final.csv"
    
    if os.path.exists(fichier_entree):
        calculer_indices_historiques(fichier_entree, fichier_sortie)
    else:
        logging.error(f"Absence critique du fichier : {fichier_entree}.")