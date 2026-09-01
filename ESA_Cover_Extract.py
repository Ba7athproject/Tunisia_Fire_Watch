import os
import logging
import geopandas as gpd
import pandas as pd
import pystac_client
import planetary_computer
import rioxarray
from datetime import datetime
from shapely.geometry import box

# -----------------------------------------------------------------------------
# Configuration OSINT & STAC
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DOSSIER_PROJET = r"C:\Ba7ath_project\Tunisia-fire-detection"
FICHIER_ENTREE = os.path.join(DOSSIER_PROJET, "dataset_ml_impute.geojson")
FICHIER_SORTIE = os.path.join(DOSSIER_PROJET, "dataset_ml_avec_sol.geojson")

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

def extraire_occupation_sol(geometry):
    """
    Extrait la classe ESA WorldCover en appliquant un buffer spatial 
    pour garantir une emprise raster valide.
    """
    try:
        # Création d'un buffer d'environ 0.0005 degrés (~50m autour du point) 
        # pour éviter l'erreur "only one point" de rioxarray.
        buffer_geom = geometry.buffer(0.0005)
        bbox = buffer_geom.bounds # (minx, miny, maxx, maxy)

        catalog = pystac_client.Client.open(STAC_API_URL, modifier=planetary_computer.sign_inplace)
        
        search = catalog.search(
            collections=["esa-worldcover"],
            bbox=bbox
        )
        items = list(search.items())
        
        if not items:
            return 0 
            
        asset_href = items[0].assets["map"].href
        
        # Lecture avec allow_one_dimensional_raster=True par sécurité
        with rioxarray.open_rasterio(asset_href) as raster:
            raster_clipped = raster.rio.clip_box(*bbox, allow_one_dimensional_raster=True)
            valeurs = raster_clipped.values.flatten()
            valeurs_valides = valeurs[valeurs > 0]
            
            if len(valeurs_valides) > 0:
                classe_dominante = int(pd.Series(valeurs_valides).mode()[0])
            else:
                classe_dominante = 0
                
        return classe_dominante

    except Exception as e:
        # En cas d'erreur ponctuelle sur une tuile, on renvoie une valeur par défaut (0 = Inconnu)
        return 0

def enrichir_dataset_sol(chemin_entree, chemin_sortie):
    """
    Enrichit le dataset de la colonne 'land_cover'.
    """
    if not os.path.exists(chemin_entree):
        logging.error(f"Fichier introuvable : {chemin_entree}")
        return

    logging.info(f"Chargement du dataset : {chemin_entree}")
    gdf = gpd.read_file(chemin_entree)
    
    classes_sol = []
    logging.info(f"Extraction de l'occupation des sols pour {len(gdf)} points de feu...")
    
    for idx, row in gdf.iterrows():
        classe = extraire_occupation_sol(row.geometry)
        classes_sol.append(classe)
        
        if (idx + 1) % 50 == 0:
            logging.info(f"Progression : {idx + 1}/{len(gdf)} points traités.")

    gdf['land_cover'] = classes_sol
    
    logging.info(f"Sauvegarde du dataset enrichi dans : {chemin_sortie}")
    gdf.to_file(chemin_sortie, driver="GeoJSON")
    logging.info("Intégration de l'occupation des sols terminée avec succès.")

if __name__ == "__main__":
    enrichir_dataset_sol(FICHIER_ENTREE, FICHIER_SORTIE)