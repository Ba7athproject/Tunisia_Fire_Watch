import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import logging
import os
import osmnx as ox  # Bibliothèque puissante pour requêter OpenStreetMap (OSINT)

# Configuration de la journalisation pour assurer la traçabilité des exécutions
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generer_grille_pays_entier(nom_pays="Tunisia", taille_cellule_m=1000, 
                               dossier_projet=r"C:\Ba7ath_project\Tunisia-fire-detection", 
                               nom_fichier="grille_tunisie_1km.geojson"):
    """
    Génère une grille spatiale vectorielle couvrant exactement les frontières d'un pays.
    
    Paramètres:
    - nom_pays : str : Nom du pays tel que reconnu par OpenStreetMap (ex: "Tunisia")
    - taille_cellule_m : int : Taille du côté de chaque cellule en mètres (ex: 1000 pour 1km²)
    - dossier_projet : str : Chemin du répertoire cible
    - nom_fichier : str : Nom du fichier de sortie GeoJSON
    
    Retour:
    - gpd.GeoDataFrame contenant la grille nationale finale.
    """
    try:
        # 1. Préparation du répertoire de travail local
        if not os.path.exists(dossier_projet):
            os.makedirs(dossier_projet)
            logging.info(f"Création du répertoire de projet : {dossier_projet}")
        
        chemin_complet_sortie = os.path.join(dossier_projet, nom_fichier)
        
        # 2. Récupération des frontières exactes depuis OpenStreetMap
        logging.info(f"Téléchargement des frontières administratives pour : {nom_pays} via OSM...")
        frontieres_pays_gdf = ox.geocode_to_gdf(nom_pays)
        
        # 3. Reprojection vers UTM Zone 32N (EPSG:32632) pour les calculs métriques
        logging.info("Reprojection en EPSG:32632 (UTM) pour garantir des cellules en mètres...")
        frontieres_metriques = frontieres_pays_gdf.to_crs("EPSG:32632")
        
        # 4. Extraction de la Bounding Box métrique du pays entier
        minx_m, miny_m, maxx_m, maxy_m = frontieres_metriques.total_bounds
        
        # 5. Création des vecteurs de coordonnées X et Y
        logging.info(f"Génération des cellules carrées de {taille_cellule_m}m...")
        x_coords = np.arange(minx_m, maxx_m, taille_cellule_m)
        y_coords = np.arange(miny_m, maxy_m, taille_cellule_m)
        
        # 6. Construction des polygones de la grille brute
        polygones = []
        for x in x_coords:
            for y in y_coords:
                cellule = Polygon([
                    (x, y), 
                    (x + taille_cellule_m, y), 
                    (x + taille_cellule_m, y + taille_cellule_m), 
                    (x, y + taille_cellule_m)
                ])
                polygones.append(cellule)
        
        # 7. Création du GeoDataFrame de la grille brute
        grille_brute_gdf = gpd.GeoDataFrame({'geometry': polygones}, crs="EPSG:32632")
        
        # 8. Nettoyage Spatial (Clipping) : Ne garder que les cellules à l'intérieur des frontières
        logging.info("Découpage de la grille pour correspondre exactement aux frontières tunisiennes (suppression de la mer et des pays voisins)...")
        grille_decoupee_gdf = gpd.clip(grille_brute_gdf, frontieres_metriques)
        
        # Réinitialisation de l'index et création d'un identifiant unique (cell_id) propre
        grille_decoupee_gdf = grille_decoupee_gdf.reset_index(drop=True)
        grille_decoupee_gdf['cell_id'] = grille_decoupee_gdf.index
        
        # 9. Reprojection finale vers WGS84 (EPSG:4326) pour usage web (Streamlit, React, Vercel)
        grille_finale_gdf = grille_decoupee_gdf.to_crs("EPSG:4326")
        
        # 10. Sauvegarde sur le disque
        logging.info(f"Sauvegarde en cours dans : {chemin_complet_sortie}")
        grille_finale_gdf.to_file(chemin_complet_sortie, driver='GeoJSON')
        
        logging.info(f"Succès ! {len(grille_finale_gdf)} cellules terrestres tunisiennes générées.")
        return grille_finale_gdf

    except Exception as e:
        logging.error(f"Erreur critique lors de l'exécution du script : {e}")
        return None

# --- Bloc principal d'exécution ---
if __name__ == "__main__":
    # Exécution de la fonction avec les paramètres spécifiques au projet
    grille_tunisie = generer_grille_pays_entier(
        nom_pays="Tunisia", 
        taille_cellule_m=1000, # Grille de 1km x 1km
        dossier_projet=r"C:\Ba7ath_project\Tunisia-fire-detection",
        nom_fichier="grille_tunisie_1km.geojson"
    )
    
    if grille_tunisie is not None:
        print("\n--- Aperçu des 5 premières cellules de la base de données ---")
        print(grille_tunisie.head())