import pandas as pd
import geopandas as gpd
import xgboost as xgb
import rasterio
from rasterio.sample import sample_gen
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extraire_valeurs_raster(points_geometry, chemin_raster):
    """Extrait les valeurs réelles d'un fichier GeoTIFF pour une liste de géométries."""
    coordonnees = [(point.x, point.y) for point in points_geometry]
    valeurs = []
    
    with rasterio.open(chemin_raster) as src:
        # L'échantillonnage (sampling) est ultra-rapide, même pour 200 000 points
        for val in src.sample(coordonnees):
            valeurs.append(val[0])
    return valeurs

def generer_carte_risques_reelle(fichier_grille, raster_ndvi, raster_ndwi, fichier_modele, fichier_sortie):
    logging.info(f"Chargement de la grille prévisionnelle : {fichier_grille}")
    gdf_grille = gpd.read_file(fichier_grille)
    
    # 1. Extraction des DONNÉES RÉELLES de végétation
    logging.info("Croisement spatial avec les images Sentinel-2 réelles...")
    # On suppose ici que tu as généré deux rasters .tif de la Tunisie pour la quinzaine en cours
    try:
        # Les centroïdes des mailles sont utilisés pour extraire la valeur du pixel satellitaire
        # Projection temporaire en UTM Zone 32N (Tunisie) pour un calcul précis en mètres
        gdf_proj = gdf_grille.to_crs(epsg=32632)
        centroides_proj = gdf_proj.geometry.centroid
        centroides = centroides_proj.to_crs(gdf_grille.crs)
        gdf_grille['ndvi'] = extraire_valeurs_raster(centroides, raster_ndvi)
        gdf_grille['ndwi'] = extraire_valeurs_raster(centroides, raster_ndwi)
    except Exception as e:
        logging.error(f"Erreur lors de la lecture des rasters satellitaires : {e}")
        logging.error("Veuillez vous assurer que les fichiers .tif existent et sont dans le même SCR (EPSG:4326).")
        return

    # 2. Chargement du modèle XGBoost entraîné
    modele_xgb = xgb.XGBClassifier()
    modele_xgb.load_model(fichier_modele)
    logging.info("✔ Modèle prédictif chargé.")

    colonnes_predictives = ['t_max', 'h_mean', 'wind_max', 'precip_sum', 'ndvi', 'ndwi']
    X_prediction = gdf_grille[colonnes_predictives].copy().astype(float)

    # 3. Calcul des probabilités
    logging.info("Calcul matriciel des probabilités d'incendie (Vérité terrain)...")
    probabilites = modele_xgb.predict_proba(X_prediction)[:, 1]
    
    gdf_grille['risque_prob'] = (probabilites * 100).round(1)
    
    # 4. Filtrage des zones à risque
    gdf_risques = gdf_grille[gdf_grille['risque_prob'] > 30].copy()
    
    gdf_risques.to_file(fichier_sortie, driver="GeoJSON")
    logging.info(f"✔ Carte des risques réels générée : {fichier_sortie}")

if __name__ == "__main__":
    # Noms des fichiers requis pour la mise en production
    generer_carte_risques_reelle(
        fichier_grille="grille_meteo_previsionnelle_20260901.geojson",
        raster_ndvi="tunisie_ndvi_actuel.tif",  # Fichier réel à télécharger 2 fois par mois
        raster_ndwi="tunisie_ndwi_actuel.tif",  # Fichier réel à télécharger 2 fois par mois
        fichier_modele="modele_xgboost_incendies_tunisie.json",
        fichier_sortie="carte_risques_demain_reel.geojson"
    )