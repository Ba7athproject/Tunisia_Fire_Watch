import logging
import os
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb
import glob

# Configuration de la journalisation pour la traçabilité de l'investigation
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extraire_valeurs_raster(points_geometry, chemin_raster: str) -> list:
    """Extrait les valeurs réelles d'un raster GeoTIFF (EPSG:4326)
    pour une liste de coordonnées géographiques (Points).
    """
    if not os.path.exists(chemin_raster):
        raise FileNotFoundError(f"Le fichier raster est introuvable : {chemin_raster}")

    coordonnees = [(point.x, point.y) for point in points_geometry]
    valeurs = []

    with rasterio.open(chemin_raster) as src:
        # Échantillonnage spatial rapide via le générateur de rasterio
        for val in src.sample(coordonnees):
            valeurs.append(float(val[0]))

    return valeurs


def generer_carte_risques_ciblee(
    fichier_grille: str,
    raster_ndvi: str,
    raster_ndwi: str,
    fichier_modele: str,
    fichier_sortie: str,
    seuil_ndvi_min: float = 0.30,
    seuil_risque_min: float = 65.0
) -> None:
    """Filtre la grille territoriale sur les zones de biomasse réelle,
    calcule les probabilités d'incendie via XGBoost et exporte
    exclusivement les mailles sous vigilance opérationnelle en format CSV.
    """
    try:
        # 1. Chargement de la grille prévisionnelle
        logger.info(f"Chargement de la grille : {fichier_grille}")
        gdf_grille = gpd.read_file(fichier_grille)
        total_initial = len(gdf_grille)

        # 2. Calcul géodésique des centroïdes (projection métrique UTM 32N pour la Tunisie)
        logger.info("Calcul des centroïdes des mailles...")
        gdf_proj = gdf_grille.to_crs(epsg=32632)
        centroides = gdf_proj.geometry.centroid.to_crs(gdf_grille.crs)

        # 3. Extraction des indicateurs satellitaires réels
        logger.info("Échantillonnage des rasters MODIS (NDVI et NDWI)...")
        gdf_grille["ndvi"] = extraire_valeurs_raster(centroides, raster_ndvi)
        gdf_grille["ndwi"] = extraire_valeurs_raster(centroides, raster_ndwi)

        # 4. Filtre de biomasse : on exclut les zones sans couvert végétal (déserts, villes)
        gdf_combustible = gdf_grille[gdf_grille["ndvi"] >= seuil_ndvi_min].copy()
        logger.info(f"Filtrage biomasse (NDVI >= {seuil_ndvi_min}) : {len(gdf_combustible):,} mailles retenues.")

        if gdf_combustible.empty:
            logger.warning("Aucune maille ne présente un couvert végétal suffisant.")
            return

        # 5. Inférence du modèle XGBoost
        logger.info(f"Chargement du modèle XGBoost : {fichier_modele}")
        modele_xgb = xgb.XGBClassifier()
        modele_xgb.load_model(fichier_modele)

        features_cols = ["t_max", "h_mean", "wind_max", "precip_sum", "ndvi", "ndwi"]
        X_pred = gdf_combustible[features_cols].copy().astype(float)

        logger.info("Calcul des probabilités de départ de feu...")
        probabilites = modele_xgb.predict_proba(X_pred)[:, 1]
        gdf_combustible["risque_prob"] = (probabilites * 100).round(1)

        # 6. Seuils de vigilance opérationnelle
        gdf_alertes = gdf_combustible[gdf_combustible["risque_prob"] >= seuil_risque_min].copy()

        conditions = [
            (gdf_alertes["risque_prob"] >= 85.0),
            (gdf_alertes["risque_prob"] >= 75.0) & (gdf_alertes["risque_prob"] < 85.0),
            (gdf_alertes["risque_prob"] >= 65.0) & (gdf_alertes["risque_prob"] < 75.0)
        ]
        classes_vigilance = ["Alerte Rouge (Extrême)", "Alerte Orange (Élevé)", "Vigilance Jaune (Modéré)"]
        gdf_alertes["niveau_vigilance"] = np.select(conditions, classes_vigilance, default="Indéterminé")

        logger.info(f"Foyers potentiels retenus après seuillage : {len(gdf_alertes):,} mailles.")

        # 7. Export CSV ultra-rapide pour Streamlit (on abandonne le GeoJSON lourd)
        logger.info("Conversion des géométries en points simples (CSV) pour le tableau de bord...")
        
        # Extraction précise des coordonnées géographiques
        gdf_alertes_proj = gdf_alertes.to_crs(epsg=32632)
        centroides_finaux = gdf_alertes_proj.geometry.centroid.to_crs(gdf_alertes.crs)
        
        gdf_alertes["lon"] = centroides_finaux.x
        gdf_alertes["lat"] = centroides_finaux.y
        
        # Suppression du polygone géométrique et sauvegarde tabulaire
        df_export = pd.DataFrame(gdf_alertes.drop(columns=["geometry"]))
        df_export.to_csv(fichier_sortie, index=False)
        
        logger.info(f"✔ Carte allégée générée avec succès : {fichier_sortie}")

    except Exception as exc:
        logger.error(f"Échec du pipeline prédictif : {exc}", exc_info=True)
        raise


if __name__ == "__main__":
    import glob
    
    # Recherche dynamique de la dernière grille météo générée
    fichiers_grille = glob.glob("grille_meteo_previsionnelle_*.geojson")
    
    if not fichiers_grille:
        logger.error("Aucune grille météo trouvée. Vérifiez Run_Automation_Pipeline.py.")
        exit(1)
        
    # Trie par ordre alphabétique/date et prend le dernier
    fichier_grille_jour = sorted(fichiers_grille)[-1]
    
    generer_carte_risques_ciblee(
        fichier_grille=fichier_grille_jour,
        raster_ndvi="tunisie_ndvi_actuel.tif",
        raster_ndwi="tunisie_ndwi_actuel.tif",
        fichier_modele="modele_xgboost_tunisia_fire.joblib", # Nom mis à jour selon tes logs
        fichier_sortie="carte_risques_demain_reel.csv",
        seuil_ndvi_min=0.30,
        seuil_risque_min=65.0
    )