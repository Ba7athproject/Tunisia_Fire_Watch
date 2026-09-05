import logging
import os
import glob
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import joblib

# -----------------------------------------------------------------------------
# Configuration de la journalisation pour la traçabilité de l'investigation
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extraire_valeurs_raster(points_geometry, chemin_raster: str) -> list:
    """Extrait les valeurs réelles d'un raster GeoTIFF (EPSG:4326)
    pour une liste de coordonnées géographiques (Points).
    Inclut la gestion des pixels NoData et des valeurs non numériques.
    """
    if not os.path.exists(chemin_raster):
        raise FileNotFoundError(f"Le fichier raster est introuvable : {chemin_raster}")

    coordonnees = [(point.x, point.y) for point in points_geometry]
    valeurs = []

    with rasterio.open(chemin_raster) as src:
        nodata = src.nodata
        for val in src.sample(coordonnees):
            v = float(val[0])
            # Si le pixel tombe sur la valeur nodata officielle ou un NaN, on neutralise à 0.0
            if nodata is not None and np.isclose(v, nodata):
                valeurs.append(0.0)
            elif np.isnan(v):
                valeurs.append(0.0)
            else:
                valeurs.append(v)

    return valeurs


def generer_carte_risques_ciblee(
    fichier_grille: str,
    raster_ndvi: str,
    raster_ndwi: str,
    fichier_modele: str,
    fichier_sortie: str,
    seuil_ndvi_min: float = 0.30,
    seuil_ndwi_max: float = 0.0,
    seuil_risque_min: float = 65.0
) -> None:
    """Filtre la grille territoriale sur les zones de biomasse terrestre réelle,
    élimine les plans d'eau maritimes et lagunaires via NDWI,
    calcule les probabilités d'incendie via XGBoost (incluant la topographie) 
    et exporte exclusivement les mailles sous vigilance opérationnelle en format CSV.
    """
    try:
        # 1. Chargement de la grille prévisionnelle
        logger.info(f"Chargement de la grille : {fichier_grille}")
        gdf_grille = gpd.read_file(fichier_grille)

        # 2. Calcul géodésique des centroïdes (projection métrique UTM 32N pour la Tunisie)
        logger.info("Calcul des centroïdes des mailles...")
        gdf_proj = gdf_grille.to_crs(epsg=32632)
        centroides = gdf_proj.geometry.centroid.to_crs(gdf_grille.crs)

        # 3. Extraction des indicateurs satellitaires réels (NDVI et NDWI)
        logger.info("Échantillonnage des rasters MODIS (NDVI et NDWI)...")
        gdf_grille["ndvi"] = extraire_valeurs_raster(centroides, raster_ndvi)
        gdf_grille["ndwi"] = extraire_valeurs_raster(centroides, raster_ndwi)

        # 4. Filtre combiné de biomasse et élimination stricte des surfaces maritimes/eau
        # L'eau libre et les zones marines côtières présentent systématiquement un NDWI >= 0
        logger.info(f"Application des filtres biophysiques (NDVI >= {seuil_ndvi_min} & NDWI < {seuil_ndwi_max})...")
        
        masque_terrestre = (
            (gdf_grille["ndvi"] >= seuil_ndvi_min) &
            (gdf_grille["ndvi"] <= 1.0) &
            (gdf_grille["ndwi"] < seuil_ndwi_max)
        )
        
        # Sécurité additionnelle : exclure d'éventuelles altitudes strictement négatives (fond marin)
        if "elevation_m" in gdf_grille.columns:
            masque_terrestre = masque_terrestre & (gdf_grille["elevation_m"] >= 0.0)

        gdf_combustible = gdf_grille[masque_terrestre].copy()
        logger.info(f"Mailles terrestres avec combustible retenues : {len(gdf_combustible):,}.")

        if gdf_combustible.empty:
            logger.warning("Aucune maille ne remplit les critères biophysiques terrestres.")
            return

        # 5. Chargement du modèle XGBoost (7 features attendues)
        logger.info(f"Chargement du modèle XGBoost (.joblib) : {fichier_modele}")
        modele_xgb = joblib.load(fichier_modele)

        # Liste stricte des 7 variables exigées par le modèle
        features_cols = ["t_max", "h_mean", "wind_max", "precip_sum", "ndvi", "ndwi", "elevation_m"]

        # 6. Blindage strict : vérification de l'intégrité de chaque colonne
        for col in features_cols:
            if col not in gdf_combustible.columns:
                valeur_secours = 250.0 if col == "elevation_m" else 0.0
                gdf_combustible[col] = valeur_secours
            else:
                valeur_secours = 250.0 if col == "elevation_m" else 0.0
                gdf_combustible[col] = gdf_combustible[col].fillna(valeur_secours)

        X_pred = gdf_combustible[features_cols].copy().astype(float)

        # 7. Inférence sécurisée
        logger.info("Calcul des probabilités de départ de feu...")
        probabilites = modele_xgb.predict_proba(X_pred)[:, 1]
        gdf_combustible["risque_prob"] = (probabilites * 100).round(1)

        # 8. Seuils de vigilance opérationnelle
        gdf_alertes = gdf_combustible[gdf_combustible["risque_prob"] >= seuil_risque_min].copy()

        conditions = [
            (gdf_alertes["risque_prob"] >= 85.0),
            (gdf_alertes["risque_prob"] >= 75.0) & (gdf_alertes["risque_prob"] < 85.0),
            (gdf_alertes["risque_prob"] >= 65.0) & (gdf_alertes["risque_prob"] < 75.0)
        ]
        classes_vigilance = ["Alerte Rouge (Extrême)", "Alerte Orange (Élevé)", "Vigilance Jaune (Modéré)"]
        gdf_alertes["niveau_vigilance"] = np.select(conditions, classes_vigilance, default="Indéterminé")

        logger.info(f"Foyers potentiels retenus après seuillage : {len(gdf_alertes):,} mailles.")

        # 9. Export CSV pour l'application front-end React / Vite
        logger.info("Conversion des géométries en points simples (CSV) pour le tableau de bord...")
        
        gdf_alertes_proj = gdf_alertes.to_crs(epsg=32632)
        centroides_finaux = gdf_alertes_proj.geometry.centroid.to_crs(gdf_alertes.crs)
        
        gdf_alertes["lon"] = centroides_finaux.x.round(5)
        gdf_alertes["lat"] = centroides_finaux.y.round(5)
        
        # Sélection des colonnes essentielles et sérialisation propre
        df_export = pd.DataFrame(gdf_alertes.drop(columns=["geometry"]))
        df_export.to_csv(fichier_sortie, index=False)
        
        logger.info(f"✔ Carte allégée générée avec succès : {fichier_sortie}")

    except Exception as exc:
        logger.error(f"Échec du pipeline prédictif : {exc}", exc_info=True)
        raise


if __name__ == "__main__":
    fichiers_grille = glob.glob("grille_meteo_previsionnelle_*.geojson")
    
    if not fichiers_grille:
        logger.error("Aucune grille météo trouvée. Vérifiez Run_Automation_Pipeline.py.")
        exit(1)
        
    fichier_grille_jour = sorted(fichiers_grille)[-1]
    logger.info(f"Grille météo dynamique identifiée : {fichier_grille_jour}")
    
    generer_carte_risques_ciblee(
        fichier_grille=fichier_grille_jour,
        raster_ndvi="tunisie_ndvi_actuel.tif",
        raster_ndwi="tunisie_ndwi_actuel.tif",
        fichier_modele="modele_xgboost_tunisia_fire.joblib",
        fichier_sortie="carte_risques_demain_reel.csv",
        seuil_ndvi_min=0.30,
        seuil_ndwi_max=0.0,
        seuil_risque_min=65.0
    )