import pystac_client
import planetary_computer
import odc.stac
import rioxarray
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generer_rasters_nationaux_rapides():
    bbox_tunisie = [7.5, 30.2, 11.6, 37.6]
    
    # Fenêtre glissante : le dernier mois pour être sûr d'avoir une composition MODIS de 8 jours complète
    date_fin = datetime.now()
    date_debut = date_fin - timedelta(days=30)
    fenetre = f"{date_debut.strftime('%Y-%m-%d')}/{date_fin.strftime('%Y-%m-%d')}"

    logging.info("Connexion à l'archive MODIS (Surface Reflectance 8-Day)...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    search = catalog.search(
        collections=["modis-09A1-061"],
        bbox=bbox_tunisie,
        datetime=fenetre
    )
    items = list(search.items())
    logging.info(f"✔ {len(items)} tuiles MODIS trouvées. Traitement en cours (ultra-rapide)...")

    # Chargement du cube de données (Red=b01, NIR=b02, SWIR=b06)
    # Chargement du cube de données (Red=b01, NIR=b02, SWIR=b06)
    cube = odc.stac.load(
        items,
        bands=["sur_refl_b01", "sur_refl_b02", "sur_refl_b06"],
        bbox=bbox_tunisie,
        crs="EPSG:4326",  # <-- Ajout indispensable pour définir le système de projection
        resolution=0.005, # Résolution de ~500m en degrés
        groupby="solar_day"
    ).astype(float)

    # Aplatissement temporel pour obtenir la vue la plus claire
    cube_median = cube.median(dim="time").compute()

    logging.info("Calcul matriciel NDVI et NDWI...")
    # Formules optiques standard (MODIS encode les valeurs hors-limites, on évite les divisions par zéro)
    nir = cube_median.sur_refl_b02
    red = cube_median.sur_refl_b01
    swir = cube_median.sur_refl_b06

    ndvi = (nir - red) / (nir + red + 1e-5)
    ndwi = (nir - swir) / (nir + swir + 1e-5)

    # Lissage des anomalies d'eau (mer) et de bordures
    ndvi = ndvi.where((ndvi >= -1) & (ndvi <= 1), 0.25)
    ndwi = ndwi.where((ndwi >= -1) & (ndwi <= 1), 0.05)

    # Attribution du référentiel spatial géographique
    ndvi.rio.write_crs("EPSG:4326", inplace=True)
    ndwi.rio.write_crs("EPSG:4326", inplace=True)

    logging.info("Sauvegarde des rasters géoréférencés sur le disque...")
    ndvi.rio.to_raster("tunisie_ndvi_actuel.tif")
    ndwi.rio.to_raster("tunisie_ndwi_actuel.tif")
    logging.info("✔ Fichiers générés avec succès. Vous pouvez maintenant exécuter le script de modèle prédictif.")

if __name__ == "__main__":
    generer_rasters_nationaux_rapides()