import logging
import pystac_client
import planetary_computer
import odc.stac
import geopandas as gpd
from shapely.geometry import Point

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def obtenir_elevation_stac(lon: float, lat: float) -> float:
    """
    Extrait l'altitude précise (en mètres) d'une coordonnée GPS
    via le Modèle Numérique de Surface Copernicus (30m).
    """
    try:
        # Bounding box microscopique autour du point
        buffer = 0.0001
        bbox = [lon - buffer, lat - buffer, lon + buffer, lat + buffer]

        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        search = catalog.search(
            collections=["cop-dem-glo-30"],
            bbox=bbox
        )
        
        items = list(search.items())
        if not items:
            logging.warning("Aucune tuile topographique trouvée pour ces coordonnées.")
            return 0.0

        # Chargement du pixel d'élévation (Bande 'data')
        cube = odc.stac.load(
            items,
            bands=["data"],
            bbox=bbox,
            resolution=30,
            crs="EPSG:4326",
            chunks={}
        )
        
        # Extraction de la valeur moyenne du pixel
        elevation = float(cube["data"].mean().compute().values)
        return round(elevation, 1)

    except Exception as exc:
        logging.error(f"Erreur lors de l'extraction topographique : {exc}")
        return 0.0

if __name__ == "__main__":
    # Test d'élévation sur le point culminant naturel de la Tunisie (Jebel ech Chambi)
    lon_test, lat_test = 8.6656, 35.2011 
    
    logging.info(f"Interrogation du modèle Copernicus pour les coordonnées : {lat_test}, {lon_test}")
    altitude = obtenir_elevation_stac(lon_test, lat_test)
    logging.info(f"Altitude détectée : {altitude} mètres.")