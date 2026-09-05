import pandas as pd
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def enrichir_topographie_historique(df_foyers: pd.DataFrame, batch_size: int = 100) -> pd.DataFrame:
    """
    Interroge l'API Open-Meteo Elevation par lots pour récupérer l'altitude.
    df_foyers doit contenir les colonnes 'latitude' et 'longitude'.
    """
    url = "https://api.open-meteo.com/v1/elevation"
    altitudes_totales = []

    logging.info(f"Début de l'extraction topographique pour {len(df_foyers)} points...")

    for i in range(0, len(df_foyers), batch_size):
        batch = df_foyers.iloc[i:i+batch_size]
        
        # Formatage des coordonnées avec une précision de 4 décimales (~11 mètres)
        lats = ",".join(batch['latitude'].round(4).astype(str).tolist())
        lons = ",".join(batch['longitude'].round(4).astype(str).tolist())
        
        params = {
            "latitude": lats,
            "longitude": lons
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # L'API renvoie une liste directe d'altitudes dans la clé 'elevation'
            elevations = data.get('elevation', [0.0] * len(batch))
            altitudes_totales.extend(elevations)
            
        except Exception as e:
            logging.error(f"Erreur sur le lot {i}-{i+batch_size}: {e}")
            altitudes_totales.extend([None] * len(batch))

    df_foyers = df_foyers.copy()
    df_foyers['elevation_m'] = altitudes_totales
    
    logging.info("Enrichissement topographique terminé.")
    return df_foyers

if __name__ == "__main__":
    # Test unitaire avec le Jebel ech Chambi et un point au niveau de la mer
    df_test = pd.DataFrame({
        'latitude': [35.2011, 36.8065],
        'longitude': [8.6656, 10.1815]
    })
    
    df_resultat = enrichir_topographie_historique(df_test)
    print(df_resultat)