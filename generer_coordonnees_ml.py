import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def preparer_base_ml(fichier_nasa, fichier_geojson):
    logging.info("1. Nettoyage de l'archive NASA FIRMS (Cas Positifs)...")
    df_pos = pd.read_csv(fichier_nasa)
    
    # Filtrer les données depuis 2016 pour assurer la compatibilité avec Sentinel-2
    df_pos['acq_date'] = pd.to_datetime(df_pos['acq_date'])
    df_pos = df_pos[df_pos['acq_date'].dt.year >= 2016]
    
    # VIIRS utilise des niveaux textuels ('low', 'nominal', 'high') au lieu de valeurs numériques
    df_pos['confidence'] = df_pos['confidence'].astype(str).str.strip().str.lower()
    df_pos = df_pos[df_pos['confidence'].isin(['nominal', 'high', 'h', 'n'])]
    
    df_pos = df_pos[['latitude', 'longitude', 'acq_date']].copy()
    df_pos['incendie'] = 1 # Variable cible : Il y a eu un feu
    
    logging.info(f"✔ {len(df_pos)} incendies avérés conservés (2016-2026).")

    logging.info("2. Génération des coordonnées aléatoires (Cas Négatifs)...")
    gdf_tunisie = gpd.read_file(fichier_geojson)
    
    # On cible les régions à fort couvert végétal pour que le modèle apprenne bien (Nord et Centre-Ouest)
    regions_forestieres = ['Jendouba', 'Beja', 'Bizerte', 'Siliana', 'Le Kef', 'Kasserine', 'Zaghouan', 'Nabeul']
    gdf_forets = gdf_tunisie[gdf_tunisie['shapeName'].isin(regions_forestieres)]
    
    # Si le filtrage échoue (noms différents), on prend toute la Tunisie
    if gdf_forets.empty:
        gdf_forets = gdf_tunisie
        
    emprise = gdf_forets.unary_union.bounds # (minx, miny, maxx, maxy)
    
    cas_negatifs = []
    n_negatifs_souhaites = len(df_pos) * 2 # On génère 2x plus de cas sans feu pour équilibrer l'IA
    
    np.random.seed(42)
    while len(cas_negatifs) < n_negatifs_souhaites:
        # Générer des coordonnées aléatoires
        lon_rand = np.random.uniform(emprise[0], emprise[2])
        lat_rand = np.random.uniform(emprise[1], emprise[3])
        point = Point(lon_rand, lat_rand)
        
        # Vérifier que le point tombe bien sur le sol tunisien
        if gdf_forets.contains(point).any():
            # Assigner une date aléatoire entre mai et octobre (saison des risques)
            annee = np.random.randint(2016, 2026)
            mois = np.random.randint(5, 11)
            jour = np.random.randint(1, 28)
            date_rand = pd.to_datetime(f"{annee}-{mois:02d}-{jour:02d}")
            
            cas_negatifs.append({
                'latitude': lat_rand,
                'longitude': lon_rand,
                'acq_date': date_rand,
                'incendie': 0 # Variable cible : Pas de feu
            })

    df_neg = pd.DataFrame(cas_negatifs)
    logging.info(f"✔ {len(df_neg)} cas normaux générés.")

    # 3. Fusion et sauvegarde
    df_final = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    df_final['acq_date'] = df_final['acq_date'].dt.strftime('%Y-%m-%d')
    
    fichier_sortie = "dataset_coordonnees_cibles.csv"
    df_final.to_csv(fichier_sortie, index=False)
    logging.info(f"✔ Squelette du dataset sauvegardé : {fichier_sortie}")

if __name__ == "__main__":
    # Noms exacts de tes fichiers
    preparer_base_ml("fire_archive_SV-C2_797614.csv", "tunisia_governorates.geojson")