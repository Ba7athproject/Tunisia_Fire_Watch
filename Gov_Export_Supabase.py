import os
import geopandas as gpd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
SUPABASE_DB_URI = os.getenv("SUPABASE_DB_URI")

# Charger ton fichier GeoJSON (assure-toi de l'avoir enregistré sous 'tunisia_governorates.geojson')
geojson_path = "tunisia_governorates.geojson"

if os.path.exists(geojson_path):
    print("Lecture du fichier GeoJSON...")
    gdf = gpd.read_file(geojson_path)
    
    # S'assurer que le système de coordonnées est bien en EPSG:4326
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
        
    engine = create_engine(SUPABASE_DB_URI)
    
    # Envoi direct dans Supabase sous forme de table PostGIS
    gdf.to_postgis('tunisia_gouvernorats', engine, if_exists='replace', index=False)
    print("✔ Table 'tunisia_governorats' créée et alimentée avec succès dans Supabase !")
else:
    print(f"❌ Le fichier {geojson_path} est introuvable dans le répertoire.")