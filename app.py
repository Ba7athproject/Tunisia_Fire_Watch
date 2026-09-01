import streamlit as st
import geopandas as gpd
import pandas as pd
import joblib
import os
import folium
from streamlit_folium import st_folium
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()
SUPABASE_DB_URI = os.getenv("SUPABASE_DB_URI")

# -----------------------------------------------------------------------------
# Configuration de la page Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tunisia Fire Watch - Dashboard Cloud OSINT",
    page_icon="🔥",
    layout="wide"
)

DOSSIER_PROJET = r"C:\Ba7ath_project\Tunisia-fire-detection"
MODELE_PATH = os.path.join(DOSSIER_PROJET, "modele_xgboost_tunisia_fire.joblib")

# -----------------------------------------------------------------------------
# Chargement des ressources depuis Supabase et le modèle local
# -----------------------------------------------------------------------------
@st.cache_resource
def charger_donnees_supabase():
    if not SUPABASE_DB_URI:
        return None
    try:
        engine = create_engine(SUPABASE_DB_URI)
        # Lecture directe de la table PostGIS dans un GeoDataFrame
        query = "SELECT * FROM foyers_actifs;"
        gdf = gpd.read_postgis(query, engine, geom_col='geom')
        return gdf
    except Exception as e:
        st.error(f"Erreur de connexion à Supabase : {e}")
        return None

@st.cache_resource
def charger_modele():
    if os.path.exists(MODELE_PATH):
        return joblib.load(MODELE_PATH)
    return None

gdf_anomalies = charger_donnees_supabase()
modele_ml = charger_modele()

# -----------------------------------------------------------------------------
# Interface Utilisateur (Sidebar & Filtres)
# -----------------------------------------------------------------------------
st.sidebar.title("🔥 Tunisia Fire Watch")
st.sidebar.markdown("Plateforme Cloud PostGIS & OSINT.")
st.sidebar.markdown("---")

if gdf_anomalies is None or gdf_anomalies.empty:
    st.warning("⚠️ Aucune donnée disponible dans la base Supabase ou échec de connexion. Exécute d'abord `Push_To_Supabase.py`.")
else:
    seuil_frp = st.sidebar.slider("Filtrer par Puissance Radiative (FRP min)", 0.0, 100.0, 1.0)
    confiance_filtre = st.sidebar.selectbox("Niveau de confiance satellite", ["Tous", "Nominal (n)", "High (h)"])
    
    # Application des filtres
    df_filtré = gdf_anomalies[gdf_anomalies['frp'] >= seuil_frp].copy()
    if confiance_filtre == "Nominal (n)":
        df_filtré = df_filtré[df_filtré['confidence'] == 'n']
    elif confiance_filtre == "High (h)":
        df_filtré = df_filtré[df_filtré['confidence'] == 'h']

    # -----------------------------------------------------------------------------
    # Corps Principal (KPIs & Carte Interactive)
    # -----------------------------------------------------------------------------
    st.title("Tableau de Bord Cloud & Veille Thermique en Temps Réel")
    st.markdown("Données synchronisées en direct depuis **Supabase PostGIS** (NASA FIRMS + Open-Meteo + Sentinel-2 + XGBoost).")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Foyers en Base", len(gdf_anomalies))
    col2.metric("Foyers filtrés", len(df_filtré))
    col3.metric("Température max", f"{gdf_anomalies['t_max'].max() if 't_max' in gdf_anomalies else 0} °C")
    col4.metric("FRP maximale", f"{gdf_anomalies['frp'].max() if 'frp' in gdf_anomalies else 0} MW")

    st.markdown("---")

    # Carte Folium
    st.subheader("🗺️ Carte Spatio-Temporelle (Source Cloud PostGIS)")
    
    m = folium.Map(location=[34.0, 9.0], zoom_start=7, tiles="OpenStreetMap")

    for _, row in df_filtré.iterrows():
        lat, lon = row.geom.y, row.geom.x
        frp = row['frp']
        t_max = row['t_max']
        wind = row['wind_max']
        ndvi = row['ndvi']
        risque = row.get('risque_prob', 0.0)
        
        couleur = "red" if risque > 75 else "orange" if risque > 40 else "green"
        
        popup_html = f"""
        <b>Zone ID:</b> {row.get('cell_id', 'N/A')}<br>
        <b>Date:</b> {str(row['acq_date'])[:10]}<br>
        <b>Risque ML:</b> {risque:.1f}%<br>
        <b>FRP (Puissance):</b> {frp} MW<br>
        <b>Température max:</b> {t_max} °C<br>
        <b>Vent max:</b> {wind} km/h<br>
        <b>NDVI (Végétation):</b> {ndvi}
        """
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=min(max(frp / 5, 4), 15),
            color=couleur,
            fill=True,
            fill_color=couleur,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)

    st_folium(m, width=1200, height=600)

    # Tableau détaillé
    with st.expander("📋 Inspecter les données stockées dans Supabase"):
        st.dataframe(df_filtré[['cell_id', 'acq_date', 'latitude', 'longitude', 'frp', 't_max', 'wind_max', 'ndvi', 'ndwi', 'risque_prob']])