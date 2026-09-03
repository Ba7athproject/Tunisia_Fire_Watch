import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

# -----------------------------------------------------------------------------
# 1. Configuration de la Plateforme ba7ath
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tunisia Fire Watch | Investigation & Prédiction",
    page_icon="🔍",
    layout="wide"
)

load_dotenv()
SUPABASE_DB_URI = os.getenv("SUPABASE_DB_URI")

col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("### **🔍 ba7ath**")
with col_title:
    st.title("🔥 Tunisia Fire Watch : Anticipation des Risques")

st.markdown("""
*Plateforme d'investigation numérique et de modélisation prédictive des anomalies thermiques en Tunisie.* 
Exploitation conjointe des archives **VIIRS**, du climat **Open-Meteo**, de la biomasse **MODIS/Sentinel-2** et de l'IA **XGBoost**.
""")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. Moteurs de Données (Mise en Cache pour Haute Performance)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_db_engine():
    if not SUPABASE_DB_URI:
        return None
    return create_engine(SUPABASE_DB_URI)

@st.cache_data(ttl=300)
def load_realtime_data():
    """Extraction des détections satellitaires actives depuis PostGIS."""
    engine = get_db_engine()
    if not engine:
        return pd.DataFrame()
    try:
        query = "SELECT cell_id, acq_date, latitude, longitude, frp, confidence, gouvernorat FROM foyers_actifs ORDER BY acq_date DESC;"
        return pd.read_sql(query, engine)
    except Exception:
        return pd.DataFrame()

@st.cache_data
def load_prediction_data():
    """Chargement et optimisation des 103 000+ mailles prédictives pour PyDeck."""
    fichier_pred = "carte_risques_demain_reel.geojson"
    if os.path.exists(fichier_pred):
        gdf = gpd.read_file(fichier_pred)
        # Extraction rapide des coordonnées pour le rendu WebGL
        gdf['lon'] = gdf.geometry.centroid.x
        gdf['lat'] = gdf.geometry.centroid.y
        # Nettoyage de la géométrie lourde pour économiser la RAM du navigateur
        df = pd.DataFrame(gdf.drop(columns=['geometry']))
        return df
    return pd.DataFrame()

@st.cache_data
def load_historical_data():
    """Chargement de l'historique national."""
    csv_path = "historique_incendies_propre.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

# -----------------------------------------------------------------------------
# 3. Interface d'Investigation : Les 3 Piliers (Temps Réel, Prédiction, Archives)
# -----------------------------------------------------------------------------
tab_pred, tab_realtime, tab_history = st.tabs([
    "🔮 Prédiction des Risques (J+1)", 
    "🔴 Surveillance Temps Réel", 
    "📚 Archives Nationales"
])

# =============================================================================
# ONGLET 1 : PRÉDICTION DES RISQUES (IA XGBOOST)
# =============================================================================
with tab_pred:
    st.subheader("Cartographie Prédictive des Départs de Feu (Modèle XGBoost)")
    df_pred = load_prediction_data()
    
    if not df_pred.empty:
        # Contrôles de filtrage
        st.sidebar.header("🎛️ Filtres Prédictifs")
        seuil_risque = st.sidebar.slider("Niveau de risque minimal (%)", 30, 100, 60)
        df_filtre_pred = df_pred[df_pred['risque_prob'] >= seuil_risque].copy()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Zones critiques détectées", f"{len(df_filtre_pred):,}")
        col2.metric("Température Max (Zone critique)", f"{df_filtre_pred['t_max'].max():.1f} °C")
        col3.metric("Vent Max (Rafales)", f"{df_filtre_pred['wind_max'].max():.1f} km/h")
        col4.metric("Stress Hydrique (NDWI min)", f"{df_filtre_pred['ndwi'].min():.3f}")
        
        # Colorimétrie dynamique : Jaune (Risque modéré) -> Rouge (Risque critique)
        df_filtre_pred['color'] = df_filtre_pred['risque_prob'].apply(
            lambda x: [255, int(255 - (x-30)*3.6), 0, 160] if x < 80 else [255, 0, 0, 200]
        )
        
        # Couche ColumnLayer pour représenter l'intensité du risque en 3D
        layer_pred = pdk.Layer(
            "ColumnLayer",
            data=df_filtre_pred,
            get_position='[lon, lat]',
            get_elevation='risque_prob * 50', # Élévation proportionnelle au risque
            elevation_scale=10,
            radius=250, # Rayon de 250m correspondant à la maille MODIS/Sentinel
            get_fill_color='color',
            pickable=True,
            auto_highlight=True,
        )
        
        view_state = pdk.ViewState(latitude=35.0, longitude=9.5, zoom=6, pitch=45)
        r_pred = pdk.Deck(
            layers=[layer_pred],
            initial_view_state=view_state,
            tooltip={"html": "<b>Risque: {risque_prob}%</b><br/>Température: {t_max}°C<br/>Vent: {wind_max} km/h<br/>Végétation (NDVI): {ndvi}"}
        )
        st.pydeck_chart(r_pred)
    else:
        st.info("La carte prédictive de demain n'a pas encore été générée par le pipeline d'automatisation.")

# =============================================================================
# ONGLET 2 : SURVEILLANCE TEMPS RÉEL (FIRMS)
# =============================================================================
with tab_realtime:
    df_foyers = load_realtime_data()
    if not df_foyers.empty:
        st.subheader("Anomalies Thermiques Actives (Satellites NASA/NOAA)")
        df_foyers['radius'] = df_foyers['frp'].apply(lambda x: min(x * 50, 3000))
        layer_realtime = pdk.Layer(
            "ScatterplotLayer",
            data=df_foyers,
            get_position='[longitude, latitude]',
            get_color='[255, 69, 0, 200]',
            get_radius='radius',
            pickable=True
        )
        r_realtime = pdk.Deck(
            layers=[layer_realtime],
            initial_view_state=pdk.ViewState(latitude=34.0, longitude=9.0, zoom=5.5),
            tooltip={"html": "<b>{gouvernorat}</b><br/>FRP: {frp} MW<br/>Confiance: {confidence}"}
        )
        st.pydeck_chart(r_realtime)
    else:
        st.info("Aucune anomalie thermique n'est actuellement signalée.")

# =============================================================================
# ONGLET 3 : ARCHIVES ET BARRAGES (CSV STATISTIQUE)
# =============================================================================
with tab_history:
    st.subheader("Bilan National et Données d'Investigation (2002-2025)")
    df_historique = load_historical_data()
    if not df_historique.empty:
        df_yearly = df_historique.groupby('annee')[['nombre', 'superficie_ha']].sum().reset_index()
        col_c1, col_c2 = st.columns(2)
        col_c1.bar_chart(df_yearly.set_index('annee')['nombre'], color="#ff4b4b")
        col_c2.line_chart(df_yearly.set_index('annee')['superficie_ha'], color="#ffa500")
        st.dataframe(df_historique.groupby('gouvernorat')['superficie_ha'].sum().sort_values(ascending=False).reset_index(), use_container_width=True)
    else:
        st.warning("Fichier historique introuvable.")

# -----------------------------------------------------------------------------
# 4. Transparence OSINT & Mentions Légales
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("⚖️ Méthodologie et Transparence OSINT"):
    st.markdown("""
    * **Collecte** : Fusion automatisée des archives ouvertes de la **NASA** (VIIRS), des réanalyses climatiques d'**Open-Meteo** (ERA5) et de l'imagerie **MODIS/Sentinel-2**.
    * **IA et Biais** : La modélisation prédictive repose sur l'algorithme open-source XGBoost. Afin d'éviter les fuites de données (Target Leakage), les variables thermiques post-incendie (FRP) ont été rigoureusement exclues de l'apprentissage.
    * **Documentation** : Les prédictions n'ont qu'une vocation d'analyse journalistique spatiale et ne se substituent pas aux alertes formelles de l'Observatoire National de l'Agriculture ou de la Protection Civile.
    """)
st.markdown("<p style='text-align: center; color: gray;'>Développé pour l'investigation numérique et le datajournalisme — Projet ba7ath (2026)</p>", unsafe_allow_html=True)