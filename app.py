import streamlit as st
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
import pydeck as pdk
import os
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Configuration de la page Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tunisia Fire Watch | ba7ath OSINT",
    page_icon="🔥",
    layout="wide"
)

load_dotenv()
SUPABASE_DB_URI = os.getenv("SUPABASE_DB_URI")

# -----------------------------------------------------------------------------
# En-tête de Marque (Logo Ba7ath & Style Journalistique)
# -----------------------------------------------------------------------------
col_logo, col_title = st.columns([1, 5])
with col_logo:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.markdown("### **🔍 ba7ath**")
with col_title:
    st.title("🔥 Tunisia Fire Watch : Veille et Analyse Territoriale des Incendies")

st.markdown("""
*Plateforme d'investigation numérique automatisée dédiée à la surveillance des anomalies thermiques en Tunisie.* 
Croisement géospatial open-source : **NASA FIRMS** (VIIRS 375m), imagerie optique **Sentinel-2** (Microsoft Planetary Computer) et modélisation prédictive par **XGBoost**.
""")
st.markdown("---")

# -----------------------------------------------------------------------------
# Connexion à Supabase et Récupération des Données
# -----------------------------------------------------------------------------
@st.cache_resource
def get_db_engine():
    if not SUPABASE_DB_URI:
        st.error("❌ ERREUR CRITIQUE : La variable SUPABASE_DB_URI est introuvable.")
        return None
    return create_engine(SUPABASE_DB_URI)

engine = get_db_engine()

@st.cache_data(ttl=300)
def load_data():
    if not engine:
        return pd.DataFrame()
    try:
        query = """
            SELECT cell_id, acq_date, latitude, longitude, frp, t_max, h_mean, 
                   wind_max, ndvi, ndwi, risque_prob, confidence, gouvernorat 
            FROM foyers_actifs 
            ORDER BY acq_date DESC;
        """
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données depuis Supabase : {e}")
        return pd.DataFrame()

df_foyers = load_data()

if df_foyers.empty:
    st.warning("⚠️ Aucun foyer thermique enregistré dans la base de données pour le moment.")
else:
    # S'assurer que acq_date est au format datetime
    df_foyers['acq_date'] = pd.to_datetime(df_foyers['acq_date'])

    # -----------------------------------------------------------------------------
    # Barre latérale : Filtres d'Investigation & Approche Historique
    # -----------------------------------------------------------------------------
    st.sidebar.header("🔍 Filtres d'Investigation")
    
    # 1. Filtre par Gouvernorat (issu du croisement PostGIS)
    gouvernorats_disponibles = sorted(df_foyers['gouvernorat'].dropna().unique().tolist())
    selected_gouvernorat = st.sidebar.selectbox(
        "Filtrer par Gouvernorat", 
        options=["Tous les gouvernorats"] + gouvernorats_disponibles
    )
    
    # 2. Filtre sur le risque minimal prédit par l'IA
    seuil_risque = st.sidebar.slider("Seuil de risque minimal (%)", 0, 100, 30)
    
    # 3. Approche Historique : Filtre temporel
    min_date = df_foyers['acq_date'].min().date()
    max_date = df_foyers['acq_date'].max().date()
    
    st.sidebar.markdown("### ⏳ Approche Historique")
    date_range = st.sidebar.date_input(
        "Période d'analyse",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Application des filtres
    df_filtered = df_foyers[df_foyers['risque_prob'] >= seuil_risque]
    
    if selected_gouvernorat != "Tous les gouvernorats":
        df_filtered = df_filtered[df_filtered['gouvernorat'] == selected_gouvernorat]
        
    if len(date_range) == 2:
        start_d, end_d = date_range
        df_filtered = df_filtered[
            (df_filtered['acq_date'].dt.date >= start_d) & 
            (df_filtered['acq_date'].dt.date <= end_d)
        ]

    # -----------------------------------------------------------------------------
    # Indicateurs clés (KPIs)
    # -----------------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Foyers Sélectionnés", len(df_filtered))
    col2.metric("Gouvernorats Impactés", df_filtered['gouvernorat'].nunique() if not df_filtered.empty else 0)
    col3.metric("Température Max Moyenne", f"{df_filtered['t_max'].mean():.1f} °C" if not df_filtered.empty else "N/A")
    col4.metric("FRP (Puissance Radiative) Max", f"{df_filtered['frp'].max():.1f} MW" if not df_filtered.empty else "N/A")

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # Visualisation Cartographique interactive (PyDeck)
    # -----------------------------------------------------------------------------
    st.subheader("🗺️ Cartographie des Foyers Thermiques et Risques Territoriaux")
    
    if not df_filtered.empty:
        def get_color(prob):
            if prob > 75:
                return [200, 30, 0, 190]   # Rouge critique
            elif prob > 45:
                return [255, 140, 0, 190] # Orange modéré
            else:
                return [50, 205, 50, 190]  # Vert faible

        df_filtered['color'] = df_filtered['risque_prob'].apply(get_color)
        df_filtered['radius'] = df_filtered['frp'].apply(lambda x: max(350, min(x * 120, 3500)))

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_filtered,
            get_position='[longitude, latitude]',
            get_color='color',
            get_radius='radius',
            pickable=True,
            auto_highlight=True,
            opacity=0.85
        )

        view_state = pdk.ViewState(
            latitude=34.0,
            longitude=9.0,
            zoom=6,
            pitch=25,
        )

        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>Date :</b> {acq_date}<br/>"
                        "<b>Gouvernorat :</b> {gouvernorat}<br/>"
                        "<b>Risque IA :</b> {risque_prob}%<br/>"
                        "<b>FRP :</b> {frp} MW<br/>"
                        "<b>Température :</b> {t_max} °C",
                "style": {"backgroundColor": "#111", "color": "#fff", "border": "1px solid #444"}
            }
        )

        st.pydeck_chart(r)
    else:
        st.info("Aucun foyer ne correspond aux filtres sélectionnés.")

    # -----------------------------------------------------------------------------
    # Approche Historique & Graphique d'Évolution
    # -----------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📈 Évolution Historique et Chronologie des Détections")
    if not df_filtered.empty:
        df_timeline = df_filtered.set_index('acq_date').resample('D').size().reset_index(name='nombre_foyers')
        st.line_chart(df_timeline.set_index('acq_date'))
    else:
        st.write("Données insuffisantes pour afficher l'historique sur la période.")

    # -----------------------------------------------------------------------------
    # Tableau de Données Détaillées & Transparence OSINT
    # -----------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("📊 Consulter les données brutes et métriques d'investigation"):
        st.dataframe(df_filtered[['acq_date', 'gouvernorat', 'latitude', 'longitude', 'risque_prob', 'frp', 't_max', 'h_mean', 'wind_max', 'confidence']])

# -----------------------------------------------------------------------------
# Mentions Légales & Pied de page (Projet ba7ath)
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
### ⚖️ Méthodologie et Mentions Légales (Projet Ba7ath)
* **Sources de données** : Ce tableau de bord exploite des données publiques en temps quasi-réel issues de la **NASA FIRMS** (capteur VIIRS SNPP 375m), de l'API météorologique **WeatherAPI**, et de l'imagerie satellitaire optique **Sentinel-2** (via Microsoft Planetary Computer).
* **Éthique & Transparence** : Les méthodologies de collecte respectent strictement les conditions d'utilisation des API ouvertes. Aucune donnée privée ou sensible n'est manipulée.
* **Avertissement** : Les scores de probabilité de risque d'incendie sont générés par un modèle d'intelligence artificielle prédictif (`XGBoost`) à des fins journalistiques, d'alerte citoyenne et de recherche. Ils ne se substituent pas aux communiqués officiels de la Protection Civile tunisienne ou des autorités forestières compétentes.
""")
st.markdown("<p style='text-align: center; color: gray;'>Développé pour l'investigation numérique et le journalisme de données — Projet Ba7ath (2026)</p>", unsafe_allow_html=True)