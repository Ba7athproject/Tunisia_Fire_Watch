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
    page_title="Tunisia Fire Watch - Dashboard OSINT",
    page_icon="🔥",
    layout="wide"
)

load_dotenv()
SUPABASE_DB_URI = os.getenv("SUPABASE_DB_URI")

# Titre et introduction journalistique
st.title("🔥 Tunisia Fire Watch : Surveillance des Foyers Thermiques")
st.markdown("""
*Système de veille automatisé et d'analyse prédictive des risques d'incendie en Tunisie.* 
Croisement de données satellitaires **NASA FIRMS** (VIIRS 375m), d'imagerie optique **Sentinel-2** et d'un modèle d'intelligence artificielle **XGBoost**.
""")

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

@st.cache_data(ttl=600)  # Cache de 10 minutes pour optimiser les performances
def charger_foyers():
    if not engine:
        return pd.DataFrame()
    try:
        query = "SELECT cell_id, acq_date, latitude, longitude, frp, t_max, h_mean, wind_max, ndvi, ndwi, risque_prob, confidence FROM foyers_actifs ORDER BY acq_date DESC;"
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données depuis Supabase : {e}")
        return pd.DataFrame()

df_foyers = charger_foyers()

if df_foyers.empty:
    st.warning("⚠️ Aucun foyer thermique enregistré dans la base de données pour le moment.")
else:
    # -----------------------------------------------------------------------------
    # Barre latérale (Filtres OSINT)
    # -----------------------------------------------------------------------------
    st.sidebar.header("🔍 Filtres d'Investigation")
    
    # Filtre sur le risque minimal prédit par le modèle
    seuil_risque = st.sidebar.slider("Seuil de risque minimal (%)", 0, 100, 30)
    
    df_filtered = df_foyers[df_foyers['risque_prob'] >= seuil_risque].copy()

    # -----------------------------------------------------------------------------
    # Indicateurs clés (KPIs)
    # -----------------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Foyers Détectés", len(df_foyers))
    col2.metric("Foyers filtrés (> " + str(seuil_risque) + "%)", len(df_filtered))
    col3.metric("Température Max Moyenne", f"{df_filtered['t_max'].mean():.1f} °C" if not df_filtered.empty else "N/A")
    col4.metric("FRP (Puissance Radiative) Max", f"{df_filtered['frp'].max():.1f} MW" if not df_filtered.empty else "N/A")

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # Visualisation Cartographique interactive (PyDeck)
    # -----------------------------------------------------------------------------
    st.subheader("🗺️ Carte Satellitaire des Foyers Actifs")
    
    if not df_filtered.empty:
        # Attribution d'une couleur dynamique selon le risque (Vert -> Orange -> Rouge)
        def get_color(prob):
            if prob > 75:
                return [200, 30, 0, 180]   # Rouge critique
            elif prob > 45:
                return [255, 140, 0, 180] # Orange modéré
            else:
                return [50, 205, 50, 180]  # Vert faible

        df_filtered['color'] = df_filtered['risque_prob'].apply(get_color)
        # Taille proportionnelle à la puissance radiative (FRP)
        df_filtered['radius'] = df_filtered['frp'].apply(lambda x: max(300, min(x * 100, 3000)))

        # Configuration de la carte centrée sur la Tunisie
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_filtered,
            get_position='[longitude, latitude]',
            get_color='color',
            get_radius='radius',
            pickable=True,
            auto_highlight=True,
            opacity=0.8
        )

        # Vue par défaut sur la Tunisie
        view_state = pdk.ViewState(
            latitude=34.0,
            longitude=9.0,
            zoom=6,
            pitch=30,
        )

        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>Date :</b> {acq_date}<br/>"
                        "<b>Risque IA :</b> {risque_prob}%<br/>"
                        "<b>FRP :</b> {frp} MW<br/>"
                        "<b>Température :</b> {t_max} °C",
                "style": {"backgroundColor": "black", "color": "white"}
            }
        )

        st.pydeck_chart(r)
    else:
        st.info("Aucun foyer ne correspond au seuil de risque sélectionné.")

    # -----------------------------------------------------------------------------
    # Tableau de Données Détaillées (Transparence OSINT)
    # -----------------------------------------------------------------------------
    with st.expander("📊 Afficher les données brutes et métriques associées"):
        st.dataframe(df_filtered[['acq_date', 'latitude', 'longitude', 'risque_prob', 'frp', 't_max', 'h_mean', 'wind_max', 'confidence']])

# -----------------------------------------------------------------------------
# Pied de page journalistique
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Développé pour l'investigation OSINT — Projet Ba7ath</p>", unsafe_allow_html=True)