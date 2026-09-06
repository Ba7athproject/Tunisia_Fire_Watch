import React, { useState, useEffect } from 'react';
import DeckGL from '@deck.gl/react';
import { ColumnLayer, ScatterplotLayer } from '@deck.gl/layers';
import Papa from 'papaparse';
import { createClient } from '@supabase/supabase-js';

// Importation exclusive de react-map-gl/maplibre pour une intégration propre sous Vite/Vercel
import Map from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';

// -------------------------------------------------------------------------
// CONFIGURATION SUPABASE (OSINT / Open Data)
// -------------------------------------------------------------------------
// Accès en lecture seule garanti par les politiques RLS de Supabase
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;
const supabase = createClient(supabaseUrl, supabaseAnonKey);

// -------------------------------------------------------------------------
// SOURCES DE DONNÉES
// -------------------------------------------------------------------------
// Flux CSV automatisé attestant de l'intégrité de la chaîne de vérification
const GITHUB_CSV_URL = "https://raw.githubusercontent.com/Ba7athproject/Tunisia_Fire_Watch/main/carte_risques_demain_reel.csv";

// Cadrage cartographique initial sur le territoire tunisien
const INITIAL_VIEW_STATE = {
  longitude: 9.5375,
  latitude: 35.5,
  zoom: 6.2,
  pitch: 45,
  bearing: 0
};

// -------------------------------------------------------------------------
// UTILITAIRES
// -------------------------------------------------------------------------
// Normalisation des indices de confiance (différences entre capteurs VIIRS et MODIS)
const formatConfidence = (conf) => {
  if (conf === null || conf === undefined || conf === '') return 'Non renseigné';
  const c = String(conf).trim().toLowerCase();
  if (c === 'l' || c === 'low') return 'Faible (Low)';
  if (c === 'n' || c === 'nominal') return 'Standard (Nominal)';
  if (c === 'h' || c === 'high') return 'Élevé (High)';
  return `${conf}%`;
};

export default function App() {
  // États de l'application
  const [predictionData, setPredictionData] = useState([]);
  const [realtimeData, setRealtimeData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [seuilRisque, setSeuilRisque] = useState(70);

  // État contrôlé de la caméra : essentiel pour synchroniser Deck.gl et MapLibre
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);

  // -------------------------------------------------------------------------
  // INGESTION DES DONNÉES (Fetch & Parsing)
  // -------------------------------------------------------------------------
  useEffect(() => {
    const fetchAllData = async () => {
      try {
        setLoading(true);

        // 1. Modélisation Prédictive (XGBoost)
        const csvResponse = await fetch(GITHUB_CSV_URL);
        if (!csvResponse.ok) throw new Error(`Erreur HTTP: ${csvResponse.status}`);

        const csvText = await csvResponse.text();
        Papa.parse(csvText, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => setPredictionData(results.data),
          error: (err) => console.error("Erreur de parsing CSV :", err)
        });

        // 2. Alertes Thermiques Temps Réel (FIRMS)
        const { data: firmsData, error: supabaseError } = await supabase
          .from('foyers_actifs')
          .select('latitude, longitude, frp, confidence, gouvernorat')
          .gte('latitude', 30.2)
          .lte('latitude', 37.5)
          .gte('longitude', 7.5)
          .lte('longitude', 11.6)
          .order('acq_date', { ascending: false });

        if (supabaseError) throw supabaseError;
        if (firmsData) setRealtimeData(firmsData);

      } catch (err) {
        console.error("Échec critique lors de l'ingestion des flux de données :", err);
      } finally {
        setLoading(false);
      }
    };

    fetchAllData();
  }, []);

  // -------------------------------------------------------------------------
  // CONTRÔLES DE NAVIGATION
  // -------------------------------------------------------------------------
  const handleZoomIn = () => setViewState(prev => ({ ...prev, zoom: Math.min(prev.zoom + 1, 14) }));
  const handleZoomOut = () => setViewState(prev => ({ ...prev, zoom: Math.max(prev.zoom - 1, 4) }));
  const handleResetView = () => setViewState(INITIAL_VIEW_STATE);

  // -------------------------------------------------------------------------
  // CONSTRUCTION DES CALQUES WEBGL (Deck.gl)
  // -------------------------------------------------------------------------
  const filteredPredictions = predictionData.filter(d => (d.risque_prob || 0) >= seuilRisque);

  const predictionLayer = new ColumnLayer({
    id: 'prediction-layer',
    data: filteredPredictions,
    diskResolution: 12,
    radius: 600,
    extruded: true,
    pickable: true,
    elevationScale: 25,
    getPosition: d => [d.lon, d.lat],
    getElevation: d => d.risque_prob || 0,
    getFillColor: d => {
      const p = d.risque_prob || 0;
      if (p < 75) return [255, 204, 0, 190];
      if (p < 85) return [255, 102, 0, 210];
      return [230, 0, 0, 240];
    }
  });

  const realtimeLayer = new ScatterplotLayer({
    id: 'realtime-layer',
    data: realtimeData,
    pickable: true,
    opacity: 0.85,
    stroked: true,
    filled: true,
    radiusScale: 12,
    radiusMinPixels: 4,
    radiusMaxPixels: 20,
    getPosition: d => [d.longitude, d.latitude],
    getFillColor: [255, 69, 0, 220],
    getLineColor: [255, 255, 255, 200],
    getRadius: d => Math.max((d.frp || 10) * 1.5, 300),
  });

  // -------------------------------------------------------------------------
  // INTERFACE UTILISATEUR & RENDU
  // -------------------------------------------------------------------------
  return (
    <div className="relative w-screen h-screen bg-slate-950 overflow-hidden font-sans">

      {/* HUD : Monitoring OSINT */}
      <div className="absolute top-4 left-4 z-20 bg-slate-900/90 backdrop-blur-md border border-slate-700/60 p-5 rounded-xl text-white shadow-2xl w-80">
        <h1 className="text-lg font-bold flex items-center gap-2 text-rose-500">
          <span>🔥</span> Tunisia Fire Watch
        </h1>
        <p className="text-xs text-slate-400 mt-1 mb-4">
          Modélisation prédictive & surveillance satellitaire
        </p>

        <div className="space-y-3">
          <div>
            <div className="flex justify-between text-xs font-semibold mb-1">
              <span>Seuil de vigilance IA</span>
              <span className="text-rose-400">{seuilRisque}%</span>
            </div>
            <input
              type="range" min="60" max="95" value={seuilRisque}
              onChange={(e) => setSeuilRisque(Number(e.target.value))}
              className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-rose-500"
            />
          </div>

          <div className="pt-2 border-t border-slate-800 text-xs flex justify-between text-slate-300">
            <span>Zones sous vigilance (J+1) :</span>
            <span className="font-bold text-amber-400">{filteredPredictions.length}</span>
          </div>

          <div className="text-xs flex justify-between text-slate-300">
            <span>Foyers thermiques actifs :</span>
            <span className="font-bold text-rose-400">{realtimeData.length}</span>
          </div>
        </div>

        {loading && (
          <div className="mt-3 text-xs text-cyan-400 animate-pulse">
            Synchronisation des flux géospatiaux...
          </div>
        )}
      </div>

      {/* HUD : Navigation Cartographique */}
      <div className="absolute top-4 right-4 z-20 flex flex-col bg-slate-900/90 backdrop-blur-md border border-slate-700/60 rounded-xl shadow-2xl overflow-hidden">
        <button onClick={handleZoomIn} title="Zoomer" className="w-10 h-10 flex items-center justify-center text-white text-lg font-bold hover:bg-slate-800 transition border-b border-slate-700/60 active:bg-slate-700">+</button>
        <button onClick={handleZoomOut} title="Dézoomer" className="w-10 h-10 flex items-center justify-center text-white text-lg font-bold hover:bg-slate-800 transition border-b border-slate-700/60 active:bg-slate-700">-</button>
        <button onClick={handleResetView} title="Réinitialiser la vue (Tunisie)" className="w-10 h-10 flex items-center justify-center text-rose-400 hover:bg-slate-800 transition active:bg-slate-700 text-xs">🏠</button>
      </div>

      {/* Rendu principal WebGL : Superposition des données et du fond de carte */}
      <div style={{ position: 'absolute', top: 0, left: 0, width: '100vw', height: '100vh', zIndex: 0 }}>
        <DeckGL
          viewState={viewState}
          onViewStateChange={e => setViewState(e.viewState)}
          controller={true}
          layers={[predictionLayer, realtimeLayer]}
          getTooltip={({ object }) => {
            if (!object) return null;

            // Info-bulle : Colonnes XGBoost
            if (object.risque_prob !== undefined) {
              return {
                html: `
                  <div style="background-color: #0f172a; color: #f8fafc; padding: 10px 14px; border-radius: 8px; font-size: 12px; line-height: 1.5; border: 1px solid #334155; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);">
                    <div style="font-weight: bold; color: #fb7185; border-bottom: 1px solid #334155; padding-bottom: 4px; margin-bottom: 6px;">
                      🔥 Indice de Risque (Modèle IA) : ${Number(object.risque_prob).toFixed(1)}%
                    </div>
                    <div>📍 Coordonnées : <b>${Number(object.lat).toFixed(3)}, ${Number(object.lon).toFixed(3)}</b></div>
                    <div>⛰️ Altitude réelle : <b>${object.elevation_m ?? '-'} m</b></div>
                    <div>🌡️ Température max : <b>${object.t_max ?? '-'} °C</b></div>
                    <div>💧 Humidité relative : <b>${object.h_mean ?? '-'} %</b></div>
                    <div>💨 Vitesse du vent : <b>${object.wind_max ?? '-'} km/h</b></div>
                    <div>🌧️ Précipitations : <b>${object.precip_sum ?? 0} mm</b></div>
                    ${object.ndvi !== undefined ? `<div>🌿 Biomasse (NDVI) : <b>${Number(object.ndvi).toFixed(2)}</b></div>` : ''}
                    ${object.ndwi !== undefined ? `<div>💦 Stress hydrique (NDWI) : <b>${Number(object.ndwi).toFixed(2)}</b></div>` : ''}
                    <div style="margin-top: 6px; padding-top: 4px; border-top: 1px dashed #334155; color: #38bdf8; font-size: 11px;">
                      Modélisation prédictive (XGBoost + MODIS)
                    </div>
                  </div>
                `
              };
            }

            // Info-bulle : Foyers FIRMS avec jointure spatiale
            if (object.frp !== undefined) {
              const frpVal = Number(object.frp || 0);
              const severityText = frpVal > 30 ? 'Intense (Critique)' : frpVal > 10 ? 'Modéré' : 'Faible';

              let meteoProche = null;
              if (predictionData && predictionData.length > 0) {
                let minDistance = Infinity;
                predictionData.forEach(p => {
                  const dist = Math.pow(p.lat - object.latitude, 2) + Math.pow(p.lon - object.longitude, 2);
                  if (dist < minDistance) {
                    minDistance = dist;
                    meteoProche = p;
                  }
                });
              }

              return {
                html: `
                  <div style="background-color: #0f172a; color: #f8fafc; padding: 10px 14px; border-radius: 8px; font-size: 12px; line-height: 1.5; border: 1px solid #e11d48; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);">
                    <div style="font-weight: bold; color: #f43f5e; border-bottom: 1px solid #e11d48; padding-bottom: 4px; margin-bottom: 6px;">
                      🔥 Foyer Actif Détecté (NASA FIRMS)
                    </div>
                    <div>📍 Gouvernorat : <b>${object.gouvernorat || 'Secteur forestier'}</b></div>
                    <div>⚡ Puissance radiative (FRP) : <b>${object.frp} MW (${severityText})</b></div>
                    <div>🛡️ Indice de confiance : <b>${formatConfidence(object.confidence)}</b></div>
                    <div>🛰️ Coordonnées GPS : <b>${Number(object.latitude).toFixed(4)}, ${Number(object.longitude).toFixed(4)}</b></div>
                    
                    ${meteoProche ? `
                    <div style="margin-top: 6px; padding-top: 6px; border-top: 1px dashed #e11d48; color: #cbd5e1; font-size: 11px;">
                      <div style="color: #38bdf8; margin-bottom: 2px;">Données environnementales de la zone :</div>
                      <div>🌡️ Température max : <b>${meteoProche.t_max ?? '-'} °C</b></div>
                      <div>💨 Vitesse du vent : <b>${meteoProche.wind_max ?? '-'} km/h</b></div>
                      <div>⛰️ Altitude : <b>${meteoProche.elevation_m ?? '-'} m</b></div>
                      <div>🌿 Biomasse (NDVI) : <b>${Number(meteoProche.ndvi).toFixed(2) ?? '-'}</b></div>
                    </div>
                    ` : ''}

                    <div style="margin-top: 6px; padding-top: 4px; border-top: 1px dashed #334155; color: #fb7185; font-size: 11px;">
                      Surveillance satellitaire en temps réel (VIIRS / MODIS)
                    </div>
                  </div>
                `
              };
            }
          }}
        >
          {/* 
            LE FIX EST ICI :
            L'opérateur spread {...viewState} transmet les coordonnées de la caméra (Lat 35, Lon 9)
            de Deck.gl vers MapLibre. Sans cela, MapLibre reste centré à Lat 0, Lon 0 (Océan Atlantique),
            ce qui explique le fond totalement noir.
          */}
          <Map
            {...viewState}
            reuseMaps
            mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
          />
        </DeckGL>
      </div>
    </div>
  );
}