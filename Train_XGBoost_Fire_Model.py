import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import joblib

# Bibliothèques Machine Learning (XGBoost & Scikit-Learn)
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# -----------------------------------------------------------------------------
# Configuration et Traçabilité (Standards OSINT)
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DOSSIER_PROJET = r"C:\Ba7ath_project\Tunisia-fire-detection"
FICHIER_DATASET = os.path.join(DOSSIER_PROJET, "dataset_ml_impute.geojson")
FICHIER_HISTORIQUE = os.path.join(DOSSIER_PROJET, "historique_incendies_propre.csv")
MODELE_SORTIE = os.path.join(DOSSIER_PROJET, "modele_xgboost_tunisia_fire.joblib")

def preparer_donnees_apprentissage(chemin_dataset: str, chemin_historique: str):
    """
    Prépare et fusionne les features météo, végétation et l'historique 
    pour constituer la matrice d'entraînement du modèle.
    """
    logging.info("Chargement du dataset imputé et de l'historique...")
    if not os.path.exists(chemin_dataset) or not os.path.exists(chemin_historique):
        logging.error("Fichiers d'entrée introuvables. Veuillez vérifier les chemins.")
        return None, None

    # 1. Chargement des anomalies (Positifs : Label = 1)
    gdf_pos = gpd.read_file(chemin_dataset)
    gdf_pos['label'] = 1 # Présence d'un feu actif validé

    # 2. Chargement de l'historique pour pondérer le risque structurel par gouvernorat
    df_hist = pd.read_csv(chemin_historique)
    # Calcul de la moyenne des surfaces brûlées par gouvernorat comme proxy de vulnérabilité historique
    vulnerabilite_gov = df_hist.groupby('gov_latin')['superficie_ha'].mean().to_dict()

    # 3. Sélection des caractéristiques (Features) explicatives
    # Le modèle apprend à partir de la météo, de la sécheresse végétale et de la puissance radiative
    features_cols = ['t_max', 'h_mean', 'wind_max', 'precip_sum', 'ndvi', 'ndwi', 'frp']
    
    # Extraction des données positives
    X_pos = gdf_pos[features_cols].copy()
    y_pos = gdf_pos['label'].copy()

    # 4. Génération de contre-exemples synthétiques (Négatifs : Label = 0)
    # Pour que le modèle comprenne ce qu'est un jour SANS feu, on perturbe légèrement les données 
    # avec des conditions météo plus clémentes et une végétation saine.
    logging.info("Génération de contre-exemples pour l'apprentissage binaire...")
    np.random.seed(42)
    n_neg = len(X_pos)
    
    X_neg = pd.DataFrame({
        't_max': np.random.uniform(25.0, 34.0, n_neg),      # Températures modérées
        'h_mean': np.random.uniform(50.0, 85.0, n_neg),     # Humidité élevée
        'wind_max': np.random.uniform(5.0, 12.0, n_neg),    # Vent faible
        'precip_sum': np.random.uniform(1.0, 15.0, n_neg),  # Présence de pluie
        'ndvi': np.random.uniform(0.4, 0.7, n_neg),         # Végétation dense et verte
        'ndwi': np.random.uniform(0.1, 0.4, n_neg),         # Sol humide
        'frp': 0.0                                          # Pas d'anomalie thermique
    })
    y_neg = pd.Series([0] * n_neg)

    # 5. Fusion des matrices X et y
    X = pd.concat([X_pos, X_neg], ignore_index=True)
    y = pd.concat([y_pos, y_neg], ignore_index=True)

    logging.info(f"Dataset prêt pour XGBoost : {len(X)} échantillons au total ({len(X_pos)} feux, {n_neg} non-feux).")
    return X, y

def entrainer_modele_xgboost(X, y, chemin_modele_sortie):
    """
    Entraîne un classificateur XGBoost avec séparation temporelle/aléatoire 
    et évalue ses performances.
    """
    # 1. Séparation Train / Test (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logging.info("Entraînement du modèle XGBoost Classifier en cours...")
    
    # 2. Configuration des hyperparamètres du modèle
    # XGBoost est choisi pour sa robustesse face aux données tabulaires hétérogènes
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    # 3. Ajustement (Fitting)
    model.fit(X_train, y_train)

    # 4. Évaluation des performances
    logging.info("Évaluation des performances sur l'ensemble de test...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Rapport de Classification ---")
    print(classification_report(y_test, y_pred))
    
    auc_score = roc_auc_score(y_test, y_proba)
    logging.info(f"Score AUC-ROC du modèle : {auc_score:.4f}")

    # 5. Sauvegarde du modèle entraîné pour l'application web
    joblib.dump(model, chemin_modele_sortie)
    logging.info(f"Modèle sauvegardé avec succès dans : {chemin_modele_sortie}")
    
    return model

# -----------------------------------------------------------------------------
# Point d'entrée principal
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logging.info("--- Démarrage du pipeline Machine Learning Tunisia Fire Watch ---")
    
    X_data, y_data = preparer_donnees_apprentissage(FICHIER_DATASET, FICHIER_HISTORIQUE)
    
    if X_data is not None and not X_data.empty:
        entrainer_modele_xgboost(X_data, y_data, MODELE_SORTIE)
        
    logging.info("--- Fin du processus d'entraînement ---")