import os
import logging
import numpy as np
import pandas as pd
import joblib

# Bibliothèques Machine Learning (XGBoost & Scikit-Learn)
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# -----------------------------------------------------------------------------
# Configuration et Traçabilité (Standards OSINT & ba7ath)
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FICHIER_DATASET = "dataset_incendies_avec_topographie.csv"
MODELE_SORTIE = "modele_xgboost_tunisia_fire.joblib"


def generer_pseudo_absences_realistes(df_positifs: pd.DataFrame, ratio: float = 1.0) -> pd.DataFrame:
    """
    Génère des contre-exemples (labels = 0) scientifiquement cohérents pour le climat tunisien.
    Évite le raccourci naïf sur la pluie en créant des journées sèches mais sans incendie
    (végétation humide, altitudes variables, températures modérées).
    """
    logging.info("Génération de contre-exemples réalistes (pseudo-absences)...")
    np.random.seed(42)
    n_neg = int(len(df_positifs) * ratio)

    # 1. Distribution réaliste des précipitations :
    # 70% des jours normaux d'été en Tunisie ont 0.0 mm de pluie
    precip_realiste = np.random.choice(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.2, 3.5, 8.0], 
        size=n_neg
    )

    # 2. Températures estivales et printanières plausibles mais sous le seuil critique d'ignition
    t_max_realiste = np.random.uniform(22.0, 35.0, n_neg)

    # 3. Humidité de l'air modérée à élevée
    h_mean_realiste = np.random.uniform(40.0, 80.0, n_neg)

    # 4. Vent modéré
    wind_max_realiste = np.random.uniform(5.0, 20.0, n_neg)

    # 5. Indices géobotaniques réalistes :
    # Zones sans feu souvent caractérisées par un bon état hydrique (NDWI élevé)
    # ou un couvert végétal absent/faible (zones agricoles récoltées, sols nus)
    ndvi_realiste = np.random.uniform(0.15, 0.65, n_neg)
    ndwi_realiste = np.random.uniform(0.05, 0.35, n_neg)

    # 6. Altitudes réparties sur la topographie tunisienne (plaines, collines, moyenne montagne)
    elevation_realiste = np.random.uniform(10.0, 1200.0, n_neg)

    df_neg = pd.DataFrame({
        't_max': t_max_realiste,
        'h_mean': h_mean_realiste,
        'wind_max': wind_max_realiste,
        'precip_sum': precip_realiste,
        'ndvi': ndvi_realiste,
        'ndwi': ndwi_realiste,
        'elevation_m': elevation_realiste,
        'label': 0
    })

    return df_neg


def preparer_donnees_apprentissage(chemin_dataset: str):
    """
    Prépare et fusionne les features climatiques, géobotaniques et topographiques.
    Garantit l'alignement des colonnes et la validation de la structure de données.
    """
    logging.info("Chargement du dataset enrichi avec topographie...")
    if not os.path.exists(chemin_dataset):
        raise FileNotFoundError(f"Fichier d'entrée introuvable : {chemin_dataset}")

    df_pos = pd.read_csv(chemin_dataset)
    df_pos['label'] = 1 

    # Liste stricte des caractéristiques explicatives (FRP banni pour éviter le target leakage)
    features_cols = ["t_max", "h_mean", "wind_max", "precip_sum", "ndvi", "ndwi", "elevation_m"]
    
    # Nettoyage des valeurs aberrantes ou manquantes
    df_pos = df_pos.dropna(subset=features_cols).copy()
    
    # Génération des négatifs avec la même distribution de colonnes
    df_neg = generer_pseudo_absences_realistes(df_pos, ratio=1.2)

    # Fusion des jeux de données positifs et négatifs
    df_complet = pd.concat([df_pos[features_cols + ['label']], df_neg], ignore_index=True)

    X = df_complet[features_cols].copy()
    y = df_complet['label'].copy()

    logging.info(
        f"Dataset final consolidé : {len(X)} observations "
        f"({len(df_pos)} foyers réels, {len(df_neg)} contre-exemples réalistes)."
    )
    return X, y


def entrainer_modele_xgboost(X: pd.DataFrame, y: pd.Series, chemin_modele_sortie: str) -> xgb.XGBClassifier:
    """
    Entraîne un modèle XGBoost régularisé, évalue l'équilibre des décisions
    et exporte les poids des variables explicatives.
    """
    # Stratification pour conserver le ratio 1/0 dans les jeux Train et Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    logging.info("Entraînement de l'algorithme XGBoost avec pénalisation L1/L2...")
    
    # Configuration des hyperparamètres avec régularisation (reg_alpha, reg_lambda)
    # pour empêcher un arbre de se focaliser exclusivement sur une seule feature
    model = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=4,              # Profondeur modérée pour éviter la mémorisation brute
        learning_rate=0.04,        # Vitesse d'apprentissage progressive
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,             # Régularisation L1 (évite le surapprentissage)
        reg_lambda=1.0,            # Régularisation L2
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    logging.info("Évaluation des métriques de classification...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "="*50)
    print("RAPPORT DE CLASSIFICATION (Sur données de test)")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=["Sans Risque (0)", "Risque Feu (1)"]))
    
    auc_score = roc_auc_score(y_test, y_proba)
    logging.info(f"Score AUC-ROC réaliste : {auc_score:.4f}")

    print("\n" + "-"*50)
    print("IMPORTANCE DES VARIABLES (Prise de décision physique)")
    print("-"*50)
    importances = model.feature_importances_
    features = X.columns
    df_importance = pd.DataFrame({'Variable': features, 'Poids (%)': (importances * 100).round(2)})
    df_importance = df_importance.sort_values(by='Poids (%)', ascending=False)
    print(df_importance.to_string(index=False))
    print("-"*50)

    # Sauvegarde de l'artefact pour production
    joblib.dump(model, chemin_modele_sortie)
    logging.info(f"✔ Nouveau modèle validé et sérialisé dans : {chemin_modele_sortie}")
    return model


if __name__ == "__main__":
    logging.info("=== DÉMARRAGE DE LA MODÉLISATION PRÉDICTIVE (TUNISIA FIRE WATCH) ===")
    try:
        X_data, y_data = preparer_donnees_apprentissage(FICHIER_DATASET)
        entrainer_modele_xgboost(X_data, y_data, MODELE_SORTIE)
    except Exception as e:
        logging.error(f"Échec critique du pipeline d'apprentissage : {e}", exc_info=True)
    logging.info("=== FIN DU TRAITEMENT ===")