import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def generer_donnees_entrainement(n_samples=5000):
    """
    Génère un jeu de données historique factice basé sur la climatologie tunisienne.
    Pour une enquête réelle, remplacez cette fonction par la lecture de votre CSV historique 
    (ex: pd.read_csv('historique_incendies_tunisie_2015_2025.csv'))
    """
    logging.info("Génération du dataset d'entraînement historique...")
    np.random.seed(42)
    
    # Génération de cas SANS incendie (Target = 0) - Conditions normales
    n_neg = int(n_samples * 0.7) # 70% du temps, il n'y a pas de feu
    df_neg = pd.DataFrame({
        't_max': np.random.normal(28, 5, n_neg),      # Températures plus clémentes
        'h_mean': np.random.normal(55, 10, n_neg),     # Humidité moyenne
        'wind_max': np.random.normal(15, 8, n_neg),    # Vent modéré
        'precip_sum': np.random.exponential(2, n_neg), # Quelques pluies
        'ndvi': np.random.normal(0.4, 0.1, n_neg),     # Végétation normale
        'ndwi': np.random.normal(0.1, 0.1, n_neg),     # Pas de stress hydrique sévère
        'incendie': 0
    })

    # Génération de cas AVEC incendie (Target = 1) - Conditions extrêmes (Sirocco, sécheresse)
    n_pos = n_samples - n_neg # 30% de cas de feux
    df_pos = pd.DataFrame({
        't_max': np.random.normal(40, 4, n_pos),       # Fortes chaleurs
        'h_mean': np.random.normal(25, 8, n_pos),      # Air très sec
        'wind_max': np.random.normal(30, 10, n_pos),   # Vents forts
        'precip_sum': np.random.exponential(0.1, n_pos), # Sécheresse
        'ndvi': np.random.normal(0.25, 0.1, n_pos),    # Végétation sèche (carburant)
        'ndwi': np.random.normal(-0.2, 0.1, n_pos),    # Stress hydrique fort
        'incendie': 1
    })

    df_historique = pd.concat([df_neg, df_pos]).sample(frac=1).reset_index(drop=True)
    
    # Nettoyage des valeurs aberrantes générées aléatoirement
    df_historique['h_mean'] = df_historique['h_mean'].clip(5, 100)
    df_historique['precip_sum'] = df_historique['precip_sum'].clip(0, 100)
    
    return df_historique

def entrainer_modele():
    # 1. Chargement des données (SANS la variable FRP)
    df = generer_donnees_entrainement(10000)
    
    X = df[['t_max', 'h_mean', 'wind_max', 'precip_sum', 'ndvi', 'ndwi']]
    y = df['incendie']

    # 2. Séparation Entraînement / Test (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    logging.info("Entraînement du modèle XGBoost Classifier...")
    
    # 3. Configuration du modèle XGBoost pour l'investigation
    # scale_pos_weight aide le modèle si les incendies sont rares par rapport aux jours normaux
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        scale_pos_weight=len(y[y==0]) / len(y[y==1]),
        random_state=42
    )

    # 4. Entraînement
    model.fit(X_train, y_train)

    # 5. Évaluation des performances
    y_pred = model.predict(X_test)
    logging.info("\n=== Rapport de Performance du Modèle ===")
    logging.info(f"Précision globale (Accuracy) : {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Pas de feu", "Incendie"]))
    
    # 6. Sauvegarde du nouveau modèle
    modele_path = "modele_xgboost_tunisia_fire.joblib"
    joblib.dump(model, modele_path)
    logging.info(f"✔ Nouveau modèle prédictif sauvegardé sous : {modele_path}")

    # 7. Importance des variables (Transparence de l'algorithme)
    importance = pd.DataFrame({
        'Variable': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n--- Importance des facteurs de risque ---")
    print(importance.to_string(index=False))

if __name__ == "__main__":
    entrainer_modele()