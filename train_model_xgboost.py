import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configuration de la journalisation pour documenter la chaîne de vérification
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def entrainer_et_evaluer_modele(fichier_dataset, fichier_modele):
    logging.info(f"Chargement du dataset final : {fichier_dataset}")
    try:
        df = pd.read_csv(fichier_dataset)
    except FileNotFoundError:
        logging.error(f"Fichier {fichier_dataset} introuvable.")
        return

    # 1. Sélection stricte des variables prédictives (Features) et de la cible (Target)
    # On exclut volontairement toute variable de confirmation thermique (FRP, brightness)
    colonnes_predictives = ['t_max', 'h_mean', 'wind_max', 'precip_sum', 'ndvi', 'ndwi']
    X = df[colonnes_predictives]
    y = df['incendie']

    # 2. Séparation des données : 80% pour l'apprentissage, 20% pour le test aveugle
    # Le paramètre stratify garantit que la proportion feux/sans-feux reste identique
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logging.info(f"Répartition - Entraînement : {len(X_train)} lignes | Test : {len(X_test)} lignes")

    # 3. Initialisation et configuration de l'algorithme XGBoost
    # scale_pos_weight aide à gérer d'éventuels déséquilibres entre cas positifs et négatifs
    ratio_desequilibre = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    modele_xgb = xgb.XGBClassifier(
        n_estimators=200,          # Nombre d'arbres de décision
        learning_rate=0.05,        # Vitesse d'apprentissage (plus bas = plus précis mais plus lent)
        max_depth=6,               # Profondeur maximale des arbres pour éviter le surapprentissage (overfitting)
        scale_pos_weight=ratio_desequilibre,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # 4. Entraînement du modèle
    logging.info("Démarrage de l'entraînement du modèle XGBoost...")
    modele_xgb.fit(X_train, y_train)
    logging.info("✔ Entraînement terminé.")

    # 5. Évaluation des performances sur les données de test (jamais vues par le modèle)
    predictions = modele_xgb.predict(X_test)
    
    logging.info("\n--- RAPPORT DE PERFORMANCE ---")
    precision_globale = accuracy_score(y_test, predictions)
    logging.info(f"Précision globale (Accuracy) : {precision_globale * 100:.2f}%\n")
    print(classification_report(y_test, predictions, target_names=['Jour Normal (0)', 'Incendie (1)']))

    # 6. Extraction de l'importance des variables (Feature Importance)
    # Permet d'expliquer pédagogiquement quels facteurs pèsent le plus dans un départ de feu
    importances = modele_xgb.feature_importances_
    df_importances = pd.DataFrame({'Variable': colonnes_predictives, 'Importance': importances})
    df_importances = df_importances.sort_values(by='Importance', ascending=False)
    
    logging.info("\n--- IMPORTANCE DES VARIABLES CLIMATIQUES ET VÉGÉTALES ---")
    for idx, row in df_importances.iterrows():
        logging.info(f"{row['Variable']:>12} : {row['Importance']*100:.1f}%")

    # 7. Sauvegarde du modèle entraîné
    modele_xgb.save_model(fichier_modele)
    logging.info(f"\n✔ Modèle prédictif sauvegardé avec succès sous : {fichier_modele}")
    logging.info("Ce fichier peut désormais être chargé dans le dashboard Streamlit ou le script d'inférence quotidien.")

if __name__ == "__main__":
    entrainer_et_evaluer_modele(
        fichier_dataset="dataset_complet_ml_final.csv",
        fichier_modele="modele_xgboost_incendies_tunisie.json"
    )