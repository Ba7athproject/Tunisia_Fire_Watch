import pandas as pd
import logging
import os

# -----------------------------------------------------------------------------
# Configuration OSINT
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DOSSIER_PROJET = r"C:\Ba7ath_project\Tunisia-fire-detection"
FICHIER_ENTREE = os.path.join(DOSSIER_PROJET, "incendies-foret-2002-2023-1.xlsx")
FICHIER_SORTIE = os.path.join(DOSSIER_PROJET, "historique_incendies_propre.csv")

def nettoyer_donnees_historiques(chemin_entree: str, chemin_sortie: str):
    """
    Charge, audite, nettoie et standardise la base de données gouvernementale 
    des incendies pour la rendre exploitable par les modèles spatiaux.
    """
    if not os.path.exists(chemin_entree):
        logging.error(f"Fichier introuvable : {chemin_entree}")
        return

    logging.info(f"Chargement du dataset historique : {chemin_entree}")
    
    try:
        # Lecture du fichier Excel (Pandas gère nativement le moteur openpyxl)
        df = pd.read_excel(chemin_entree, sheet_name=0)
        
        # 1. Traitement des valeurs manquantes (Imputation)
        # On sait que Gafsa 2021 a un NaN avec 0ha brûlé.
        valeurs_nulles = df.isnull().sum().sum()
        if valeurs_nulles > 0:
            logging.info(f"Correction de {valeurs_nulles} valeur(s) manquante(s) dans le jeu de données.")
            df['nombre'] = df['nombre'].fillna(0)
            
        # 2. Conversion de type (Float -> Int)
        df['nombre'] = df['nombre'].astype(int)
        
        # 3. Standardisation toponymique (Arabe -> Latin) pour jointure GIS
        dictionnaire_gov = {
            'تونس': 'Tunis', 'أريانة': 'Ariana', 'بن عروس': 'Ben Arous', 'منوبة': 'Manouba',
            'نابل': 'Nabeul', 'زغوان': 'Zaghouan', 'بنزرات': 'Bizerte', 'باجة': 'Beja',
            'جندوبة': 'Jendouba', 'الكاف': 'Kef', 'سليانة': 'Siliana', 'القصرين': 'Kasserine',
            'سيدي بوزيد': 'Sidi Bouzid', 'القيروان': 'Kairouan', 'قفصة': 'Gafsa', 'سوسة': 'Sousse',
            'صفاقس': 'Sfax', 'المهدية': 'Mahdia', 'المنستير': 'Monastir', 'قابس': 'Gabes',
            'توزر': 'Tozeur', 'تطاوين': 'Tataouine', 'مدنين': 'Medenine', 'قبلي': 'Kebili'
        }
        
        # Ajout d'une colonne standardisée sans écraser la donnée source originale (Principe de transparence)
        df['gov_latin'] = df['gouvernorat'].map(dictionnaire_gov)
        
        # Vérification des correspondances
        gov_non_mappes = df[df['gov_latin'].isnull()]
        if not gov_non_mappes.empty:
            logging.warning(f"Attention, certains toponymes n'ont pas été traduits : {gov_non_mappes['gouvernorat'].unique()}")
        
        # 4. Restructuration finale
        # On réordonne les colonnes pour plus de lisibilité
        colonnes_finales = ['annee', 'gov_latin', 'gouvernorat', 'nombre', 'superficie_ha']
        df_propre = df[colonnes_finales].copy()
        
        # 5. Sauvegarde en CSV (format ouvert, léger et interopérable)
        df_propre.to_csv(chemin_sortie, index=False, encoding='utf-8')
        logging.info(f"Dataset nettoyé sauvegardé avec succès ({len(df_propre)} lignes) : {chemin_sortie}")
        
    except Exception as e:
        logging.error(f"Une erreur est survenue lors du nettoyage : {e}")

# -----------------------------------------------------------------------------
# Point d'exécution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    nettoyer_donnees_historiques(FICHIER_ENTREE, FICHIER_SORTIE)