# **Documentation Scientifique et Méthodologique : Tunisia Fire Watch**

*Projet ba7ath*

## **1\. Objectif et Cadre Éthique (OSINT)**

Le système Tunisia Fire Watch est une plateforme de datajournalisme spatial conçue pour anticiper et cartographier les risques d'incendies sur le territoire tunisien. La méthodologie repose exclusivement sur l'exploitation de sources ouvertes (OSINT) légales et publiques. L'approche garantit une traçabilité totale de la chaîne de vérification, de l'acquisition du pixel satellitaire brut jusqu'à l'inférence de la probabilité de risque, sans aucune boîte noire.

## **2\. Nomenclature des Sources de Données (Open Data)**

| Source / Capteur | Fournisseur | Résolution | Fréquence | Rôle dans l'Architecture |
| :---- | :---- | :---- | :---- | :---- |
| **ERA5 (Open-Meteo)** | ECMWF | \~11 km | Horaire (Historique & Prévisions) | Fournit le cadre atmosphérique (Température maximale, Humidité moyenne, Vent maximum, Précipitations). |
| **MODIS (MOD09A1.061)** | NASA / Microsoft Planetary Computer | 500 m | 8 jours (Composite) | Permet le calcul de la biomasse (NDVI) et du stress hydrique (NDWI) via la réflectance de surface. |
| **VIIRS (VNP14IMGTDL\_NRT)** | NASA FIRMS | 375 m | Temps réel (NRT) | Sert de vérité terrain (cibles) pour l'entraînement du modèle et le suivi en direct des anomalies thermiques. |

## 

## **3\. Ingénierie des Données et Prévention des Biais (Feature Engineering)**

La préparation des données a été structurée pour isoler les facteurs causaux et empêcher toute fuite de données (*Target Leakage*).

* **Exclusion des marqueurs post-ignition :** Les variables satellitaires mesurant l'intensité d'un feu déjà déclaré (FRP \- *Fire Radiative Power*, Température de brillance *Bright\_ti4* / *Bright\_ti5*) ont été rigoureusement exclues du jeu de données d'apprentissage. Le modèle apprend uniquement des conditions environnementales préalables.  
    
* **Calcul des indices géobotaniques :**  
  

  * **NDVI** (Densité végétale) : $\\frac{NIR \- RED}{NIR \+ RED}$  
      
  * **NDWI** (Stress hydrique) : $\\frac{NIR \- SWIR}{NIR \+ SWIR}$  
      
* **Filtre absolu de biomasse combustible :** Le système de production applique un masque spatial strict ($NDVI \\ge 0.30$). Cette règle physique empêche le modèle de calculer un risque d'incendie sur des zones minérales, urbaines, ou désertiques (chotts, sebkhas) ne disposant pas du combustible nécessaire à la propagation d'un feu de forêt.


## **4\. Modélisation Prédictive (Machine Learning)**

L'algorithme de classification supervisée retenu est **XGBoost** (Extreme Gradient Boosting), sélectionné pour sa robustesse face aux données non linéaires et aux jeux de données déséquilibrés.

* **Hyperparamètres de base :** n\_estimators=200, learning\_rate=0.05, max\_depth=6.  
    
* **Gestion du déséquilibre des classes :** Utilisation du paramètre scale\_pos\_weight pour pondérer mathématiquement les rares événements positifs (jours d'incendie) face à l'écrasante majorité des jours normaux.  
    
* **Importance des variables (Feature Importance) :** Le modèle a identifié l'humidité moyenne (27.9%) et la densité végétale (26.9%) comme les facteurs de déclenchement primaires, validant la pertinence physique de l'algorithme face au climat tunisien.


## **5\. Classification Opérationnelle et Seuillage**

Pour éviter la saturation de l'interface par un bruit de fond statistique, les probabilités brutes générées par l'algorithme sont traduites en classes de vigilance exploitables par les journalistes et analystes :

* **Vigilance Jaune (Risque Modéré) :** Probabilité comprise entre $65\\%$ et $74.9\\%$.  
    
* **Alerte Orange (Risque Élevé) :** Probabilité comprise entre $75\\%$ et $84.9\\%$.  
    
* **Alerte Rouge (Risque Extrême) :** Probabilité $\\ge 85\\%$.  
    
* Toute probabilité inférieure à $65\\%$ est considérée comme non significative et exclue du rendu final.


## **6\. Architecture de Déploiement et Optimisation (Haute Performance)**

* **Format d'échange :** Remplacement du format vectoriel lourd (GeoJSON) par un format tabulaire plat (CSV) stockant les centroïdes géographiques. Ce changement structurel élimine la surcharge de la mémoire du navigateur client.  
    
* **Rendu Spatial :** Utilisation de la bibliothèque PyDeck (WebGL) via Streamlit pour un rendu 3D immédiat de plusieurs dizaines de milliers de points.  
    
* **Automatisation (CRON) :** Pipeline hébergé sur GitHub Actions exécutant une boucle de collecte (Data Ingestion) et d'inférence (Predictive Mapping) toutes les 6 heures, avec écriture automatisée dans le dépôt public pour rafraîchissement transparent du tableau de bord.


## **7\. Perspectives et Améliorations Futures**

Le protocole scientifique prévoit l'intégration de nouvelles variables pour affiner l'inférence :

1. **Topographie (MNT) :** Intégration de l'élévation et du degré de pente via la mission SRTM (30m) pour anticiper les dynamiques de convection.  
     
2. **Facteurs Anthropiques :** Mesure de la distance spatiale aux infrastructures humaines (réseaux routiers, lignes électriques, zones agricoles) via OpenStreetMap.  
     
3. **Profondeur temporelle :** Calcul des anomalies thermiques cumulées (jours consécutifs de sécheresse).  
   