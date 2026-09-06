## **Protocole Scientifique et Modélisation Prédictive : Tunisia Fire Watch**

La fiabilité d'une enquête basée sur les données spatiales repose sur la transparence absolue de sa méthodologie. Le modèle prédictif de **Tunisia Fire Watch** s'articule autour d'un algorithme de type Gradient Boosting (XGBoost) alimenté par un croisement de données satellitaires, météorologiques et topographiques.

### **1\. Sources des Données (Open Data et OSINT)**

L'entraînement du modèle repose sur un historique géospatial structuré, croisant les foyers d'incendie avérés avec les conditions environnementales antérieures.

* **Vérité Terrain (Target Variable) \- NASA FIRMS :** Les occurrences historiques de feux sont extraites des bases de données FIRMS (Fire Information for Resource Management System). Le système exploite les relevés des capteurs **VIIRS** (résolution spatiale de 375 m) et **MODIS** (1 km). La variable cible $Y$ est binaire, où $Y=1$ indique la détection d'une anomalie thermique avec un indice de confiance élevé (Nominal/High), et $Y=0$ représente l'absence d'anomalie.  
* **Conditions Météorologiques (Features Dynamiques) :** Les variables climatiques historiques et prédictives (à J+1) sont extraites via des API météorologiques ouvertes (comme Open-Meteo). Les variables retenues sont :  
  * $T\_{max}$ : Température maximale journalière (°C).  
  * $H\_{mean}$ : Humidité relative moyenne (%).  
  * $W\_{max}$ : Vitesse maximale du vent (km/h).  
  * $P\_{sum}$ : Cumul des précipitations journalières (mm).  
* **Indices Biologiques (Features Statiques/Saisonnières) :** Les données multispectrales issues des satellites Sentinel-2 ou Landsat 8/9 permettent de quantifier la biomasse et le stress hydrique.  
  * **NDVI (Normalized Difference Vegetation Index) :** Mesure la densité et la santé de la végétation.  
    $$NDVI \= \\frac{NIR \- RED}{NIR \+ RED}$$  
  * **NDWI (Normalized Difference Water Index) :** Évalue la teneur en eau de la végétation (vulnérabilité à la combustion).  
    $$NDWI \= \\frac{NIR \- SWIR}{NIR \+ SWIR}$$  
    *(NIR \= Proche Infrarouge, RED \= Rouge visible, SWIR \= Infrarouge à ondes courtes).*  
* **Topographie (Features Spatiales) :** Le modèle intègre un Modèle Numérique de Terrain (MNT/DEM) tel que SRTM (Shuttle Radar Topography Mission) pour extraire l'altitude ($E\_{m}$) et le degré de pente, la topographie influençant fortement la propagation des flammes.

### **2\. Ingénierie des Caractéristiques (Feature Engineering)**

Avant d'alimenter le modèle, les données brutes subissent un pipeline de transformation spatial et temporel strict :

1. **Maillage Géospatial (Grid Creation) :** Le territoire est découpé en une matrice de mailles hexagonales ou carrées (ex: H3 de Uber ou bounding boxes régulières).  
2. **Jointure Spatio-Temporelle :** Pour chaque maille et chaque jour $t$, les données météorologiques et les indices de végétation sont agrégés. Si un feu est détecté par FIRMS dans cette maille au jour $t$, la ligne est labellisée $Y=1$.  
3. **Gestion du Déséquilibre des Classes (Imbalance) :** Les jours avec incendies ($Y=1$) représentent une infime minorité par rapport aux jours sans incendies ($Y=0$). Le jeu de données est équilibré soit par suréchantillonnage (SMOTE), soit de manière algorithmique via l'hyperparamètre scale\_pos\_weight de XGBoost, calculé selon le ratio $\\frac{N\_{negatives}}{N\_{positives}}$.

### **3\. Modélisation Algorithmique : XGBoost**

L'algorithme **eXtreme Gradient Boosting (XGBoost)** a été privilégié en raison de sa performance supérieure sur les données tabulaires hétérogènes et sa capacité à modéliser des relations non linéaires complexes (comme l'interaction entre une forte pente, un vent fort et un faible NDWI).  
XGBoost est un ensemble d'arbres de décision construits séquentiellement. À chaque itération $t$, le modèle ajoute un nouvel arbre $f\_t$ qui tente de corriger les erreurs des arbres précédents en minimisant la fonction d'objectif suivante :

$$\\mathcal{L}^{(t)} \= \\sum\_{i=1}^n l(y\_i, \\hat{y}\_i^{(t-1)} \+ f\_t(x\_i)) \+ \\Omega(f\_t)$$

* $l$ représente la fonction de perte (Log Loss pour la classification binaire).  
* $\\Omega(f\_t) \= \\gamma T \+ \\frac{1}{2} \\lambda \\sum\_{j=1}^T w\_j^2$ est le terme de régularisation qui pénalise la complexité de l'arbre pour éviter le surapprentissage (overfitting).

Le modèle génère une probabilité continue $P(Y=1 \\vert{} X)$ comprise entre 0 et 1\. Cette probabilité est traduite dans le dashboard en risque\_prob (exprimé en pourcentage).

### **4\. Évaluation et Validation Scientifique**

L'exactitude (Accuracy) est une métrique trompeuse pour la détection d'incendies en raison du déséquilibre des classes. L'évaluation du modèle repose sur une matrice de confusion analysée à travers les métriques suivantes :

* **Rappel (Recall / Sensibilité) :** La capacité du modèle à identifier correctement tous les feux réels. Un rappel élevé est prioritaire en investigation pour minimiser les faux négatifs (zones à risque ignorées).  
* **Précision (Precision) :** La proportion de feux prédits qui se sont réellement déclarés, minimisant ainsi les fausses alertes.  
* **F1-Score :** La moyenne harmonique de la précision et du rappel.  
* **ROC-AUC (Area Under the Receiver Operating Characteristic Curve) :** Évalue la capacité globale du modèle à distinguer les classes positives et négatives à différents seuils de probabilité.

### **5\. Inférence en Production**

En phase d'exploitation (production), le pipeline est exécuté quotidiennement via GitHub Actions. Les prévisions météorologiques à J+1 sont récupérées, croisées avec la topographie et les derniers indices végétatifs, puis injectées dans le modèle XGBoost pré-entraîné. Les probabilités résultantes sont exportées dans le fichier carte\_risques\_demain\_reel.csv, prêt à être lu et rendu en 3D par l'interface React / Deck.gl.