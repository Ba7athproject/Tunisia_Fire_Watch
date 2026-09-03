import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Lire le gros fichier issu de VIIRS
df = pd.read_csv("dataset_coordonnees_cibles.csv")
df['acq_date'] = pd.to_datetime(df['acq_date'])
df['mois'] = df['acq_date'].dt.month

df_pos = df[df['incendie'] == 1]
df_neg = df[df['incendie'] == 0]

# 2. Extraire 2500 feux et 5500 situations normales (stratifié par mois)
df_pos_sample, _ = train_test_split(df_pos, train_size=2500, stratify=df_pos['mois'], random_state=42)
df_neg_sample, _ = train_test_split(df_neg, train_size=5500, stratify=df_neg['mois'], random_state=42)

# 3. Assembler et sauvegarder
df_final = pd.concat([df_pos_sample, df_neg_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
df_final = df_final.drop(columns=['mois'])
df_final['acq_date'] = df_final['acq_date'].dt.strftime('%Y-%m-%d')

df_final.to_csv("dataset_echantillon_ml.csv", index=False)
print(f"✔ Échantillon créé : {len(df_final)} lignes prêtes pour la météo.")