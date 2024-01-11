import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import random
import csv

# Charger le CSV avec les données
#df = pd.read_csv('train_100_WR.csv', delimiter=';')
df = pd.read_csv('dev_100_WR.csv', delimiter=';')

# Compter le nombre de critiques pour chaque classe
nombre_critiques_par_classe = df['Note'].value_counts().to_dict()
print(nombre_critiques_par_classe)

# Obtenir le nombre de critiques pour la classe de référence
#nombre_critiques_reference = 49999 # Pour chaque classe du train
nombre_critiques_reference = 7999 # Pour chaque classe du dev

# Définir une fonction pour sous-échantillonner ou suréchantillonner une classe spécifique
def equilibrer_classe(classe, nombre_critiques_reference, df):
        df_classe = df[df['Note'] == classe]
        df_classe_sous_echantillonne = df_classe.head(nombre_critiques_reference)
        return df_classe_sous_echantillonne

# Appliquer la fonction à chaque classe
dfs_equilibres = [equilibrer_classe(classe, nombre_critiques_reference, df) for classe in nombre_critiques_par_classe.keys()]

# Concaténer les résultats en un seul dataframe
df_equilibre = pd.concat(dfs_equilibres, axis=0)

# Mélanger le dataframe final
df_equilibre = df_equilibre.sample(frac=1).reset_index(drop=True)

# Sauvegarder le résultat dans un fichier CSV avec le même format
#df_equilibre.to_csv('final_train.csv', sep=';', index=False)
df_equilibre.to_csv('final_dev.csv', sep=';', index=False)