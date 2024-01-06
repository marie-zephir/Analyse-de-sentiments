import pandas as pd

chemin_fichier = 'predictions.txt'

# Charger le fichier texte dans un DataFrame en spécifiant le délimiteur (espace dans ce cas)
df = pd.read_csv(chemin_fichier, delimiter=' ', header=None, names=['review', 'note'])

# Remplacer les points par des virgules dans la colonne 'note'
#df['note'] =df['note'].astype(float)/2
df['note'] =df['note'].astype(str).str.replace('.', ',')

df.to_csv(chemin_fichier, sep=' ', header=False, index=False)
