import pandas as pd
from unidecode import unidecode

# Fonction pour retirer les accents
def remove_accents(text):
    return unidecode(text)

# Chemin vers le fichier CSV d'entrée
#input_csv_path = 'testingcode3.csv'

#input_csv_path = 'train.csv'

#input_csv_path = 'dev.csv'

#input_csv_path = 'test.csv'

#input_csv_path = 'Note_train_0_5_processed.csv'

#input_csv_path = 'Note_train_1_0_processed.csv'

input_csv_path = 'Note_train_1_5_processed.csv'

# Charger le fichier CSV en utilisant pandas
df = pd.read_csv(input_csv_path, delimiter=';')

# Appliquer la fonction remove_accents à la colonne 'Commentaire'
df['Commentaire'] = df['Commentaire'].apply(remove_accents)

# Sauvegarder le DataFrame modifié dans un nouveau fichier CSV
df.to_csv(input_csv_path, index=False, sep=';')

print(f"Le fichier modifié a été enregistré sous {input_csv_path}")
