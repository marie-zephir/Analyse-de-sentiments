import pandas as pd
import random

# Chargement du premier CSV
#premier_csv_path = 'Note_train_0_5_processed.csv'
#premier_csv_path = 'Note_train_1_0_processed.csv'
premier_csv_path = 'Note_train_1_5_processed.csv'

df_premier = pd.read_csv(premier_csv_path, delimiter=';')

# Chargement du deuxième CSV
#deuxieme_csv_path = 'train.csv'
deuxieme_csv_path = 'dev.csv'
df_deuxieme = pd.read_csv(deuxieme_csv_path, delimiter=';')

# Nombre n de lignes à ajouter de manière aléatoire
#n = 11669 # Pour classe 0.5 dans train.csv
#n = 19779 # Pour classe 1.0 dans train.csv
#n = 23064 # Pour classe 1.5 dans train.csv
#n = 2169 # Pour classe 0.5 dans dev.csv
#n = 3458 # Pour classe 1.0 dans dev.csv
n = 4040 # Pour classe 1.5 dans dev.csv
donnees_a_ajouter = df_premier.head(n)

indices_deuxieme = list(df_deuxieme.index)
random.shuffle(indices_deuxieme)

# Insérez les lignes sélectionnées de manière aléatoire dans le deuxième CSV
for index, row in donnees_a_ajouter.iterrows():
    indice_insertion = random.choice(indices_deuxieme)
    df_deuxieme = pd.concat([df_deuxieme.iloc[:indice_insertion], row.to_frame().T, df_deuxieme.iloc[indice_insertion:]], ignore_index=True)

# Enlevez les lignes ajoutées du premier CSV
df_premier = df_premier.iloc[n:]

# Sauvegardez les modifications dans les fichiers CSV
df_premier.to_csv(premier_csv_path, index=False, sep=';')
df_deuxieme.to_csv(deuxieme_csv_path, index=False, sep=';')
