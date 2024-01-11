import pandas as pd
import re

with open('liste.de.mots.francais.txt', 'r', encoding='utf-8') as f:
    mots_francais_set = set(f.read().splitlines())

def est_mot_francais(mot):
    return mot.lower() in mots_francais_set

def corriger_mot(mot):
    # Utilise une expression régulière pour supprimer les lettres répétitives
    mot_corrigé = re.sub(r'(.)\1+', r'\1', mot)
    if not est_mot_francais(mot_corrigé):
        mot_corrigé = ''
    
    return mot_corrigé

def corriger_commentaire(commentaire):
    # Utiliser une fonction lambda pour appliquer les corrections et filtrer efficacement
    mots_corriges = commentaire.split()
    mots_corriges = [mot if est_mot_francais(mot) else corriger_mot(mot) for mot in mots_corriges]
    commentaire_corrigé = ' '.join(mots_corriges)
    commentaire_corrigé = re.sub(r'\s+', ' ', commentaire_corrigé).strip()
    # Utiliser join pour concaténer la liste résultante en une seule chaîne
    return commentaire_corrigé

#input_file_path = 'testingcode.csv'
#output_file_path = 'testingcode2.csv'

#input_file_path = 'train.csv'
#output_file_path = 'train.csv'

#input_file_path = 'dev.csv'
#output_file_path = 'dev.csv'

#input_file_path = 'test.csv'
#output_file_path = 'test.csv'

#input_file_path = 'Note_train_0_5_processed.csv'
#output_file_path = 'Note_train_0_5_processed.csv'

#input_file_path = 'Note_train_1_0_processed.csv'
#output_file_path = 'Note_train_1_0_processed.csv'

input_file_path = 'Note_train_1_5_processed.csv'
output_file_path = 'Note_train_1_5_processed.csv'

df = pd.read_csv(input_file_path, sep=';')

# Appliquer la correction sur la colonne 'Commentaire'
df['Commentaire'] = df['Commentaire'].apply(corriger_commentaire)

df = df[df['Commentaire'] != '']

# Sauvegarder le résultat dans un nouveau fichier CSV
df.to_csv(output_file_path, sep=';', index=False)
