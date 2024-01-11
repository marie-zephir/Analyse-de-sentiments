import pandas as pd
import spacy
from concurrent.futures import ProcessPoolExecutor

# Charger le modèle Spacy
nlp = spacy.load('fr_core_news_lg', disable=['parser', 'ner'])

# Fonction pour appliquer la lemmatisation à un texte
def lemmatize_text(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in doc])

# Fonction pour lemmatiser une colonne en parallèle
def parallel_lemmatization(column):
    with ProcessPoolExecutor() as executor:
        result = list(executor.map(lemmatize_text, column))
    return result

if __name__ == '__main__':
    # Charger le CSV
    #input_csv_path = 'testingcode2.csv'
    #output_csv_path = 'testingcode3.csv'

    #input_csv_path = 'train.csv'
    #output_csv_path = 'train.csv'

    #input_csv_path = 'dev.csv'
    #output_csv_path = 'dev.csv'

    #input_csv_path = 'test.csv'
    #output_csv_path = 'test.csv'

    #input_csv_path = 'Note_train_0_5_processed.csv'
    #output_csv_path = 'Note_train_0_5_processed.csv'

    #input_csv_path = 'Note_train_1_0_processed.csv'
    #output_csv_path = 'Note_train_1_0_processed.csv'

    input_csv_path = 'Note_train_1_5_processed.csv'
    output_csv_path = 'Note_train_1_5_processed.csv'

    df = pd.read_csv(input_csv_path, sep=';')

    # Vérifier si la colonne 'Commentaire' existe
    if 'Commentaire' in df.columns:
        # Supprimer les lignes avec 'Commentaire' vide
        df = df.dropna(subset=['Commentaire'])

        # Lemmatisation en parallèle de la colonne 'Commentaire'
        df['Commentaire'] = parallel_lemmatization(df['Commentaire'])

        df['Commentaire'] = df['Commentaire'].str.lower()

        # Enregistrer le résultat dans le même CSV
        df.to_csv(output_csv_path, sep=';', index=False)

        print(f"Résultat enregistré dans {output_csv_path}")
    else:
        print("La colonne 'Commentaire' n'existe pas dans le fichier CSV.")
