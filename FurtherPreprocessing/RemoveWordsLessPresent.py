import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# Charger le fichier CSV
#df = pd.read_csv('testingcode4.csv', delimiter=';')
#df = pd.read_csv('train.csv', delimiter=';')
df = pd.read_csv('dev.csv', delimiter=';')
#df = pd.read_csv('test.csv', delimiter=';')

# Fonction pour traiter un commentaire et renvoyer les mots fréquents
def process_comment(comment):
    words = comment.split()
    return words

# Compter la fréquence des mots dans tous les commentaires
word_freq = {}
for comment in df['Commentaire']:
    words = process_comment(comment)
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1

# Filtrer les mots présents moins de n fois
rare_words = set(word for word, freq in word_freq.items() if freq < 100) # ->>> for train, dev datasets
#rare_words = set(word for word, freq in word_freq.items() if freq < 50) # ->>> for test dataset

# Fonction pour filtrer les commentaires
def filter_comments(row):
    index, data = row
    words = process_comment(data['Commentaire'])
    # Filtrer les mots rares
    filtered_words = [word for word in words if word not in rare_words]
    # Reconstruct the comment
    filtered_comment = ' '.join(filtered_words)
    # Vérifier si le commentaire est vide
    if filtered_comment.strip() == '':
        return None
    else:
        data['Commentaire'] = filtered_comment
        return data

# Fonction principale
def main():
    # Paralléliser le traitement des commentaires
    with ProcessPoolExecutor() as executor:
        filtered_rows = list(executor.map(filter_comments, df.iterrows()))

    # Filtrer les lignes nulles et reconstruire le DataFrame
    filtered_df = pd.DataFrame(filter(lambda x: x is not None, filtered_rows))

    # Sauvegarder le résultat dans un nouveau fichier CSV
    #filtered_df.to_csv('testingcode5.csv', index=False, sep=';')
    #filtered_df.to_csv('train_100_WR.csv', index=False, sep=';')
    filtered_df.to_csv('dev_100_WR.csv', index=False, sep=';')
    #filtered_df.to_csv('final_test.csv', index=False, sep=';')

# Point d'entrée principal pour le programme
if __name__ == '__main__':
    main()
