import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Charger le train et le dev
train_data = pd.read_csv('train.csv', sep=';')
dev_data = pd.read_csv('dev.csv', sep=';')

# Concaténer les données de train et de dev pour créer un vocabulaire global
all_data = pd.concat([train_data, dev_data])
all_data['Commentaire'].fillna('', inplace=True)

# Créer un vecteur TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(all_data['Commentaire'])

# Sauvegarder la matrice TF-IDF pour le train
tfidf_train = tfidf_matrix[:len(train_data)]
pd.DataFrame(tfidf_train.toarray()).to_csv('svm_train_tfidf.csv', index=False)

# Sauvegarder la matrice TF-IDF pour le dev
tfidf_dev = tfidf_matrix[len(train_data):]
pd.DataFrame(tfidf_dev.toarray()).to_csv('svm_dev_tfidf.csv', index=False)

# Sauvegarder les notes correspondantes pour le train
train_labels = train_data['Note']
train_labels.to_csv('svm_train_labels.csv', index=False)

# Sauvegarder les notes correspondantes pour le dev
dev_labels = dev_data['Note']
dev_labels.to_csv('svm_dev_labels.csv', index=False)
