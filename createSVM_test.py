import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Charger le test
test_data = pd.read_csv('test.csv', sep=';')
test_data['Commentaire'].fillna('', inplace=True)

# Créer un vecteur TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(test_data['Commentaire'])

# Sauvegarder la matrice TF-IDF pour le test
tfidf_train = tfidf_matrix[:len(test_data)]
pd.DataFrame(tfidf_train.toarray()).to_csv('svm_test_tfidf.csv', index=False)

# Sauvegarder les reviewIds correspondants pour le test
test_reviewId = test_data['ReviewId']
test_reviewId.to_csv('svm_test_reviews.csv', index=False)