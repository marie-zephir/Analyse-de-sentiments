import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.optimizers import RMSprop, Adam
from gensim.models.fasttext import FastText
import numpy as np
from tensorflow.keras.utils import to_categorical

# Charger les données d'entraînement et de validation
train_data = pd.read_csv('train.csv', sep=';')
val_data = pd.read_csv('dev.csv', sep=';')
train_data['Commentaire'].fillna('', inplace=True)
val_data['Commentaire'].fillna('', inplace=True)

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_data['Note'])
val_labels = label_encoder.fit_transform(val_data['Note'])

train_labels_one_hot = to_categorical(train_labels)
val_labels_one_hot = to_categorical(val_labels)

# Paramètres du Tokenizer
embedding_dim = 100 
max_words = 10000
max_len = 100

# Utiliser le Tokenizer pour convertir les critiques en séquences d'entiers
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_data['Commentaire'])

# Convertir les critiques en séquences d'entiers
train_sequences = tokenizer.texts_to_sequences(train_data['Commentaire'])
val_sequences = tokenizer.texts_to_sequences(val_data['Commentaire'])

# Appliquer le padding pour que toutes les séquences aient la même longueur
train_sequences = pad_sequences(train_sequences, maxlen=max_len)
val_sequences = pad_sequences(val_sequences, maxlen=max_len)

# Entraîner le modèle FastText
fasttext_model = FastText(sentences=[text.split() for text in train_data['Commentaire']], vector_size=embedding_dim, window=5, min_count=1, workers=4, sg=1)

# Construire la matrice d'embedding pour les données d'entraînement
embedding_matrix_train = np.zeros((len(train_sequences), max_len, embedding_dim))
for i, sequence in enumerate(train_sequences):
    for j, word_index in enumerate(sequence):
        if word_index != 0:  # 0 est utilisé pour le padding
            word = tokenizer.index_word[word_index]
            embedding_matrix_train[i, j, :] = fasttext_model.wv[word]

# Construire la matrice d'embedding pour les données de validation
embedding_matrix_val = np.zeros((len(val_sequences), max_len, embedding_dim))
for i, sequence in enumerate(val_sequences):
    for j, word_index in enumerate(sequence):
        if word_index != 0:
            word = tokenizer.index_word[word_index]
            embedding_matrix_val[i, j, :] = fasttext_model.wv[word]

# Paramètres du modèle
lstm_units = 128

# Construction du modèle
model = Sequential()
model.add(Bidirectional(LSTM(units=lstm_units, dropout=0.2, recurrent_dropout=0.2), input_shape=(max_len, embedding_dim)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compiler le modèle
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(embedding_matrix_train, train_labels_one_hot, epochs=10, batch_size=128, validation_data=(embedding_matrix_val, val_labels_one_hot))

# Évaluation du modèle
loss, accuracy = model.evaluate(embedding_matrix_val, val_labels_one_hot)
print(f'Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}')

model.save('ModelEncodeDataset2.h5')

# Charger les données de test
test_data = pd.read_csv('test.csv', sep=';')
test_data['Commentaire'].fillna('', inplace=True)

# Convertir les commentaires en séquences numériques
test_sequences = tokenizer.texts_to_sequences(test_data['Commentaire'])

# Appliquer le padding pour les séquences de test
test_sequences = pad_sequences(test_sequences, maxlen=max_len)

# Construire la matrice d'embedding pour les données de test
embedding_matrix_test = np.zeros((len(test_sequences), max_len, embedding_dim))
for i, sequence in enumerate(test_sequences):
    for j, word_index in enumerate(sequence):
        if word_index != 0:
            word = tokenizer.index_word[word_index]
            embedding_matrix_test[i, j, :] = fasttext_model.wv[word]

# Faire des prédictions sur les données de test
test_predictions = model.predict(embedding_matrix_test)

# Convertir les prédictions en notes (indices de la classe prédite)
predicted_labels = test_predictions.argmax(axis=1)

# Récupérer les ReviewId du fichier de test
review_ids = test_data['ReviewId']

# Créer un DataFrame avec ReviewId et les notes prédites
result_df = pd.DataFrame({'ReviewId': review_ids, 'Note': predicted_labels})

# Enregistrer le résultat dans un fichier texte sans header
result_df.to_csv('predictions2.txt', sep=' ', header=False, index=False)
