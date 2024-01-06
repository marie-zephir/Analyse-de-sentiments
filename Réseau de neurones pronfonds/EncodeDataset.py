import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, CamembertTokenizer, CamembertModel
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

# Paramètres du modèle
embedding_dim = 50
lstm_units = 100

# Construction du modèle
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(units=lstm_units, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(train_sequences, train_labels_one_hot, epochs=5, batch_size=32, validation_data=(val_sequences, val_labels_one_hot))

# Évaluation du modèle
loss, accuracy = model.evaluate(val_sequences, val_labels_one_hot)
print(f'Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}')

model.save('ModelEncodeDataset3.h5')

# Charger les données de test
test_data = pd.read_csv('test.csv', sep=';')
test_data['Commentaire'].fillna('', inplace=True)

# Convertir les commentaires en séquences numériques
test_sequences = tokenizer.texts_to_sequences(test_data['Commentaire'])

# Remplir les séquences pour obtenir des séquences de longueur uniforme
test_sequences = pad_sequences(test_sequences, maxlen=max_length)

# Faire des prédictions sur les données de test
test_predictions = model.predict(test_sequences)

# Convertir les prédictions en notes (indices de la classe prédite)
predicted_labels = test_predictions.argmax(axis=1)

# Récupérer les ReviewId du fichier de test
review_ids = test_data['ReviewId']

# Créer un DataFrame avec ReviewId et les notes prédites
result_df = pd.DataFrame({'ReviewId': review_ids, 'Note': predicted_labels})

# Enregistrer le résultat dans un fichier texte sans header
result_df.to_csv('predictions3.txt', sep=' ', header=False, index=False)

