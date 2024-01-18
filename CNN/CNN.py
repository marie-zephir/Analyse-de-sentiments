import pandas as pd
import numpy as np
from gensim.models import FastText
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Dense, Flatten, LSTM, MaxPooling1D, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import to_categorical

# Charger les données depuis les fichiers CSV
train_data = pd.read_csv('final_train.csv', delimiter=';')
dev_data = pd.read_csv('final_dev.csv', delimiter=';')

# Concaténer les données d'entraînement et de développement
data = pd.concat([train_data, dev_data], ignore_index=True)

# Prétraiter les données
texts = data['Commentaire'].tolist()
labels = data['Note'].tolist()

# Convertir les labels en encodage one-hot
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
one_hot_labels = to_categorical(encoded_labels)

# Créer et entraîner le modèle FastText
vector_size = 256
window = 5

fasttext_model = FastText(vector_size=vector_size,window=window, min_count=1, workers=4)
fasttext_model.build_vocab(texts)
fasttext_model.train(texts, total_examples=fasttext_model.corpus_count, epochs=fasttext_model.epochs)

fasttext_model.save('fasttext_model.model')

test_size = len(dev_data) / len(data)

# Diviser les données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(texts, one_hot_labels, test_size=test_size, random_state=42)

# Préparer les données pour l'entraînement du modèle
max_no_tokens = 100
x_vectors = fasttext_model.wv

x_train_data = np.zeros((len(x_train), max_no_tokens, vector_size), dtype=np.float32)
x_test_data = np.zeros((len(x_test), max_no_tokens, vector_size), dtype=np.float32)

for i, text in enumerate(x_train):
    for t, token in enumerate(text.split()):
        if t >= max_no_tokens:
            break
        if token in x_vectors:
            x_train_data[i, t, :] = x_vectors[token]

for i, text in enumerate(x_test):
    for t, token in enumerate(text.split()):
        if t >= max_no_tokens:
            break
        if token in x_vectors:
            x_test_data[i, t, :] = x_vectors[token]

# Construire le modèle
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same', input_shape=(100, vector_size)))
model.add(Conv1D(32, kernel_size=3, activation='elu', padding='same'))
model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=3))
model.add(Bidirectional(LSTM(512, dropout=0.2, recurrent_dropout=0.3)))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compiler le modèle
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

# Entraîner le modèle
batch_size = 32
no_epochs =2

model.fit(x_train_data, y_train, epochs=no_epochs, batch_size=batch_size, validation_data=(x_test_data, y_test))

# Évaluer le modèle
loss, accuracy = model.evaluate(x_test_data, y_test, batch_size=batch_size)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Sauvegarder le modèle
model.save('sentiment_analysis_model.h5')
