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
from keras.callbacks import EarlyStopping, TensorBoard

# Charger les données d'entraînement et de validation
train_data = pd.read_csv('final_train.csv', sep=';')
dev_data = pd.read_csv('final_dev.csv', sep=';')


# Paramètres du Tokenizer
max_len = 100

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

fasttext_model.save('fasttext_model3.model')

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

lstm_units = 128

# Construction du modèle
model = Sequential()
model.add(Bidirectional(LSTM(units=lstm_units, dropout=0.2, recurrent_dropout=0.2), input_shape=(max_len, vector_size)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compiler le modèle
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(x_train_data, y_train, epochs=5, batch_size=128, validation_data=(x_test_data, y_test))

# Évaluation du modèle
loss, accuracy = model.evaluate(x_test_data, y_test)
print(f'Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}')

model.save('ModelEncodeDataset3.h5')