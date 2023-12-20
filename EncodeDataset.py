import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, CamembertTokenizer, CamembertModel
from tensorflow.keras.utils import to_categorical

# Charger les données d'entraînement et de validation
train_data = pd.read_csv('train.csv', sep=';')
val_data = pd.read_csv('dev.csv', sep=';')
train_data['Commentaire'].fillna('', inplace=True)
val_data['Commentaire'].fillna('', inplace=True)

# Charger les notes pré-encodées en valeurs nominales de 0 à 10
train_labels = pd.csv_read('svm_train_labels_encoded.csv')
val_labels = pd.csv_read('svm_dev_labels_encoded.csv')

#label_encoder = LabelEncoder()
#train_labels = label_encoder.fit_transform(train_data['Note'])
#val_labels = label_encoder.fit_transform(val_data['Note'])

#train_labels_one_hot = to_categorical(train_labels)
#val_labels_one_hot = to_categorical(val_labels)


# Tokenizer pour convertir les commentaires en séquences numériques
#tokenizer = Tokenizer()
#tokenizer.fit_on_texts(train_data['Commentaire'])
#train_sequences = tokenizer.texts_to_sequences(train_data['Commentaire'])
#val_sequences = tokenizer.texts_to_sequences(val_data['Commentaire'])


# Remplir les séquences pour obtenir des séquences de longueur uniforme
#max_length = max(max(len(seq) for seq in train_sequences), max(len(seq) for seq in val_sequences))
#train_sequences = pad_sequences(train_sequences, maxlen=max_length, padding='post')
#val_sequences = pad_sequences(val_sequences, maxlen=max_length, padding='post')

# Charger le tokenizer Camembert
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

# Tokeniser les commentaires
train_tokens = tokenizer(list(train_data['Commentaire']), padding=True, truncation=True, return_tensors='tf')
val_tokens = tokenizer(list(val_data['Commentaire']), padding=True, truncation=True, return_tensors='tf')

print(train_labels_one_hot.shape)
print(val_labels_one_hot.shape)
print(train_tokens['input_ids'].shape[1])
print(val_tokens['input_ids'].shape[1])

# Construction du modèle
model=Sequential()
model.add(Embedding(input_dim=(tokenizer.vocab_size+1), output_dim=100, input_length=train_tokens['input_ids'].shape[1]))
model.add(LSTM(100,activation='relu',return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_tokens['input_ids'], train_labels, epochs=10, batch_size=32, validation_data=(val_tokens['input_ids'], val_labels))

# Charger les données de test
test_data = pd.read_csv('test.csv', sep=';')
test_data['Commentaire'].fillna('', inplace=True)

# Convertir les commentaires en séquences numériques
#test_sequences = tokenizer.texts_to_sequences(test_data['Commentaire'])

# Remplir les séquences pour obtenir des séquences de longueur uniforme
#test_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='post')

test_tokens = tokenizer(list(test_data['Commentaire']), padding=True, truncation=True, return_tensors='pt')

# Faire des prédictions sur les données de test
test_predictions = model.predict(test_tokens)

# Décodez les prédictions en utilisant inverse_transform du label_encoder
decoded_predictions = label_encoder.inverse_transform(test_predictions.round().astype(int).flatten())

results_df = pd.DataFrame({'Review_id': test_data['Review_id'], 'Predicted_Note': decoded_predictions})
results_df.to_csv('predictions.txt', index=False)

