# %%
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter
import re
import string
import io

# %%
def read_file(file):
    with open(file) as f:
        contents = f.read()
            
    #contents = contents.replace("\\u2019", "'").replace("\\u002c", "").replace(":D", "").replace("xD", "").replace(":)", "")
    #contents = contents.translate(str.maketrans('', '', string.punctuation))
    table = [line.split("\t")[1:] for line in contents.lower().split("\n")[10:-1]]
    print(table[0])
    return pd.DataFrame(table, columns = ['texte','classe'])

# %%
"""
## Chargement des données
Pour lancer le chargement de données veuillez mettre les données dans un dossier data : 'data/donnees_appr_dev/train.xml' 

"""

# %%
# loading one line of data only, file is too large :/
def load_data():
    train_xml, dev_xml = '', ''
    f_train = open('data/donnees_appr_dev/train.xml', "r")
    f_dev = open('data/donnees_appr_dev/dev.xml', "r")
    train_xml = f_train.readlines()
    dev_xml = f_dev.readlines()
    train_xml = ''.join(train_xml)
    dev_xml += ''.join(dev_xml)
    train = pd.read_xml(io.StringIO(train_xml))
    dev = pd.read_xml(io.StringIO(dev_xml))
    return train, dev

# %%
train_df, dev_df = load_data()

# %%
train_df

# %%
"""
## Tokenization
"""

# %%
def yield_tokens(tokenizer, data_iter):
    for text in data_iter:
        yield tokenizer(str(text))

def get_vocab(data):
    data_iter = iter(data)
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(tokenizer, data_iter), specials=["<unk>", "<link>", "<hashtag>", "<mention>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

# %%
"""
## Nettoyage des données
"""

# %%
def clean_table(table):
  #table = table.drop(columns=['id'])
  table = table.dropna()
  table["texte"] = table["texte"].apply(lambda x: re.sub(r'https?:\/\/\S+', '<link>', x))
  table["texte"] = table["texte"].apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '<link>', x))
  table["texte"] = table["texte"].apply(lambda x: re.sub(r"#(\w+)", '<hashtag>', x))
  table["texte"] = table["texte"].apply(lambda x: re.sub(r'/@([a-z\d_]+)/ig', '<mention>', x))
  return table

# %%
train_df, test_df = clean_table(train_df), clean_table(test_df)

# %%
"""
## Récupération du vocabulaire
"""

# %%
vocab = get_vocab(pd.concat([train_df,test_df])["texte"])

# %%
print(len(vocab.get_stoi().keys()))

# %%
train_df.head()

# %%
fr = spacy.load('fr')


# %%
"""
## Encodage du texte
"""

# %%
def encode_corpus(data_iter, vocab):
    encoded = []
    tokenizer = get_tokenizer('spacy', 'fr')
    
    for text in data_iter:
      tokenized_sentence = tokenizer(str(text))
      tokenized_counter = Counter(tokenized_sentence)
      encoded.append([tokenized_counter[word] for word in list(vocab.get_stoi().keys())])
    return encoded

# %%
encoded_train = encode_corpus(iter(train_df['texte']), vocab)
encoded_test = encode_corpus(iter(test_df['texte']), vocab)

# %%
encoded_train[0]

# %%
train_df['texte'][0]

# %%
"""
## Encodage des labels  
"""

# %%
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(["negative", "objective", "mixed", "positive"])
train_df["classe"] = le.transform(train_df["classe"]) - 1

# %%
test_df["classe"] = le.transform(test_df["classe"]) - 1

# %%
"""
## Fichiers SVM
"""

# %%
def write_svm(encoded_set, orig_set, text_file):
  f = open(text_file,'w')
  for idx, line in enumerate(encoded_set):
    l = str(orig_set["classe"][idx]) + " "
    for i in range(len(line)):
      if line[i]!=0:
        l += str(i)+":"+str(line[i])+" "
    l += '\n'
    f.write(l)
  return idx

# %%
write_svm(encoded_train, train_df, "data_deft2017/"+"train_svm.txt")

# %%
write_svm(encoded_test, test_df, "data_deft2017/"+"test_svm.txt")

# %%
"""
## Modèle SVM de LibLinear
"""

# %%
!pip install -U liblinear-official

# %%
from liblinear.liblinearutil import *

# %%
import numpy as np
import scipy

# %%
y_train, x_train = svm_read_problem("data_deft2017/"+"train_svm.txt")
prob = problem(y_train, x_train)
param = parameter('-s 3 -c 1 -q')
m = train(y_train, x_train, '-c 2')
m = train(prob, '-w1 5 -c 3')
m = train(prob, '-s 0 -c 1 -B 1')
m = train(prob, param)
CV_ACC = train(y_train, x_train, '-v 3')
best_C, best_p, best_rate = train(y_train, x_train, '-C -s 0')
m = train(y_train, x_train, '-c {0} -s 0'.format(best_C))
p_label, p_acc, p_val = predict(y_train, x_train, m)

# %%
y_test, x_test = svm_read_problem("data_deft2017/"+"test_svm.txt")
p_label, p_acc, p_val = predict(y_test, x_test, m)

# %%
"""

"""

# %%
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# %%
len(p_label)
multilabel_confusion_matrix(test_df["classe"], p_label,labels=[-1, 0, 1, 2])

# %%
print(classification_report(test_df["classe"], p_label))

# %%
ACC, MSE, SCC = evaluations(y_test, p_label)

# %%
print("ACC", ACC,"| MSE", MSE,"| SCC", SCC)

# %%
"""
## Réseau de neurones

"""

# %%
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
import tensorflow as tf

# %%
train_df["texte"] = encoded_train
test_df["texte"] = encoded_test

# %%
X_train = pd.DataFrame(encoded_train)
X_test = pd.DataFrame(encoded_test)

# %%
y_train = train_df["classe"]
y_test = test_df["classe"]

# %%
X_train.shape

# %%
y_train += 1

# %%
y_train

# %%
from sklearn.preprocessing import OneHotEncoder

# %%
one_hot_enc = OneHotEncoder()
y_train_ohe = one_hot_enc.fit_transform(train_df['classe'].to_numpy().reshape(-1,1)).toarray()

# %%
#Model Building
model = Sequential()
model.add(Embedding(X_train.shape[1],120, input_length = X_train.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(X_train.shape[1],dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(352, activation='LeakyReLU'))
model.add(Dense(4, activation='softmax')) 
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
#Model Training
model.fit(X_train, y_train_ohe, epochs = 20, batch_size=32, verbose =1)
#Model Testing
model.evaluate(X_test,y_train_ohe)

# %%
multilabel_confusion_matrix(test_df["classe"], p_label,labels=[-1, 0, 1])

# %%
print(classification_report(test_df["classe"], p_label))