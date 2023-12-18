# %%
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter
import re
import string
import io
from sklearn import preprocessing
from datetime import datetime


# %%

"""
## Chargement des données
Pour lancer le chargement de données veuillez mettre les données dans un dossier data : 'data/donnees_appr_dev/train.xml' 

"""

# loading one line of data only, file is too large :/
def load_data():
    train_xml, dev_xml = '', ''
    f_train = open('../data/xml/train.xml', "r")
    #f_dev = open('../data/xml/dev.xml', "r")
    train_xml = f_train.readlines()
    #dev_xml = f_dev.readlines()
    train_xml = ''.join(train_xml)
    #dev_xml += ''.join(dev_xml)
    train = pd.read_xml(io.StringIO(train_xml))
    #dev = pd.read_xml(io.StringIO(dev_xml))
    return train

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
  # Dictionary containing all  regex patterns with the corresponding substitutions
  sub_dic = {r'https?:\/\/\S+': '<link>',
             r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)": '<link>',
             r"#(\w+)": '<hashtag>',
             r'/@([a-z\d_]+)/ig': '<mention>'}
  # drop null values
  table = table.dropna()
  table['Commentaire'] = table['Commentaire'].apply(lambda x: re.sub(sub_dic, x))
  return table

# %%
#train_df, test_df = clean_table(train_df), clean_table(test_df)

# %%
"""
## Encodage du texte
"""

# %%
def encode_corpus(data_iter, vocab):
    encoded = []
    tokenizer = get_tokenizer('spacy', 'fr_core_news_md')
    
    for text in data_iter:
      tokenized_sentence = tokenizer(str(text))
      tokenized_counter = Counter(tokenized_sentence)
      encoded.append([tokenized_counter[word] for word in list(vocab.get_stoi().keys())])
    return encoded


# %%
"""
## Encodage des labels  
"""

"""
## Récupération du vocabulaire
"""
train_df, test_df = pd.read_csv("../data/csv/train.csv", sep=';'), pd.read_csv("../data/csv/dev.csv", sep=';')

start_time = datetime.now()
vocab = get_vocab(pd.concat([train_df,test_df])['Commentaire'])
time_elapsed = datetime.now() - start_time

print('Time for creating vocab (hh:mm:ss.ms) {}'.format(time_elapsed))
print('vocab size = {}'.format(len(vocab.get_stoi().keys())))
# %%
#print((vocab.get_stoi().keys()))

"""# %%
train_df.head()

# %%
fr = spacy.load('fr')
"""
# %%
start_time = datetime.now()
encoded_train = encode_corpus(iter(train_df['Commentaire']), vocab)
time_elapsed = datetime.now() - start_time

print('Time for encoded train (hh:mm:ss.ms) {}'.format(time_elapsed))

start_time = datetime.now()
encoded_test = encode_corpus(iter(test_df['Commentaire']), vocab)
time_elapsed = datetime.now() - start_time

print('Time for encoded dev (hh:mm:ss.ms) {}'.format(time_elapsed))
"""
# %%
encoded_train[0]

# %%
train_df['Commentaire'][0]
# %%
le = preprocessing.LabelEncoder()
le.fit([i for i in range(1,11)])
train_df['Note'] = le.transform(train_df['Note']) - 1

# %%
test_df['Note'] = le.transform(test_df['Note']) - 1

"""
# %%
"""
## Fichiers SVM
"""

# %%
def write_svm(encoded_set, orig_set, text_file):
  f = open(text_file,'w')
  for idx, line in enumerate(encoded_set):
    l = str(orig_set['Note'][idx]) + " "
    for i in range(len(line)):
      if line[i]!=0:
        l += str(i)+":"+str(line[i])+" "
    l += '\n'
    f.write(l)
  return idx

# %%
#write_svm(encoded_train, train_df, "data_deft2017/"+"train_svm.txt")

# %%
#write_svm(encoded_test, test_df, "data_deft2017/"+"test_svm.txt")

