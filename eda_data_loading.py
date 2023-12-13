import io
import pandas as pd
import re
import string

f_dev = open(
    '/home/rim/Downloads/Corpus dapprentissage corpus de développement-20231107/donnees_appr_dev/donnees_appr_dev/dev.xml',
    "r")
df_dev = pd.read_xml(io.StringIO(f_dev))


f_train = open(
    '/home/rim/Downloads/Corpus dapprentissage corpus de développement-20231107/donnees_appr_dev/donnees_appr_dev/train.xml',
    "r")
df_train = pd.read_xml(io.StringIO(f_train))

df_dev=df_dev.drop(columns=['commentaire', 'name'])
df_train=df_train.drop(columns=['commentaire', 'name'])

df_dev.to_csv('data/eda_dev.csv', index=False)
df_train.to_csv('data/eda_train.csv', index=False)

