import re
import os
import random
import tarfile
import urllib
import torchtext
import pandas as pd

import csv

class MyDataset(torchtext.data.Dataset):
    @classmethod
    def from_csv(cls, text_field, label_field, csv_path, **kwargs):
        examples = []
        df = pd.read_csv(csv_path, sep=';')
        for index, row in df.iterrows():
            text, label = str(row['Commentaire']), (int(float(str(row['Note']).replace(",", "."))*2))
            
            #print("commentaire", row['Commentaire'])
            #print("note", label)
            examples.append(torchtext.data.Example.fromlist([text, label], fields=[('text', text_field), ('label', label_field)]))
            #d = data.Example.fromlist(examples, fields=[('text', text_field), ('label', label_field)])

        return cls(examples=examples, fields=[('text', text_field), ('label', label_field)], **kwargs)

    @classmethod
    def test_from_csv(cls, text_field, csv_path, **kwargs):
        examples = []
        df = pd.read_csv(csv_path, sep=';')
        for index, row in df.iterrows():
            text = str(row['Commentaire'])
            examples.append(torchtext.data.Example.fromlist([text, None], fields=[('text', text_field), ('label', None)]))

        return cls(examples=examples, fields=[('text', text_field)], **kwargs)


class CustomIterator(torchtext.data.Iterator):
    def random_shuffler(self, data, batch_size):
        indices = list(range(len(data)))
        random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            yield indices[i:i + batch_size]

