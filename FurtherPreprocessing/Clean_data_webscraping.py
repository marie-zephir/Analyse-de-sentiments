import csv
import xml.etree.ElementTree as ET
import re
import nltk
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
from concurrent.futures import ProcessPoolExecutor

nlp = spacy.load("fr_core_news_sm")

def process_csv(input_csv, output_csv):
    with open(input_csv, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)  # Skip header
        rows = [(row[0], row[1]) for row in reader]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_row, rows))

    results = [result for result in results if result[1] != '']

    with open(output_csv, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Note", "Commentaire"])
        writer.writerows(results)

def process_row(row):
    note, commentaire = row
    clean_comment = clean_comments(commentaire)
    #lemmatized_comment = lemmatize_comments(clean_comment)
    #if(lemmatized_comment is None):
    #    sentence = ""
    if(clean_comment is None):
        sentence = ""
    else:
        #words = lemmatized_comment.split()
        words = clean_comment.split()
        filtered_words = [word for word in words if len(word) > 2]
        sentence = ' '.join(filtered_words)
    return note, sentence

def clean_comments(text):
    clean = re.sub(r'\n', ' ', text)
    clean = re.sub(';', ' ', clean)
    clean = re.sub(r'http\S+', ' ', clean)
    clean = re.sub(r'[0-9]', ' ', clean)
    clean = re.sub(r'[^\w\s]', ' ', clean)
    clean = clean.lower()
    clean = ' '.join(word for word in clean.split() if word not in STOP_WORDS)
    return clean

def lemmatize_comments(text):
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_tokens)

if __name__ == "__main__":
    input_file_train_0_5 = 'Note_train_0_5.csv'
    output_file_train_0_5 = 'Note_train_0_5_processed.csv'

    input_file_train_1_0 = 'Note_train_1_0.csv'
    output_file_train_1_0 = 'Note_train_1_0_processed.csv'

    input_file_train_1_5 = 'Note_train_1_5.csv'
    output_file_train_1_5 = 'Note_train_1_5_processed.csv'

    process_csv(input_file_train_0_5, output_file_train_0_5)
    process_csv(input_file_train_1_0, output_file_train_1_0)
    process_csv(input_file_train_1_5, output_file_train_1_5)


    #input_file_dev = 'testingcode.xml'
    #output_file_dev = 'testingcode.csv'
    #convert_to_csv(input_file_dev, output_file_dev)