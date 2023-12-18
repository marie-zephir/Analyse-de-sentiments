import csv
import xml.etree.ElementTree as ET
import re
import nltk
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
from concurrent.futures import ProcessPoolExecutor

nlp = spacy.load("fr_core_news_sm")

def convert_to_csv(input_file, output_file):
    tree = ET.parse(input_file)
    root = tree.getroot()
    rows = []

    for element in root:
        note = element.find('note').text.strip()
        commentaire = element.find('commentaire').text.strip() if element.find('commentaire').text is not None else ""

        rows.append((note, commentaire))

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_row, rows))

    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Note", "Commentaire"])
        writer.writerows(results)

def process_row(row):
    note, commentaire = row
    clean_comment = clean_comments(commentaire)
    lemmatized_comment = lemmatize_comments(clean_comment)
    if(lemmatized_comment is None):
        sentence = ""
    else:
        words = lemmatized_comment.split()
        filtered_words = [word for word in words if len(word) > 2]
        sentence = ' '.join(filtered_words)
    return note, sentence

def clean_comments(text):
    clean = re.sub(r'\n', ' ', text)
    clean = re.sub(';', '', clean)
    clean = re.sub(r'http\S+', ' ', clean)
    clean = re.sub(r'[0-9]', '', clean)
    clean = re.sub(r'[^\w\s]', ' ', clean)
    clean = clean.lower()
    clean = " ".join(clean.split())
    clean = ' '.join(word for word in clean.split() if word not in STOP_WORDS)
    return clean

def lemmatize_comments(text):
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_tokens)

if __name__ == "__main__":
    input_file_train = 'train.xml'
    output_file_train = 'train.csv'

    input_file_dev = 'dev.xml'
    output_file_dev = 'dev.csv'

    convert_to_csv(input_file_train, output_file_train)
    convert_to_csv(input_file_dev, output_file_dev)
