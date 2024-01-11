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
        reviewId = element.find('review_id').text.strip() if element.find('review_id').text is not None else ""
        commentaire = element.find('commentaire').text.strip() if element.find('commentaire').text is not None else ""

        rows.append((reviewId, commentaire))

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_row, rows))

    results = [result for result in results if result[1] != '']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["ReviewId", "Commentaire"])
        writer.writerows(results)

def process_row(row):
    reviewId, commentaire = row
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
    return reviewId, sentence

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
    
    input_file_test = 'test.xml'
    output_file_test = 'test.csv'

    convert_to_csv(input_file_test, output_file_test)


