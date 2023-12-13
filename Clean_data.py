import csv
import xml.etree.ElementTree as ET
import re

def convert_to_csv(input_file, output_file):
    tree = ET.parse(input_file)
    root = tree.getroot()
    classes = []
    commentaires = []

    for note in root.iter('note'):
        if(note.text.strip() == "0,5"):
            classes.append("1")
        elif(note.text.strip() == "1,0"):
            classes.append("2")
        elif(note.text.strip() == "1,5"):
            classes.append("3")
        elif(note.text.strip() == "2,0"):
            classes.append("4")
        elif(note.text.strip() == "2,5"):
            classes.append("5")
        elif(note.text.strip() == "3,0"):
            classes.append("6")
        elif(note.text.strip() == "3,5"):
            classes.append("7")
        elif(note.text.strip() == "4,0"):
            classes.append("8")
        elif(note.text.strip() == "4,5"):
            classes.append("9")
        elif(note.text.strip() == "5,0"):
            classes.append("10")
    for comment in root.iter('commentaire'):
        if(comment.text is None):
            commentaires.append("")
        else:
            commentaires.append(clean_comments(comment.text))
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Note", "Commentaire"])
        for i in range(len(classes)):
            writer.writerow([classes[i], commentaires[i]])

def clean_comments(text):
    clean = re.sub(r'\n', ' ', text)
    clean = re.sub(';', '', clean)
    clean = re.sub(r'http\S+', ' ', clean)
    clean = re.sub(r'[0-9]', '', clean)
    clean = re.sub(r'[^\w\s]', ' ', clean)
    clean = clean.lower()
    clean = " ".join(clean.split())
    return clean

if __name__ == "__main__":
    input_file_train = 'train.xml'
    output_file_train = 'train.csv'

    input_file_dev = 'dev.xml'
    output_file_dev = 'dev.csv'

    convert_to_csv(input_file_train, output_file_train)
    convert_to_csv(input_file_dev, output_file_dev)