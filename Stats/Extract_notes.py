import xml.etree.ElementTree as ET
import csv
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Nous avons plusieurs classes : 
tree = ET.parse('train.xml')
root = tree.getroot()
notes = []
classes = []
movies = []
review_ids = []
names=[]
user_ids=[]
lines = 133192


def func(value):
    newValue = value.replace(",", "")
    newValue2 = newValue.replace('"', "")
    return ''.join(newValue2.splitlines())

def topoint(value):
    newValue = value.replace(",", ".")
    return float(newValue)

for note in root.iter('note'):
    notes.append(topoint(note.text.strip()))
    if(note.text.strip() == "0,5"):
        classes.append("1")
    elif(note.text.strip()  == "1,0"):
        classes.append("2")
    elif(note.text.strip()  == "1,5"):
        classes.append("3")
    elif(note.text.strip()  == "2,0"):
        classes.append("4")
    elif(note.text.strip()  == "2,5"):
        classes.append("5")
    elif(note.text.strip()  == "3,0"):
        classes.append("6")
    elif(note.text.strip()  == "3,5"):
        classes.append("7")
    elif(note.text.strip()  == "4,0"):
        classes.append("8")
    elif(note.text.strip()  == "4,5"):
        classes.append("9")
    elif(note.text.strip() == "5,0"):
        classes.append("10")

for movie in root.iter('movie'):
    movies.append(movie.text)

for review_id in root.iter('review_id'):
    review_ids.append(review_id.text)

for name in root.iter('name'):
    if(name.text is None):
        names.append("")
    else:
        names.append(func(name.text.strip()))

for user_id in root.iter('user_id'):
    user_ids.append(user_id.text)

    
with open('notes1.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["Film", "Critique_id", "Nom_utilisateur","Utilisateur_id", "Note", "Classe","Commentaire" ])
    for i in range(lines):
        writer.writerow([movies[i], review_ids[i],names[i], user_ids[i], notes[i], classes[i],commentaires[i]])


with open('notes2.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["Film", "Critique_id", "Nom_utilisateur","Utilisateur_id", "Note", "Classe","Commentaire" ])
    for i in range(lines, lines+lines):
        writer.writerow([movies[i], review_ids[i],names[i], user_ids[i], notes[i], classes[i],commentaires[i]])



with open('notes3.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["Film", "Critique_id", "Nom_utilisateur","Utilisateur_id", "Note", "Classe","Commentaire" ])
    for i in range(lines+lines, lines+lines+lines):
        writer.writerow([movies[i], review_ids[i],names[i], user_ids[i], notes[i], classes[i],commentaires[i]])


with open('notes4.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["Film", "Critique_id", "Nom_utilisateur","Utilisateur_id", "Note", "Classe","Commentaire" ])
    for i in range(lines+lines+lines, lines+lines+lines+lines):
        writer.writerow([movies[i], review_ids[i],names[i], user_ids[i], notes[i], classes[i],commentaires[i]])
    
with open('notes5.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["Film", "Critique_id", "Nom_utilisateur","Utilisateur_id", "Note", "Classe","Commentaire" ])
    for i in range(lines+lines+lines+lines, len(notes)):
        writer.writerow([movies[i], review_ids[i],names[i], user_ids[i], notes[i], classes[i],commentaires[i]])
