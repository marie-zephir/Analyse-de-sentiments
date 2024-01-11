import requests
from bs4 import BeautifulSoup
import pandas as pd

def obtenir_critiques_positives_allocine(url):
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        critiques_positives = []

        # Sélectionnez les éléments div avec les classes "review-card-review-holder"
        critiques_elements = soup.select('.review-card-review-holder')
        
        for critique_element in critiques_elements:
            # Sélectionnez l'élément span avec la classe "stareval-note" à l'intérieur de chaque critique
            note_element = critique_element.select_one('.stareval-note')

            # Vérifiez si la note est égale à "0,5"
            if note_element and note_element.text.strip() == '1,5': #pour train : 0,5 ; 1,0 ; 1,5
                # Si oui, récupérez le texte de l'élément div avec la classe "content-txt review-card-content"
                texte_critique = critique_element.select_one('.content-txt.review-card-content')
                if texte_critique:
                    critiques_positives.append(texte_critique.get_text(strip=True).replace(';', ''))

        return critiques_positives
    else:
        print(f"Erreur lors de la requête : {response.status_code}")
        return None

def ajouter_critiques_au_csv_pandas(critiques_positives):
    # Chargez le fichier CSV existant ou créez-le s'il n'existe pas encore
    try:
        df = pd.read_csv('Note_train_1_5.csv', delimiter=';')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Note', 'Commentaire'])

    # Ajoutez les critiques positives au DataFrame
    nouvelles_critiques = pd.DataFrame({'Note': [1,5] * len(critiques_positives), #Changer Note à  0,5 ; 1,0 ; 1,5 
                                         'Commentaire': critiques_positives})
    df = pd.concat([df, nouvelles_critiques], ignore_index=True)

    # Écrivez le DataFrame dans le fichier CSV en mode append
    df.to_csv('Note_train_1_5.csv', index=False, sep=';', encoding='utf-8')

    # Obtenez le nombre total de lignes dans le fichier
    nombre_lignes = len(df)

    return nombre_lignes

# Parcours des pages de 2 à 552
for page_number in range(2, 553):
    url_page = f"https://www.allocine.fr/films/notes/?page={page_number}"
    response_page = requests.get(url_page)
    
    if response_page.status_code == 200:
        soup_page = BeautifulSoup(response_page.text, 'html.parser')
        
        # Trouver tous les liens vers les films
        liens_films = soup_page.select('.meta-title-link')
        
        for lien_film in liens_films:
            href_value = lien_film.get('href', '')
            
            # Vérifiez si la valeur de l'attribut href est non vide
            if href_value:
                # Construire l'URL complet du film
                url_film = "https://www.allocine.fr" + lien_film['href'].replace('.html', '/critiques/spectateurs/star-1/')
                url_film = url_film.replace('_gen_cfilm=', '-')
                
                # Obtenir les critiques positives du film
                critiques_positives = obtenir_critiques_positives_allocine(url_film)
                
                if critiques_positives:
                    # Ajouter les critiques au CSV
                    nombre_lignes = ajouter_critiques_au_csv_pandas(critiques_positives)
                    print(f"{nombre_lignes} lignes ajoutées au fichier CSV pour {url_film}")
                else:
                    print(f"Aucune critique positive trouvée pour {url_film}")
            else:
                print("Aucun attribut href trouvé pour le lien.")

print("Parcours terminé.")
