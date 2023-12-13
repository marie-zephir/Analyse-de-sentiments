import io
import pandas as pd
import re
import string

#train_df = pd.read_csv('data/train.csv')
#df = pd.read_csv('data/dev.csv', index_col=False, dtype={'note': 'string', 'commentaire': 'string'})


# encode in utf8
# remove punctuation
# convert to lowercase
# make a list of all the emojis in the string
# create regex finder for date, number, price formats
# create regex finder for repeted consecutive letters (more than 2 consecutive letters)
# make a dictionnary of all adjectives in the string

emojis = [['😀' , 'Grinning Face', 'U+1F600'],
    ['😃' , 'Grinning Face with Big Eyes', 'U+1F603'],
    ['😄' , 'Grinning Face with Smiling Eyes', 'U+1F604'],
    ['😁' , 'Beaming Face with Smiling Eyes', 'U+1F601'],
    ['😆' , 'Grinning Squinting Face', 'U+1F606'],
    ['😅' , 'Grinning Face with Sweat', 'U+1F605'],
    ['😂' , 'Face with Tears of Joy', 'U+1F602'],
    ['🤣' , 'Rolling on the Floor Laughing', 'U+1F923'],
    ['😊' , 'Smiling Face with Smiling Eyes', 'U+1F60A'],
    ['😇' , 'Smiling Face with Halo', 'U+1F607']]


# Function to replace emojis with "EMOJI"
def replace_emojis(text):
    return re.sub("|".join(emojis[:][0]), "<EM/>", text)


# Function to extract adjectives
def extract_adjectives(text):
    adjectives = {}
    words = text.split()
    for word in words:
        if word.lower() in adjectives_list:
            adjectives[word] = adjectives.get(word, 0) + 1
    return adjectives

# Read the dataframe from a CSV file or any other data source
# Replace 'your_data.csv' with your actual data source
f_dev = open(
    '/home/rim/Downloads/Corpus dapprentissage corpus de développement-20231107/donnees_appr_dev/donnees_appr_dev/dev.xml',
    "r")
dev_xml = re.sub("<(movie|review_id|name|user_id)>(.*)</(movie|review_id|name|user_id)>", '', f_dev.read())
df = pd.read_xml(io.StringIO(dev_xml))

# Specify the name of the column you want to process
column_name = 'commentaire'

df = (df.head())
print(df)

# Define a list of adjectives to identify
adjectives_list = ['awesome', 'amazing', 'fantastic', 'great', 'wonderful', 'beautiful']

# Step 1: Encode in UTF-8
#df[column_name] = df[column_name].apply(lambda x: x.encode('utf-8').decode('utf-8'))

# Step 2: Remove punctuation
df[column_name] = df[column_name].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# Step 3: Convert to lowercase
df[column_name] = df[column_name].str.lower()

# Step 4: Replace emojis with "EMOJI"
df[column_name] = df[column_name].apply(replace_emojis)

# Step 5: Create regex finder for date, number, and price formats
date_regex = r'\d{2}/\d{2}/\d{4}'
number_regex = r'\b\d+\b'
price_regex = r'\$\d+(\.\d{2})?'
lbreak = '\n'

df['date_format'] = df[column_name].apply(lambda x: re.findall(date_regex, x))
df['number_format'] = df[column_name].apply(lambda x: re.findall(number_regex, x))
df['price_format'] = df[column_name].apply(lambda x: re.findall(price_regex, x))

# Step 6: Create regex finder for repeated consecutive letters (more than 2 consecutive letters)
repeated_letters_regex = r'(\w)\1{2,}'

df['repeated_letters'] = df[column_name].apply(lambda x: re.findall(repeated_letters_regex, x))

# Step 7: Make a dictionary of all adjectives in the string
df['adjectives'] = df[column_name].apply(extract_adjectives)

# Print or save the resulting dataframe
print(df)
# To save to a new CSV file, you can use df.to_csv('output.csv', index=False)

df.to_csv('data/head_dev.csv')