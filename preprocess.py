import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from unidecode import unidecode
import re

nltk.download('stopwords')
ro_stopwords = stopwords.words('romanian')
sn = SnowballStemmer(language='romanian')

def to_lower(data):
    return data.lower()

def remove_diacritics(data):
    return unidecode(data)

def remove_symbols(data):
    symbols = "!?@\"#$%*+-.[\]/:;<=>^_`{|}~\n\'&()\t\r,"
    for i in symbols:
        data = data.replace(i, ' ')
    return data

def remove_stopwords(data):
    data = data.split()

    new_data = ""
    for word in data:
        if word not in ro_stopwords:
            new_data = new_data + word + " "
    return new_data

def remove_numbers(data):
    data = re.sub(r'[0-9]+', '', data)
    return data

def stem(data):
    data = data.split()
    new_data = ""
    for word in data:
        new_data = new_data + sn.stem(word) + " "
    return new_data