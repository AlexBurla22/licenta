import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from unidecode import unidecode
import re

class PreProcessor():
    def __init__(self):
        nltk.download('stopwords')
        self.ro_stopwords = stopwords.words('romanian')
        self.ro_stopwords.append('buna')
        self.ro_stopwords.append('bună')
        self.ro_stopwords.append('ziua')
        self.ro_stopwords.append('seara')
        self.ro_stopwords.append('seară')
        self.ro_stopwords.append('azi')
        self.ro_stopwords.append('astazi')
        self.ro_stopwords.append('astăzi')
        self.ro_stopwords.append('si')
        self.sn = SnowballStemmer(language='romanian')

    def to_lower(self, data):
        return data.lower()

    def remove_diacritics(self, data):
        return unidecode(data)

    def remove_symbols(self, data):
        symbols = "!?@\"#$%*+-.[\]/:;<=>^_`{|}~\n\'&()\t\r,"
        for i in symbols:
            data = data.replace(i, ' ')
        return data

    def remove_stopwords(self, data):
        data = data.split()

        new_data = ""
        for word in data:
            if word not in self.ro_stopwords:
                new_data = new_data + word + " "
        return new_data

    def remove_numbers(self, data):
        data = re.sub(r'[0-9]+', '', data)
        return data

    def remove_singular_letters(self, data):
        data = data.split()
        new_data = ''
        for word in data:
            if len(word) > 1:
                new_data += word + ' '
        return new_data

    def stem(self, data):
        data = data.split()
        new_data = ""
        for word in data:
            new_data = new_data + self.sn.stem(word) + " "
        return new_data