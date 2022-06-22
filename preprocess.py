from nltk.stem.snowball import SnowballStemmer
from unidecode import unidecode
import re
import spacy

class PreProcessor():
    def __init__(self):
        with open('stopwords.txt') as f:
            self.ro_stopwords = f.read().splitlines()
        self.sn = SnowballStemmer(language='romanian')
        self.sp = spacy.load('ro_core_news_sm')

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

    def remove_nonwords(self, data):
        data = data.split()
        new_data = ''
        for word in data:
            if len(word) > 3:
                new_data += word + ' '
        return new_data

    def stem(self, data):
        data = data.split()
        new_data = ""
        for word in data:
            new_data = new_data + self.sn.stem(word) + " "
        return new_data
    
    def lem(self, data):
        tokens = self.sp(data)
        new_data =''
        for token in tokens:
            new_data += token.lemma_.lower() + ' '
        return new_data