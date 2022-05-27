import pandas as pd
import numpy as np
from preprocess import PreProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

classifier = LinearSVC()
vectorizer = TfidfVectorizer()
pp = PreProcessor()
corpus = []

def predict_raport(text):
    text = pp.to_lower(text)
    text = pp.remove_symbols(text)
    text = pp.remove_stopwords(text)
    text = pp.remove_diacritics(text)
    text = pp.remove_numbers(text)
    text = pp.remove_nonwords(text)
    text = pp.stem(text)
    text_x = vectorizer.transform([text])
    text_y = classifier.predict(text_x)
    t = departs.loc[departs['cheie'] == text_y[0]]
    return t

def read_all_csv(departs):
    all_files = pd.DataFrame(columns=['link', 'number', 'text', 'cheie_depart'])
    for depart in departs:
        file = pd.read_csv('data/rapoarte' + str(depart) + '.csv')
        file['cheie_depart'] = depart
        all_files = pd.concat([all_files, file], ignore_index=True)
    return all_files

def create_corpus(data):
    for row in range(0, data.shape[0]):
        raport = data['text'][row]
        raport = pp.to_lower(raport)
        raport = pp.remove_symbols(raport)
        raport = pp.remove_diacritics(raport)
        raport = pp.remove_stopwords(raport)
        raport = pp.remove_numbers(raport)
        raport = pp.remove_nonwords(raport)
        raport = pp.stem(raport)
        corpus.append(raport)

departs = pd.read_csv('data/departamente.csv')
all_data = read_all_csv(departs.loc[:, 'cheie'])
all_data = all_data.sample(frac = 1, random_state=99).reset_index(drop=True)

create_corpus(all_data)
tfidf_vect = vectorizer.fit_transform(corpus)

names = vectorizer.get_feature_names_out()
with open('features.txt', 'w') as f:
    for word in names:
        f.write(word + '\n')

X = tfidf_vect.toarray()
y = all_data.loc[:,'cheie_depart'].values
y = [int(i) for i in y]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=99)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#print(confusion_matrix(y_test, y_pred))
#print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names = departs['nume_departament']))