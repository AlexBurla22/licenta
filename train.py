import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from preprocess import PreProcessor

def read_all_csv(departs):
    all_files = pd.DataFrame(columns=['link', 'number', 'text', 'cheie_depart'])
    for depart in departs:
        file = pd.read_csv('data/rapoarte' + str(depart) + '.csv')
        file['cheie_depart'] = depart
        all_files = pd.concat([all_files, file], ignore_index=True)
    
    return all_files

departs = pd.read_csv('data/departamente.csv')
all_data = read_all_csv(departs.loc[:, 'cheie'])
all_data = all_data.sample(frac = 1, random_state = 0).reset_index(drop=True)

#print(all_data)
corpus = []

pp = PreProcessor()

for row in range(0, all_data.shape[0]):
    raport = all_data['text'][row]
    raport = pp.to_lower(raport)
    raport = pp.remove_symbols(raport)
    raport = pp.remove_stopwords(raport)
    raport = pp.remove_diacritics(raport)
    raport = pp.remove_numbers(raport)
    raport = pp.remove_singular_letters(raport)
    raport = pp.stem(raport)
    corpus.append(raport)

from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()

response = cv.fit_transform(corpus)
print(response)
X = response.toarray()
y = all_data.loc[:,'cheie_depart'].values
print(X.shape)

#print(np.concatenate((y_encoded.reshape(len(y_encoded), 1), y.reshape(len(y), 1)), 1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
a = accuracy_score(y_test, y_pred)
print(a)