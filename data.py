import pandas as pd
import numpy as np
import preprocess

all_data = pd.read_excel('data/rapoarte.xlsx')

corpus = []

for row in range(0, all_data.shape[0]):
    raport = all_data["Text Incident"][row]
    raport = preprocess.to_lower(raport)
    raport = preprocess.remove_symbols(raport)
    raport = preprocess.remove_stopwords(raport)
    raport = preprocess.remove_diacritics(raport)
    raport = preprocess.remove_numbers(raport)
    raport = preprocess.stem(raport)
    corpus.append(raport)

from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=3400)

response = cv.fit_transform(corpus)
print(response)
X = response.toarray()
y = all_data.loc[:,'Categorie'].values
print(X.shape)

#print(np.concatenate((y_encoded.reshape(len(y_encoded), 1), y.reshape(len(y), 1)), 1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
a = accuracy_score(y_test, y_pred)
print(a)