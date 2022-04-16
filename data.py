import pandas as pd
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

all_data = pd.read_excel('data/rapoarte2.xlsx')
temp = all_data.iloc[:, 2:4].values
nltk.download('stopwords')

corpus = []

all_stopwords = stopwords.words('romanian')
sn = SnowballStemmer('romanian')

for i in range(0, all_data.shape[0]):
    raport = re.sub('[^a-zA-Z]', ' ', all_data["Text Incident"][i])
    raport = raport.lower()
    raport = raport.split()
    raport = [sn.stem(word) for word in raport if not word in set(all_stopwords)]
    raport = ' '.join(raport)
    corpus.append(raport)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(strip_accents='unicode')

X = cv.fit_transform(corpus).toarray()
y = all_data.loc[:,'Categorie'].values
print(X)

from sklearn.preprocessing import LabelEncoder, StandardScaler
encoder = LabelEncoder()
y = encoder.fit_transform(y)

#print(np.concatenate((y_encoded.reshape(len(y_encoded), 1), y.reshape(len(y), 1)), 1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
a = accuracy_score(y_test, y_pred)
print(a)