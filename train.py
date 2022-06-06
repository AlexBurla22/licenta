import pandas as pd
import numpy as np
from preprocess import PreProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import pickle

classifierSVM = LinearSVC(C=0.3)
classifierNB = MultinomialNB(alpha=0.1)
vectorizer = TfidfVectorizer(max_features=7000)
pp = PreProcessor()
corpus = []

def read_all_csv(departs):
    all_files = pd.DataFrame(columns=['link', 'number', 'text', 'cheie_depart'])
    for depart in departs:
        if(depart != 0):
            file = pd.read_csv('data/rapoarte' + str(depart) + '.csv')
            file['cheie_depart'] = depart
            all_files = pd.concat([all_files, file], ignore_index=True)
    return all_files

def print_gridsearch(model, parameters, X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(estimator = model,
        param_grid=parameters,
        scoring='accuracy',
        cv=10)
    grid_search.fit(X_train, y_train)
    best_acc = grid_search.best_score_
    best_params = grid_search.best_params_
    print('Best accuracy: {:.2f} %'.format(best_acc * 100))
    print('Best parameters: ', best_params)

def print_cross_validation(model, X_train, y_train, folds=10):
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = folds)
    print("Acurracy for model: {:.2f} %".format(accuracies.mean() * 100))
    print("Standard deviation for model {:.2f} %".format(accuracies.std() * 100))

def print_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_true, y_pred))

def print_classification_report(y_true, y_pred, targets):
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred, target_names=targets))

def save_model(filename, model):
    pickle.dump(model, open(filename, 'wb'))

def create_corpus(data):
    for row in range(0, data.shape[0]):
        raport = data['text'][row]
        raport = pp.to_lower(raport)
        raport = pp.remove_symbols(raport)
        #raport = pp.remove_diacritics(raport)
        raport = pp.remove_numbers(raport)
        raport = pp.remove_nonwords(raport)
        raport = pp.stem(raport)
        #raport = pp.lem(raport)
        raport = pp.remove_diacritics(raport)
        raport = pp.remove_stopwords(raport)
        corpus.append(raport)

departs = pd.read_csv('data/departamente.csv')
all_data = read_all_csv(departs.loc[:, 'cheie'])
all_data = all_data.sample(frac = 1, random_state = 1000).reset_index(drop=True)

create_corpus(all_data)

X = corpus
y = all_data.loc[:,'cheie_depart'].values
y = [int(i) for i in y]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

names = vectorizer.get_feature_names_out()
with open('features.txt', 'w', encoding='utf-8') as f:
    for word in names:
        f.write(word + '\n')

classifierSVM.fit(X_train, y_train)
y_pred = classifierSVM.predict(X_test)

classifierNB.fit(X_train, y_train)
y2_pred = classifierNB.predict(X_test)

save_model('models/vectorizer.sav', vectorizer)

#save_model('models/linearSVM.sav', classifierSVM)
#save_model('models/multinomialNB.sav', classifierNB)

#print_classification_report(y_test, y_pred, departs['nume_departament'])
#print_classification_report(y_test, y2_pred, departs['nume_departament'])

#print_cross_validation(classifierSVM, X_train, y_train)
#print_cross_validation(classifierNB, X_train, y_train)

#parameters = [{'C': np.linspace(0.1, 1, 10)}]
#print_gridsearch(classifierSVM, parameters, X_train, y_train)

#parameters = [{'alpha': np.linspace(0.1, 1, 10)}]
#print_gridsearch(classifierNB, parameters, X_train, y_train)