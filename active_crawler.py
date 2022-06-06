import pickle
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

filename = 'models/linearSVM.sav'
classifierSVM = pickle.load(open(filename, 'rb'))

filename = 'models/multinomialNB.sav'
classifierNB = pickle.load(open(filename, 'rb'))