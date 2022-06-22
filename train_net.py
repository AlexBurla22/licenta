import pandas as pd
import numpy as np
import sklearn
from preprocess import PreProcessor
import tensorflow as tf
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from keras import Sequential
import matplotlib.pyplot as plt
import seaborn as sb
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Conv1D, GlobalAveragePooling1D, SimpleRNN, Dropout

pp = PreProcessor()
corpus = []

def plot_confusion_matrix(y_true, y_pred, labels):
    from sklearn.metrics import confusion_matrix
    mc = confusion_matrix(y_true, y_pred)
    sb.set()
    ax = sb.heatmap(mc, cmap='GnBu', annot=True, fmt='d', cbar=False)
    ax.set_xlabel("Clasă predicționată")
    ax.set_ylabel("Clasă adevărată")
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels, rotation=360)
    plt.tight_layout()
    plt.show()

def read_all_csv(departs):
    all_files = pd.DataFrame(columns=['link', 'number', 'text', 'cheie_depart'])
    for depart in departs:
        if(depart != 0):
            file = pd.read_csv('data/rapoarte' + str(depart) + '.csv', nrows=5000)
            file['cheie_depart'] = depart
            all_files = pd.concat([all_files, file], ignore_index=True)
    return all_files

def create_corpus(data):
    for row in range(0, data.shape[0]):
        raport = data['text'][row]
        raport = pp.to_lower(raport)
        raport = pp.remove_symbols(raport)
        raport = pp.remove_numbers(raport)
        raport = pp.remove_nonwords(raport)
        raport = pp.remove_diacritics(raport)
        raport = pp.remove_stopwords(raport)
        #raport = pp.lem(raport)
        raport = pp.stem(raport)
        corpus.append(raport)

departs = pd.read_csv('data/departamente.csv')
all_data = read_all_csv(departs.loc[:, 'cheie'])
all_data = all_data.sample(frac = 1, random_state = 1000).reset_index(drop=True)

create_corpus(all_data)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

labels = all_data.loc[:,'cheie_depart'].values
le.fit_transform(labels)
# with open('models/labels.pickle', 'wb') as handle:
#     pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(corpus, labels, test_size = 0.25)

y_train=le.transform(y_train)
y_test=le.transform(y_test)

from keras.preprocessing.text import Tokenizer
tkz = Tokenizer()

tkz.fit_on_texts(X_train)

with open('models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tkz, handle, protocol=pickle.HIGHEST_PROTOCOL)

X_train = tkz.texts_to_sequences(X_train)
X_test = tkz.texts_to_sequences(X_test)

vocab_size = len(tkz.word_index) + 1  # Adding 1 because of reserved 0 index

max_len = 600

X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding='post', maxlen=max_len)

#encoded_reviews = [tf.keras.preprocessing.text.one_hot(d, vocab_size) for d in corpus]
#print(max(len(r) for r in encoded_reviews))
#padded_reviews = tf.keras.preprocessing.sequence.pad_sequences(encoded_reviews, maxlen=max_length, padding='post')

model = Sequential()
embedding_dim = 50
model.add(Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=max_len))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(departs.index), activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train,
            epochs=10,
            verbose=False,
            validation_data=(X_test, y_test),
            batch_size=128)

model.save('models/net')

# ac = model.evaluate(X_train, y_train, verbose=False)
# print("Accuracy train: {:.4f}".format(ac))
# ac = model.evaluate(X_test, y_test, verbose=False)
# print("Accuracy test:  {:.4f}".format(ac))

predNET = model.predict(X_test, verbose=False)
predNET = predNET.argmax(axis=1)
predNET = le.inverse_transform(predNET)
predNET = np.array([int(i) for i in predNET])

y_true=le.inverse_transform(y_test)
y_true=np.array([int(i) for i in y_true])

print(classification_report(y_true, predNET, target_names=departs['nume_departament']))
plot_confusion_matrix(y_true, predNET, labels=departs['nume_departament'])