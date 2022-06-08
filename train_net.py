import pandas as pd
import numpy as np
import sklearn
from preprocess import PreProcessor
import tensorflow as tf
import pickle

from keras import Sequential
#from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Conv1D, GlobalAveragePooling1D

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
all_data = all_data.sample(frac = 1, random_state = 100).reset_index(drop=True)

create_corpus(all_data)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

labels = all_data.loc[:,'cheie_depart'].values
encoded_labels = le.fit_transform(labels)
with open('models/labels.pickle', 'wb') as handle:
    pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(corpus, encoded_labels, test_size = 0.25)

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=7000)

tokenizer.fit_on_texts(X_train)

with open('models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
print(X_train[2])

max_len = 600

X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding='post', maxlen=max_len)

#encoded_reviews = [tf.keras.preprocessing.text.one_hot(d, vocab_size) for d in corpus]
#print(max(len(r) for r in encoded_reviews))
#padded_reviews = tf.keras.preprocessing.sequence.pad_sequences(encoded_reviews, maxlen=max_length, padding='post')

model = Sequential()
#embedding_layer = Embedding(input_dim=vocab_size,output_dim=8,input_length=max_length)
embedding_dim = 18
model.add(Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=max_len))
#model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(25, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(len(departs.index), activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(X_train, y_train,
                     epochs=100,
                     verbose=False,
                     validation_data=(X_test, y_test),
                     batch_size=10)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

model.save('models/net')
#model.fit(padded_reviews, labels, epochs=100, verbose=0)