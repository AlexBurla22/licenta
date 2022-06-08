import pickle
import pandas as pd
import numpy as np
import tensorflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from keras.preprocessing.text import Tokenizer
from keras import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Conv1D, GlobalAveragePooling1D
from preprocess import PreProcessor
import firebase_admin
from firebase_admin import credentials, db
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time 

url_login = 'https://sibiu.eventya.eu/users/login'
url_waiting = 'https://sibiu.eventya.eu/app/incidents/incidents?status_key=9'
url_app = 'https://sibiu.eventya.eu/app'
url_root = 'https://sibiu.eventya.eu'

email=''
password=''
db_URL = ''

departs = pd.read_csv('data/departamente.csv')
pp = PreProcessor()

reports = []
reports_number = []
reports_text = []
options = Options()
options.headless = True
driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

vectorizer = pickle.load(open('models/vectorizer.sav','rb'))
classifierSVM = pickle.load(open('models/linearSVM.sav', 'rb'))
classifierNB = pickle.load(open('models/multinomialNB.sav', 'rb'))
tokenizer = pickle.load(open('models/tokenizer.pickle', 'rb'))
le = pickle.load(open('models/labels.pickle', 'rb'))
net = tensorflow.keras.models.load_model('models/net')

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, { 'databaseURL': db_URL})
db_ref = db.reference('/ClassifiedReports')

def get_depart_name_by_key(key):
    return departs.loc[departs['cheie'] == key, 'nume_departament'].iloc[0]

def scrape(url):
    print('scraping...')
    temp_reps=[]
    temp_reps_number=[]
    db_content = db_ref.get()
    for depart in departs.loc[:, 'cheie']:
        driver.get(url+'&department_key=' + str(depart))
        time.sleep(2)
        divs = driver.find_element(By.ID, 'list_incidents').find_elements(By.CLASS_NAME, 'list-group-item')
        for div in divs:
            number = int(div.find_element(By.TAG_NAME, 'p').text[1:])
            link = div.find_element(By.TAG_NAME, 'a').get_attribute('href')
            if not is_number_in_database(db_content, number):
                temp_reps_number.append(number)
                temp_reps.append(link)     
                print('scraped link for report nr. ', number)
    return temp_reps, temp_reps_number

def is_number_in_database(db_content, number):
    if not db_content:
        return False
    else:
        for item in db_content.items():
            if item[1]['number'] == number:
                return True
    return False

def login(url_login, email, password):
    driver.get(url_login)
    driver.find_element(By.ID, 'user_email').send_keys(email)
    driver.find_element(By.ID,'user_password').send_keys(password)
    driver.find_element(By.ID,'sign_in').click()

def logout(url_app):
    driver.get(url_app)
    driver.find_element(By.ID, 'user_dropdown_toggle').click()
    driver.find_element(By.XPATH, '/html/body/header/nav/div/div[3]/div/a[3]').click()
    driver.close()

def preprocess_raport(raport):
    raport = pp.to_lower(raport)
    raport = pp.remove_symbols(raport)
    #raport = pp.remove_diacritics(raport)
    raport = pp.remove_numbers(raport)
    raport = pp.remove_nonwords(raport)
    #raport = pp.lem(raport)
    raport = pp.remove_diacritics(raport)
    raport = pp.remove_stopwords(raport)
    raport = pp.stem(raport)
    return raport

def process():
    print('scraping text...')
    reports_text.clear()
    for report in reports:
        driver.get(report)
        time.sleep(2)
        text_divs = driver.find_element(By.ID, 'user_messages_container').find_elements(By.CLASS_NAME, 'list-group-item')
        msg_text = text_divs[1].text
        reports_text.append(msg_text)
        
def classify():
    processed_texts = []
    if len(reports_text) > 0:
        for text in reports_text:
            text = preprocess_raport(text)
            processed_texts.append(text)
        tokenized_text = tokenizer.texts_to_sequences(processed_texts)
        tokenized_text = tensorflow.keras.preprocessing.sequence.pad_sequences(tokenized_text, padding='post', maxlen=600)
        vect_text = vectorizer.transform(processed_texts)

        predNET = net.predict(tokenized_text)
        predNET = predNET.argmax(axis=1)
        predNET = le.inverse_transform(predNET)
        predNET = np.array([int(i) for i in predNET])
        return classifierSVM.predict(vect_text), classifierNB.predict(vect_text), predNET
    else:
        return [], [], []

def write_to_db():
    if len(reports) > 0:
        for i in range(len(reports)):
            db_obj = { 'link': str(reports[i]),
                        'number': reports_number[i],
                        'PredSVM': str(predsSVM[i]),
                        'PredNB': str(predsNB[i]),
                        'PredNET': str(predsNET[i]) }
            db_ref.push(db_obj)
    
def send_emails():
    yield

login(url_login=url_login, email=email, password=password)
while True:
    reports, reports_number = scrape(url=url_waiting)
    process()
    predsSVM, predsNB, predsNET=classify()
    write_to_db()
    send_emails()