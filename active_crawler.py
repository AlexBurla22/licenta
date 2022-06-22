import pickle
import pandas as pd
import numpy as np
import firebase_admin
import tensorflow
import time
import smtplib
from email.mime.text import MIMEText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from keras.preprocessing.text import Tokenizer
from keras import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Conv1D, GlobalAveragePooling1D
from preprocess import PreProcessor
from firebase_admin import credentials, db
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from collections import Counter
from random import randint
from unidecode import unidecode
url_login = 'https://sibiu.eventya.eu/users/login'
url_waiting = 'https://sibiu.eventya.eu/app/incidents/incidents?status_key=9'
url_app = 'https://sibiu.eventya.eu/app'
url_root = 'https://sibiu.eventya.eu'

email='alexandru.burla@ulbsibiu.ro'
mail_test='test.crawler99@gmail.com'
password=''
db_URL = 'https://crawlerdb-320d6-default-rtdb.europe-west1.firebasedatabase.app/'

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

def write_departs_in_db():
    db_ref=db.reference('/Departments')
    for i in range(0, len(departs)):
        db_obj = {
            'departID': str(departs['cheie'][i]),
            'departName': departs['nume_departament'][i],
            'departMail': mail_test
        }
        db_ref.push(db_obj)

def get_depart_name_by_key(key):
    return departs.loc[departs['cheie'] == key, 'nume_departament'].iloc[0]

def scrape(url):
    print('\nscraping links...')
    temp_reps=[]
    temp_reps_number=[]
    db_ref = db.reference('/ClassifiedReports')
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
                print('--scraped link for report nr. ', number)
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
    print('\nscraping text...')
    reports_text.clear()
    for report in reports:
        print('--scraping text from link ' + report)
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

def write_to_db(results):
    db_ref = db.reference('/ClassifiedReports')
    if len(reports) > 0:
        for i in range(len(reports)):
            db_obj = { 'link': str(reports[i]),
                        'number': reports_number[i],
                        'departmentID' : str(results[i]) }
            db_ref.push(db_obj)

def get_email_by_department_key(key):
    db_ref=db.reference('/Departments')
    db_cont=db_ref.get()
    if not db_cont:
        return ''
    else:
        for item in db_cont.items():
            if item[1]['departID'] == key:
                return str(item[1]['departMail'])
    return ''

def get_name_by_department_key(key):
    db_ref=db.reference('/Departments')
    db_cont=db_ref.get()
    if not db_cont:
        return ''
    else:
        for item in db_cont.items():
            if item[1]['departID'] == key:
                return str(item[1]['departName'])
    return ''

def send_emails(results):
    for i in range(len(reports)):
        time.sleep(3)
        send_email(str(reports[i]), str(results[i]))

def send_email(content, key):
    sen='alexandru.burla@ulbsibiu.ro'
    depart_name=get_name_by_department_key(key)
    dest=get_email_by_department_key(key)
    text="""\Buna ziua, \n\tA aparut o sesizare noua cu linkul:\n""" + content

    m = MIMEText(text, 'plain')
    m['To']=dest
    m['From']=sen
    m['Subject']='Sesizare noua pentru departament: ' + unidecode(depart_name)

    s = smtplib.SMTP('localhost', 1025)
    s.sendmail(sen, [dest], m.as_string())
    print('Sent email to department: ' + depart_name + ' with email address: ' + dest)
    s.quit()

def predict(predSVM, predNB, predNET):
    res = []
    for i in range(len(reports)):
        temp = Counter([predSVM[i], predNB[i], predNET[i]])
        if len(temp.keys()) > 2:
            res.append(list(temp.keys())[randint(0, 2)])
        elif len(temp) < 2:
            res.append(predNET[i])
        else:
            max_freq = max(temp.values())
            res.append(list(temp.keys())[list(temp.values()).index(max_freq)])
    return res

login(url_login=url_login, email=email, password=password)
write_departs_in_db()
while True:
    reports, reports_number = scrape(url=url_waiting)
    process()
    predsSVM, predsNB, predsNET = classify()
    results = predict(predsSVM, predsNB, predsNET)
    write_to_db(results)
    send_emails(results)
    time.sleep(100)