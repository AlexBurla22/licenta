from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time 
import pandas as pd
import os.path

url_login = 'https://sibiu.eventya.eu/users/login'
url_app = 'https://sibiu.eventya.eu/app'
url_root = 'https://sibiu.eventya.eu'
url_incidents = 'https://sibiu.eventya.eu/app/incidents/incidents'

email = ''
password = ''

options = Options()
options.headless = True

driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

def sort_csv(depart):
    path = 'data/rapoarte' + depart + '.csv'
    if os.path.exists(path):
        reps = pd.read_csv(path)
        reps = reps.sort_values(['number'], ascending = False)
        reps.to_csv(path, index=False)


def login():
    driver.get(url_login)
    driver.find_element(By.ID, 'user_email').send_keys(email)
    driver.find_element(By.ID,'user_password').send_keys(password)
    driver.find_element(By.ID,'sign_in').click()

def logout():
    driver.get(url_app)
    driver.find_element(By.ID, 'user_dropdown_toggle').click()
    driver.find_element(By.XPATH, '/html/body/header/nav/div/div[3]/div/a[3]').click()
    driver.close()

def scrape_text(url):
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    sesizare_lista = soup.find('div', {'class': 'list-group list-group-flush'})
    items = sesizare_lista.find_all('div', {'class': 'list-group-item'})
    time.sleep(2)
    return items[1].text

def get_last_number(department_key):
    path = 'data/rapoarte' + department_key + '.csv'
    if os.path.exists(path):
        with open(path) as file:
            for i, line in enumerate(file):
                if i == 1:
                    return line.split(',')[1]
    
    return 0

def scrape():
    url_to_scrape = url_incidents + '?status_key=11&department_key='
    driver.get(url_to_scrape)

    html_page = driver.page_source
    soup = BeautifulSoup(html_page, 'html.parser')

    departamente = soup.find('select', {'id': 'department_key'})
    valori = departamente.find_all('option')

    valori_departamente = []
    nume_departamente = []
    for valoare in valori:
        if valoare['value'] != '31':
        #print('0' if valoare['value'] == '' else valoare['value'], valoare.text)
            valori_departamente.append('0' if valoare['value'] == '' else valoare['value'])
            nume_departamente.append(valoare.text)

    depts = pd.DataFrame(
        {
            'nume_departament': nume_departamente,
            'cheie': valori_departamente
        }
    )
    #print(depts)
    depts.to_csv('data/departamente.csv', index=False)

    for val in valori_departamente:
        lista_link = []
        lista_numar = []
        lista_text = []

        numar_ultim_raport = get_last_number(val)
        print('crawling with department key: ' + val)
        if val == '0':
            url_to_scrape = url_incidents + '?status_key=11&department_key='
        else:
            url_to_scrape = url_incidents + '?status_key=11&department_key=' + val
        driver.get(url_to_scrape)
        
        not_scraped = True

        while not_scraped:
            html_page = driver.page_source
            #print(html_page)
            current_url = driver.current_url
            soup = BeautifulSoup(html_page, 'html.parser')
            rapoarte = soup.find('div', {'id': 'list_incidents'}).find_all('div', recursive=False)

            for raport in rapoarte:
                media_div = raport.find('div', {'class': 'media'})
                link = url_root + media_div.find('a', {'class': 'stretched-link'})['href']
                number = media_div.find('div', {'class': 'col-lg-2 col-md-4 col-sm-4'}).find('p', {'class': 'mb-0'}).text
                number = number[1:]

                if number == numar_ultim_raport:
                    not_scraped = False
                    break

                #print(link, number, '0' if val[0] == '' else val[0])
                lista_link.append(link)
                lista_numar.append(number)
                lista_text.append(scrape_text(link))
                print('scraped ' + link)

            driver.get(current_url)

            button_group = soup.find('div', {'class': 'btn-group btn-box'})
            button_next = button_group.find('i', {'class': 'fa fa-chevron-right'}).parent
            clase = button_next.attrs['class']

            clase_string = ''
            for clasa in clase:
                clase_string += clasa 

            if 'disabled' in clase_string or not_scraped == False:
                break
            else:
                driver.find_element(By.XPATH, '/html/body/main/div/div/div[2]/div[2]/div/div[2]/div/a[2]/button').click()
                time.sleep(2)
        
        reps = pd.DataFrame(
            {
                'link': lista_link,
                'number': lista_numar,
                'text': lista_text
            }
        )
        #print(rapoarte)

        path = 'data/rapoarte' + val+ '.csv'
        if os.path.exists(path):
            reps.to_csv(path, mode='a', index=False, header=False)
        else:
            reps.to_csv(path, index=False)

        sort_csv(val)

login()
scrape()
logout()