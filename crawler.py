from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import hashlib

url_login = 'https://sibiu.eventya.eu/users/login'
url_app = 'https://sibiu.eventya.eu/app'
email = ''
password = ''

options = Options()
options.headless = True

driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

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

def scrape():
    driver.get('https://sibiu.eventya.eu/app/incidents/incidents?status_key=11')
    html_page = driver.page_source
    #print(html_page)
    soup = BeautifulSoup(html_page, 'html.parser')
    rapoarte = soup.find_all('div', {'class': 'list-group-item list-group-item-action text-muted' })
    for raport in rapoarte:
        media_div = raport.find('div', {'class': 'media'})
        link=media_div.find('a', {'class': 'stretched-link'})['href']
        number=media_div.find('div', {'class': 'class="col-lg-2 col-md-4 col-sm-4"'})
        print()
        break

login()
scrape()
logout()