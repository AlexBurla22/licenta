import requests
from bs4 import BeautifulSoup
import hashlib

URL = "https://sibiucityapp.ro/ro/places?cat_type=expriente-in-sibiu_1541573835"
page = requests.get(URL)

print(page.text)
soup = BeautifulSoup(page.content, 'html.parser')
results = soup.find(id="places_list_container")
print(results)

# job_elements = results.find_all("div", class_="card-content")

# python_jobs = results.find_all(
#     "h2", string=lambda text: "python" in text.lower()
# )

# python_job_elements = [
#     h2_element.parent.parent.parent for h2_element in python_jobs
# ]

jobs_hash = hashlib.md5(results.encode('utf-8'))

print(jobs_hash.hexdigest(), end='\n'*2)

# for python_job_element in python_job_elements:
#     title_element = python_job_element.find("h2", class_="title")
#     company_element = python_job_element.find("h3", class_="company")
#     location_element = python_job_element.find("p", class_="location")
#     links = python_job_element.find_all("a")

#     for link in links:
#         if link.text == "Apply":
#             link_url = link["href"]
#             print(link_url)

#     print(title_element.text.strip())
#     print(company_element.text.strip())
#     print(location_element.text.strip())
#     print()