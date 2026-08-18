# Data Collection & Creation Using Web Scraping- Static and Dynamic Webpages 

# Import Required Libraries
import requests 
from bs4 import BeautifulSoup
import pandas as pd

# Scraping a static webpage
url ="https://www.geeksforgeeks.org/aptitude/aptitude-questions-and-answers/"
response = requests.get(url)
soup= BeautifulSoup(response.text, "html.parser")
print("=== Page Title ===")
print(soup.title.text)

# Extract all Headings
print("==== Headings ====")
headings= soup.find_all("h2")
for heading in headings:
    print(heading.text)

# Extract Hyperlinks
print(" ==== Hyperlinks =====")
links=soup.find_all("a")
for link in links:
    print(link.get("href"))

# Store Data in a Pandas Dataframe
titles=["News1","News2","News3"]
df=pd.DataFrame({
    "Titles":titles
})
print("=== Data Frame ===")
print(df)

# Save dataset as CSV
df.to_csv("news.csv",index=False)
print("\nCSV File saved successfully")

# Export Dataset to Excel
df.to_excel("dataset.xlsx", index= False)
print("EXcel file saved successfully")

# Web scraping using selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
driver= webdriver.Chrome()
driver.get("https://www.geeksforgeeks.org/aptitude/aptitude-questions-and-answers/")
print("\n === Dynamic Hadings===")
elements = driver.find_elements(By.TAG_NAME, "h2")


for item in elements:
    print(item.text)

driver.quit()
