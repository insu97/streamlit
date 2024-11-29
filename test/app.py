import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


@st.experimental_singleton
def get_driver():
    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


st.title("Web Scraping with Selenium and Streamlit")

url = st.text_input("Enter the URL to scrape:")

if url:
    driver = get_driver()
    driver.get(url)

    try:
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        st.write("Page title:", driver.title)
        st.code(driver.page_source[:1000])  # Display first 1000 characters of page source
    finally:
        driver.quit()