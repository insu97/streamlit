import streamlit as st
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


@st.cache_resource
def get_driver():
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    return webdriver.Firefox(options=options)


st.title("Web Scraping with Selenium and Streamlit")

url = st.text_input("Enter the URL to scrape:")

if url:
    try:
        driver = get_driver()
        driver.get(url)

        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        st.write("Page title:", driver.title)
        st.code(driver.page_source[:1000])  # Display first 1000 characters of page source
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        if 'driver' in locals():
            driver.quit()