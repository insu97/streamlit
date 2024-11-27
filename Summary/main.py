import time
import streamlit as st
# seleniumbase
from selenium import webdriver
from seleniumbase import Driver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager




def get_driver():
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run Chrome in headless mode
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=chrome_options)
    except Exception as e:
        st.write(f"Chrome initialization failed: {e}")
        try:
            firefoxOptions = Options()
            firefoxOptions.add_argument("--headless")
            service = Service(GeckoDriverManager().install())
            return webdriver.Firefox(options=firefoxOptions, service=service)
        except Exception as e:
            print(f"Firefox initialization failed: {e}")
            st.error("Unable to initialize a browser driver.")
            return None


url = "https://sports.chosun.com/football/?action=worldfootball"

# streamlit
st.title('해외축구 뉴스 요약')

key_word = st.text_input("검색하고 싶은 키워드를 입력하세요!")
view_count = st.text_input("보고싶은 기사의 수를 입력하세요!", 1)

if st.button("검색 시작"):
    if not key_word:
        st.error("키워드를 입력해주세요!!")
    else:
        driver = get_driver()
        driver.get(url)
        driver.quit()

st.write("완료!")