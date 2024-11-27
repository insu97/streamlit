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
        return webdriver.Chrome()
    except:
        firefoxOptions = Options()
        firefoxOptions.add_argument("--headless")
        service = Service(GeckoDriverManager().install())
        return webdriver.Firefox(options=firefoxOptions, service=service)


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
        driver.maximize_window()
        driver.get(url)
        driver.quit()

st.write("완료!")