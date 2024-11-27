import time
import os, sys
import streamlit as st

from selenium import webdriver
from selenium.webdriver import FirefoxOptions
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

@st.cache_resource
def installff():
  os.system('sbase install geckodriver')
  os.system('ln -s /home/appuser/venv/lib/python3.7/site-packages/seleniumbase/drivers/geckodriver /home/appuser/venv/bin/geckodriver')

_ = installff()

opts = FirefoxOptions()
opts.add_argument("--headless")

options = Options()
options.add_argument('--disable-gpu')
options.add_argument('--headless')

url = "https://sports.chosun.com/football/?action=worldfootball"

# streamlit
st.title('해외축구 뉴스 요약')

key_word = st.text_input("검색하고 싶은 키워드를 입력하세요!")
view_count = st.text_input("보고싶은 기사의 수를 입력하세요!", 1)

if st.button("검색 시작"):
    if not key_word:
        st.error("키워드를 입력해주세요!!")
    else:
        driver = webdriver.Firefox(options=opts)
        driver.maximize_window()
        driver.get(url)
        driver.quit()

st.write("완료!")