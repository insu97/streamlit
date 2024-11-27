import time
import os
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

def install_dependencies():
    st.write("시스템 의존성 설치 중...")
    os.system("sudo apt-get update")
    os.system("sudo apt-get install -y libglib2.0-0 libnss3 libgconf-2-4")
    st.write("설치 완료!")

if st.button("의존성 설치"):
    try:
        install_dependencies()
    except Exception as e:
        st.error(f"설치 중 오류 발생: {e}")


# Chrome 설정
def get_chrome_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # webdriver-manager 경로를 명시적으로 설정
    chromedriver_path = "/home/appuser/.wdm/drivers/chromedriver/linux64/114.0.5735.90/chromedriver"
    if not os.path.exists(chromedriver_path):
        raise FileNotFoundError(f"Chromedriver 경로를 찾을 수 없습니다: {chromedriver_path}")

    service = Service(chromedriver_path)
    return webdriver.Chrome(service=service, options=chrome_options)


url = "https://sports.chosun.com/football/?action=worldfootball"

# streamlit
st.title('해외축구 뉴스 요약')

key_word = st.text_input("검색하고 싶은 키워드를 입력하세요!")
view_count = st.text_input("보고싶은 기사의 수를 입력하세요!", 1)

if st.button("검색 시작"):
    if not key_word:
        st.error("키워드를 입력해주세요!!")
    else:
        driver = get_chrome_driver()
        driver.get(url)
        driver.quit()

st.write("완료!")