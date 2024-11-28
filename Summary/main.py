import streamlit as st
from bs4 import BeautifulSoup
from selenium import webdriver
from urllib.request import Request, urlopen
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# streamlit 설정
st.set_page_config(layout="wide")

# streamlit
st.title('해외축구 뉴스 요약')

key_word = st.text_input("검색하고 싶은 키워드를 입력하세요!")

view_count = st.text_input("보고싶은 기사의 수를 입력하세요!")

if st.button("검색 시작하기"):
    # 크롤링 시작
    url = 'https://sports.chosun.com/football/?action=worldfootball'
    options = webdriver.ChromeOptions()
    # Chrome 옵션 설정
    options.add_argument("--headless")  # 헤드리스 모드
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")  # Docker 환경에서 유용

    # ChromeDriver 실행
    # service = Service(ChromeDriverManager().install())
    # driver = webdriver.Chrome(options=options)
    GOOGLE_CHROME_BIN = "/app/.apt/usr/bin/google-chrome"
    CHROMEDRIVER_PATH = "/app/.chromedriver/bin/chromedriver"

    chrome_options = webdriver.ChromeOptions()
    chrome_options.binary_location = GOOGLE_CHROME_BIN
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH, chrome_options=chrome_options)
    driver.maximize_window()
    driver.get(url)
    driver.implicitly_wait(20)

    driver.find_element(By.XPATH, '/html/body/div[2]/nav/div[2]/ul/li[3]/button/i').click()
    driver.implicitly_wait(20)

    if key_word:
        driver.find_element(By.XPATH, '//*[@id="keyword"]').send_keys(key_word)
        driver.find_element(By.XPATH, '//*[@id="sfrm"]/button/i').click()
        driver.implicitly_wait(20)
    else:
        st.write("아직 검색어를 입력하지 않았네요.. 검색어를 입력하신 뒤 다시 클릭해주세요!!")
        raise NotImplemented

    if view_count:
        view_count = int(view_count)
        for i in range(1, view_count + 1):
            if i == 1:
                driver.find_element(By.XPATH,
                                    '/html/body/div[4]/div[1]/div/div[2]/div[1]/div[2]/div/div/div/a').send_keys(
                    Keys.CONTROL + Keys.ENTER)
                driver.implicitly_wait(20)
            else:
                driver.find_element(By.XPATH,
                                    f'/html/body/div[4]/div[1]/div/div[2]/div[1]/div[2]/div[{i}]/div/div/a').send_keys(
                    Keys.CONTROL + Keys.ENTER)
                driver.implicitly_wait(20)

            driver.switch_to.window(driver.window_handles[-1])
            info_url = driver.current_url  # 현재 URL
            info_req = Request(info_url, headers={'User-Agent': 'Mozilla/5.0'})
            info_html = urlopen(info_req).read()
            bs_obj = BeautifulSoup(info_html, 'html.parser')

            title = bs_obj.find('h1', {'class': 'article-title'}).text
            article = bs_obj.find('div', {'class': 'article-Box'}).text

            parser = PlaintextParser.from_string(article, Tokenizer("korean"))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, 3)  # 요약할 문장 개수

            st.write("Title : ", title)
            st.write("URL : ", info_url)
            for sentence in summary:
                st.write(sentence)

            st.markdown("---")

            driver.close()
            driver.switch_to.window(driver.window_handles[-1])
    else:
        st.write("보고싶은 기사의 수를 작성하시고 다시 눌러주세요!")
        raise NotImplemented

    driver.quit()
