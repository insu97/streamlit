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
from selenium.webdriver.chrome.options import Options
from webdriver_manager.core.os_manager import ChromeType

with st.echo():
    @st.cache_resource
    def get_driver():
        return webdriver.Chrome(
            service=Service(
                ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()
            ),
            options=options,
        )
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--headless")

    # streamlit 설정
    # st.set_page_config(layout="wide")

    # streamlit
    st.title('해외축구 뉴스 요약')

    key_word = st.text_input("검색하고 싶은 키워드를 입력하세요!")
    view_count = st.text_input("보고싶은 기사의 수를 입력하세요!")

    if st.button("검색 시작하기"):
        if not key_word:
            st.error("검색어를 입력해 주세요!")
        elif not view_count:
            st.error("보고 싶은 기사의 수를 입력해 주세요!")
        else:
            # 크롤링 시작
            url = 'https://sports.chosun.com/football/?action=worldfootball'

            # ChromeDriver 실행
            driver = get_driver()
            driver.maximize_window()
            driver.get(url)

            # 명시적 대기 사용
            try:
                WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[2]/nav/div[2]/ul/li[3]/button/i'))).click()

                if key_word:
                    search_box = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '//*[@id="keyword"]')))
                    search_box.send_keys(key_word)
                    driver.find_element(By.XPATH, '//*[@id="sfrm"]/button/i').click()

                WebDriverWait(driver, 20).until(EC.presence_of_all_elements_located((By.XPATH, '/html/body/div[4]/div[1]/div/div[2]/div[1]/div[2]/div')))

                for i in range(1, int(view_count) + 1):
                    driver.find_element(By.XPATH, f'/html/body/div[4]/div[1]/div/div[2]/div[1]/div[2]/div[{i}]/div/div/a').send_keys(Keys.CONTROL + Keys.ENTER)
                    driver.implicitly_wait(10)
                    driver.switch_to.window(driver.window_handles[-1])

                    # 현재 URL에서 기사 내용 추출
                    info_url = driver.current_url
                    info_req = Request(info_url, headers={'User-Agent': 'Mozilla/5.0'})
                    info_html = urlopen(info_req).read()
                    bs_obj = BeautifulSoup(info_html, 'html.parser')

                    title = bs_obj.find('h1', {'class': 'article-title'}).text
                    article = bs_obj.find('div', {'class': 'article-Box'}).text

                    # 요약 처리
                    parser = PlaintextParser.from_string(article, Tokenizer("korean"))
                    summarizer = LsaSummarizer()
                    summary = summarizer(parser.document, 3)

                    st.write("Title : ", title)
                    st.write("URL : ", info_url)
                    for sentence in summary:
                        st.write(sentence)

                    st.markdown("---")

                    driver.close()
                    driver.switch_to.window(driver.window_handles[-1])
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
            finally:
                driver.quit()
