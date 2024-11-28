from bs4 import BeautifulSoup
from selenium import webdriver
from urllib.request import Request, urlopen
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from selenium.webdriver.chrome.options import Options

print("해외축구 뉴스 요약")
print('-------------------------------------')
keyword = str(input("Enter a keyword: "))
view_count = int(input("Enter a view count: "))
print('-------------------------------------')

url = 'https://sports.chosun.com/football/?action=worldfootball'
options = Options()
options.add_argument('--no-sandbox')
options.add_argument('--headless')
options.add_argument('--disable-dev-shm-usage')

options.add_argument("start-maximized");
options.add_argument("disable-infobars");
options.add_argument("--disable-extensions");
options.add_argument("--disable-gpu");
options.add_argument("--disable-dev-shm-usage");

# ChromeDriver 실행
driver = webdriver.Chrome(options=options)
# driver.maximize_window()
driver.get(url)
driver.implicitly_wait(20)

driver.find_element(By.XPATH, '/html/body/div[2]/nav/div[2]/ul/li[3]/button/i').click()
driver.implicitly_wait(20)

if keyword:
    driver.find_element(By.XPATH, '//*[@id="keyword"]').send_keys(keyword)
    driver.find_element(By.XPATH, '//*[@id="sfrm"]/button/i').click()
    driver.implicitly_wait(20)
else:
    print("아직 검색어를 입력하지 않았네요.. 검색어를 입력하신 뒤 다시 클릭해주세요!!")
    raise NotImplemented

if view_count:
    view_count = int(view_count)
    for i in range(1, view_count + 1):
        if i == 1:
            driver.find_element(By.XPATH,
                                "/html/body/div[4]/div[1]/div/div[2]/div[1]/div[2]/div/div/div/a").send_keys(
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

        # 마침표 기준으로 텍스트를 분리하고, 각 문장을 새로운 줄에 넣기
        text_lines = article.split('.')

        # 각 문장 뒤에 마침표를 추가하고 줄바꿈 처리
        article = '\n'.join([line.strip() + '.' for line in text_lines if line.strip()])

        parser = PlaintextParser.from_string(article, Tokenizer("korean"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, 3)  # 요약할 문장 개수


        print("Title : ", title)
        print("URL : ", info_url)
        for sentence in summary:
            print(sentence)

        print('-------------------------------------')

        driver.close()
        driver.switch_to.window(driver.window_handles[-1])
else:
    print("보고싶은 기사의 수를 작성하시고 다시 눌러주세요!")
    raise NotImplemented

driver.quit()

input("Press Enter to exit the program...")

# Enter 키를 누르면 프로그램 종료
print("Goodbye!")