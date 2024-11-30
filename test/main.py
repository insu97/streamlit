﻿import re
import json
import urllib.parse
import pandas as pd
import urllib.request
import streamlit as st
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from transformers import pipeline

summarizer = pipeline("summarization")

# 텍스트 요약 함수
def summarize_text(text, word_count=50):
    summary = summarizer(text, max_length=word_count, min_length=25, do_sample=False)
    return summary[0]['summary_text']

st.title("NEWS 크롤링 및 요약")

web_df = pd.DataFrame(columns = ("Title", "link", "postdate","Description"))

keyword = st.text_input("검색할 키워드를 입력하세요!")
count = st.text_input("보고싶은 뉴스의 수를 입력하세요!")

start_button = st.button("검색 시작!")

if start_button:
    if not keyword:
        st.error("검색할 키워드를 입력해주세요!!")
    else:
        client_id = "xyGqbTdeDOTYIhtY2kwB"
        client_secret = "_b69A669rX"
        encText = urllib.parse.quote(keyword)
        url = f"https://openapi.naver.com/v1/search/blog?query={encText}&display={count}"  # JSON 결과 URL
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", client_id)
        request.add_header("X-Naver-Client-Secret", client_secret)

        response = urllib.request.urlopen(request)
        rescode = response.getcode()

        if rescode == 200:
            response_body = response.read()
            response_dict = json.loads(response_body.decode('utf-8'))
            items = response_dict['items']

            remove_tag = re.compile('<.*?>')  # html 태그 제거 정규식

            for item in items:
                title = re.sub(remove_tag, '', item['title'])  # 태그 제거
                link = item['link']
                postdate = item['postdate']
                postdate = pd.to_datetime(postdate, format='%Y%m%d').strftime('%Y년 %m월 %d일')

                info_req = Request(link, headers={'User-Agent': 'Mozilla/5.0'})
                info_html = urlopen(info_req).read()
                bs_obj = BeautifulSoup(info_html, 'html.parser')

                iframe = bs_obj.find('iframe', id='mainFrame')
                description = []
                if iframe:
                    iframe_src = iframe.get('src')
                    iframe_url = "https://blog.naver.com" + iframe_src  # 네이버 블로그는 절대 경로 사용

                    # 두 번째 요청: iframe의 URL에서 본문 가져오기
                    iframe_req = Request(iframe_url, headers={'User-Agent': 'Mozilla/5.0'})
                    iframe_html = urlopen(iframe_req).read()
                    iframe_soup = BeautifulSoup(iframe_html, 'html.parser')

                    # 본문 컨테이너 찾기
                    content = iframe_soup.find('div', {'class': 'se-main-container'})  # 스마트에디터 본문
                    if content:
                        description.append(content.get_text(strip=True))
                    else:
                        st.write("본문을 가져올 수 없습니다. HTML 구조를 확인하세요.")
                else:
                    # 스마트에디터 구조 (최신 블로그)
                    content = None

                    if bs_obj.find('div', {'class': 'se-main-container'}):
                        content = bs_obj.find('div', {'class': 'se-main-container'})
                    elif bs_obj.find('div', {'id': 'postViewArea'}):  # 이전 블로그 구조 (구버전 블로그)
                        content = bs_obj.find('div', {'id': 'postViewArea'})
                    elif bs_obj.find('div', {'class': 'contents_style'}):  # 다른 경우 (필요시 추가)
                        content = bs_obj.find('div', {'class': 'contents_style'})

                    # 본문 내용 추출
                    if content:
                        full_text = content.get_text(strip=True)  # 텍스트만 추출하고 공백 제거
                        description.append(full_text)
                    else:
                        st.write("본문을 찾을 수 없습니다. HTML 구조를 확인하세요.")

                web_df.loc[len(web_df)] = [title, link, postdate, description]  # 데이터프레임 추가
            # 리스트 형태의 Description을 문자열로 병합하고 \u200b 제거
            web_df['Description'] = web_df['Description'].apply(
                lambda x: ' '.join(x).replace('\u200b', '') if isinstance(x, list) else x)

            for i in range(len(web_df)):
                st.write("Title : ", web_df['Title'][i])
                st.write("Link : ", web_df['link'][i])
                st.write("Postdate : ", web_df['postdate'][i])
                article = web_df['Description'][i]
                # 마침표 기준으로 텍스트를 분리하고, 각 문장을 새로운 줄에 넣기
                text_lines = article.split('.')
                # 각 문장 뒤에 마침표를 추가하고 줄바꿈 처리
                article = '\n'.join([line.strip() + '.' for line in text_lines if line.strip()])

                summary = summarize_text(article, word_count=50)  # 요약할 단어 수 설정
                st.write("Summary: ", summary)

                # parser = PlaintextParser.from_string(article, Tokenizer("korean"))
                # summarizer = LsaSummarizer()
                # summary = summarizer(parser.document, 3)  # 요약할 문장 개수
                # for sentence in summary:
                #     st.write(sentence)
                #     st.write("---")
        else:
            st.write(f"Error Code: {rescode}")