import streamlit as st
import os, sys

def installff():
  os.system('sbase install geckodriver')
  os.system('ln -s /home/appuser/venv/lib/python3.7/site-packages/seleniumbase/drivers/geckodriver /home/appuser/venv/bin/geckodriver')

_ = installff()
from selenium import webdriver
from selenium.webdriver import FirefoxOptions
opts = FirefoxOptions()
# opts.add_argument("--headless")
browser = webdriver.Firefox(options=opts)

url = 'https://sports.chosun.com/football/?action=worldfootball'

browser.get(url)
st.write(browser.page_source)