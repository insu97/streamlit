#!/bin/bash

# Chrome 설치
dpkg --install /google-chrome-stable_114.0.5735.90-1_amd64.deb

# 의존성 해결
apt-get install -f
