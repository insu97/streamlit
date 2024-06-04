import streamlit as st
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from function import *
import pandas as pd
st.set_page_config(layout="wide")

st.title("Deep-Learning-from-Scratch01")

tab1, tab2 = st.tabs(["tab1", "tab2"])

with st.sidebar:
    btn_load = st.button("mnist_data 불러오기")
    btn_flatten = st.button("flatten 작업")

    with st.form(key='scaler'):
        choice_scaler = ['StandardScaler', 'MinMaxScaler', 'Normalizer']
        select_scaler = st.selectbox('Scaler 선택', choice_scaler, placeholder="scaler를 선택해주세요", index=None)
        submit_scaler = st.form_submit_button(label='Submit')
    
    seed_value = st.slider('seed', min_value=1, max_value=100, value=42)
    np.random.seed(seed_value)

    btn_network = st.button("network 생성 및 학습")



with tab1:
    col1, col2 = st.columns([1,1])
    with col1 :
        st.markdown('''
                    - 활성화 함수
                        - 은닉층 : ReLU 함수 사용
                        - 출력층
                            1. 회귀 문제 : 항등함수
                            2. 이진 분류 : 시그모이드
                            3. 다중 분류 : 소프트맥스
                    ''')
        st.markdown('''
                    - 손실 함수
                        - 회귀 문제 : MSE, RMSE
                        - 다중 분류 : CEE
                    ''')
        st.markdown('''
                    - 매개변수 갱신
                        1. 수치 미분 사용
                        2. 경사 하강법
                        3. 확률적 경사하강법
                        4. 모멘텀
                        5. Adagrad
                        6. Adam
                        7. Nesterov
                        8. RMSprop
                    ''')
        st.markdown('''
                    - 가중치의 초깃값
                        1. 선형 : Xavier 초깃값 사용
                        2. ReLU 사용 -> He 초깃값 사용&nbsp;&nbsp;
                    ''')
        st.markdown('''
                    - 배치 정규화 : Affine -> Batch Norm -> ReLU
                    ''')
        st.markdown('''
                    - 오버피팅 시
                        1. 드롭아웃
                        2. 가중치 감소 [ 손실 함수에 가중치의 L2노름을 더한 가중치 감소 방법 ]
                    ''')
        st.markdown('''
                    - 적절한 하이퍼파라미터 값 찾기
                        1. 데이터 분할
                            - 훈련 데이터 : 데이터 학습
                            - 검증 데이터 : 하이퍼파라미터 성능 평가
                            - 시험 데이터 : 신경망의 범용 성능 평가
                    ''')
        st.markdown('''
                    - 하이퍼파라미터 최적화 : 그리드 서치보다는 무작위로 샘플링해 탐색하는 편이 좋다고 알려짐
                    ''')
with tab2:

    def data_save(x_train, y_train, x_test, y_test):
        st.session_state.x_train = x_train
        st.session_state.y_train = y_train
        st.session_state.x_test = x_test
        st.session_state.y_test = y_test
    
    def data_load():
        x_train = st.session_state.x_train
        y_train = st.session_state.y_train
        x_test = st.session_state.x_test
        y_test = st.session_state.y_test

        return x_train, y_train, x_test, y_test

    # 데이터 불러오기
    if btn_load:
        mnist = keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        st.write("x_train.shape : ", x_train.shape, "y_train.shape : ", y_train.shape, "x_test.shape : ", x_test.shape, "y_test.shape : ", y_test.shape)

        data_save(x_train, y_train, x_test, y_test)
    
    # 데이터 평탄화
    if btn_flatten:
        x_train, y_train, x_test, y_test = data_load()

        st.write("flatten 작업")
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)
        st.write("x_train.shape : ", x_train.shape, "y_train.shape : ", y_train.shape, "x_test.shape : ", x_test.shape, "y_test.shape : ", y_test.shape)

        data_save(x_train, y_train, x_test, y_test)
    
    # 데이터 스케일링
    if submit_scaler:

        if select_scaler == None :
            st.write("아직 scaler를 선택하지 않았습니다!")
        else:
            x_train, y_train, x_test, y_test = data_load()

            fig = px.histogram(x_train[0], title="scaler 전")
            st.plotly_chart(fig, theme=None)

            if select_scaler == 'StandardScaler':
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
            elif select_scaler == 'MinMaxScaler':
                scaler = MinMaxScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
            elif select_scaler == 'Normalizer':
                scaler = Normalizer()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
            
            fig = px.histogram(x_train[0], title="scaler 후")
            st.plotly_chart(fig, theme=None)

            data_save(x_train, y_train, x_test, y_test)
    
    if btn_network:
        x_train, y_train, x_test, y_test = data_load()
        net = MnistNet(input_size=784, hidden_size=50, output_size=10)
        iters_num = 10000
        train_size = x_train.shape[0]
        batch_size = 100
        learning_rate = 0.1

        train_loss_list = []
        train_acc_list = []
        test_acc_list = []

        iter_per_epoch = max(train_size / batch_size, 1)

        for i in range(iters_num):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            y_batch = y_train[batch_mask]

            grad = net.gradient(x_batch, y_batch)

            # 갱신
            for key in ('W1', 'b1', 'W2', 'b2'):
                net.params[key] -= learning_rate * grad[key]

            loss = net.loss(x_batch, y_batch)
            train_loss_list.append(loss)

            if i % iter_per_epoch == 0:
                train_acc = net.accuracy(x_train, y_train)
                test_acc = net.accuracy(x_test, y_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)

        col1, col2, col3 = st.columns([1,1,2])
        with col1:
            for i in train_acc_list:
                st.write("train_acc : ", i)
        with col2:
            for j in test_acc_list:
                st.write("test_acc : ", j)
        with col3:
            acc = pd.DataFrame({
                "train_acc":train_acc_list,
                "test_acc":test_acc_list
            })

            fig = px.line(acc, title="정확도")
            st.plotly_chart(fig, theme=None)

            fig = px.line(train_loss_list, title="train_loss")
            st.plotly_chart(fig, theme=None)
