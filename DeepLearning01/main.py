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
from stqdm import stqdm
st.set_page_config(layout="wide")

st.title("Deep-Learning-from-Scratch01")

with st.sidebar:
    btn_load = st.button("mnist_data 불러오기")
    btn_flatten = st.button("flatten 작업 및 train, val, test split")

    with st.form(key='scaler'):
        choice_scaler = ['StandardScaler', 'MinMaxScaler', 'Normalizer']
        select_scaler = st.selectbox('Scaler 선택', choice_scaler, placeholder="scaler를 선택해주세요", index=None)
        submit_scaler = st.form_submit_button(label='Submit')
    
    seed_value = st.slider('seed', min_value=1, max_value=100, value=42)
    np.random.seed(seed_value)

    with st.form(key='optimizer'):
        choice_optimizer = ['SGD','Momentum','Nesterov','AdaGrad','RMSprop','Adam']
        select_optimizer = st.selectbox('optimizer 선택', choice_optimizer, placeholder="optimizer를 선택해주세요", index=None)
        submit_optimizer = st.form_submit_button(label='Submit')

    with st.form(key='w_d_lambda'):
        choice_w_d_lambda = [0, 0.01, 0.001, 0.0001, 0.00001]
        select_w_d_lambda= st.selectbox('lambda 선택', choice_w_d_lambda, placeholder="weight_decay_lambda를 선택해주세요", index=None)
        submit_w_d_lambda = st.form_submit_button(label='Submit')
    
    dropout_ratio = st.slider('dropout_ratio', min_value=0.0, max_value=0.5, value=0.25)
    st.session_state.dropout_ratio = dropout_ratio

    BatchNorm = st.radio("use_batchnorm", [True, False])
    st.session_state.BatchNorm = BatchNorm

    btn_network = st.button("network 생성 및 학습")

    st.write("---")
    btn_hyperparameter = st.button("매개변수 최적화")

    btn_h_run = st.button("최적화된 매개변수로 학습")

col1, col2 = st.columns(2)

def data_save_1(x_train, y_train, x_test, y_test):
    st.session_state.x_train = x_train
    st.session_state.y_train = y_train
    st.session_state.x_test = x_test
    st.session_state.y_test = y_test

def data_save_2(x_train, y_train, x_test, y_test, x_val, y_val):
    st.session_state.x_train = x_train
    st.session_state.y_train = y_train
    st.session_state.x_test = x_test
    st.session_state.y_test = y_test
    st.session_state.x_val = x_val
    st.session_state.y_val = y_val

def data_load_1():
    x_train = st.session_state.x_train
    y_train = st.session_state.y_train
    x_test = st.session_state.x_test
    y_test = st.session_state.y_test

    return x_train, y_train, x_test, y_test

def data_load_2():
    x_train = st.session_state.x_train
    y_train = st.session_state.y_train
    x_test = st.session_state.x_test
    y_test = st.session_state.y_test
    x_val = st.session_state.x_val
    y_val = st.session_state.y_val

    return x_train, y_train, x_test, y_test, x_val, y_val


with col1:

    # 데이터 불러오기
    if btn_load:
        mnist = keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        data_save_1(x_train, y_train, x_test, y_test)

    # 데이터 평탄화
    if btn_flatten:
        x_train, y_train, x_test, y_test = data_load_1()

        st.write("flatten 작업 및 train, val, test_set 나누기")
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)

        x_train = x_train[:1000]
        y_train = y_train[:1000]

        x_train, y_train = shuffle_dataset(x_train, y_train)

        validation_rate = 0.2
        validation_num = int(x_train.shape[0] * validation_rate)

        x_val = x_train[:validation_num]
        y_val = y_train[:validation_num]
        x_train = x_train[validation_num:]
        y_train = y_train[validation_num:]

        data_save_2(x_train, y_train, x_test, y_test, x_val, y_val)

    # 데이터 스케일링
    if submit_scaler:

        if select_scaler == None :
            st.write("아직 scaler를 선택하지 않았습니다!")
        else:
            x_train, y_train, x_test, y_test, x_val, y_val = data_load_2()

            fig = px.histogram(x_train[0], title="scaler 전")
            st.plotly_chart(fig, theme=None)

            if select_scaler == 'StandardScaler':
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_val = scaler.transform(x_val)
                x_test = scaler.transform(x_test)
            elif select_scaler == 'MinMaxScaler':
                scaler = MinMaxScaler()
                x_train = scaler.fit_transform(x_train)
                x_val = scaler.transform(x_val)
                x_test = scaler.transform(x_test)
            elif select_scaler == 'Normalizer':
                scaler = Normalizer()
                x_train = scaler.fit_transform(x_train)
                x_val = scaler.transform(x_val)
                x_test = scaler.transform(x_test)
            
            fig = px.histogram(x_train[0], title="scaler 후")
            st.plotly_chart(fig, theme=None)

            data_save_2(x_train, y_train, x_test, y_test, x_val, y_val) 

    # 매개변수 갱신
    if submit_optimizer:

        if select_optimizer == None:
            st.write("아직 optimizer를 선택하지 않았습니다!")
        else: # 'SGD','Momentum','Nesterov','AdaGrad','RMSprop','Adam'
            st.session_state.optimizer_ = select_optimizer

    # 가중치 감소
    if submit_w_d_lambda:
        if submit_w_d_lambda == None:
            st.write("아직 weight_decay_lambda를 선택하지 않았습니다!")
        else:
            st.session_state.weight_decay_lambda = select_w_d_lambda

    # 데이터 학습
    if btn_network:
        x_train, y_train, x_test, y_test = data_load_1()
        net = net = MultiLayerNetExtend(784, [100], 10, activation='relu', weight_init_std='relu', weight_decay_lambda=st.session_state.weight_decay_lambda,
                                            use_dropout = True, dropout_ratio = st.session_state.dropout_ratio, use_batchnorm=st.session_state.BatchNorm)
        iters_num = 100
        train_size = x_train.shape[0]
        batch_size = 100
        learning_rate = 0.1

        train_loss_list = []
        train_acc_list = []
        test_acc_list = []

        iter_per_epoch = max(train_size / batch_size, 1)

        optimizer = st.session_state.optimizer_

        if optimizer == 'SGD':
            optimizer = SGD()
        elif optimizer == 'Momentum':
            optimizer = Momentum()
        elif optimizer == 'Nesterov':
            optimizer = Nesterov()
        elif optimizer == 'AdaGrad':
            optimizer = AdaGrad()
        elif optimizer == 'RMSprop':
            optimizer = RMSprop()
        elif optimizer == 'Adam':
            optimizer = Adam()

        for i in range(iters_num):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            y_batch = y_train[batch_mask]

            grad = net.gradient(x_batch, y_batch)
            params = net.params

            optimizer.update(params, grad)

            loss = net.loss(x_batch, y_batch)
            train_loss_list.append(loss)

            if i % iter_per_epoch == 0:
                train_acc = net.accuracy(x_train, y_train)
                test_acc = net.accuracy(x_test, y_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)

        # col1, col2, col3 = st.columns([1,1,2])
        # with col1:
        #     for i in train_acc_list:
        #         st.write("train_acc : ", i)
        # with col2:
        #     for j in test_acc_list:
        #         st.write("test_acc : ", j)
        # with col3:
        #     acc = pd.DataFrame({
        #         "train_acc":train_acc_list,
        #         "test_acc":test_acc_list
        #     })

        acc = pd.DataFrame({
            "train_acc":train_acc_list,
            "test_acc":test_acc_list
        })

        fig = px.line(acc, title="정확도")
        st.plotly_chart(fig, theme=None)

        fig = px.line(train_loss_list, title="train_loss")
        st.plotly_chart(fig, theme=None)

    # 매개변수 최적화
    def __train(lr, weight_decay, epocs=50):
        network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100],
                                output_size=10, weight_decay_lambda=weight_decay)
        trainer = Trainer(network, x_train, y_train, x_val, y_val,
                        epochs=epocs, mini_batch_size=100,
                        optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
        trainer.train()

        return trainer.test_acc_list, trainer.train_acc_list
    
    if btn_hyperparameter:
        x_train, y_train, x_test, y_test, x_val, y_val = data_load_2()

        # 하이퍼파라미터 무작위 탐색======================================
        optimization_trial = 20
        results_val = {}
        results_train = {}
        for _ in stqdm(range(optimization_trial)):
            # 탐색한 하이퍼파라미터의 범위 지정===============
            weight_decay = 10 ** np.random.uniform(-8, -4)
            lr = 10 ** np.random.uniform(-6, -2)
            # ================================================

            val_acc_list, train_acc_list = __train(lr, weight_decay)
            # print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
            key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
            results_val[key] = val_acc_list
            results_train[key] = train_acc_list
        
        # 그래프 그리기========================================================
        st.write("=========== Hyper-Parameter Optimization Result ===========")
        graph_draw_num = 20
        col_num = 5
        row_num = int(np.ceil(graph_draw_num / col_num))
        i = 0

        # for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):

        #     plt.subplot(row_num, col_num, i+1)
        #     plt.title("Best-" + str(i+1))
        #     plt.ylim(0.0, 1.0)
        #     if i % 5: plt.yticks([])
        #     plt.xticks([])
        #     x = np.arange(len(val_acc_list))
        #     plt.plot(x, val_acc_list)
        #     plt.plot(x, results_train[key], "--")
        #     i += 1

        #     if i >= graph_draw_num:
        #         break

        # plt.show()

        fig, axs = plt.subplots(row_num, col_num, figsize=(15, 10))
        fig.tight_layout(pad=3.0)

        for key, val_acc_list in sorted(results_val.items(), key=lambda x: x[1][-1], reverse=True):
            # st.write("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)
            ax = axs[i // col_num, i % col_num]
            ax.set_title("Best-" + str(i + 1))
            ax.set_ylim(0.0, 1.0)
            if i % col_num != 0:
                ax.set_yticks([])
            ax.set_xticks([])
            x = np.arange(len(val_acc_list))
            ax.plot(x, val_acc_list, label='Validation')
            ax.plot(x, results_train[key], "--", label='Training')
            ax.legend()

            if i == 0 :
                st.session_state.best_parameters = key

            i += 1

            if i >= graph_draw_num:
                break

        # 남은 subplot을 빈칸으로 채우기
        for j in range(i, row_num * col_num):
            fig.delaxes(axs[j // col_num, j % col_num])

        st.pyplot(fig)

    # 하이퍼 파라미터로 학습 후 평가
    if btn_h_run:
        x_train, y_train, x_test, y_test, x_val, y_val = data_load_2()
        lr = st.session_state.best_parameters.split(',')[0].split(':')[1]
        weight_decay = st.session_state.best_parameters.split(',')[1].split(':')[1]

        lr = float(lr)
        weight_decay = float(weight_decay)

        x_test, y_test = shuffle_dataset(x_test, y_test)

        x_test = x_test[:300]
        y_test = y_test[:300]

        network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100],
                            output_size=10, weight_decay_lambda=weight_decay)
        trainer = Trainer(network, x_train, y_train, x_test, y_test,
                        epochs=50, mini_batch_size=100,
                        optimizer='sgd', optimizer_param={'lr': lr}, verbose=False)
        trainer.train()

        acc = pd.DataFrame({
            "train_acc":trainer.train_acc_list,
            "test_acc":trainer.test_acc_list
        })

        fig = px.line(acc, title="정확도")
        st.plotly_chart(fig, theme=None)

        data_save_2(x_train, y_train, x_test, y_test, x_val, y_val)

with col2:
    st.write("Seed : ", seed_value)
    if "x_train" in st.session_state.to_dict().keys():
        st.write("x_train.shape : ", st.session_state.x_train.shape)
    if "y_train" in st.session_state.to_dict().keys():
        st.write("y_train.shape : ", st.session_state.y_train.shape)
    if "x_test" in st.session_state.to_dict().keys():
        st.write("x_test.shape : ", st.session_state.x_test.shape)
    if "y_test" in st.session_state.to_dict().keys():
        st.write("y_test.shape : ", st.session_state.y_test.shape)
    try:
        st.write("x_val.shape : ", st.session_state.x_val.shape)
        st.write("y_val.shape : ", st.session_state.y_val.shape)
    except:
        pass
    try:
        st.write("Scaler : ", select_scaler)
    except:
        pass
    try:
        st.write("Optimizer : ", select_optimizer)
    except:
        pass
    try:
        st.write("weight_decay_lambda : ", select_w_d_lambda)
    except:
        pass
    st.write("Dropout ratio : ", dropout_ratio)
    st.write("use_batchNorm : ", BatchNorm)
