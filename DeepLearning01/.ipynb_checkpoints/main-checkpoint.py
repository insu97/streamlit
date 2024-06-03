import streamlit as st
st.set_page_config(layout="wide")

st.title("Deep-Learning-from-Scratch01")

tab1, tab2 = st.tabs(["tab1", "tab2"])

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