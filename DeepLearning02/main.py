import streamlit as st
from function import *
import numpy as np
import matplotlib.pyplot as plt
import ptb
import pickle

st.set_page_config(layout="wide")
st.title("Deep-Learning-from-Scratch02")

ch02, ch03 , ch04= st.tabs(['ch02', 'ch03', 'ch04'])

with ch02:

    col1, col2= st.columns([2,1])

    with col1:
        text = st.text_input("text_1", "You say goodbye and I say hello.")

        #---
        corpus, word_to_id, id_to_word = preprocess(text)
        C = create_co_matrix(corpus=corpus, vocab_size=len(word_to_id), window_size=1)
        #---

        col1_1, col1_2, col1_3= st.columns([1,1,1])

        with col1_1:
            st.write("word_to_id : ", word_to_id)
        
        with col1_2:
            st.write("id_to_word : ", id_to_word)

        with col1_3:
            st.write("corpus : ", corpus)

        st.write('---')

        col1_2_1, col1_2_2 = st.columns([1,1])

        with col1_2_1:
            st.markdown("## 벡터 간 유사도")
            x = st.text_input("x", "you")
            y = st.text_input("y", "i")
        
            try:
                cos_similarity(C[word_to_id[x]], C[word_to_id[y]])
                st.write("cos_similarity : ", cos_similarity(C[word_to_id[x]], C[word_to_id[y]]))
            except:
                pass
        with col1_2_2:
            
            W = ppmi(C)
            st.markdown("## PPMI")
            st.write(W)

        st.write("---")

        col1_3_1, col1_3_2 = st.columns([1,1])

        with col1_3_1:
            try :
                st.markdown("## SVD에 의한 차원 감소")
                U, S, V = np.linalg.svd(W)

                st.write("U", U)
            except:
                pass
        
        with col1_3_2:
            # Matplotlib을 사용하여 그래프 그리기
            fig, ax = plt.subplots()
            for word, word_id in word_to_id.items():
                ax.annotate(word, (U[word_id][0], U[word_id][1]))
            ax.scatter([point[0] for point in U], [point[1] for point in U], alpha=0.5)

            # Streamlit을 사용하여 그래프 표시
            st.pyplot(fig)
        
        
            
    with col2:
        
        st.write("동시발생행렬 : ", C)    
        
        st.write("---")

        st.markdown("## 유사 단어 랭킹")

        z = st.text_input("word", "you")

        st.write(most_similar(z, word_to_id, id_to_word, C, top=5))

        st.write("---")

        st.markdown("## PTB dataset")

        if st.button("PTB 학습하기"):
            
            corpus, word_to_id, id_to_word = ptb.load_data('train')
            st.write("말뭉치 크기 : ", len(corpus))

            window_size = 2
            wordvec_size = 100

            vocab_size = len(word_to_id)
            C = create_co_matrix(corpus, vocab_size, window_size)

            st.write("PPMI 계산...")
            W = ppmi(C)

            st.write("SVD 계산...")
            try:
                from sklearn.utils.extmath import randomized_svd
                U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
            except ImportError:
                U, S, V = np.linalg.svd(W)
            
            word_vecs = U[:, :wordvec_size]

            querys = ['you', 'year', 'car', 'toyota']

            for query in querys:
                most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

    st.write('---')

with ch03:
    col1, col2= st.columns([2,1])

    with col1:
        text = st.text_input("text_2", "You say goodbye and I say hello.")

        st.write('---')

        corpus, word_to_id, id_to_word = preprocess(text)
        contexts, target = create_contexts_target(corpus, window_size=1)

        col1_1, col1_2, col1_3, col1_4 = st.columns([1,1,2,2])
        
        with col1_1:
            st.write("contexts", contexts)
        with col1_2:
            st.write("target", target)
        
        vocab_size = len(word_to_id)
        target = convert_one_hot(target, vocab_size)
        contexts = convert_one_hot(contexts, vocab_size)

        with col1_3:
            st.markdown("### contexts")
            for i in range(len(contexts)):
                st.write(contexts[i])
        with col1_4:
            st.write("target", target)

        st.write('---')
    
    with col2:
        st.markdown("## CBOW")
        window_size = 1
        hidden_size = 5
        batch_size = 3
        max_epoch = 1000

        st.write("window_size : ", window_size)
        st.write("hidden_size : ", hidden_size)
        st.write("batch_size : ", batch_size)
        st.write("max_epoch : ", max_epoch)

        model = SimpleCBOW(vocab_size, hidden_size)
        optimizer = Adam()
        trainer = Trainer(model, optimizer)

        trainer.fit(contexts, target, max_epoch, batch_size)
        trainer.plot()

        # word_vecs = model.word_vecs
        # for word_id, word in id_to_word.items():
        #     st.write(word, word_vecs[word_id])

with ch04:
    col1, col2= st.columns([2,1])

    with col1:
        text = st.text_input("text_3", "You say goodbye and I say hello.")

        st.write('---')

        corpus, word_to_id, id_to_word = preprocess(text)
        contexts, target = create_contexts_target(corpus, window_size=1)

        col1_1, col1_2, col1_3 = st.columns([1,1,1])

        with col1_1:
            st.write("corpus : ", corpus)

        with col1_2:
            power = 0.75
            sample_size = 2

            st.write("power : ", power)
            st.write("sample_size : ", sample_size)
            target = np.array([1,3,0])
            st.write("target : ", target)
        
        with col1_3:
            st.markdown("### negative sample")
            sampler = UnigramSampler(corpus, power, sample_size)
            negative_sample = sampler.get_negative_sample(target)

            st.write(negative_sample)

        st.write("---")
    
    with col2:
        st.markdown("## PTB DataSet")
        # # 하이퍼파라미터 설정
        # window_size = 5
        # hidden_size = 100
        # batch_size = 100
        # max_epoch = 10

        # # 데이터 읽기
        # corpus, word_to_id, id_to_word = ptb.load_data('train')[:100]
        # vocab_size = len(word_to_id)

        # contexts, target = create_contexts_target(corpus, window_size)

        # # 모델 등 생성
        # model = CBOW(vocab_size, hidden_size, window_size, corpus)
        # # model = SkipGram(vocab_size, hidden_size, window_size, corpus)
        # optimizer = Adam()
        # trainer = Trainer(model, optimizer)

        # # 학습 시작
        # trainer.fit(contexts, target, max_epoch, batch_size)
        # trainer.plot()

        # # 나중에 사용할 수 있도록 필요한 데이터 저장
        # word_vecs = model.word_vecs
        # params = {}
        # params['word_vecs'] = word_vecs.astype(np.float16)
        # params['word_to_id'] = word_to_id
        # params['id_to_word'] = id_to_word
        # pkl_file = 'cbow_params.pkl'  # or 'skipgram_params.pkl'
        # with open(pkl_file, 'wb') as f:
        #     pickle.dump(params, f, -1)

        # import os
        # entries = os.listdir()

        # for entry in entries:
        #     st.write(entry)

        
        try:
            pkl_file = 'cbow_params.pkl'
            # pkl_file = 'skipgram_params.pkl'

            with open(pkl_file, 'rb') as f:
                params = pickle.load(f)
                word_vecs = params['word_vecs']
                word_to_id = params['word_to_id']
                id_to_word = params['id_to_word']
        except:
            pkl_file = 'DeepLearning02/cbow_params.pkl'
            # pkl_file = 'skipgram_params.pkl'

            with open(pkl_file, 'rb') as f:
                params = pickle.load(f)
                word_vecs = params['word_vecs']
                word_to_id = params['word_to_id']
                id_to_word = params['id_to_word']

        # 가장 비슷한(most similar) 단어 뽑기
        querys = ['you', 'year', 'car', 'toyota']
        # for query in querys:
        #     most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

        col2_1, col2_2 = st.columns([1,1])

        with col2_1:
            most_similar('you', word_to_id, id_to_word, word_vecs, top=5)
            st.write('---')
            most_similar('year', word_to_id, id_to_word, word_vecs, top=5)
            st.write('---')
        
        with col2_2:
            most_similar('car', word_to_id, id_to_word, word_vecs, top=5)
            st.write('---')
            most_similar('toyota', word_to_id, id_to_word, word_vecs, top=5)
            st.write('---')

    with col1:
        st.markdown("## PTB DATA 유추문제")
        # 유추(analogy) 작업
        col1_1, col1_2, col1_3, col1_4 = st.columns([1,1,1,1])

        with col1_1:
            analogy('king', 'man', 'queen',  word_to_id, id_to_word, word_vecs)
        with col1_2:
            analogy('take', 'took', 'go',  word_to_id, id_to_word, word_vecs)
        with col1_3:
            analogy('car', 'cars', 'child',  word_to_id, id_to_word, word_vecs)
        with col1_4:
            analogy('good', 'better', 'bad',  word_to_id, id_to_word, word_vecs)

    st.write('---')