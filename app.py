import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
# 사이킷런 임포트 > 레이블인코더, 원핫인코더
from sklearn.compose import ColumnTransformer  
# 컬럼 트렌스폼
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def main():
    st.title('K-Means 클러스터링 앱')

    # csv 파일 업로드
    csv_file = st.file_uploader('csv 파일 업로드', type=['csv'])

    if csv_file is not None : 
        # 업로드한 csv 파일을 데이터프레임으로 읽기
        df = pd.read_csv(csv_file)
        st.dataframe(df)

    # 구글 코랩에서 했던 코드 복붙해옴
        st.subheader('Nan 데이터 확인')
        st.dataframe(df.isna().sum())

        st.subheader('결측값 처리한 결과')
        df = df.dropna()
        df.reset_index(inplace=True, drop= True)  
        st.dataframe(df)


        # 유저가 선택한 것만 보여주기
        st.subheader('클러스터링에 사용할 컬럼 선택')
        selected_columns = st.multiselect('X로 사용할 컬럼을 선택하세요', df.columns)

        if len(selected_columns) != 0 :
            X = df[selected_columns]
            st.dataframe(X)

            # 숫자로된 새 데이터프레임 만들기
            # 레이블인코딩, 원핫인코딩 하는 for문
            X_new = pd.DataFrame() # 새 데이터프레임 만듬

            for name in X.columns:            
            # 만약 해당 데이터가 문자열(오브젝트)이면, 데이터의 종류가 몇개인지 확인하기
                if X[name].dtype == object: 

                    if X[name].nunique() >=3:
                    # 종류가 3개 이상이면 원핫인코딩 한다
                        ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], 
                                        remainder='passthrough')
                        col_names = sorted(X[name].unique())
                        X_new[col_names] = ct.fit_transform(X[name].to_frame())

                    else:
                    # 아니면 레이블인코딩 한다
                        label_encoder = LabelEncoder()
                        X_new[name] = label_encoder.fit_transform(X[name])
                        
                else: 
                    X_new[name] = X[name]
                    # 만약 숫자데이터라면 여기서 처리됨
                    # 그냥 가져온다

            st.subheader('문자열은 숫자로 바꿔줍니다.')
            st.dataframe(X_new)

            # 피처 스케일링 하기
            st.subheader('피처 스케일링 합니다.')
            sclaer = MinMaxScaler()
            X_new = sclaer.fit_transform(X_new)
            st.dataframe(X_new)
    # 여기까지 하면 데이터 준비는 끝...


        # WCSS 값 구하기
        # 유저가 입력한 파일의 데이터 개수를 세어서 
        # 해당 데이터의 개수가 10보다 작으면, 데이터의 개수까지만 WCSS를 구하고
        # 10보다 크면 10개로 하기
        # 그 이상으로 정하면 에러가 뜸. 데이터는 8개까지 있는데 10묶음하라고 하면...

        print(X_new.shape[0]) 
        # 8개
        if X_new.shape[0] < 10: # 가져온 데이터의 개수가 10보다 작으면
            max_count = X_new.shape[0] 
        else:
            max_count = 10

        wcss = []
        for k in range(1, max_count+1) :
            kmeans = KMeans(n_clusters=k, random_state=5, n_init='auto') # 그룹값을 1부터 10까지 묶어보기
            kmeans.fit(X_new)  # X_scaled 데이터 사용
            wcss.append(kmeans.inertia_)  # WCSS 리스트 안에 차곡차곡 값이 저장됨

        st.write(wcss)

        # 엘보우에 있는 개수 알아보기
        x = np.arange(1, max_count+1)

        fig = plt.figure()
        plt.plot(x, wcss) 
        plt.title('The Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')

        st.pyplot(fig)


        # 유저에게 개수 입력받기
        st.subheader('클러스터링 개수 선택')
        k = st.number_input('K를 선택', 1, max_count, value=3) 
                            # 1부터 max_count까지, 기본값은 3
        kmeans = KMeans(n_clusters=k, random_state=5, n_init='auto')
        y_pred = kmeans.fit_predict(X_new)
        df['Group'] = y_pred
        
        st.subheader('그루핑 정보 표시')
        st.dataframe(df)


        st.subheader('보고싶은 그룹을 선택하세요.')
        group_number = st.number_input('그룹번호 선택', 0, k-1)
        st.dataframe(df.loc[df['Group']])== group_number

        df.to_csv('result.csv', index=False)


if __name__ == '__main__' :
    main()