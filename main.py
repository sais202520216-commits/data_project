import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="자전거 사고 다발지역 분석",
    layout="wide"
)

st.title("🚲 자전거 사고 다발지역 데이터 분석")
st.write("CSV 파일을 업로드하면 자동으로 분석합니다.")

# CSV 파일 업로드
uploaded_file = st.file_uploader(
    "자전거 사고 CSV 파일 업로드",
    type=["csv"]
)

if uploaded_file is not None:
    # 데이터 불러오기
    df = pd.read_csv(uploaded_file, encoding="utf-8")

    st.subheader("📄 데이터 미리보기")
    st.dataframe(df.head())

    # 결측치 확인
    st.subheader("❗ 결측치 개수")
    st.write(df.isnull().sum())

    # 숫자형 컬럼만 선택
    numeric_df = df.select_dtypes(include="number")

    st.subheader("📊 숫자형 데이터 기초 통계")
    st.dataframe(numeric_df.describe())

    # 상관관계 히트맵
    if numeric_df.shape[1] >= 2:
        st.subheader("🔍 상관관계 히트맵")

        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )
        st.pyplot(fig)

        # 산점도
        st.subheader("📈 변수 간 산점도")
        x_col = st.selectbox("X축 변수", numeric_df.columns)
        y_col = st.selectbox("Y축 변수", numeric_df.columns)

        fig2, ax2 = plt.subplots()
        ax2.scatter(numeric_df[x_col], numeric_df[y_col])
        ax2.set_xlabel(x_col)
        ax2.set_ylabel(y_col)
        ax2.set_title(f"{x_col} vs {y_col}")
        st.pyplot(fig2)

    else:
        st.warning("숫자형 컬럼이 부족하여 상관관계 분석을 할 수 없습니다.")

else:
    st.info("CSV 파일을 업로드해 주세요.")
