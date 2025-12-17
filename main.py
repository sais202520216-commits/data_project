import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------
# 1. 스트림릿 기본 설정
# ---------------------------------
st.set_page_config(
    page_title="교통수단 사고율과 시간대의 상관관계",
    layout="wide"
)

st.title("교통수단 사고율과 시간대의 상관관계")

# ---------------------------------
# 2. 데이터 불러오기
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("한국도로교통공단_자전거사고 다발지역 개별사고 정보_20201231.csv", encoding="cp949")
    return df

df = load_data()

st.subheader("📌 원본 데이터")
st.dataframe(df.head())

# ---------------------------------
# 3. 문자 → 숫자로 변환
# ---------------------------------
st.subheader("🔢 문자 데이터 숫자로 변환")

df_numeric = df.copy()

for col in df_numeric.columns:
    # 데이터 타입이 문자(object)이면
    if df_numeric[col].dtype == "object":
        # 문자 → 숫자 코드로 변환
        df_numeric[col], _ = pd.factorize(df_numeric[col])

st.write("✔ 문자형 컬럼을 모두 숫자로 변환 완료")
st.dataframe(df_numeric.head())

# ---------------------------------
# 4. IQR 이상치 제거 함수
# ---------------------------------
st.subheader("📉 이상치(IQR) 처리")

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return data[(data[column] >= lower) & (data[column] <= upper)]

# 숫자 컬럼 선택
num_columns = df_numeric.select_dtypes(include=np.number).columns
selected_col = st.selectbox("이상치 제거할 컬럼 선택", num_columns)

df_clean = remove_outliers_iqr(df_numeric, selected_col)

st.write(f"이상치 제거 전 데이터 수: {len(df_numeric)}")
st.write(f"이상치 제거 후 데이터 수: {len(df_clean)}")

# ---------------------------------
# 5. 시각화 (이상치 비교)
# ---------------------------------
st.subheader("📊 이상치 제거 전/후 비교")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

sns.boxplot(y=df_numeric[selected_col], ax=ax[0])
ax[0].set_title("이상치 제거 전")

sns.boxplot(y=df_clean[selected_col], ax=ax[1])
ax[1].set_title("이상치 제거 후")

st.pyplot(fig)
