import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import platform

# --- 한글 폰트 설정 ---
def set_korean_font():
    if platform.system() == 'Darwin': plt.rc('font', family='AppleGothic')
    elif platform.system() == 'Windows': plt.rc('font', family='Malgun Gothic')
    else: plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

st.set_page_config(page_title="교통사고 분석 대시보드", layout="wide")
st.title("🚴 자전거 사고 데이터 분석: 시간대와 사고의 상관관계")

# ---------------------------------
# 1. 데이터 로드
# ---------------------------------
@st.cache_data
def load_data():
    # 파일명은 깃허브 저장소에 함께 올릴 파일명과 일치해야 합니다.
    df = pd.read_csv("한국도로교통공단_자전거사고 다발지역 개별사고 정보_20201231.csv", encoding="cp949")
    # 시간 정보 추출 (사고일시: 2020051514 -> 14시)
    df['시간대'] = df['사고일시'].astype(str).str[-2:].astype(int)
    return df

df_raw = load_data()

# ---------------------------------
# 2. 결측치 처리 (Missing Values)
# ---------------------------------
st.subheader("1️⃣ 결측치 처리")
if df_raw.isnull().sum().sum() > 0:
    df_clean = df_raw.fillna(df_raw.mode().iloc[0])
    st.write("✔ 결측치를 최빈값으로 대체 완료")
else:
    df_clean = df_raw.copy()
    st.write("✔ 결측치가 없습니다.")

# ---------------------------------
# 3. 데이터 변환 및 이상치 제거 (Outliers)
# ---------------------------------
st.subheader("2️⃣ 수치 변환 및 이상치 제거")

# 상관계수 계산을 위해 범주형 데이터를 숫자로 변환
df_numeric = df_clean.copy()
severity_map = {'사망': 4, '중상': 3, '경상': 2, '부상신고': 1}
df_numeric['사고심각도'] = df_clean['사고내용'].map(severity_map)

# IQR 기반 이상치 제거 함수 (시간대나 사고내용은 범주형이므로 주로 수치형 '사고번호' 등 대신 전반적인 정제 적용)
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# 여기서는 예시로 '시간대' 컬럼 기준 정제 (사실 시간대는 0-23이라 이상치가 거의 없으나 프로세스 포함)
df_no_outlier = remove_outliers_iqr(df_numeric, '시간대')
st.write(f"데이터 정제 완료: {len(df_raw)}행 → {len(df_no_outlier)}행")

# ---------------------------------
# 4. 정규화 (Normalization)
# ---------------------------------
st.subheader("3️⃣ 데이터 정규화 (Min-Max Scaling)")
# 상관관계 분석을 위해 주요 수치 컬럼 정규화
scaler = MinMaxScaler()
cols_to_scale = ['시간대', '사고심각도']
df_scaled = df_no_outlier.copy()
df_scaled[cols_to_scale] = scaler.fit_transform(df_no_outlier[cols_to_scale])
st.dataframe(df_scaled[cols_to_scale].head())

# ---------------------------------
# 5. 시각화: 시간대와 사고율의 상관관계
# ---------------------------------
st.subheader("4️⃣ 상관관계 시각화")

col1, col2 = st.columns(2)

with col1:
    st.write("📊 **시간대별 사고 발생 건수**")
    fig1, ax1 = plt.subplots()
    sns.histplot(df_no_outlier['시간대'], bins=24, kde=True, color='skyblue', ax=ax1)
    ax1
