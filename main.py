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
    # 인코딩 에러 방지를 위해 utf-8 시도 후 실패 시 euc-kr 사용
    try:
        df = pd.read_csv("한국도로교통공단_자전거사고 다발지역 개별사고 정보_20201231.csv", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv("한국도로교통공단_자전거사고 다발지역 개별사고 정보_20201231.csv", encoding="euc-kr")
    
    # [수정된 부분] '발생시간대' 컬럼에서 '시'를 제거하고 숫자로 변환
    # 예: '07시' -> 7
    df['시간대'] = df['발생시간대'].str.replace('시', '').astype(int)
    return df

df_raw = load_data()

# ---------------------------------
# 2. 결측치 처리 (Missing Values)
# ---------------------------------
st.subheader("1️⃣ 결측치 처리")
# 모든 결측치를 최빈값으로 채우거나 제거
df_clean = df_raw.dropna(subset=['시간대', '사고내용']) # 주요 분석 컬럼 결측치 제거
df_clean = df_clean.fillna(df_clean.mode().iloc[0])
st.write(f"✔ 결측치 처리 완료 (남은 데이터: {len(df_clean)}행)")

# ---------------------------------
# 3. 데이터 변환 및 이상치 제거 (Outliers)
# ---------------------------------
st.subheader("2️⃣ 수치 변환 및 이상치 제거")

# 상관계수 계산을 위해 범주형 데이터를 숫자로 변환
df_numeric = df_clean.copy()
severity_map = {'사망': 4, '중상': 3, '경상': 2, '부상신고': 1}
df_numeric['사고심각도'] = df_clean['사고내용'].map(severity_map).fillna(1)

# IQR 기반 이상치 제거 (예: 가해자연령 등 수치형 데이터 대상)
# 여기서는 예시로 '사망자수' 등의 수치를 사용하거나 필요 시 추가 가능
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# 시간대는 0~23 사이이므로 이상치가 없으나, 학습을 위해 연령 등에서 제거 가능
# (현재 데이터셋 구조에 맞춰 '시간대'는 그대로 유지하고 진행)
df_no_outlier = df_numeric.copy() 

# ---------------------------------
# 4. 정규화 (Normalization)
# ---------------------------------
st.subheader("3️⃣ 데이터 정규화 (Min-Max Scaling)")
scaler = MinMaxScaler()
cols_to_scale = ['시간대', '사고심각도']
df_scaled = df_no_outlier.copy()
df_scaled[cols_to_scale] = scaler.fit_transform(df_no_outlier[cols_to_scale])
st.write("상관분석을 위해 시간대와 심각도 데이터를 0~1 사이로 정규화했습니다.")
st.dataframe(df_scaled[['시간대', '사고심각도']].head())

# ---------------------------------
# 5. 시각화
# ---------------------------------
st.subheader("4️⃣ 시간대와 사고율의 상관관계 시각화")

col1, col2 = st.columns(2)

with col1:
    st.write("📊 **시간대별 사고 발생 빈도**")
    fig1, ax1 =
