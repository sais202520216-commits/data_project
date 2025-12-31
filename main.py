import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 전처리
@st.cache_data
def load_analysis_data():
    # 데이터 불러오기 (환경에 맞게 경로 수정)
    df = pd.read_csv("한국도로교통공단_자전거사고 다발지역 개별사고 정보_20201231.csv", encoding="cp949")
    
    # '사고일시'에서 시간 정보만 추출 (예: 2020051514 -> 14시)
    # 데이터 형식에 따라 추출 방식이 다를 수 있으니 확인이 필요합니다.
    df['시간대'] = df['사고일시'].astype(str).str[-4:-2].astype(int)
    
    # 사고 심각도 점수화 (상관관계 분석용)
    # 사망: 4, 중상: 3, 경상: 2, 부상신고: 1 등으로 수치화
    severity_map = {'사망': 4, '중상': 3, '경상': 2, '부상신고': 1}
    df['사고심각도'] = df['사고내용'].map(severity_map).fillna(0)
    
    return df

df_analysis = load_analysis_data()

st.header("⏰ 시간대별 사고 상관관계 분석")

# ---------------------------------------------------------
# 2. 시간대별 사고 건수 시각화
# ---------------------------------------------------------
st.subheader("1. 시간대별 사고 발생 빈도")
fig1, ax1 = plt.subplots(figsize=(12, 5))
sns.countplot(data=df_analysis, x='시간대', palette='viridis', ax=ax1)
ax1.set_title("시간대별 사고 발생 건수", fontsize=15)
ax1.set_xlabel("시간 (0-23시)")
ax1.set_ylabel("사고 건수")
st.pyplot(fig1)

st.info("💡 보통 출퇴근 시간대(08-09시, 17-19시)에 사고 발생 빈도가 가장 높게 나타납니다.")

# ---------------------------------------------------------
# 3. 시간대 vs 사고 심각도 상관관계 (Heatmap)
# ---------------------------------------------------------
st.subheader("2. 시간대와 사고 내용의 상관관계 (Heatmap)")

# 시간대와 사고내용으로 피벗 테이블 생성
pivot_df = df_analysis.groupby(['시간대', '사고내용']).size().unstack(fill_value=0)

# 비율로 변환 (각 시간대별로 어떤 사고가 많이 발생하는지)
pivot_norm = pivot_df.div(pivot_df.sum(axis=1), axis=0)

fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.heatmap(pivot_norm.T, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax2)
ax2.set_title("시간대별 사고 내용 분포 (비율)", fontsize=15)
st.pyplot(fig2)

# ---------------------------------------------------------
# 4. 수치적 상관계수 확인
# ---------------------------------------------------------
st.subheader("3. 통계적 상관계수 (Correlation)")
# 시간과 사고심각도(수치) 간의 피어슨 상관계수 계산
corr_value = df_analysis[['시간대', '사고심각도']].corr().iloc[0, 1]

st.write(f"**시간대와 사고 심각도 간의 상관계수:** `{corr_value:.4f}`")
st.write("> 상관계수가 0에 가깝다면 시간대와 사고의 '심각도' 자체는 직접적인 선형 관계가 낮음을 의미합니다. 하지만 '빈도'와는 밀접한 관계가 있을 수 있습니다.")

# ---------------------------------
# [추가] 결측치 확인 및 처리
# ---------------------------------
st.subheader("🔍 결측치 확인 및 처리")

# 1. 결측치 현황 파악
null_counts = df.isnull().sum()
if null_counts.sum() > 0:
    st.warning("데이터에 결측치가 존재합니다.")
    st.dataframe(null_counts[null_counts > 0].reset_index().rename(columns={0: '결측치 수', 'index': '컬럼명'}))
    
    # 2. 결측치 처리 방법 선택
    method = st.radio("결측치 처리 방법 선택", ["제거(Drop)", "최빈값으로 채우기(Fill with Mode)"])
    
    if method == "제거(Drop)":
        df = df.dropna()
        st.success("결측치가 포함된 행을 모두 제거했습니다.")
    else:
        # 문자열 데이터가 많으므로 최빈값(Mode)으로 채우는 것이 일반적입니다.
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        st.success("모든 결측치를 해당 컬럼의 최빈값으로 채웠습니다.")
else:
    st.success("데이터에 결측치가 없습니다! ✅")

st.write(f"현재 남은 데이터 수: {len(df)}")
