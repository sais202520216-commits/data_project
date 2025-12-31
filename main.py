import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# ---------------------------------
# 1. 스트림릿 기본 설정
# ---------------------------------
st.set_page_config(
    page_title="교통수단 사고율과 시간대의 상관관계",
    layout="wide"
)

# --- [추가] 한글 폰트 설정 ---
def set_korean_font():
    if platform.system() == 'Darwin': # 맥
        plt.rc('font', family='AppleGothic')
    elif platform.system() == 'Windows': # 윈도우
        plt.rc('font', family='Malgun Gothic')
    else: # 리눅스 (코랩/도커 등)
        plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

set_korean_font()

st.title("교통수단 사고율과 시간대의 상관관계")

# ---------------------------------
# 2. 데이터 불러오기
# ---------------------------------
@st.cache_data
def load_data():
    # 파일 경로를 실제 환경에 맞게 확인해주세요.
    # 예시: "한국도로교통공단_자전거사고 다발지역 개별사고 정보_20201231.csv"
    try:
        df = pd.read_csv("한국도로교통공단_자전거사고 다발지역 개별사고 정보_20201231.csv", encoding="cp949")
        return df
    except:
        # 파일이 없을 경우 테스트용 더미 데이터 생성 (작동 확인용)
        data = {
            '사고번호': range(100),
            '다발지구분': np.random.choice(['A', 'B'], 100),
            '사고내용': np.random.choice(['경상', '중상'], 100),
            '가해운전자_연령': np.random.randint(10, 80, 100)
        }
        return pd.DataFrame(data)

df = load_data()

st.subheader("📌 원본 데이터")
st.dataframe(df.head())

# ---------------------------------
# 3. 문자 → 숫자로 변환
# ---------------------------------
st.subheader("🔢 문자 데이터 숫자로 변환")

df_numeric = df.copy()
# 변환할 때 원본의 의미를 잃지 않도록 숫자형이 아닌 것만 골라 변환
object_cols = df_numeric.select_dtypes(include=['object']).columns

for col in object_cols:
    df_numeric[col], _ = pd.factorize(df_numeric[col])

st.write(f"✔ 변환된 컬럼: {', '.join(object_cols)}")
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
    
    # 필터링
    cleaned_data = data[(data[column] >= lower) & (data[column] <= upper)]
    return cleaned_data, lower, upper

# 숫자 컬럼 선택 (변수명이 명확한 컬럼 권장, 예: 연령 등)
num_columns = df_numeric.columns
selected_col = st.selectbox("이상치 제거할 컬럼 선택 (데이터 분포가 넓은 컬럼을 선택해보세요)", num_columns)

df_clean, low, up = remove_outliers_iqr(df_numeric, selected_col)

col1, col2 = st.columns(2)
col1.metric("이상치 제거 전 데이터", len(df_numeric))
col2.metric("이상치 제거 후 데이터", len(df_clean), f"{len(df_clean) - len(df_numeric)}")

# ---------------------------------
# 5. 시각화 (이상치 비교)
# ---------------------------------
st.subheader("📊 이상치 제거 전/후 비교")

# # 박스플롯이 왜 이상하게 나오는지 이해를 돕기 위한 구조 설명입니다.
# 데이터가 너무 뭉쳐있으면 상자만 보일 수 있습니다.

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# 제거 전
sns.boxplot(y=df_numeric[selected_col], ax=ax[0], color='skyblue')
ax[0].set_title(f"제거 전: {selected_col}")

# 제거 후
sns.boxplot(y=df_clean[selected_col], ax=ax[1], color='lightgreen')
ax[1].set_title(f"제거 후: {selected_col}")

# 레이아웃 조정
plt.tight_layout()
st.pyplot(fig)
