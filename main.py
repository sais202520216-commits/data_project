import streamlit as st
import pandas as pd
import seaborn as sns
# ---------------------------
# 1. 데이터 불러오기
# ---------------------------
uploaded_file = st.file_uploader("엑셀 파일 업로드 (fitness data.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.subheader("📌 데이터 미리보기")
    st.dataframe(df)

    # ---------------------------
    # 2. 체지방률과 상관관계 분석
    # ---------------------------

    if "체지방률" in df.columns:
        st.subheader("📊 체지방률과 상관관계가 높은 속성")

        corr_series = df.corr(numeric_only=True)["체지방률"].sort_values(ascending=False)
        st.write(corr_series)

        # ---------------------------
        # 3. 산점도 그리기
        # ---------------------------
        st.subheader("📈 산점도 그래프")

        x_col = st.selectbox("X축에 사용할 변수를 선택하세요", df.columns)
        if pd.api.types.is_numeric_dtype(df[x_col]):
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_col, y="체지방률", ax=ax)
            ax.set_title(f"{x_col} vs 체지방률")
            st.pyplot(fig)
        else:
            st.warning("선택한 X 변수는 숫자형이 아닙니다.")

        # ---------------------------
        # 4. 히트맵
        # ---------------------------
        st.subheader("🔥 상관관계 히트맵")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    else:
        st.error("데이터에 '체지방률' 열이 존재하지 않습니다. 엑셀 파일을 확인해주세요.")
