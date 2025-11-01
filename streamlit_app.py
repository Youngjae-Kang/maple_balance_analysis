# 파일 저장 모두 이 파일에 저장해라!
import streamlit as st
import pandas as pd
import numpy as np

st.title("간단한 Streamlit 예제")
st.write("데이터와 그래프를 보여주는 예제입니다")

# 샘플 데이터
data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['A', 'B', 'C']
)

st.line_chart(data) # 데이터를 넣은 것. 선 그래프
# 터미널에 streamlit run 파일명 해서 실행
