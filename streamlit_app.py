import streamlit as st
import pandas as pd

def introduce_myself() :
    st.markdown("# Introduction")
    st.markdown("## 한줄소개")
    st.write("안녕하세요. 저는 강영재입니다✨")
    st.markdown("### 전공")
    st.write("**주전공**: 영어영문학과")
    st.write("**복수전공**: 경제학부, 경영학과")
    st.markdown("### 하고 있는 일")
    st.write("1. 영어영문학과 과대")
    st.write("2. 서울 6개 대학 영어영문학과 연합 동아리 UR'E 학술기획국장")
    st.write("3. 넥슨 헬로메이트 1기")
    st.write("4. SAM멘토링")
    st.write("5. 서울대 중앙 게임동아리 snugdc 부원")
    st.markdown("### 하고 싶은 일")
    st.write("**게임기획자**🎮")
    st.markdown("### 취미")
    st.write("- 책 읽기: 최근에는 윌리엄 해즐릿의 <<혐오의 즐거움에 대하여>>를 읽고 있어요.")
    st.write("- 음악 듣기: 최근에는 데이식스의 '드디어 끝나갑니다'에 빠졌어요!")


def time_table():
    data = {"월": ["재무제표분석과 기업가치평가", "", ""], "화": ["", "행태경제학", "건강경제학"], "수": ["", "", ""], "목": ["", "경제통계학", ""], "금": ["컴퓨팅탐색", "", "정보문화기술입문"]}
    df = pd.DataFrame(data)

    st.markdown("# 나의 수업 시간표")
    st.markdown("## 정적 시간표 (st.table)")
    st.table(df)

    json_data = {"재무제표분석과 기업가치평가": {"교수": "황이석", "강의실": "58동 supexhall"}, "행태경제학": {"교수": "최승주", "강의실": "우석경제관 107호"}, "건강경제학": {"교수": "홍석철", "강의실": "우석경제관 107호"}, "경제통계학": {"교수": "류근관", "강의실": "우석경제관 107호"}, "컴퓨팅 탐색": {"교수": "변해선", "강의실": "26동"}, "정보문화기술입문": {"교수": "은진수", "강의실": "83동"}}
    st.write("## 수업 정보")
    st.json(json_data)

    st.write("## 이번학기 요약")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="수강과목 수", value="6", delta="+1")

    with col2:
        st.metric(label="이번 학기 학점", value="18", delta="+3학점")


selected = "자기소개"

li = ["자기소개", "시간표"]

st.sidebar.header("자기소개👩‍🦰와 시간표📃")

for item in li: # 딕셔너리의 key와 value를 다 가져옴
    if st.sidebar.button(item, key=item):  # 각 섹션을 버튼으로 표시
        selected = item # 무엇이 선택됐는지 확인

if selected == "자기소개":
    introduce_myself()
elif selected == "시간표":
    time_table()


