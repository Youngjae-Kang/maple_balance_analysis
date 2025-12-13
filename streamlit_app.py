import statsmodels.api as sm
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =======================================================
# 폰트 설정 (한글 깨짐 방지)
# =======================================================
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지
# =======================================================

# --- 1. 설정 및 데이터 로드 ---
DATA_FILE = "maple_analysis_data_N450_final.csv" # 데이터 수집 스크립트의 파일명과 일치해야 함

st.set_page_config(layout="wide")
st.title("메이플스토리 : 직업별 성장 효율 분석")
st.markdown("---")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
        return df
    except FileNotFoundError:
        st.error(f"'{DATA_FILE}' 파일을 찾을 수 없습니다. 데이터 수집 스크립트 실행 후 다시 시도하세요.")
        return None

df = load_data()

if df is not None:
    
    # --- A. 데이터 필수 컬럼 진단 및 오류 처리 ---
    required_cols = ['직업분류', '전투력', '주스탯', '보스_몬스터_데미지', '크리티컬_데미지', '방어율_무시']    
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"❌ 데이터프레임에 필수 컬럼이 누락되었습니다: {missing_cols}")
        st.code(f"현재 데이터 컬럼 목록: {list(df.columns)}", language='python')
        st.stop()
    # --- 진단 및 오류 처리 끝 ---

    # --- B. 데이터 클리닝 및 타입 변환 (Object dtype 오류 해결) ---
    
    # 1. 숫자형 컬럼 리스트 정의
    numeric_cols = ['전투력', '주스탯', '보스_몬스터_데미지', '크리티컬_데미지', '방어율_무시']

    for col in numeric_cols:
        if col in df.columns:
            # 숨겨진 문자/쉼표/공백 제거 및 강제 변환
            if df[col].dtype == 'object':
                 df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. NaN (결측치) 처리: 회귀 분석 전에 결측치가 있는 행 제거
    df.dropna(subset=numeric_cols + ['직업분류'], inplace=True) 
    
    # 3. 사이드바 정보 표시 (클리닝 후의 최종 샘플 수)
    st.sidebar.header("분석 정보")
    st.sidebar.metric("총 샘플 수", len(df))
    st.sidebar.metric("분석 대상 직업 분류", df['직업분류'].nunique())

# --- 2. 데이터 전처리 및 회귀분석 ---
    
    # 종속 변수 (Y) 설정
    Y = np.log1p(df['전투력']) 
    Y.reset_index(drop=True, inplace=True) 
    
    # 직업 분류 더미 변수화 (X_job)
    df['직업분류'] = df['직업분류'].astype(str)
    X_job = pd.get_dummies(df['직업분류'], drop_first=True, prefix='Job')
    
    # OLS 모델은 bool 타입을 처리하지 못하므로, 0과 1의 int 타입으로 변환합니다.
    for col in X_job.columns:
        if X_job[col].dtype == 'bool':
            X_job[col] = X_job[col].astype(int)
    
    # 통제 변수 (X_control) 설정
    X_control = df[['주스탯', '보스_몬스터_데미지', '크리티컬_데미지', '방어율_무시']]

    # 스케일링
    scaler = StandardScaler()
    X_control_scaled = scaler.fit_transform(X_control)
    X_control_scaled = pd.DataFrame(X_control_scaled, columns=X_control.columns)
    
    # 최종 독립 변수 병합
    X = pd.concat([X_job.reset_index(drop=True), X_control_scaled.reset_index(drop=True)], axis=1)
    
    # 상수항 추가
    X = sm.add_constant(X, has_constant='add')
    
    # 모델 적합
    try:
        model = sm.OLS(Y, X).fit()
    except Exception as e:
        st.error(f"❌ 회귀 모델 적합 중 심각한 오류 발생: {e}")
        st.warning("데이터의 모든 값이 숫자인지 확인하십시오. 오류가 계속되면 데이터를 샘플링하거나 줄여보십시오.")
        st.write("--- 최종 독립 변수 X의 DTYPES ---")
        st.dataframe(X.dtypes.to_frame(name='Dtype'))
        st.write("--- X 변수 상위 5개 행 (확인용) ---")
        st.dataframe(X.head())
        st.stop()


# 목차 구성
toc = {
    "예상 효율 시뮬레이션": [],
    "0. 메알못의 메이플스토리 분석기": [
        "0.1. 연구 동기",
        "0.2. 메이플스토리란?",
        "0.3. 메이플스토리의 직업"
    ],
    "1. 분석 방법": [
        "1.1.-1.4. 분석 방법"
    ],
    "2. 다중 선형 회귀분석 결과": [
        "2.1. 직업 분류 상세 정보",
        "2.2. 회귀 분석 결과",
        "2.3. 그래프"
    ]
}

# 상위 챕터 선택
chapter = st.sidebar.selectbox("📂 챕터 선택", list(toc.keys()))

# 하위 섹션 선택
section = st.sidebar.radio("📑 섹션 선택", toc[chapter])

# 본문 출력

# ----------------------------------------------------
# --- ★★★ 새로운 기능 추가: 예상 효율 분석 시뮬레이션 ★★★ ---
# ----------------------------------------------------

# 2. --- 컨테이너 (버튼과 결과만 포함) ---
if chapter == "예상 효율 시뮬레이션":
    st.header("🎯 내 캐릭터 효율 예측 시뮬레이션")

    # 직업 목록 및 기준 직업 설정 (이전 로직 유지)
    all_jobs = df['직업분류'].unique().tolist()
    try:
        modeled_jobs = [col.replace('Job_', '') for col in model.params.index if col.startswith('Job_')]
        remaining_jobs = [job for job in all_jobs if job not in modeled_jobs]
        # 모델 학습에서 제외된 '원래' 기준 직업군 (더미 변수 drop_first=True로 인해 절편에 흡수됨)
        original_reference_job = remaining_jobs[0] if remaining_jobs else all_jobs[0]
        analysis_jobs = [original_reference_job] + modeled_jobs
    except (NameError, AttributeError):
        st.warning("경고: 회귀 모델 학습이 완료되지 않아 시뮬레이션을 실행할 수 없습니다. 데이터 로드 및 모델 학습 단계를 확인해주세요.")
        st.stop()
    except AttributeError:
        # model.params가 없는 경우 (학습 실패 등)
        st.warning("경고: 회귀 모델 학습이 실패하여 시뮬레이션을 실행할 수 없습니다.")
        st.stop()

    # 1. --- 입력 필드 섹션 (컨테이너 밖에 위치) ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("나의 스펙 입력 (통제 변수)")
        # 평균치를 기본값으로 사용하여 사용자가 쉽게 입력할 수 있도록 합니다.
        default_stat = int(df['주스탯'].mean())
        input_main_stat = st.number_input("주스탯", min_value=1, value=default_stat, step=10000)
        
        default_boss = int(df['보스_몬스터_데미지'].mean())
        input_boss_dmg = st.number_input("보스 몬스터 데미지 (%)", min_value=0, value=default_boss, step=10)
        
        default_crit = int(df['크리티컬_데미지'].mean())
        input_crit_dmg = st.number_input("크리티컬 데미지 (%)", min_value=0, value=default_crit, step=5)
        
        default_def = int(df['방어율_무시'].mean())
        input_def_ignore = st.number_input("방어율 무시 (%)", min_value=0, value=default_def, step=5)
        
    with col2:
        st.subheader("비교 대상 선택")

        user_reference_job = st.selectbox(
        "**비교 기준** 직업군 선택",
        options=analysis_jobs,
        index=analysis_jobs.index(original_reference_job) if original_reference_job in analysis_jobs else 0,
        key="user_ref_job"
    )
        
    # 비교 대상 직업군 선택
        target_job = st.selectbox(
            "**비교 대상** 직업군 선택",
            options=analysis_jobs,
            index=analysis_jobs.index(original_reference_job) if original_reference_job in analysis_jobs else 0,
            key="target_job"
        )
        # 빈 공간 채우기
        st.markdown("---")
        st.markdown("모델에 사용된 스케일러와 통제 변수 순서를 맞춰야 정확한 예측이 가능합니다.")
    
    analysis = st.button("예상 효율 분석하기", use_container_width=True)
    
    if analysis:
        
        # [수정] 기준 직업과 대상 직업이 같은지 확인
        if user_reference_job == target_job:
            st.warning("경고: 비교 기준 직업과 비교 대상 직업이 동일합니다. 다른 직업을 선택해 주세요.")
            st.stop()
        
        # 1. 입력 데이터를 DataFrame으로 구성 (원래 통제 변수 순서대로)
        input_data = pd.DataFrame({
            '주스탯': [input_main_stat], 
            '보스_몬스터_데미지': [input_boss_dmg],
            '크리티컬_데미지': [input_crit_dmg],
            '방어율_무시': [input_def_ignore]
        })

        # 2. 통제 변수 스케일링 (이미 fit된 scaler 사용)
        scaled_control = scaler.transform(input_data[X_control.columns])
        scaled_control_df = pd.DataFrame(scaled_control, columns=X_control.columns)
        
        # 3. 예측 로직 함수 정의: 특정 직업의 로그 전투력을 예측하는 함수
        def predict_log_power(job_name, scaled_data):
            job_dummies_cols = [f'Job_{job}' for job in modeled_jobs]
            job_dummies_data = pd.DataFrame(0, index=[0], columns=job_dummies_cols)
            
            # 예측하려는 직업이 original_reference_job이 아닐 경우에만 해당 더미 변수를 1로 설정
            if job_name != original_reference_job:
                target_col = f'Job_{job_name}'
                if target_col in job_dummies_cols:
                    job_dummies_data[target_col] = 1

            # 최종 예측용 X 행렬 구성
            X_pred_raw = pd.concat([scaled_data, job_dummies_data], axis=1)
            X_pred = sm.add_constant(X_pred_raw, has_constant='add')
            X_pred = X_pred[model.params.index] # OLS 모델 컬럼 순서 강제 적용
            
            return model.predict(X_pred)[0]

        # 4. 예측값 계산
        # 사용자 기준 직업 예측
        predicted_log_ref = predict_log_power(user_reference_job, scaled_control_df)
        predicted_power_ref = np.expm1(predicted_log_ref)
        
        # 목표 직업 예측
        predicted_log_target = predict_log_power(target_job, scaled_control_df)
        predicted_power_target = np.expm1(predicted_log_target)
        
        # 5. 전투력 차이 계산
        power_diff = predicted_power_target - predicted_power_ref
        power_ratio = (predicted_power_target / predicted_power_ref) - 1
        
        # --- 결과 출력 ---
        st.subheader(f"📊 **{user_reference_job} vs {target_job}** 예상 효율 분석 결과")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.metric(f"기준 ({user_reference_job}) 예상 전투력", f"{predicted_power_ref:,.0f}")
        with col_res2:
            st.metric(f"대상 ({target_job}) 예상 전투력", f"{predicted_power_target:,.0f}")
        with col_res3:
            st.metric(
                "상대적 효율 (기준 대비)",
                f"{power_ratio:.1%}", 
                delta=f"{power_diff:,.0f} 차이" 
            )
        
        # 해석 제공
        if power_ratio > 0.05:
            st.success(f"✅ {target_job} 직업군이 {user_reference_job} 대비 **약 {power_ratio:.1%}** 더 높은 전투력을 가질 것으로 예상됩니다. (스펙 통제)")
        elif power_ratio < -0.05:
            st.error(f"❌ {target_job} 직업군이 {user_reference_job} 대비 **약 {-power_ratio:.1%}** 더 낮은 전투력을 가질 것으로 예상됩니다. (스펙 통제)")
        else:
            st.info(f"💡 {target_job} 직업군과 {user_reference_job} 직업군 간의 전투력 효율 차이는 크지 않을 것으로 예상됩니다. ({power_ratio:.1%})")


elif section == "0.1. 연구 동기":
    st.header("0. 메알못의 메이플스토리 분석기")
    st.subheader("0.1. 연구 동기")
    st.write("""
             최근 게임 산업에 관심이 생겼습니다.
             \n게임은 인생에 도움이 되지 않는 취미로 여겨지기도 하지만,
             \n하나의 게임을 자세히 뜯어보면 놀라울 정도로 체계적인 설계가 숨어있었습니다.
             \n수많은 게임 중 누구나 한 번쯤 들어봤을 **메이플스토리**.
             \n특히 다양한 직업이 존재하는 만큼 직업 간 밸런스가 어떻게 유지되고 있는지 궁금해졌습니다.
             \n게임에서 **밸런스**는 특정 직업이나 스킬 등이 구조적으로 유리하지 않도록 설계되는 것을 의미합니다.
             \n그렇다면 메이플스토리의 직업 밸런스는 실제 데이터에서도 그렇게 나타날까요?
             \n이러한 질문에 답하기 위해 전투력과 종합적인 스탯 데이터를 활용하여
             \n직업 간 차이가 통계적으로 유의미한지 회귀 분석을 통해 살펴보려고 합니다.
             
             \n\n *※주의: 저는 메이플스토리를 깊게 플레이해본 적 없는 '메알못'입니다.*
             """)
elif section == "0.2. 메이플스토리란?":
    st.header("0. 메알못의 메이플스토리 분석기")
    st.subheader("0.2. 메이플스토리란?")
    st.write("""
             넥슨에서 서비스하는 대표적인 MMORPG 게임!
             \n2003년에 출시된 이후 20년 넘게 운영되고 있는 장수 게임입니다.
             \n귀여운 2D 도트 그래픽과 복잡한 성장 구조가 매력입니다.
             
             \n캐릭터를 키우고, 장비를 강화하고, 더 강한 보스를 잡아야 합니다.
             \n이 과정에서 유저는 필수적으로 선택해야 하는 것이 있습니다.
             \n바로 **직업**입니다.
             \n
             """)
elif section == "0.3. 메이플스토리의 직업":
    st.header("0. 메알못의 메이플스토리 분석기")
    st.subheader("0.3. 메이플스토리의 직업")
    st.write("""
             메이플스토리에는 많은 직업이 있습니다.
             \n분석에 사용된 **전사, 마법사, 궁수, 도적, 해적** 5개 분류와 세부 직업 목록입니다.
             """)

    all_jobs_map = {
        "전사": [
            "히어로", "팔라딘", "다크나이트", "소울마스터", "미하일", "아란", 
            "데몬슬레이어", "데몬어벤져", "카이저", "아델", "블래스터", "제로"
        ],
        "마법사": [
            "아크메이지(불,독)", "아크메이지(썬,콜)", "비숍", "플레임위자드", 
            "에반", "루미너스", "배틀메이지", "일리움", "라라", "키네시스"
        ],
        "궁수": [
            "보우마스터", "신궁", "윈드브레이커", "메르세데스", "와일드헌터", "카인"
        ],
        "도적": [
            "나이트로드", "섀도어", "듀얼블레이드", "나이트워커", "팬텀", "카데나", "칼리"
        ],
        "해적": [
            "바이퍼", "캡틴", "스트라이커", "은월", "엔젤릭버스터", "아크", "제논", "캐논마스터"
        ]
    }

    data = []
    for group, jobs in all_jobs_map.items():
        data.append({'큰 직업 분류': group, '세부 직업 목록': ', '.join(sorted(jobs))})

    full_job_df = pd.DataFrame(data)

    st.dataframe(full_job_df, use_container_width=True)


elif section == "1.1.-1.4. 분석 방법":
    # 1. 분석 방법
    st.header("1. 분석 방법")
    st.subheader("1.1. 가설")
    st.write("""
             #### 귀무가설
             \n다른 스펙 요인을 통제할 경우, 메이플스토리의 직업은 전투력에 **통계적으로 유의미한 영향을 미치지 않는다**.
             \n#### 대립가설
             \n다른 스펙 요인을 통제하더라도, 메이플스토리의 직업은 전투력에 **통계적으로 유의미한 영향을 미친다**.
             \n▷ 본 분석에서 직업에 따른 전투력 차이가 통계적으로 유의하지 않을 경우, 직업 간 밸런스가 비교적 잘 유지되고 있다고 해석합니다.
             """)

    st.subheader("1.2. 데이터 수집")
    st.write("""
             #### 데이터 수집
             \n넥슨 open api에서 메이플스토리 데이터를 수집했습니다.
             \n#### 표본
             \n다섯 개의 월드(스카니아, 루나, 베라, 크로아, 엘리시움)의 상위 2000명 중 각각 무작위로 60명씩 추출
             \n#### 과정
             \n상위 2000명 닉네임 수집 → 닉네임을 통해 ocid 추출 → ocid로 종합 스탯 추출
             \n▷ 서버 편향을 줄이기 위해 성격이 다른 월드를 혼합하여 표본 구성
             \n▷ 장비 등 종합 스탯 이외의 영향을 최소화하기 위해 상위권에서 표본 추출
             \n▷ 수치가 0이거나 비공개여서 사용할 수 없는 데이터를 제외하여 총 **273개**의 표본 획득
             \n▷ api의 하루 호출량이 1000번이므로, 월드당 30개로 결정
             """)
    
    all_variables = {
        "주스탯": "각 직업이 사용하는 핵심 능력치의 합 (전사 STR, 마법사 INT, 궁수 DEX, 도적 LUK)",
        "보스 몬스터 데미지": "보스 몬스터에게 주는 피해량을 추가로 증가시키는 스탯",
        "방어율 무시": "몬스터의 방어력을 무시하고 공격할 수 있는 비율",
        "크리티컬 데미지": "치명타 발생 시 피해량 증가 비율",
        "직업군": "메이플스토리의 직업을 5가지로 나눈 것 (궁수, 도적, 마법사, 전사, 해적)"
    }

    data = []
    for words, meaning in all_variables.items():
        data.append({'변수 목록': words, '설명': meaning})

    variables_df = pd.DataFrame(data)
    
    st.subheader("1.3. 변수")
    st.write("""
             - 독립변수: 직업군, 주스탯, 보스 몬스터 데미지, 크리티컬 데미지, 방어율 무시
             \n- 종속변수: 전투력
             \n ▷ 직업군은 더미변수로 변환 (해당 직업군일 경우 1, 아닐 경우 0)
             \n ▷ 직업군 이외의 변수는 통제변수로 활용
             """)
    
    st.dataframe(variables_df, use_container_width=True)

    st.subheader("1.4. 분석 방법")
    st.write("""
             #### 다중 선형 회귀 모델로 분석
             \n1. 궁수 직업군과 타 직업군을 비교
             \n2. 선형 회귀로 분석하기 위해 로그플러스원 변환 [np.log1p(df['전투력'])]
             \n3. StandardScaler를 통해 변수간 스케일 조정 (표준화)
             \n4. 변수간 다중공선성 파악 [variance_inflation_factor]
             \n#### 결과 분석 방법
             \n1. p-value가 0.05 이하인 경우 유의하다고 판정
             \n2. 잔차 산점도: 모델이 데이터의 선형성 가정을 잘 충족하는지 시작적으로 검토
             \n3. 상관관계 히트맵: 변수 간 상관관계 파악
             """)

    st.markdown("---")

elif section == "2.1. 직업 분류 상세 정보":
    st.header("2. 다중 선형 회귀분석 결과")
    st.markdown("**종속 변수:** $\ln(\text{전투력}+1)$ (로그 변환)")
    
    # 실제 제외된 기준 그룹을 찾아 표시합니다.
    all_groups = sorted(df['직업분류'].unique()) # 모든 직업 분류를 정렬
    if all_groups:
        reference_group = all_groups[0] # 정렬 순서상 첫 번째 그룹이 drop_first=True에 의해 제외됨
        st.markdown(f"**기준 직업 분류:** **{reference_group}** (더미 변수에서 제외된 그룹)")
    else:
        st.markdown("**기준 직업 분류:** 데이터에 직업 분류가 없습니다.")


    # --- 3. Streamlit 대시보드 구성 --- (이 섹션 시작 전에 추가)

    # 0. 직업 분류 매핑 테이블 생성
    st.subheader("2.1. 직업 분류 상세 정보")
    st.markdown("분석 대상 데이터를 기반으로 한 **큰 직업 분류**와 이에 속하는 **세부 직업** 목록입니다.")

    # 1. '직업분류'와 '직업' 컬럼만을 선택하고 중복 제거
    job_mapping_df = df[['직업분류', '직업']].drop_duplicates().reset_index(drop=True)

    # 2. '직업분류'를 기준으로 그룹화하여 세부 직업을 리스트로 묶음
    # 파이썬의 groupby와 join 함수를 사용하여 세부 직업을 쉼표(,)로 연결
    grouped_jobs = job_mapping_df.groupby('직업분류')['직업'].apply(lambda x: ', '.join(sorted(x))).reset_index()

    # 3. Streamlit에 표로 출력
    grouped_jobs.columns = ['큰 직업 분류', '세부 직업 목록']
    st.dataframe(grouped_jobs, use_container_width=True)

elif section == "2.2. 회귀 분석 결과":
    st.header("2. 다중 선형 회귀분석 결과")
    st.markdown("**종속 변수:** $\ln(\text{전투력}+1)$ (로그 변환)")
    st.subheader("2.2. 회귀 분석 결과")

    st.code(model.summary().as_text(), language='text')

    st.write("""#### 결과 해석
             \nR-squared: 0.72""")


    st.write("#### 직업 분류별 전투력 영향 (회귀계수)")
    st.markdown("계수가 높을수록 기준 그룹(궁수) 대비 효율이 좋습니다.")
    
    job_coeffs = model.params[model.params.index.str.startswith('Job')]
    job_pvalues = model.pvalues[model.pvalues.index.str.startswith('Job')]
    
    coeff_df = pd.DataFrame({
        '회귀계수': job_coeffs,
        'P-value': job_pvalues
    })
    
    coeff_df['유의성'] = np.where(coeff_df['P-value'] < 0.05, '유의함 (p < 0.05)', '유의하지 않음')
    coeff_df = coeff_df.sort_values(by='회귀계수', ascending=False)
    
    st.dataframe(coeff_df)
    
    
    # Streamlit 네이티브 차트를 위해 Index를 컬럼으로 변환
    coeff_df_chart = coeff_df.reset_index()
    coeff_df_chart.columns = ['직업 분류', '회귀 계수', 'P-value', '유의성']

    # 네이티브 차트 출력 (색상 구분이 어려우므로 별도 마크다운 설명 필요)
    st.bar_chart(
        coeff_df_chart, 
        x='직업 분류', 
        y='회귀 계수', 
        color='회귀 계수' # 값에 따라 자동으로 색상을 다르게 표시
    )

elif section == "2.3. 그래프":
    st.header("2. 다중 선형 회귀분석 결과")
    st.subheader("2.3. 그래프")
        
    # 1. 잔차 산점도
    st.write("#### 모델 진단: 잔차 산점도")
    st.write("""예측값($\hat{Y}$)에 따른 잔차($Y - \hat{Y}$)의 분포를 나타냅니다.
             \n잔차들이 0을 중심으로 **무작위로** 분포해야 선형 회귀 모델의 가정이 충족됩니다.
             """)

    # 모델의 예측값 계산
    predicted_Y = model.fittedvalues
    # 모델의 잔차(Residuals) 계산
    residuals = model.resid
    
    # 네이티브 차트를 위해 두 배열을 DataFrame으로 병합
    residual_chart_df = pd.DataFrame({
        '예측된 로그 전투력': predicted_Y,
        '잔차': residuals
    })

    st.scatter_chart(
        residual_chart_df,
        x='예측된 로그 전투력',
        y='잔차',
    )

    st.write("""
             잔차가 예측값 전반에 걸쳐 **무작위적으로 분포**하고 있습니다.
             \n▷ 선형 회귀모형의 기본 가정이 크게 위배되지 않는 것으로 판단됩니다.
             """)


    # 2. 다중공선성 진단 (VIF)
    st.write("#### 다중공선성 진단")
    st.write("주요 통제 변수(스펙) 간의 다중공선성을 확인합니다. 다중공선성이 10을 초과하면 변수 간 상관성이 매우 높습니다.")
    
    try:
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_control.columns
        vif_data["VIF"] = [variance_inflation_factor(X_control_scaled.values, i) for i in range(len(X_control_scaled.columns))]
        
        st.dataframe(vif_data.sort_values(by="VIF", ascending=False))
    except Exception as e:
        st.warning(f"VIF 계산 중 오류 발생: {e}. (주로 샘플 부족 또는 모든 값이 0인 경우 발생)")

    # 3. 통제 변수 상관관계 히트맵
    st.write("#### 통제 변수 간의 상관관계: 히트맵")
    
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(X_control.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    ax_corr.set_title("스펙 변수 간 상관관계")
    st.pyplot(fig_corr)

