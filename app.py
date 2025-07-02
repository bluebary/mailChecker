import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import logging
from dbmanager import DatabaseManager

# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 파일 핸들러 설정
file_handler = logging.FileHandler('streamlit.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# 콘솔 핸들러 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

# 포매터 설정
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 핸들러 추가 (중복 방지)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def setup_app() -> None:
    """
    애플리케이션 초기 설정
    """
    logger.info("Streamlit 애플리케이션 초기 설정 시작")
    st.set_page_config(
        page_title="이메일 분석 대시보드",
        page_icon="📧",
        layout="wide"
    )
    st.title("📧 이메일 분석 대시보드")
    logger.info("Streamlit 애플리케이션 초기 설정 완료")

def setup_sidebar(db_manager: DatabaseManager) -> str:
    """
    사이드바 구성 및 분석 뷰 선택 기능

    Args:
        db_manager (DatabaseManager): 데이터베이스 관리자 인스턴스

    Returns:
        str: 선택된 뷰 이름
    """
    logger.info("사이드바 설정 시작")
    st.sidebar.title("🔍 분석 뷰 선택")
    
    try:
        # 표시할 뷰 목록 생성
        base_views = ["모든 결과", "모델별 통계"]
        model_names = db_manager.get_model_names()
        model_views = [f"{name} 결과" for name in model_names]
        available_views = base_views + model_views
        
        selected_view = st.sidebar.selectbox(
            "조회할 데이터를 선택하세요.",
            available_views
        )
        logger.info(f"사용자가 뷰 선택: {selected_view}")
    except Exception as e:
        logger.error(f"뷰 목록 불러오기 오류: {e}")
        st.sidebar.error(f"뷰 목록을 불러오는 중 오류가 발생했습니다: {e}")
        selected_view = None

    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        이 대시보드는 이메일 분석 결과를 다양한 관점에서 조회하고,
        'human_verified_spam' 컬럼을 편집하여 사람의 판별 결과를 
        데이터베이스에 기록할 수 있습니다.
        """
    )

    return selected_view

def display_dataframe(
    db_manager: DatabaseManager, 
    data: pd.DataFrame, 
    is_editable: bool, 
    view_name: str
) -> None:
    """
    데이터프레임 표시 및 편집 기능

    Args:
        db_manager (DatabaseManager): 데이터베이스 관리자 인스턴스
        data (pd.DataFrame): 표시할 데이터프레임
        is_editable (bool): 데이터 편집 기능 활성화 여부
        view_name (str): 현재 뷰의 이름 (고유한 위젯 키 생성에 사용)
    """
    st.subheader(f"데이터: {view_name}")

    if data.empty:
        st.write("표시할 데이터가 없습니다.")
        return

    # 필터링 및 정렬 옵션
    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("검색어", key=f"search_{view_name}")
    with col2:
        if is_editable and "human_verified_spam" in data.columns:
            spam_filter = st.selectbox(
                "스팸 필터", 
                ["모두", "스팸", "정상", "미확인"], 
                key=f"filter_{view_name}"
            )
        else:
            spam_filter = "모두"

    # 데이터 필터링
    filtered_data = data.copy()
    if search_term:
        # 모든 컬럼을 문자열로 변환하여 검색
        filtered_data = filtered_data[
            filtered_data.astype(str).apply(
                lambda row: row.str.contains(search_term, case=False, na=False).any(), 
                axis=1
            )
        ]

    if is_editable and "human_verified_spam" in filtered_data.columns and spam_filter != "모두":
        if spam_filter == "스팸":
            filtered_data = filtered_data[filtered_data["human_verified_spam"] == True]
        elif spam_filter == "정상":
            filtered_data = filtered_data[filtered_data["human_verified_spam"] == False]
        elif spam_filter == "미확인":
            filtered_data = filtered_data[filtered_data["human_verified_spam"].isnull()]

    # 페이지네이션
    rows_per_page = 100  # 페이지당 행 수를 100으로 고정
    total_rows = len(filtered_data)
    total_pages = max(1, (total_rows + rows_per_page - 1) // rows_per_page)
    
    # 세션 상태에서 현재 페이지 가져오기 또는 초기화
    page_key = f"page_{view_name}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    
    page = st.session_state[page_key]

    # 필터링 등으로 인해 전체 페이지 수가 줄었을 경우, 현재 페이지 번호를 조정
    if page > total_pages:
        page = total_pages
        st.session_state[page_key] = page

    start_idx = (page - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, total_rows)
    page_data = filtered_data.iloc[start_idx:end_idx]

    # 페이지 정보와 이전/다음 버튼을 한 줄에 표시
    col1, col2, col3 = st.columns([0.8, 0.05, 0.05])
    with col1:
        st.write(f"전체 {total_rows}개 중 {start_idx + 1}~{end_idx}개 표시 (페이지 {page}/{total_pages})")
    
    with col2:
        if st.button("이전", key=f"prev_{view_name}", disabled=(page <= 1)):
            st.session_state[page_key] -= 1
            st.rerun()

    with col3:
        if st.button("다음", key=f"next_{view_name}", disabled=(page >= total_pages)):
            st.session_state[page_key] += 1
            st.rerun()

    # human_verified_spam 컬럼이 있는 경우 편집 기능 추가
    if is_editable:
        if 'id' not in page_data.columns:
            st.error("테이블에 'id' 컬럼이 없어 편집할 수 없습니다.")
            st.dataframe(page_data)
            return

        # 원본 데이터의 bool 및 None 값을 유지하기 위해 astype 대신 직접 변환
        editor_data = page_data.copy()
        editor_data['human_verified_spam'] = editor_data['human_verified_spam'].apply(
            lambda x: None if pd.isna(x) else bool(x)
        )
        
        # 표시할 데이터에서 'id' 컬럼 제외
        display_editor_data = editor_data.drop(columns=['id'], errors='ignore')
        disabled_columns = [col for col in display_editor_data.columns if col != 'human_verified_spam']
        
        edited_df = st.data_editor(
            display_editor_data,
            column_config={
                "human_verified_spam": st.column_config.CheckboxColumn(
                    "사용자 확인 스팸",
                    default=False,
                )
            },
            disabled=disabled_columns,
            hide_index=True,
            key=f"editor_{view_name}_{page}"
        )

        if edited_df is not None:
            # 변경사항 감지
            # st.data_editor는 None을 False로 반환할 수 있으므로, 원본과 비교 시 주의
            original_series = page_data['human_verified_spam'].apply(lambda x: None if pd.isna(x) else bool(x))
            edited_series = pd.Series(edited_df['human_verified_spam'], index=page_data.index)
            
            # 직접 비교를 통해 변경된 행 찾기
            changed_mask = original_series != edited_series
            changed_indices = changed_mask[changed_mask].index

            if len(changed_indices) > 0:
                # 첫 번째 변경된 행의 인덱스를 안전하게 가져옴
                changed_idx = changed_indices[0]
                
                # 해당 인덱스가 page_data에 존재하는지 확인
                if changed_idx in page_data.index:
                    row_id = page_data.loc[changed_idx, 'id']
                    
                    # edited_df에서 해당 행의 위치를 찾기
                    edited_row_position = page_data.index.get_loc(changed_idx)
                    if edited_row_position < len(edited_df):
                        new_spam_status = edited_df.iloc[edited_row_position]['human_verified_spam']
                        
                        # dbmanager의 새 메서드 호출
                        logger.info(f"사용자 데이터 편집 시도 - ID: {row_id}, 새 스팸 상태: {new_spam_status}")
                        if db_manager.update_human_verification(row_id, new_spam_status):
                            logger.info(f"사용자 데이터 편집 성공 - ID: {row_id}")
                            st.success(f"ID {row_id}의 스팸 상태가 '{new_spam_status}' (으)로 업데이트되었습니다.")
                            st.rerun()
                        else:
                            logger.error(f"사용자 데이터 편집 실패 - ID: {row_id}")
                            st.error(f"ID {row_id} 업데이트 실패")
                    else:
                        st.error("편집된 데이터에서 해당 행을 찾을 수 없습니다.")
                else:
                    st.error("변경된 행의 인덱스가 유효하지 않습니다.")
    else:
        # 표시할 데이터에서 'id' 컬럼 제외
        st.dataframe(page_data.drop(columns=['id'], errors='ignore'), hide_index=True)

def display_visualizations(db_manager: DatabaseManager, data: pd.DataFrame, view_name: str) -> None:
    """
    데이터 시각화

    Args:
        db_manager (DatabaseManager): 데이터베이스 관리자 인스턴스
        data (pd.DataFrame): 시각화할 데이터
        view_name (str): 현재 뷰의 이름
    """
    st.subheader("데이터 시각화")
    
    if data.empty:
        st.write("시각화할 데이터가 없습니다.")
        return

    # 모델별 통계 뷰인 경우와 일반 결과 뷰인 경우 다른 시각화 제공
    if view_name == "모델별 통계":
        # 모델별 통계 뷰에서는 cm_data를 직접 가져옴
        cm_data = db_manager.get_confusion_matrix_data()
        display_model_stats_visualizations(data, cm_data)
    else:
        display_results_visualizations(data)

def display_model_stats_visualizations(stats_data: pd.DataFrame, cm_data: pd.DataFrame) -> None:
    """
    모델별 통계 뷰를 위한 시각화

    Args:
        stats_data (pd.DataFrame): 모델별 통계 데이터 (model_stats_view)
        cm_data (pd.DataFrame): Confusion Matrix 데이터
    """
    st.write("#### 모델별 성능 분석")
    tab1, tab2, tab3 = st.tabs(["분류 통계", "Confusion Matrix", "성능 지표 비교"])

    with tab1:
        st.write("##### 모델별 이메일 분류 통계")
        if not stats_data.empty and 'model' in stats_data.columns:
            # 모델별로 통합하여 차트 데이터 준비
            model_summary = {}
            for model in stats_data['model'].unique():
                model_data = stats_data[stats_data['model'] == model]
                model_summary[model] = {
                    '평균 신뢰도': model_data['avg_reliability'].mean(),
                    '평균 분석 시간': model_data['avg_duration'].mean(),
                    '총 분석 건수': model_data['total_emails'].sum()
                }
            
            summary_df = pd.DataFrame(model_summary).T
            st.write("**모델별 통합 통계**")
            st.bar_chart(summary_df[['평균 신뢰도', '평균 분석 시간']])
            st.dataframe(summary_df.style.format({
                '평균 신뢰도': '{:.3f}',
                '평균 분석 시간': '{:.3f}',
                '총 분석 건수': '{:.0f}'
            }))
        else:
            st.info("모델 통계 데이터가 충분하지 않습니다.")

    with tab2:
        st.write("##### 모델별 통합 Confusion Matrix")
        if not cm_data.empty:
            # 모든 모델의 리스트 가져오기
            models = sorted(cm_data['model'].unique())
            
            if models:
                num_columns = 3
                for i in range(0, len(models), num_columns):
                    cols = st.columns(num_columns)
                    for j in range(num_columns):
                        model_idx = i + j
                        if model_idx < len(models):
                            model = models[model_idx]
                            with cols[j]:
                                st.subheader(f"{model}")
                                
                                # 해당 모델의 first_spam과 second_spam 데이터 가져오기
                                model_data = cm_data[cm_data['model'] == model]
                                first_spam_data = model_data[model_data['analysis_type'] == 'first_spam']
                                second_spam_data = model_data[model_data['analysis_type'] == 'second_spam']
                                
                                if len(first_spam_data) == 0 and len(second_spam_data) == 0:
                                    st.info(f"{model} 모델에 대한 데이터가 없습니다.")
                                    continue
                                
                                # 통합 Confusion Matrix 생성
                                total_tn = 0
                                total_fp = 0
                                total_fn = 0
                                total_tp = 0
                                
                                analysis_labels = []
                                
                                if len(first_spam_data) > 0:
                                    first_data = first_spam_data.iloc[0]
                                    total_tn += first_data['TN']
                                    total_fp += first_data['FP']
                                    total_fn += first_data['FN']
                                    total_tp += first_data['TP']
                                    analysis_labels.append("1차")
                                
                                if len(second_spam_data) > 0:
                                    second_data = second_spam_data.iloc[0]
                                    total_tn += second_data['TN']
                                    total_fp += second_data['FP']
                                    total_fn += second_data['FN']
                                    total_tp += second_data['TP']
                                    analysis_labels.append("2차")
                                
                                # 통합된 Confusion Matrix
                                cm_combined = [
                                    [int(total_tn), int(total_fp)],
                                    [int(total_fn), int(total_tp)]
                                ]
                                
                                # 히트맵 생성
                                fig = go.Figure(data=go.Heatmap(
                                    z=cm_combined,
                                    x=['예측: 정상', '예측: 스팸'],
                                    y=['실제: 정상', '실제: 스팸'],
                                    colorscale='Blues',
                                    showscale=True,
                                    text=[[f"{val}" for val in row] for row in cm_combined],
                                    texttemplate="%{text}",
                                    textfont={"size": 16}
                                ))
                                
                                analysis_type_str = " + ".join(analysis_labels) if analysis_labels else "데이터 없음"
                                fig.update_layout(
                                    width=400, 
                                    height=400,
                                    title=f"<b>{model}</b><br><sub>({analysis_type_str} 분석 통합)</sub>",
                                    margin=dict(t=60, l=10, r=10, b=10)
                                )
                                st.plotly_chart(fig, key=f"{model}_integrated_cm")
                                
                                # 통계 정보 표시
                                total_samples = total_tn + total_fp + total_fn + total_tp
                                if total_samples > 0:
                                    accuracy = (total_tp + total_tn) / total_samples
                                    st.write(f"**통합 정확도: {accuracy:.3f}**")
                                    st.write(f"총 샘플 수: {total_samples}")
            else:
                st.info("Confusion Matrix를 생성할 데이터가 없습니다.")
        else:
            st.info("Confusion Matrix를 생성할 데이터가 없습니다.")

    with tab3:
        st.write("##### 모델별 성능 지표 통합 비교")
        if not cm_data.empty:
            # 모델별로 성능 지표 통합 계산
            model_performance = {}
            
            for model in cm_data['model'].unique():
                model_data = cm_data[cm_data['model'] == model]
                
                # 모든 분석 타입의 결과를 통합
                total_tn = model_data['TN'].sum()
                total_fp = model_data['FP'].sum()
                total_fn = model_data['FN'].sum()
                total_tp = model_data['TP'].sum()
                
                # 성능 지표 계산
                total_samples = total_tn + total_fp + total_fn + total_tp
                if total_samples > 0:
                    accuracy = (total_tp + total_tn) / total_samples
                    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    model_performance[model] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1_score
                    }
            
            if model_performance:
                perf_df = pd.DataFrame(model_performance).T
                
                # 성능 지표 꺾은선 그래프
                fig = go.Figure()
                colors = ['blue', 'red', 'green', 'orange']
                
                for i, metric in enumerate(perf_df.columns):
                    fig.add_trace(go.Scatter(
                        x=perf_df.index, 
                        y=perf_df[metric],
                        mode='lines+markers',
                        name=metric.upper(),
                        line=dict(color=colors[i % len(colors)], width=3),
                        marker=dict(size=8)
                    ))
                
                fig.update_layout(
                    title='모델별 통합 성능 지표 비교',
                    xaxis_title='모델',
                    yaxis_title='성능 점수',
                    legend_title='성능 지표',
                    height=500,
                    yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig, use_container_width=True, key="integrated_performance")
                
                # 성능 지표 테이블
                st.write("**모델별 통합 성능 지표**")
                st.dataframe(perf_df.style.format("{:.3f}"))
                
                # 모델 순위
                st.write("**모델 순위 (F1 Score 기준)**")
                ranking = perf_df.sort_values('f1_score', ascending=False)
                ranking_display = ranking[['f1_score']].copy()
                ranking_display['순위'] = range(1, len(ranking_display) + 1)
                ranking_display = ranking_display[['순위', 'f1_score']]
                st.dataframe(ranking_display.style.format({'f1_score': '{:.3f}'}))
            else:
                st.info("성능 지표를 계산할 데이터가 없습니다.")
        else:
            st.info("성능 지표를 계산할 데이터가 없습니다.")

def display_results_visualizations(data: pd.DataFrame) -> None:
    """
    일반 결과 뷰를 위한 시각화

    Args:
        data (pd.DataFrame): 이메일 분석 결과 데이터
    """
    tab1, tab2, tab3 = st.tabs(["스팸 분류 비교", "분석 신뢰도/시간 분포", "발신자 분포"])

    with tab1:
        st.write("##### 스팸 분류 상호 비교")
        
        # 비교할 컬럼 목록
        spam_cols = [col for col in ["first_spam", "second_spam", "human_verified_spam"] if col in data.columns]

        if not spam_cols:
            st.info("비교할 스팸 분류 컬럼이 없습니다.")
        else:
            # 각 분류별 스팸, 정상, 미확인 개수 집계
            summary_data = {}
            col_rename_map = {
                "first_spam": "1차 분석",
                "second_spam": "2차 분석",
                "human_verified_spam": "사용자 확인"
            }

            for col in spam_cols:
                col_name = col_rename_map.get(col, col)
                # astype('boolean')으로 변환된 컬럼은 True/False/pd.NA 값을 가짐
                summary_data[col_name] = {
                    '스팸': data[col].eq(True).sum(),
                    '정상': data[col].eq(False).sum(),
                    '미확인': data[col].isnull().sum()
                }
            
            # 데이터프레임 생성
            comparison_df = pd.DataFrame(summary_data)
            
            # 카테고리 순서 정렬
            category_order = ['스팸', '정상', '미확인']
            # 현재 df에 있는 카테고리만 필터링
            ordered_categories = [cat for cat in category_order if cat in comparison_df.index]
            comparison_df = comparison_df.reindex(ordered_categories)

            st.bar_chart(comparison_df)

            st.write("---")
            st.write("##### Confusion Matrix (vs. 사용자 확인)")

            if 'human_verified_spam' not in data.columns or data['human_verified_spam'].isnull().all():
                st.info("사용자 확인 데이터가 없어 Confusion Matrix를 생성할 수 없습니다.")
            else:
                cm_data = data.dropna(subset=['human_verified_spam'])
                
                def create_cm_fig(pred_col, actual_col, title):
                    """Confusion Matrix 플롯 생성"""
                    if pred_col not in cm_data.columns:
                        return None
                    
                    tn = ((cm_data[pred_col] == False) & (cm_data[actual_col] == False)).sum()
                    fp = ((cm_data[pred_col] == True) & (cm_data[actual_col] == False)).sum()
                    fn = ((cm_data[pred_col] == False) & (cm_data[actual_col] == True)).sum()
                    tp = ((cm_data[pred_col] == True) & (cm_data[actual_col] == True)).sum()
                    
                    z = [[int(tn), int(fp)], [int(fn), int(tp)]]
                    x = ['예측: 정상', '예측: 스팸']
                    y = ['실제: 정상', '실제: 스팸']

                    fig = ff.create_annotated_heatmap(
                        z, x=x, y=y, colorscale='Blues', showscale=False
                    )
                    fig.update_layout(
                        title_text=f'<b>{title}</b>',
                        width=300, height=300,
                        margin=dict(t=50, l=20, r=20, b=20)
                    )
                    return fig

                col1, col2 = st.columns(2)
                with col1:
                    fig1 = create_cm_fig('first_spam', 'human_verified_spam', '1차 분석')
                    if fig1:
                        st.plotly_chart(fig1, use_container_width=True, key="first_spam_cm")
                    else:
                        st.info("1차 분석 데이터가 없습니다.")
                
                with col2:
                    fig2 = create_cm_fig('second_spam', 'human_verified_spam', '2차 분석')
                    if fig2:
                        st.plotly_chart(fig2, use_container_width=True, key="second_spam_cm")
                    else:
                        st.info("2차 분석 데이터가 없습니다.")

    with tab2:
        st.write("##### 분석 신뢰도 및 시간 분포")

        reliability_cols = [col for col in ['first_reliability', 'second_reliability'] if col in data.columns]
        duration_cols = [col for col in ['first_duration', 'second_duration'] if col in data.columns]

        if not reliability_cols and not duration_cols:
            st.info("분석 신뢰도 또는 시간 데이터가 없어 분포를 표시할 수 없습니다.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.write("###### 신뢰도 분포 (Histogram)")
                if reliability_cols:
                    fig = go.Figure()
                    for col in reliability_cols:
                        fig.add_trace(go.Histogram(x=data[col], name=col, nbinsx=20, opacity=0.75))
                    fig.update_layout(barmode='overlay', xaxis_title="신뢰도", yaxis_title="빈도")
                    st.plotly_chart(fig, use_container_width=True, key="reliability_hist")
                else:
                    st.info("신뢰도 데이터가 없습니다.")

            with col2:
                st.write("###### 분석 시간 분포 (Box Plot)")
                if duration_cols:
                    fig = go.Figure()
                    for col in duration_cols:
                        fig.add_trace(go.Box(y=data[col], name=col))
                    fig.update_layout(yaxis_title="분석 시간 (초)")
                    st.plotly_chart(fig, use_container_width=True, key="duration_box")
                else:
                    st.info("분석 시간 데이터가 없습니다.")
            
            st.write("###### 신뢰도 vs. 분석 시간 (Scatter Plot)")
            if reliability_cols and duration_cols:
                # 1차, 2차 분석 데이터가 모두 있는 경우에만 산점도 표시
                if 'first_reliability' in data.columns and 'first_duration' in data.columns:
                    st.write("1차 분석")
                    fig1 = go.Figure(data=go.Scatter(
                        x=data['first_reliability'],
                        y=data['first_duration'],
                        mode='markers',
                        marker=dict(opacity=0.6)
                    ))
                    fig1.update_layout(xaxis_title="신뢰도", yaxis_title="분석 시간 (초)")
                    st.plotly_chart(fig1, use_container_width=True, key="first_scatter")

                if 'second_reliability' in data.columns and 'second_duration' in data.columns:
                    st.write("2차 분석")
                    fig2 = go.Figure(data=go.Scatter(
                        x=data['second_reliability'],
                        y=data['second_duration'],
                        mode='markers',
                        marker=dict(opacity=0.6, color='red')
                    ))
                    fig2.update_layout(xaxis_title="신뢰도", yaxis_title="분석 시간 (초)")
                    st.plotly_chart(fig2, use_container_width=True, key="second_scatter")
            else:
                st.info("신뢰도와 분석 시간 데이터가 모두 있어야 산점도를 표시할 수 있습니다.")

    with tab3:
        st.write("##### 발신자 기반 분석")
        if 'from_email' in data.columns:
            # 상위 10개 발신자 도메인 추출
            top_senders = data['from_email'].value_counts().nlargest(10).index.tolist()
            filtered_data = data[data['from_email'].isin(top_senders)]

            # 발신자 도메인별 스팸 분류 결과 비교
            spam_comparison_cols = [col for col in ["first_spam", "second_spam", "human_verified_spam"] if col in filtered_data.columns]

            if not spam_comparison_cols:
                st.info("비교할 스팸 분류 컬럼이 없습니다.")
            else:
                sender_summary = {}
                for sender in top_senders:
                    sender_data = filtered_data[filtered_data['from_email'] == sender]
                    sender_summary[sender] = {
                        '스팸': sender_data['human_verified_spam'].eq(True).sum(),
                        '정상': sender_data['human_verified_spam'].eq(False).sum(),
                        '미확인': sender_data['human_verified_spam'].isnull().sum()
                    }
                
                sender_comparison_df = pd.DataFrame(sender_summary).T
                sender_comparison_df = sender_comparison_df.reindex(ordered_categories)

                st.write("###### 발신자 도메인별 스팸 분류 결과")
                st.bar_chart(sender_comparison_df)

                # 발신자 도메인별 분석 신뢰도 및 시간
                reliability_duration_cols = [col for col in ['first_reliability', 'second_reliability', 'first_duration', 'second_duration'] if col in filtered_data.columns]

                if not reliability_duration_cols:
                    st.info("신뢰도 또는 분석 시간 데이터가 없습니다.")
                else:
                    st.write("###### 발신자 도메인별 분석 신뢰도 및 시간")
                    reliability_duration_df = filtered_data.groupby('from_email')[reliability_duration_cols].mean().reset_index()

                    # 신뢰도 및 분석 시간 분포 시각화
                    fig = go.Figure()

                    for col in reliability_duration_cols:
                        if 'reliability' in col:
                            fig.add_trace(go.Box(
                                x=reliability_duration_df[col],
                                name=col,
                                marker_color='green',
                                boxmean='sd'
                            ))
                        else:
                            fig.add_trace(go.Box(
                                x=reliability_duration_df[col],
                                name=col,
                                marker_color='red',
                                boxmean='sd'
                            ))

                    fig.update_layout(
                        title="발신자 도메인별 분석 신뢰도 및 시간",
                        xaxis_title="신뢰도 / 분석 시간",
                        yaxis_title="값",
                        legend_title="컬럼",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True, key="sender_reliability_duration")
        else:
            st.info("'from_email' 컬럼이 없어 발신자 기반 분석을 수행할 수 없습니다.")

def main() -> None:
    """
    메인 애플리케이션 함수
    """
    logger.info("메인 애플리케이션 시작")
    setup_app()
    
    # 컨텍스트 관리자를 사용하여 DB 연결 관리
    with DatabaseManager("email_analysis.db") as db_manager:
        selected_view = setup_sidebar(db_manager)
        
        if selected_view:
            logger.info(f"선택된 뷰로 데이터 로드: {selected_view}")
            st.header(f"📊 {selected_view}")
            
            data = pd.DataFrame()
            is_editable = False
            show_visuals = False

            # 선택된 뷰에 따라 데이터 로드
            if selected_view == "모든 결과":
                logger.info("모든 결과 데이터 로드 중")
                data = db_manager.get_all_results()
                is_editable = True
                logger.info(f"모든 결과 데이터 로드 완료 - 행 수: {len(data)}")
            elif selected_view == "모델별 통계":
                logger.info("모델별 통계 데이터 로드 중")
                data = db_manager.get_model_stats()
                # 모델별 통계에서는 human_verified_spam 컬럼 제거
                if 'human_verified_spam' in data.columns:
                    data = data.drop(columns=['human_verified_spam'])
                show_visuals = True  # 통계 뷰에서만 시각화 표시
                logger.info(f"모델별 통계 데이터 로드 완료 - 행 수: {len(data)}")
            elif selected_view.endswith(" 결과"):
                model_name = selected_view.replace(" 결과", "")
                logger.info(f"특정 모델 결과 데이터 로드 중: {model_name}")
                data = db_manager.get_model_results(model_name)
                is_editable = True
                logger.info(f"모델 {model_name} 결과 데이터 로드 완료 - 행 수: {len(data)}")
            
            # first_spam, second_spam 컬럼을 1/0에서 True/False로 변환
            spam_map = {1: True, 0: False, 1.0: True, 0.0: False}
            if "first_spam" in data.columns:
                data["first_spam"] = data["first_spam"].map(spam_map).astype("boolean")
            if "second_spam" in data.columns:
                data["second_spam"] = data["second_spam"].map(spam_map).astype("boolean")
            
            logger.info("데이터 변환 완료, UI 렌더링 시작")
            # 데이터프레임 표시 및 편집 UI
            display_dataframe(db_manager, data, is_editable, selected_view)
            
            # 시각화 표시
            if show_visuals:
                logger.info("데이터 시각화 렌더링 시작")
                st.markdown("---")
                display_visualizations(db_manager, data, selected_view)
                logger.info("데이터 시각화 렌더링 완료")
        else:
            logger.warning("선택된 뷰가 없음")

if __name__ == "__main__":
    main()
