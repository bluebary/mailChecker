import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from dbmanager import DatabaseManager

def setup_app() -> None:
    """
    애플리케이션 초기 설정
    """
    st.set_page_config(
        page_title="이메일 분석 대시보드",
        page_icon="📧",
        layout="wide"
    )
    st.title("📧 이메일 분석 대시보드")

def setup_sidebar(db_manager: DatabaseManager) -> str:
    """
    사이드바 구성 및 분석 뷰 선택 기능

    Args:
        db_manager (DatabaseManager): 데이터베이스 관리자 인스턴스

    Returns:
        str: 선택된 뷰 이름
    """
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
    except Exception as e:
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
            
            changed_rows = original_series.compare(edited_series)

            if not changed_rows.empty:
                # compare 결과에서 첫 번째 변경된 행의 인덱스를 가져옴
                changed_idx = changed_rows.index[0]
                
                row_id = page_data.loc[changed_idx, 'id']
                new_spam_status = edited_df.loc[changed_idx, 'human_verified_spam']
                
                # dbmanager의 새 메서드 호출
                if db_manager.update_human_verification(row_id, new_spam_status):
                    st.success(f"ID {row_id}의 스팸 상태가 '{new_spam_status}' (으)로 업데이트되었습니다.")
                    st.rerun()
                else:
                    st.error(f"ID {row_id} 업데이트 실패")
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
            analysis_types = stats_data['analysis_type'].unique()
            
            for analysis_type in analysis_types:
                st.write(f"**{analysis_type.capitalize()} 분석**")
                type_data = stats_data[stats_data['analysis_type'] == analysis_type]
                
                if not type_data.empty:
                    chart_data = type_data.set_index('model')[['avg_reliability', 'avg_duration']]
                    st.bar_chart(chart_data)
                else:
                    st.info(f"{analysis_type} 분석에 대한 데이터가 없습니다.")
        else:
            st.info("모델 통계 데이터가 충분하지 않습니다.")

    with tab2:
        st.write("##### 모델별 Confusion Matrix")
        if not cm_data.empty:
            analysis_types = cm_data['analysis_type'].unique()
            for analysis_type in analysis_types:
                st.write(f"---")
                st.subheader(f"{analysis_type.capitalize()} 분석")
                type_cm_data = cm_data[cm_data['analysis_type'] == analysis_type]
                
                if not type_cm_data.empty:
                    models = sorted(type_cm_data['model'].unique())
                    
                    num_columns = 3
                    for i in range(0, len(models), num_columns):
                        cols = st.columns(num_columns)
                        for j in range(num_columns):
                            model_idx = i + j
                            if model_idx < len(models):
                                model = models[model_idx]
                                with cols[j]:
                                    row = type_cm_data[type_cm_data['model'] == model].iloc[0]
                                    cm = [[row['TN'], row['FP']], [row['FN'], row['TP']]]
                                    
                                    fig = ff.create_annotated_heatmap(
                                        z=cm,
                                        x=['Normal', 'Spam'],
                                        y=['Actual Normal', 'Actual Spam'],
                                        colorscale='Blues',
                                        showscale=False
                                    )
                                    fig.update_layout(
                                        width=250, height=250,
                                        title=f"<b>{model}</b>",
                                        margin=dict(t=40, l=10, r=10, b=10)
                                    )
                                    st.plotly_chart(fig)
                else:
                    st.info(f"{analysis_type} 분석에 대한 Confusion Matrix 데이터가 없습니다.")
        else:
            st.info("Confusion Matrix를 생성할 데이터가 없습니다. (사용자 확인 데이터 필요)")

    with tab3:
        st.write("##### 모델별 성능 지표 비교 (꺾은선 그래프)")
        if not cm_data.empty:
            # 성능 지표 계산
            cm_data['accuracy'] = (cm_data['TP'] + cm_data['TN']) / (cm_data['TP'] + cm_data['TN'] + cm_data['FP'] + cm_data['FN'])
            cm_data['precision'] = cm_data['TP'] / (cm_data['TP'] + cm_data['FP'])
            cm_data['recall'] = cm_data['TP'] / (cm_data['TP'] + cm_data['FN'])
            cm_data['f1_score'] = 2 * (cm_data['precision'] * cm_data['recall']) / (cm_data['precision'] + cm_data['recall'])
            cm_data.fillna(0, inplace=True)

            analysis_types = cm_data['analysis_type'].unique()
            for analysis_type in analysis_types:
                st.write(f"**{analysis_type.capitalize()} 분석 성능**")
                type_perf_data = cm_data[cm_data['analysis_type'] == analysis_type]
                
                if not type_perf_data.empty:
                    chart_data = type_perf_data.set_index('model')[['accuracy', 'precision', 'recall', 'f1_score']]
                    
                    fig = go.Figure()
                    for metric in chart_data.columns:
                        fig.add_trace(go.Scatter(
                            x=chart_data.index, 
                            y=chart_data[metric],
                            mode='lines+markers',
                            name=metric
                        ))
                    
                    fig.update_layout(
                        title=f'{analysis_type.capitalize()} 분석 성능 지표',
                        xaxis_title='모델',
                        yaxis_title='점수',
                        legend_title='성능 지표',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(chart_data.style.format("{:.3f}"))
                else:
                    st.info(f"{analysis_type} 분석에 대한 성능 데이터가 없습니다.")
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
            
    with tab2:
        st.write("##### 분석 신뢰도 및 시간 상호 비교")

        reliability_cols_map = {
            'first_reliability': '1차 신뢰도',
            'second_reliability': '2차 신뢰도'
        }
        duration_cols_map = {
            'first_duration': '1차 분석 시간',
            'second_duration': '2차 분석 시간'
        }

        available_reliability_cols = [col for col in reliability_cols_map if col in data.columns]
        available_duration_cols = [col for col in duration_cols_map if col in data.columns]

        if not available_reliability_cols and not available_duration_cols:
            st.info("분석 관련 컬럼이 없어 분포를 표시할 수 없습니다.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.write("신뢰도(Reliability) 비교")
                if available_reliability_cols:
                    reliability_df = data[available_reliability_cols].rename(columns=reliability_cols_map)
                    st.line_chart(reliability_df, height=250, use_container_width=True)
                else:
                    st.info("신뢰도 데이터가 없습니다.")
            
            with col2:
                st.write("분석 시간(Duration) 비교")
                if available_duration_cols:
                    duration_df = data[available_duration_cols].rename(columns=duration_cols_map)
                    st.line_chart(duration_df, height=250, use_container_width=True)
                else:
                    st.info("분석 시간 데이터가 없습니다.")

    with tab3:
        st.write("##### 발신 도메인별 이메일 수 (상위 10)")
        if 'sender_domain' in data.columns:
            sender_counts = data['sender_domain'].value_counts().nlargest(10)
            st.bar_chart(sender_counts)
        else:
            st.info("'sender_domain' 컬럼이 없어 발신자 분포를 표시할 수 없습니다.")

def main() -> None:
    """
    메인 애플리케이션 함수
    """
    setup_app()
    
    # 컨텍스트 관리자를 사용하여 DB 연결 관리
    with DatabaseManager("email_analysis.db") as db_manager:
        selected_view = setup_sidebar(db_manager)
        
        if selected_view:
            st.header(f"📊 {selected_view}")
            
            data = pd.DataFrame()
            is_editable = False
            show_visuals = False

            # 선택된 뷰에 따라 데이터 로드
            if selected_view == "모든 결과":
                data = db_manager.get_all_results()
                is_editable = True
            elif selected_view == "모델별 통계":
                data = db_manager.get_model_stats()
                show_visuals = True  # 통계 뷰에서만 시각화 표시
            elif selected_view.endswith(" 결과"):
                model_name = selected_view.replace(" 결과", "")
                data = db_manager.get_model_results(model_name)
                is_editable = True
            
            # first_spam, second_spam 컬럼을 1/0에서 True/False로 변환
            spam_map = {1: True, 0: False, 1.0: True, 0.0: False}
            if "first_spam" in data.columns:
                data["first_spam"] = data["first_spam"].map(spam_map).astype("boolean")
            if "second_spam" in data.columns:
                data["second_spam"] = data["second_spam"].map(spam_map).astype("boolean")

            # 데이터프레임 표시 및 편집 UI
            display_dataframe(db_manager, data, is_editable, selected_view)
            
            # 시각화 표시
            if show_visuals:
                st.markdown("---")
                display_visualizations(db_manager, data, selected_view)

if __name__ == "__main__":
    main()
