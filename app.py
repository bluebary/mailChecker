import streamlit as st
import pandas as pd
from dbmanager import DatabaseManager

def setup_app() -> None:
    """
    애플리케이션 초기 설정
    """
    st.set_page_config(
        page_title="이메일 분석 데이터베이스 뷰어",
        page_icon="📧",
        layout="wide"
    )
    st.title("이메일 분석 데이터베이스 뷰어")

def setup_sidebar(db_manager: DatabaseManager) -> str:
    """
    사이드바 구성 및 테이블 선택 기능

    Args:
        db_manager (DatabaseManager): 데이터베이스 관리자 인스턴스

    Returns:
        str: 선택된 테이블 이름
    """
    st.sidebar.title("테이블 선택")
    try:
        tables = db_manager.get_tables()
        selected_table = st.sidebar.selectbox("테이블 선택", tables)
    except Exception as e:
        st.sidebar.error(f"테이블 목록을 불러오는 중 오류가 발생했습니다: {e}")
        selected_table = None

    st.sidebar.markdown("---")
    st.sidebar.info("이 뷰어는 이메일 분석 데이터베이스를 조회하고 human_verified_spam 컬럼을 편집할 수 있습니다.")

    return selected_table

def display_table_data(db_manager: DatabaseManager, table_name: str) -> None:
    """
    테이블 데이터 표시 및 편집 기능

    Args:
        db_manager (DatabaseManager): 데이터베이스 관리자 인스턴스
        table_name (str): 표시할 테이블 이름
    """
    schema = db_manager.get_table_schema(table_name)
    has_spam_column = any(col[0] == "human_verified_spam" for col in schema)

    st.subheader(f"테이블: {table_name}")

    # 데이터 조회
    data = db_manager.get_table_data(table_name, limit=10000)

    if data.empty:
        st.write("테이블에 데이터가 없습니다.")
        return

    # 필터링 및 정렬 옵션
    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("검색어")
    with col2:
        if has_spam_column:
            spam_filter = st.selectbox("스팸 필터", ["모두", "스팸만", "정상만"])
        else:
            spam_filter = "모두"

    # 데이터 필터링
    filtered_data = data
    if search_term:
        filtered_data = filtered_data[filtered_data.astype(str).apply(
            lambda row: row.str.contains(search_term, case=False, na=False).any(), axis=1)]

    if has_spam_column and spam_filter != "모두":
        spam_value = spam_filter == "스팸만"
        filtered_data = filtered_data[filtered_data["human_verified_spam"] == spam_value]

    # 페이지네이션
    rows_per_page = st.slider("페이지당 행 수", 10, 100, 20)
    total_rows = len(filtered_data)
    total_pages = max(1, (total_rows + rows_per_page - 1) // rows_per_page)
    page = st.number_input("페이지", 1, total_pages, 1)

    start_idx = (page - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, total_rows)
    page_data = filtered_data.iloc[start_idx:end_idx].copy()

    st.write(f"전체 {total_rows}개 중 {start_idx + 1}~{end_idx}개 표시 (페이지 {page}/{total_pages})")

    # human_verified_spam 컬럼이 있는 경우 편집 기능 추가
    if has_spam_column:
        if 'id' not in page_data.columns:
            st.error("테이블에 'id' 컬럼이 없어 편집할 수 없습니다.")
            st.dataframe(page_data)
            return

        editor_data = page_data.copy()
        editor_data['human_verified_spam'] = editor_data['human_verified_spam'].astype(bool)
        disabled_columns = [col for col in editor_data.columns if col != 'human_verified_spam']
        edited_data = st.data_editor(
            editor_data,
            disabled=disabled_columns,
            hide_index=True,
            key=f"data_editor_{table_name}_{page}"
        )
        if edited_data is not None:
            edited_df = pd.DataFrame(edited_data, index=page_data.index)
            changed_mask = (page_data['human_verified_spam'].astype(bool) != edited_df['human_verified_spam'])
            if changed_mask.any():
                changed_idx = changed_mask.idxmax()
                row_id = page_data.loc[changed_idx, 'id']
                new_spam_status = edited_df.loc[changed_idx, 'human_verified_spam']
                if db_manager.update_spam_flag(table_name, row_id, new_spam_status):
                    st.success(f"ID {row_id}의 스팸 상태가 업데이트되었습니다.")
                    st.rerun()
                else:
                    st.error(f"ID {row_id} 업데이트 실패")
    else:
        st.dataframe(page_data, hide_index=True)

def display_visualizations(db_manager: DatabaseManager, table_name: str) -> None:
    """
    테이블 데이터 시각화

    Args:
        db_manager (DatabaseManager): 데이터베이스 관리자 인스턴스
        table_name (str): 표시할 테이블 이름
    """
    st.subheader("데이터 시각화")
    schema = db_manager.get_table_schema(table_name)
    data = db_manager.get_table_data(table_name, limit=50000)
    if data.empty:
        st.write("시각화할 데이터가 없습니다.")
        return
    tab1, tab2, tab3 = st.tabs(["스팸/정상 비율", "분석 신뢰도/시간 분포", "발신자 분포"])
    with tab1:
        if "human_verified_spam" in data.columns:
            st.write("##### 스팸/정상 이메일 비율")
            spam_count = data["human_verified_spam"].sum()
            normal_count = len(data) - spam_count
            chart_data = pd.DataFrame({
                "분류": ["스팸 (True)", "정상 (False)"],
                "개수": [spam_count, normal_count]
            })
            st.bar_chart(chart_data.set_index("분류"))
        else:
            st.write("'human_verified_spam' 컬럼이 없어 비율을 표시할 수 없습니다.")
    with tab2:
        # 분석 신뢰도/시간 분포 (first/second)
        duration_cols = [col for col in data.columns if 'duration' in col]
        reliability_cols = [col for col in data.columns if 'reliability' in col]
        if duration_cols or reliability_cols:
            st.write("##### 분석 신뢰도/시간 분포")
            for col in duration_cols:
                st.line_chart(data[col], height=200)
            for col in reliability_cols:
                st.line_chart(data[col], height=200)
        else:
            st.write("분석 관련 컬럼이 없어 분포를 표시할 수 없습니다.")
    with tab3:
        sender_columns = [col[0] for col in schema if 'sender' in col[0].lower()]
        if sender_columns:
            sender_col = sender_columns[0]
            st.write(f"##### 발신자별 이메일 수 (상위 10)")
            if sender_col in data.columns:
                sender_counts = data[sender_col].value_counts().nlargest(10)
                st.bar_chart(sender_counts)
            else:
                st.write(f"'{sender_col}' 컬럼을 찾을 수 없습니다.")
        else:
            st.write("'sender' 관련 컬럼이 없어 발신자 분포를 표시할 수 없습니다.")

def main() -> None:
    """
    메인 애플리케이션 함수
    """
    setup_app()
    db_manager = DatabaseManager("email_analysis.db")
    db_manager.connect()
    selected_table = setup_sidebar(db_manager)
    if selected_table:
        display_table_data(db_manager, selected_table)
        st.markdown("---")
        display_visualizations(db_manager, selected_table)
    db_manager.close()

if __name__ == "__main__":
    main()
