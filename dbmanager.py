import sqlite3
import pandas as pd
import logging
from typing import List, Tuple, Any, Optional

# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 파일 핸들러 설정
file_handler = logging.FileHandler('dbmanager.log', encoding='utf-8')
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

class DatabaseManager:
    """SQLite 데이터베이스 연결 및 쿼리 관리 클래스"""

    def __init__(self, db_path: str) -> None:
        """
        데이터베이스 관리자 초기화

        Args:
            db_path (str): 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None
        logger.info(f"DatabaseManager 초기화됨 - DB 경로: {db_path}")

    def connect(self) -> None:
        """
        데이터베이스 연결 설정
        """
        if self.connection:
            logger.debug("데이터베이스 연결이 이미 존재함")
            return
        try:
            self.connection = sqlite3.connect(self.db_path)
            logger.info(f"데이터베이스 연결 성공: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"데이터베이스 연결 오류: {e}")
            print(f"데이터베이스 연결 오류: {e}")
            self.connection = None

    def close(self) -> None:
        """
        데이터베이스 연결 해제
        """
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("데이터베이스 연결 해제됨")

    def __enter__(self) -> 'DatabaseManager':
        """컨텍스트 관리자 진입"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """컨텍스트 관리자 종료"""
        self.close()

    def get_tables(self, include_views: bool = False) -> List[str]:
        """
        데이터베이스의 모든 테이블 또는 뷰 목록 반환

        Args:
            include_views (bool): True이면 뷰를 포함합니다.

        Returns:
            List[str]: 테이블/뷰 이름 목록
        """
        if not self.connection:
            self.connect()
        
        if include_views:
            query = "SELECT name FROM sqlite_master WHERE type IN ('table', 'view');"
        else:
            query = "SELECT name FROM sqlite_master WHERE type='table';"
            
        cursor = self.connection.cursor()
        cursor.execute(query)
        tables = [table[0] for table in cursor.fetchall()]
        return tables

    def get_table_data(self, table_name: str, limit: int = 1000, offset: int = 0) -> pd.DataFrame:
        """
        지정된 테이블의 데이터 조회

        Args:
            table_name (str): 테이블 이름
            limit (int): 조회할 최대 행 수
            offset (int): 시작 오프셋

        Returns:
            pd.DataFrame: 테이블 데이터를 담은 DataFrame
        """
        if not self.connection:
            self.connect()
        query = f"SELECT * FROM '{table_name}' LIMIT {limit} OFFSET {offset}"
        try:
            df = pd.read_sql_query(query, self.connection)
            return df
        except pd.io.sql.DatabaseError as e:
            print(f"테이블 데이터 읽기 오류: {e}")
            return pd.DataFrame()

    def get_table_schema(self, table_name: str) -> List[Tuple[str, str]]:
        """
        테이블 스키마 정보 조회

        Args:
            table_name (str): 테이블 이름

        Returns:
            List[Tuple[str, str]]: 컬럼 이름과 데이터 타입 정보 리스트
        """
        if not self.connection:
            self.connect()
        query = f"PRAGMA table_info('{table_name}');"
        cursor = self.connection.cursor()
        cursor.execute(query)
        schema = [(row[1], row[2]) for row in cursor.fetchall()]
        return schema

    def update_human_verification(self, email_id: str, is_spam: bool) -> bool:
        """
        특정 이메일에 대한 사용자의 스팸 여부 판단을 모든 관련 테이블에 업데이트합니다.
        
        이 메서드는 트랜잭션을 사용하여 모델별 테이블과 all_results 테이블의
        'human_verified_spam' 컬럼을 원자적으로 업데이트합니다.

        Args:
            email_id (str): 업데이트할 이메일의 ID
            is_spam (bool): 새로운 스팸 플래그 값 (True 또는 False)

        Returns:
            bool: 업데이트 성공 여부
        """
        logger.info(f"사용자 스팸 검증 업데이트 시작 - ID: {email_id}, 스팸 여부: {is_spam}")
        
        if not self.connection:
            self.connect()
        
        try:
            model_names = self.get_model_names()
        except Exception as e:
            logger.error(f"모델 이름 조회 중 오류: {e}")
            print(f"모델 이름 조회 중 오류: {e}")
            return False

        try:
            cursor = self.connection.cursor()
            cursor.execute("BEGIN TRANSACTION")

            # 1. all_results 테이블 업데이트
            query_all = "UPDATE all_results SET human_verified_spam = ? WHERE id = ?"
            cursor.execute(query_all, (is_spam, email_id))

            # 2. 각 모델별 테이블 업데이트
            for model_name in model_names:
                table_name = model_name.replace('-', '_')
                query_model = f'UPDATE "{table_name}" SET human_verified_spam = ? WHERE id = ?'
                cursor.execute(query_model, (is_spam, email_id))
            
            self.connection.commit()
            logger.info(f"사용자 스팸 검증 업데이트 성공 - ID: {email_id}")
            return True
        except sqlite3.Error as e:
            logger.error(f"스팸 플래그 업데이트 오류: {e}")
            print(f"스팸 플래그 업데이트 오류: {e}")
            if self.connection:
                self.connection.rollback()
            return False

    def get_model_names(self) -> List[str]:
        """
        데이터베이스에 저장된 모든 모델의 이름 목록을 반환합니다.
        'all_results' 테이블에서 고유한 모델 이름을 조회합니다.
        
        Returns:
            List[str]: 모델 이름 목록
        """
        query = "SELECT DISTINCT model FROM all_results;"
        df = self.execute_query(query)
        if not df.empty:
            return df['model'].tolist()
        return []

    def get_all_results(self, limit: int = 1000, offset: int = 0) -> pd.DataFrame:
        """
        모든 모델의 통합 분석 결과를 'all_results_view' 뷰에서 조회합니다.
        
        Args:
            limit (int): 조회할 최대 행 수
            offset (int): 시작 오프셋
            
        Returns:
            pd.DataFrame: 통합 분석 결과
        """
        query = f"SELECT * FROM all_results_view LIMIT {limit} OFFSET {offset}"
        return self.execute_query(query)

    def get_model_results(self, model_name: str, limit: int = 1000, offset: int = 0) -> pd.DataFrame:
        """
        특정 모델의 통합 분석 결과를 해당 모델의 결합 뷰에서 조회합니다.
        
        Args:
            model_name (str): 조회할 모델의 이름
            limit (int): 조회할 최대 행 수
            offset (int): 시작 오프셋
            
        Returns:
            pd.DataFrame: 특정 모델의 분석 결과
        """
        view_name = f"view_{model_name.replace('-', '_')}_combined"
        query = f"SELECT * FROM {view_name} LIMIT {limit} OFFSET {offset}"
        return self.execute_query(query)

    def get_model_stats(self) -> pd.DataFrame:
        """
        모델별 통계 정보를 조회합니다. (human_verified_spam 컬럼 제외)
        
        Returns:
            pd.DataFrame: 모델별 통계 정보
        """
        query = """
        WITH stats_data AS (
            SELECT 
                model,
                'first_spam' as analysis_type,
                first_reliability as reliability,
                first_duration as duration
            FROM all_results
            WHERE first_reliability IS NOT NULL AND first_duration IS NOT NULL
            UNION ALL
            SELECT 
                model,
                'second_spam' as analysis_type,
                second_reliability as reliability,
                second_duration as duration
            FROM all_results
            WHERE second_reliability IS NOT NULL AND second_duration IS NOT NULL
        )
        SELECT 
            model, 
            analysis_type,
            COUNT(*) as total_emails,
            AVG(reliability) as avg_reliability,
            AVG(duration) as avg_duration,
            MIN(reliability) as min_reliability,
            MAX(reliability) as max_reliability
        FROM stats_data
        GROUP BY model, analysis_type
        """
        return self.execute_query(query)

    def get_confusion_matrix_data(self) -> pd.DataFrame:
        """
        모델 및 분석 유형별 Confusion Matrix 계산에 필요한 데이터를 조회합니다.
        각 모델의 first_spam과 second_spam 결과 분포를 계산합니다.
        (TP, TN, FP, FN은 간단한 분포 통계로 대체)

        Returns:
            pd.DataFrame: 모델, 분석 유형, TP, TN, FP, FN을 포함하는 데이터프레임
        """
        query = """
        WITH combined_predictions AS (
            SELECT
                model,
                'first_spam' as analysis_type,
                first_spam AS spam_prediction
            FROM all_results
            WHERE first_spam IS NOT NULL
            UNION ALL
            SELECT
                model,
                'second_spam' as analysis_type,
                second_spam AS spam_prediction
            FROM all_results
            WHERE second_spam IS NOT NULL
        )
        SELECT
            model,
            analysis_type,
            SUM(CASE WHEN spam_prediction = 1 THEN 1 ELSE 0 END) as TP,
            SUM(CASE WHEN spam_prediction = 0 THEN 1 ELSE 0 END) as TN,
            0 as FP,
            0 as FN
        FROM combined_predictions
        GROUP BY model, analysis_type
        HAVING COUNT(*) > 0
        """
        return self.execute_query(query)

    def execute_query(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """
        사용자 정의 쿼리 실행
        
        Args:
            query (str): SQL 쿼리 문자열
            params (tuple): 쿼리 매개변수
        
        Returns:
            pd.DataFrame: 쿼리 결과를 담은 DataFrame
        """
        logger.debug(f"쿼리 실행 시작: {query[:100]}..." if len(query) > 100 else f"쿼리 실행 시작: {query}")
        
        if not self.connection:
            self.connect()
        try:
            df = pd.read_sql_query(query, self.connection, params=params)
            logger.info(f"쿼리 실행 성공 - 결과 행 수: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"쿼리 실행 오류: {e}")
            print(f"쿼리 실행 오류: {e}")
            return pd.DataFrame()
