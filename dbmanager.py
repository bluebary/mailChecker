import sqlite3
import pandas as pd
from typing import List, Tuple, Any, Optional

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

    def connect(self) -> None:
        """
        데이터베이스 연결 설정
        """
        try:
            self.connection = sqlite3.connect(self.db_path)
        except sqlite3.Error as e:
            print(f"데이터베이스 연결 오류: {e}")
            self.connection = None

    def close(self) -> None:
        """
        데이터베이스 연결 해제
        """
        if self.connection:
            self.connection.close()
            self.connection = None

    def get_tables(self) -> List[str]:
        """
        데이터베이스의 모든 테이블 목록 반환

        Returns:
            List[str]: 테이블 이름 목록
        """
        if not self.connection:
            self.connect()
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

    def update_spam_flag(self, table_name: str, row_id: Any, is_spam: bool) -> bool:
        """
        human_verified_spam 플래그 업데이트

        Args:
            table_name (str): 테이블 이름
            row_id (Any): 업데이트할 행의 ID
            is_spam (bool): 새로운 스팸 플래그 값

        Returns:
            bool: 업데이트 성공 여부
        """
        if not self.connection:
            self.connect()
        pk_column = 'id'
        try:
            cursor = self.connection.cursor()
            query = f'UPDATE "{table_name}" SET human_verified_spam = ? WHERE "{pk_column}" = ?'
            cursor.execute(query, (is_spam, row_id))
            self.connection.commit()
            return True
        except sqlite3.Error as e:
            print(f"스팸 플래그 업데이트 오류: {e}")
            self.connection.rollback()
            return False

    def execute_query(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """
        사용자 정의 쿼리 실행
        
        Args:
            query (str): SQL 쿼리 문자열
            params (tuple): 쿼리 매개변수
        
        Returns:
            pd.DataFrame: 쿼리 결과를 담은 DataFrame
        """
        if not self.connection:
            self.connect()
        try:
            df = pd.read_sql_query(query, self.connection, params=params)
            return df
        except Exception as e:
            print(f"쿼리 실행 오류: {e}")
            return pd.DataFrame()
