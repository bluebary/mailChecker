#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
이메일 데이터 처리 및 데이터베이스 변환 유틸리티

이 스크립트는 다음 두 가지 주요 기능을 수행합니다.
1. 이메일 원본 파일(.eml)을 파싱하여 JSON 형식으로 변환하고 저장합니다.
2. 파싱된 이메일 JSON과 분석 결과 JSON을 SQLite 데이터베이스로 가져옵니다.
"""

import os
import re
import json
import glob
import sqlite3
import logging
import argparse
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# --- 상수 정의 ---
# 경로 관련
DEFAULT_MAIL_DIR = "mailData"
DEFAULT_SAVE_DIR = "saveData"
DEFAULT_RESULT_DIR = "results"
DEFAULT_ERROR_DIR = "failList"
DEFAULT_DB_PATH = "email_analysis.db"

# 로깅 관련
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# --- 로깅 설정 ---
def setup_logging(log_file: Optional[str] = None, log_level: int = logging.INFO) -> None:
    """
    로깅 설정을 초기화합니다.

    Args:
        log_file (Optional[str]): 로그를 저장할 파일 경로.
        log_level (int): 로깅 레벨.
    """
    logging.basicConfig(level=log_level, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logging.getLogger('').addHandler(file_handler)
    logging.info(f"로깅 설정 완료. 레벨: {logging.getLevelName(log_level)}")


class EmailParser:
    """
    이메일 원본 파일을 파싱하고 JSON으로 저장하는 클래스
    """
    def __init__(self, mail_root: str, save_root: str, error_root: str):
        """
        EmailParser를 초기화합니다.

        Args:
            mail_root (str): 이메일 원본 파일이 있는 루트 디렉토리.
            save_root (str): 파싱된 JSON 파일을 저장할 루트 디렉토리.
            error_root (str): 오류 로그를 저장할 디렉토리.
        """
        self.mail_root = mail_root
        self.save_root = save_root
        self.error_root = error_root
        os.makedirs(self.save_root, exist_ok=True)
        os.makedirs(self.error_root, exist_ok=True)
        logging.info(f"EmailParser 초기화 완료. 메일 경로: {self.mail_root}, 저장 경로: {self.save_root}")

    def process_all_emails(self) -> Dict[str, Any]:
        """
        메일 루트 디렉토리의 모든 이메일 파일을 처리합니다.

        Returns:
            Dict[str, Any]: 처리 결과 통계.
        """
        stats = {"총_파일_수": 0, "성공": 0, "실패": 0}
        for root, _, files in os.walk(self.mail_root):
            for file in files:
                if file == '.DS_Store':
                    continue
                
                file_path = os.path.join(root, file)
                stats["총_파일_수"] += 1
                
                try:
                    json_data = self._convert_to_json(file_path)
                    if json_data:
                        self._save_json(json_data)
                        stats["성공"] += 1
                    else:
                        stats["실패"] += 1
                except Exception as e:
                    stats["실패"] += 1
                    logging.error(f"파일 처리 중 심각한 오류 발생: {file_path} - {e}")
                    self._save_error_log(file_path, str(e))

                if stats["총_파일_수"] % 100 == 0:
                    logging.info(f"진행 상황: {stats['총_파일_수']} 파일 처리됨 (성공: {stats['성공']}, 실패: {stats['실패']})")
        
        return stats

    def _convert_to_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        단일 이메일 텍스트 파일을 JSON 형식으로 변환합니다.

        Args:
            file_path (str): 이메일 텍스트 파일의 경로.

        Returns:
            Optional[Dict[str, Any]]: 변환된 이메일 데이터 또는 실패 시 None.
        """
        try:
            content = self._read_email_content(file_path)
            if content is None:
                raise ValueError("파일 내용을 읽을 수 없습니다.")

            headers_section, body = self._split_header_body(content)
            email_data = self._parse_headers(headers_section)
            email_data['body'] = body.strip()
            
            self._refine_email_data(email_data, file_path)
            
            return email_data
        except Exception as e:
            logging.error(f"파일 {file_path} 변환 중 오류 발생: {e}")
            self._save_error_log(file_path, str(e))
            return None

    def _read_email_content(self, file_path: str) -> Optional[str]:
        """다양한 인코딩으로 이메일 파일을 읽습니다."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        try:
            with open(file_path, 'rb') as f:
                return f.read().decode('utf-8', errors='replace')
        except Exception as e:
            logging.warning(f"바이너리 읽기 실패: {file_path}, 오류: {e}")
            return None

    def _split_header_body(self, content: str) -> Tuple[str, str]:
        """헤더와 본문을 분리합니다."""
        if '\n\n' in content:
            parts = content.split('\n\n', 1)
            return parts[0], parts[1] if len(parts) > 1 else ""
        else:
            lines = content.split('\n')
            return '\n'.join(lines[:10]), '\n'.join(lines[10:])

    def _parse_headers(self, headers_section: str) -> Dict[str, Any]:
        """헤더 섹션을 파싱하여 딕셔너리로 반환합니다."""
        email_data = {}
        current_header = None
        current_value = ""
        for line in headers_section.split('\n'):
            if line.startswith('//'): continue
            if ':' in line and not line.startswith((' ', '\t')):
                if current_header:
                    email_data[current_header.lower()] = current_value.strip()
                match = re.match(r'^([^:]+):\s*(.*)', line)
                if match:
                    current_header, current_value = match.groups()
            else:
                if current_header:
                    current_value += " " + line.strip()
        if current_header:
            email_data[current_header.lower()] = current_value.strip()
        return email_data

    def _refine_email_data(self, email_data: Dict[str, Any], file_path: str) -> None:
        """파싱된 이메일 데이터를 정제하고 추가 정보를 삽입합니다."""
        # 날짜 파싱
        if 'date' in email_data:
            try:
                date_str = email_data['date']
                date_clean = re.sub(r'\s*\(.+\)$', '', date_str)
                date_clean = re.sub(r'\s*[-+]\d{4}$', '', date_clean).strip()
                parsed_date = datetime.strptime(date_clean, '%a, %d %b %Y %H:%M:%S')
                email_data['parsed_date'] = parsed_date.isoformat()
            except (ValueError, TypeError):
                email_data['parsed_date'] = None

        # 주소 필드 리스트로 변환
        for field in ['to', 'cc', 'bcc']:
            if field in email_data and email_data[field]:
                email_data[field] = [addr.strip() for addr in email_data[field].split(',') if addr.strip()]
            else:
                email_data[field] = []

        # 고유 ID 및 메타데이터 추가
        email_data['id'] = str(uuid.uuid4())
        email_data['original_path'] = os.path.relpath(file_path, self.mail_root)
        email_data['mail_info'] = self._extract_mail_category_info(file_path)

    def _extract_mail_category_info(self, file_path: str) -> Dict[str, str]:
        """파일 경로에서 메일 카테고리 정보를 추출합니다."""
        info = {"user": "", "folder": "", "category": ""}
        if self.mail_root in file_path:
            rel_path = os.path.relpath(file_path, self.mail_root)
            parts = rel_path.split(os.sep)
            if len(parts) >= 1: info["user"] = parts[0]
            if len(parts) >= 2: info["folder"] = parts[1]
            
            folder_lower = info["folder"].lower()
            if 'sent' in folder_lower: info["category"] = "sent"
            elif 'inbox' in folder_lower: info["category"] = "inbox"
            elif 'deleted' in folder_lower: info["category"] = "deleted"
            else: info["category"] = "other"
        return info

    def _save_json(self, data: Dict[str, Any]) -> None:
        """데이터를 연/월 구조의 폴더에 JSON 파일로 저장합니다."""
        try:
            save_dir = self.save_root
            if 'parsed_date' in data and data['parsed_date']:
                dt = datetime.fromisoformat(data['parsed_date'])
                save_dir = os.path.join(self.save_root, dt.strftime('%Y'), dt.strftime('%m'))
            else:
                save_dir = os.path.join(self.save_root, "unknown_date")
            
            os.makedirs(save_dir, exist_ok=True)
            
            file_name = f"{data['id']}.json"
            output_path = os.path.join(save_dir, file_name)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logging.debug(f"JSON 저장 완료: {output_path}")
        except Exception as e:
            logging.error(f"JSON 저장 실패: {data.get('id', 'N/A')}, 오류: {e}")

    def _save_error_log(self, file_path: str, error_message: str) -> None:
        """실패한 파일의 오류 정보를 로그 파일에 기록합니다."""
        error_file = os.path.join(self.error_root, "parser_fail_list.txt")
        current_time = datetime.now().strftime(DATE_FORMAT)
        log_entry = f"[{current_time}] {file_path} - {error_message}\n"
        try:
            with open(error_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            logging.error(f"오류 로그 파일 작성 실패: {error_file}, 오류: {e}")

# --- DatabaseManager 클래스는 여기에 추가될 예정 ---

class DatabaseManager:
    """
    JSON 파일을 SQLite 데이터베이스로 가져오는 클래스
    """
    def __init__(self, db_path: str, save_dir: str, result_dir: str, year: Optional[int] = None):
        """
        DatabaseManager를 초기화합니다.

        Args:
            db_path (str): 데이터베이스 파일 경로.
            save_dir (str): 파싱된 이메일 JSON 파일이 있는 디렉토리.
            result_dir (str): 분석 결과 JSON 파일이 있는 디렉토리.
            year (Optional[int]): 처리할 특정 연도. None이면 모든 연도를 처리.
        """
        self.db_path = db_path
        self.save_dir = save_dir
        self.result_dir = result_dir
        self.year = year
        self.conn: Optional[sqlite3.Connection] = None
        logging.info(f"DatabaseManager 초기화 완료. DB 경로: {self.db_path}, 연도 필터: {self.year or '없음'}")

    def __enter__(self):
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def import_all_data(self, overwrite: bool = False) -> None:
        """
        모든 JSON 데이터를 데이터베이스로 가져오는 전체 프로세스를 실행합니다.

        Args:
            overwrite (bool): 기존 데이터베이스 파일을 덮어쓸지 여부.
        """
        if os.path.exists(self.db_path) and overwrite:
            logging.warning(f"기존 데이터베이스 파일을 삭제합니다: {self.db_path}")
            os.remove(self.db_path)
        elif os.path.exists(self.db_path) and not overwrite:
            logging.warning(f"데이터베이스 파일이 이미 존재합니다. 덮어쓰려면 --overwrite 옵션을 사용하세요.")
            return

        email_files = self._scan_json_files(self.save_dir)
        result_files = self._scan_json_files(self.result_dir, is_result_file=True)
        
        if not email_files and not result_files:
            logging.warning("데이터베이스로 가져올 파일이 없습니다.")
            return

        self._create_database_schema(result_files)
        
        self._insert_email_data(email_files)
        self._insert_result_data(result_files)
        
        self._create_views(result_files)
        logging.info("모든 데이터 가져오기 작업이 완료되었습니다.")

    def _scan_json_files(self, directory: str, is_result_file: bool = False) -> List[Dict[str, Any]]:
        """
        지정된 디렉토리에서 JSON 파일을 스캔하고 메타데이터를 추출합니다.

        Args:
            directory (str): 스캔할 디렉토리 경로.
            is_result_file (bool): 분석 결과 파일인지 여부.

        Returns:
            List[Dict[str, Any]]: 파일 메타데이터 목록.
        """
        file_infos = []
        search_path = os.path.join(directory, str(self.year), "**", "*.json") if self.year else os.path.join(directory, "**", "*.json")
        
        for file_path in glob.glob(search_path, recursive=True):
            if is_result_file:
                file_name = os.path.basename(file_path)
                model_info = self._extract_model_info(file_name)
                if model_info:
                    file_infos.append({
                        "path": file_path,
                        "model": model_info["model"],
                        "analysis_type": model_info["type"],
                    })
            else:
                file_infos.append({"path": file_path})
        
        logging.info(f"{directory}에서 {len(file_infos)}개의 유효한 JSON 파일을 찾았습니다.")
        return file_infos

    def _extract_model_info(self, file_name: str) -> Optional[Dict[str, str]]:
        """파일명에서 모델명과 분석 유형을 추출합니다."""
        patterns = [
            r"(.+?)_(first|second)_(\d{4}-\d{2}-\d{2})\.json$",
            r"spam_analysis_(.+?)_(first|second)_results_(\d{8})_\d{6}\.json$"
        ]
        for pattern in patterns:
            match = re.match(pattern, file_name)
            if match:
                return {"model": match.group(1), "type": match.group(2)}
        logging.warning(f"파일명 형식이 일치하지 않습니다: {file_name}")
        return None

    def _create_database_schema(self, model_infos: List[Dict[str, Any]]) -> None:
        """데이터베이스 스키마를 생성합니다."""
        if not self.conn: return
        cursor = self.conn.cursor()
        logging.info("데이터베이스 스키마 생성 시작")
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            id TEXT PRIMARY KEY,
            original_path TEXT,
            parsed_date TEXT,
            sender TEXT,
            receiver TEXT,
            subject TEXT,
            body TEXT
        )""")

        created_tables = set()
        for info in model_infos:
            model_name = info['model'].replace('-', '_')
            if model_name in created_tables: continue
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {model_name} (
                id TEXT PRIMARY KEY,
                first_spam BOOLEAN, first_duration REAL, first_reliability REAL,
                second_spam BOOLEAN, second_duration REAL, second_reliability REAL,
                human_verified_spam BOOLEAN,
                FOREIGN KEY (id) REFERENCES emails(id)
            )""")
            created_tables.add(model_name)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS all_results (
            id TEXT, model TEXT,
            first_spam BOOLEAN, first_duration REAL, first_reliability REAL,
            second_spam BOOLEAN, second_duration REAL, second_reliability REAL,
            human_verified_spam BOOLEAN,
            PRIMARY KEY (id, model),
            FOREIGN KEY (id) REFERENCES emails(id)
        )""")
        self.conn.commit()
        logging.info("데이터베이스 스키마 생성 완료")

    def _insert_email_data(self, email_files: List[Dict[str, Any]]) -> None:
        """파싱된 이메일 JSON 데이터를 emails 테이블에 삽입합니다."""
        if not self.conn: return
        logging.info(f"{len(email_files)}개의 이메일 JSON 파일 삽입 시작.")
        cursor = self.conn.cursor()
        
        for file_info in email_files:
            try:
                with open(file_info['path'], 'r', encoding='utf-8') as f:
                    email = json.load(f)
                
                cursor.execute("""
                INSERT OR REPLACE INTO emails (id, original_path, parsed_date, sender, receiver, subject, body)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    email.get('id'), email.get('original_path'), email.get('parsed_date'),
                    email.get('from'), ", ".join(email.get('to', [])),
                    email.get('subject'), email.get('body')
                ))
            except (json.JSONDecodeError, KeyError) as e:
                logging.error(f"이메일 JSON 파일 처리 실패: {file_info['path']}, 오류: {e}")
        
        self.conn.commit()
        logging.info("이메일 데이터 삽입 완료.")

    def _insert_result_data(self, result_files: List[Dict[str, Any]]) -> None:
        """분석 결과 JSON 데이터를 해당 모델 테이블과 all_results 테이블에 삽입합니다."""
        if not self.conn: return
        logging.info(f"{len(result_files)}개의 분석 결과 JSON 파일 삽입 시작.")
        cursor = self.conn.cursor()

        for file_info in result_files:
            try:
                with open(file_info["path"], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                model_name_sql = file_info['model'].replace('-', '_')
                model_name = file_info['model']
                analysis_type = file_info['analysis_type']

                for _, emails in data.items():
                    for email in emails:
                        email_id = email.get('id')
                        if not email_id: continue

                        spam = email.get('spam')
                        duration = email.get('duration')
                        reliability = email.get('reliability')
                        human_verified = email.get('human_verified_spam')

                        # 모델별 테이블 및 all_results 테이블 업데이트
                        col_prefix = f"{analysis_type}"
                        cursor.execute(f"""
                        INSERT INTO {model_name_sql} (id, {col_prefix}_spam, {col_prefix}_duration, {col_prefix}_reliability, human_verified_spam)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            {col_prefix}_spam=excluded.{col_prefix}_spam,
                            {col_prefix}_duration=excluded.{col_prefix}_duration,
                            {col_prefix}_reliability=excluded.{col_prefix}_reliability,
                            human_verified_spam=COALESCE(excluded.human_verified_spam, {model_name_sql}.human_verified_spam)
                        """, (email_id, spam, duration, reliability, human_verified))

                        cursor.execute(f"""
                        INSERT INTO all_results (id, model, {col_prefix}_spam, {col_prefix}_duration, {col_prefix}_reliability, human_verified_spam)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id, model) DO UPDATE SET
                            {col_prefix}_spam=excluded.{col_prefix}_spam,
                            {col_prefix}_duration=excluded.{col_prefix}_duration,
                            {col_prefix}_reliability=excluded.{col_prefix}_reliability,
                            human_verified_spam=COALESCE(excluded.human_verified_spam, all_results.human_verified_spam)
                        """, (email_id, model_name, spam, duration, reliability, human_verified))

            except Exception as e:
                logging.error(f"분석 결과 파일 처리 실패: {file_info['path']}, 오류: {e}")
        
        self.conn.commit()
        logging.info("분석 결과 데이터 삽입 완료.")

    def _create_views(self, model_infos: List[Dict[str, Any]]) -> None:
        """데이터베이스 뷰를 생성합니다."""
        if not self.conn: return
        cursor = self.conn.cursor()
        logging.info("데이터베이스 뷰 생성 시작")

        # 통합 결과 뷰
        cursor.execute("DROP VIEW IF EXISTS all_results_view")
        cursor.execute("""
        CREATE VIEW all_results_view AS
        SELECT 
            e.id, e.sender, e.subject, r.model, 
            r.first_spam, r.first_duration, r.first_reliability, 
            r.second_spam, r.second_duration, r.second_reliability, r.human_verified_spam
        FROM emails e JOIN all_results r ON e.id = r.id
        """)

        # 모델별 통계 뷰
        cursor.execute("DROP VIEW IF EXISTS model_stats_view")
        cursor.execute("""
        CREATE VIEW model_stats_view AS
        SELECT model, 'first' as analysis_type, COUNT(id) as total, AVG(first_reliability) as avg_reliability, AVG(first_duration) as avg_duration
        FROM all_results WHERE first_reliability IS NOT NULL GROUP BY model
        UNION ALL
        SELECT model, 'second' as analysis_type, COUNT(id) as total, AVG(second_reliability) as avg_reliability, AVG(second_duration) as avg_duration
        FROM all_results WHERE second_reliability IS NOT NULL GROUP BY model
        """)
        
        self.conn.commit()
        logging.info("데이터베이스 뷰 생성 완료.")

# --- 메인 실행 로직 ---
def main():
    """메인 프로그램 함수"""
    parser = argparse.ArgumentParser(description="이메일 데이터 처리 및 DB 변환 유틸리티")
    parser.add_argument("command", choices=["parse-emails", "import-to-db", "all"], help="실행할 작업")
    
    # 경로 관련 인자
    parser.add_argument("--mail-dir", default=DEFAULT_MAIL_DIR, help=f"이메일 원본 디렉토리 (기본값: {DEFAULT_MAIL_DIR})")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help=f"파싱된 JSON 저장 디렉토리 (기본값: {DEFAULT_SAVE_DIR})")
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR, help=f"분석 결과 JSON 디렉토리 (기본값: {DEFAULT_RESULT_DIR})")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help=f"SQLite DB 파일 경로 (기본값: {DEFAULT_DB_PATH})")
    parser.add_argument("--error-dir", default=DEFAULT_ERROR_DIR, help=f"오류 로그 디렉토리 (기본값: {DEFAULT_ERROR_DIR})")

    # 기능 관련 인자
    parser.add_argument("--year", type=int, help="처리할 특정 연도 (YYYY 형식)")
    parser.add_argument("--overwrite", action="store_true", help="기존 데이터베이스가 있으면 덮어씁니다.")
    parser.add_argument("--log-file", help="로그를 저장할 파일 경로")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="로깅 레벨")

    args = parser.parse_args()

    # 로깅 설정
    setup_logging(args.log_file, getattr(logging, args.log_level.upper()))

    try:
        if args.command in ["parse-emails", "all"]:
            logging.info("--- 이메일 파싱 작업 시작 ---")
            parser_inst = EmailParser(args.mail_dir, args.save_dir, args.error_dir)
            stats = parser_inst.process_all_emails()
            logging.info(f"이메일 파싱 완료. 총 {stats['총_파일_수']}개 파일 중 성공 {stats['성공']}개, 실패 {stats['실패']}개")

        if args.command in ["import-to-db", "all"]:
            logging.info("--- 데이터베이스 가져오기 작업 시작 ---")
            with DatabaseManager(args.db_path, args.save_dir, args.result_dir, args.year) as db_manager:
                db_manager.import_all_data(args.overwrite)
            logging.info("데이터베이스 가져오기 완료.")
        
        logging.info("모든 작업이 성공적으로 완료되었습니다.")

    except Exception as e:
        logging.error(f"작업 중 심각한 오류 발생: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()
