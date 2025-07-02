#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
이메일 분석 결과 JSON 파일을 SQLite 데이터베이스로 변환하는 유틸리티

이 모듈은 result 폴더의 JSON 파일을 파싱하여 SQLite 데이터베이스로 변환합니다.
각 모델별, 분석 유형별로 테이블을 생성하고 데이터를 삽입합니다.
"""

import os
import re
import json
import glob
import sqlite3
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple

# 상수 정의
DEFAULT_RESULT_DIR = "results"
DEFAULT_DB_PATH = "email_analysis.db"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_file: Optional[str] = None, log_level: int = logging.INFO) -> None:
    """로깅 설정을 초기화합니다.
    
    콘솔과 선택적으로 파일에 로그를 출력하도록 로깅 설정을 구성합니다.
    
    Args:
        log_file: 로그를 저장할 파일 경로. None이면 파일 로깅을 하지 않습니다.
        log_level: 로깅 레벨 (예: logging.INFO, logging.DEBUG)
    
    Returns:
        None
    """
    logging.basicConfig(level=log_level, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    
    # 파일 로깅 추가 설정
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logging.getLogger('').addHandler(file_handler)
    
    logging.info(f"로깅 설정 완료. 레벨: {logging.getLevelName(log_level)}")


def parse_arguments() -> argparse.Namespace:
    """명령줄 인자를 파싱합니다.
    
    프로그램 실행에 필요한 인자를 정의하고 명령줄에서 파싱합니다.
    
    Returns:
        argparse.Namespace: 파싱된 인자들
    """
    parser = argparse.ArgumentParser(
        description="스팸 분석 결과 JSON 파일을 SQLite 데이터베이스로 변환합니다."
    )
    
    parser.add_argument(
        "-i", "--input-dir",
        default=DEFAULT_RESULT_DIR,
        help=f"JSON 파일이 있는 디렉토리 경로 (기본값: {DEFAULT_RESULT_DIR})"
    )
    
    parser.add_argument(
        "-o", "--output-db",
        default=DEFAULT_DB_PATH,
        help=f"출력할 SQLite 데이터베이스 파일 경로 (기본값: {DEFAULT_DB_PATH})"
    )
    
    parser.add_argument(
        "-l", "--log-file",
        help="로그를 저장할 파일 경로"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="로깅 레벨 (기본값: INFO)"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 데이터베이스 파일이 있으면 덮어씁니다."
    )
    
    return parser.parse_args()


def extract_model_info(file_name: str) -> Optional[Dict[str, str]]:
    """파일명에서 모델명과 분석 유형을 추출합니다.
    
    다음 형식의 파일명을 지원합니다:
    1. 'model_name_first_YYYY-MM-DD.json' 또는 'model_name_second_YYYY-MM-DD.json'
    2. 'spam_analysis_model_name_first_results_YYYYMMDD_HHMMSS.json'
    
    Args:
        file_name: JSON 파일명
    
    Returns:
        Dict[str, str]: 모델명, 분석 유형, 날짜를 포함한 딕셔너리 또는 형식이 일치하지 않으면 None
    """
    # 정규표현식 패턴 1: model_name_first_YYYY-MM-DD.json 또는 model_name_second_YYYY-MM-DD.json
    pattern1 = r"(.+?)_(first|second)_(\d{4}-\d{2}-\d{2})\.json$"
    match = re.match(pattern1, file_name)
    
    if match:
        return {
            "model": match.group(1),
            "type": match.group(2),
            "date": match.group(3)
        }
    
    # 정규표현식 패턴 2: spam_analysis_model_name_first_results_YYYYMMDD_HHMMSS.json
    pattern2 = r"spam_analysis_(.+?)_(first|second)_results_(\d{8})_\d{6}\.json$"
    match = re.match(pattern2, file_name)
    
    if match:
        # YYYYMMDD 형식을 YYYY-MM-DD 형식으로 변환
        date_str = match.group(3)
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        return {
            "model": match.group(1),
            "type": match.group(2),
            "date": formatted_date
        }
    
    return None


def scan_result_files(result_dir: str) -> List[Dict[str, Any]]:
    """결과 디렉토리에서 모든 JSON 파일을 스캔하고 메타데이터를 추출합니다.
    
    주어진 디렉토리에서 JSON 파일을 찾고, 파일명에서 모델 정보를 추출합니다.
    
    Args:
        result_dir: 결과 파일이 저장된 디렉토리 경로
        
    Returns:
        List[Dict[str, Any]]: 파일 메타데이터 목록 (파일 경로, 모델명, 분석 유형 등)
    """
    file_infos = []
    json_files = glob.glob(os.path.join(result_dir, "*.json"))
    
    logging.info(f"{len(json_files)}개의 JSON 파일을 발견했습니다.")
    
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        # 파일명에서 모델명과 분석 유형을 추출
        model_info = extract_model_info(file_name)
        if model_info:
            file_info = {
                "path": file_path,
                "model": model_info["model"],
                "analysis_type": model_info["type"],
                "date": model_info["date"]
            }
            file_infos.append(file_info)
            logging.debug(f"파일 정보 추출: {file_info}")
        else:
            logging.warning(f"파일명 형식이 일치하지 않습니다: {file_name}")
    
    logging.info(f"{len(file_infos)}개의 유효한 파일을 찾았습니다.")
    
    return file_infos


def create_database_schema(db_path: str, model_infos: List[Dict[str, Any]]) -> None:
    """데이터베이스 스키마를 생성합니다.
    
    이 함수는 SQLite 데이터베이스 파일을 생성하고, 기본 이메일 테이블과
    각 모델별 결과 테이블을 생성합니다. 이미 존재하는 경우 테이블을 다시 생성하지 않습니다.
    또한 모든 모델의 결과를 통합하는 all_results 테이블을 생성합니다.
    
    Args:
        db_path: 데이터베이스 파일 경로
        model_infos: 모델 메타데이터 목록. 각 항목은 모델명과 분석 유형을 포함합니다.
        
    Returns:
        None
        
    Raises:
        sqlite3.Error: 데이터베이스 연결 또는 쿼리 실행 오류 발생 시
    """
    try:
        logging.info(f"데이터베이스 스키마 생성 시작: {db_path}")
        
        db_dir = os.path.dirname(os.path.abspath(db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 메인 이메일 테이블 생성
        logging.info("메인 이메일 테이블 생성")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            id TEXT PRIMARY KEY,
            sender TEXT,
            sender_domain TEXT,
            receiver TEXT,
            subject TEXT
        )
        """)
        
        # 모델별 결과 테이블 생성 (모델명 기준, first/second 결과 모두 포함)
        created_tables = set()
        for model_info in model_infos:
            model_name = model_info['model'].replace('-', '_')
            if model_name in created_tables:
                continue
            logging.info(f"모델 결과 테이블 생성: {model_name}")
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {model_name} (
                id TEXT PRIMARY KEY,
                first_spam BOOLEAN,
                first_duration REAL,
                first_reliability REAL,
                second_spam BOOLEAN,
                second_duration REAL,
                second_reliability REAL,
                human_verified_spam BOOLEAN,
                FOREIGN KEY (id) REFERENCES emails(id)
            )
            """)
            created_tables.add(model_name)
        
        # all_results 테이블 생성
        logging.info("통합 결과 테이블 (all_results) 생성")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS all_results (
            id TEXT,
            model TEXT,
            first_spam BOOLEAN,
            first_duration REAL,
            first_reliability REAL,
            second_spam BOOLEAN,
            second_duration REAL,
            second_reliability REAL,
            human_verified_spam BOOLEAN,
            PRIMARY KEY (id, model),
            FOREIGN KEY (id) REFERENCES emails(id)
        )
        """)
        
        conn.commit()
        conn.close()
        logging.info("데이터베이스 스키마 생성 완료")
        
    except sqlite3.Error as e:
        logging.error(f"데이터베이스 스키마 생성 중 오류 발생: {str(e)}")
        if 'conn' in locals() and conn:
            try:
                conn.close()
            except:
                pass
        raise


def insert_data_from_json(db_path: str, file_info: Dict[str, Any]) -> Tuple[int, int]:
    """JSON 파일에서 데이터를 읽어 데이터베이스에 삽입합니다.
    
    단일 JSON 파일의 데이터를 파싱하여 이메일 기본 정보는 emails 테이블에,
    모델별 분석 결과는 해당 모델의 테이블에 삽입합니다.
    중복된 데이터는 UPSERT 방식으로 처리합니다.
    
    Args:
        db_path: 데이터베이스 파일 경로
        file_info: 파일 메타데이터 (경로, 모델명, 분석 유형 등)
        
    Returns:
        Tuple[int, int]: 삽입된 이메일 수와 모델 결과 수를 담은 튜플
        
    Raises:
        sqlite3.Error: 데이터베이스 연결 또는 쿼리 실행 오류 발생 시
        json.JSONDecodeError: JSON 파싱 오류 발생 시
    """
    try:
        logging.info(f"JSON 데이터 삽입 시작: {file_info['path']}")
        
        with open(file_info["path"], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        model_name = file_info['model'].replace('-', '_')
        model = file_info['model']
        analysis_type = file_info['analysis_type']  # 'first' 또는 'second'
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        conn.execute("BEGIN TRANSACTION")
        
        email_count = 0
        model_result_count = 0
        
        # 데이터 순회 및 삽입
        for year, emails in data.items():
            if not isinstance(emails, list):
                logging.warning(f"{year} 데이터가 리스트 형식이 아닙니다. 스킵합니다.")
                continue
            for email in emails:
                if not isinstance(email, dict) or 'id' not in email:
                    logging.warning("유효하지 않은 이메일 데이터를 발견했습니다. 스킵합니다.")
                    continue
                email_id = email.get('id')
                if not email_id:
                    logging.warning("이메일 ID가 없는 데이터를 발견했습니다. 스킵합니다.")
                    continue
                
                # 이메일 기본 정보 삽입 (UPSERT)
                cursor.execute("""
                INSERT INTO emails (id, sender, sender_domain, receiver, subject)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    sender = COALESCE(excluded.sender, sender),
                    sender_domain = COALESCE(excluded.sender_domain, sender_domain),
                    receiver = COALESCE(excluded.receiver, receiver),
                    subject = COALESCE(excluded.subject, subject)
                """, (
                    email_id,
                    email.get('sender'),
                    email.get('sender_domain'),
                    email.get('receiver'),
                    email.get('subject')
                ))
                email_count += 1
                
                # 모델별 테이블 및 all_results 테이블에 분석 결과 삽입
                human_verified_spam = email.get('human_verified_spam')
                spam = email.get('spam')
                duration = email.get('duration')
                reliability = email.get('reliability')

                if analysis_type == 'first':
                    # 모델별 테이블 업데이트
                    cursor.execute(f"""
                    INSERT INTO {model_name} (id, first_spam, first_duration, first_reliability, human_verified_spam)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        first_spam=COALESCE(excluded.first_spam, {model_name}.first_spam),
                        first_duration=COALESCE(excluded.first_duration, {model_name}.first_duration),
                        first_reliability=COALESCE(excluded.first_reliability, {model_name}.first_reliability),
                        human_verified_spam=COALESCE(excluded.human_verified_spam, {model_name}.human_verified_spam)
                    """, (email_id, spam, duration, reliability, human_verified_spam))
                    
                    # all_results 테이블 업데이트
                    cursor.execute("""
                    INSERT INTO all_results (id, model, first_spam, first_duration, first_reliability, human_verified_spam)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id, model) DO UPDATE SET
                        first_spam=COALESCE(excluded.first_spam, all_results.first_spam),
                        first_duration=COALESCE(excluded.first_duration, all_results.first_duration),
                        first_reliability=COALESCE(excluded.first_reliability, all_results.first_reliability),
                        human_verified_spam=COALESCE(excluded.human_verified_spam, all_results.human_verified_spam)
                    """, (email_id, model, spam, duration, reliability, human_verified_spam))

                elif analysis_type == 'second':
                    # 모델별 테이블 업데이트
                    cursor.execute(f"""
                    INSERT INTO {model_name} (id, second_spam, second_duration, second_reliability, human_verified_spam)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        second_spam=COALESCE(excluded.second_spam, {model_name}.second_spam),
                        second_duration=COALESCE(excluded.second_duration, {model_name}.second_duration),
                        second_reliability=COALESCE(excluded.second_reliability, {model_name}.second_reliability),
                        human_verified_spam=COALESCE(excluded.human_verified_spam, {model_name}.human_verified_spam)
                    """, (email_id, spam, duration, reliability, human_verified_spam))

                    # all_results 테이블 업데이트
                    cursor.execute("""
                    INSERT INTO all_results (id, model, second_spam, second_duration, second_reliability, human_verified_spam)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id, model) DO UPDATE SET
                        second_spam=COALESCE(excluded.second_spam, all_results.second_spam),
                        second_duration=COALESCE(excluded.second_duration, all_results.second_duration),
                        second_reliability=COALESCE(excluded.second_reliability, all_results.second_reliability),
                        human_verified_spam=COALESCE(excluded.human_verified_spam, all_results.human_verified_spam)
                    """, (email_id, model, spam, duration, reliability, human_verified_spam))

                model_result_count += 1
        conn.commit()
        conn.close()
        logging.info(f"데이터 삽입 완료: {email_count}개 이메일, {model_result_count}개 분석 결과")
        return email_count, model_result_count
    except sqlite3.Error as e:
        logging.error(f"데이터베이스 삽입 중 오류 발생: {str(e)}")
        if 'conn' in locals() and conn:
            conn.rollback()
            conn.close()
        raise
    except json.JSONDecodeError as e:
        logging.error(f"JSON 파싱 중 오류 발생: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"데이터 삽입 중 예상치 못한 오류 발생: {str(e)}")
        if 'conn' in locals() and conn:
            conn.rollback()
            conn.close()
        raise


def insert_all_data(db_path: str, file_infos: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """모든 JSON 파일의 데이터를 데이터베이스에 삽입합니다.
    
    주어진 파일 목록의 모든 JSON 데이터를 처리하여 데이터베이스에 삽입합니다.
    
    Args:
        db_path: 데이터베이스 파일 경로
        file_infos: 파일 메타데이터 목록
        
    Returns:
        Tuple[int, int, int]: 성공한 파일 수, 실패한 파일 수, 삽입된 총 이메일 수를 담은 튜플
    """
    logging.info(f"전체 데이터 삽입 시작 ({len(file_infos)}개 파일)")
    
    success_count = 0
    error_count = 0
    total_emails = 0
    total_results = 0
    
    for idx, file_info in enumerate(file_infos, 1):
        try:
            logging.info(f"[{idx}/{len(file_infos)}] 파일 처리 중: {os.path.basename(file_info['path'])}")
            email_count, result_count = insert_data_from_json(db_path, file_info)
            success_count += 1
            total_emails += email_count
            total_results += result_count
        except Exception as e:
            logging.error(f"파일 처리 실패: {os.path.basename(file_info['path'])}, 오류: {str(e)}")
            error_count += 1
    
    logging.info(f"전체 데이터 삽입 완료: 성공 {success_count}개, 실패 {error_count}개, 총 {total_emails}개 이메일, {total_results}개 분석 결과")
    return success_count, error_count, total_emails


def create_views(db_path: str, model_infos: List[Dict[str, Any]]) -> None:
    """데이터베이스 뷰를 생성합니다.
    
    이 함수는 각 모델별 통합 뷰, 전체 통합 뷰, 통계 뷰를 생성합니다.
    
    Args:
        db_path: 데이터베이스 파일 경로
        model_infos: 모델 메타데이터 목록.
        
    Returns:
        None
        
    Raises:
        sqlite3.Error: 데이터베이스 연결 또는 쿼리 실행 오류 발생 시
    """
    try:
        logging.info("뷰 생성 시작")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        unique_models = set(info["model"] for info in model_infos)
        
        # 모델별 뷰 생성
        for model in unique_models:
            model_name = model.replace('-', '_')
            view_name = f"view_{model_name}_combined"
            logging.info(f"모델 뷰 생성: {view_name}")
            cursor.execute(f"DROP VIEW IF EXISTS {view_name}")
            cursor.execute(f"""
            CREATE VIEW {view_name} AS
            SELECT 
                e.id,
                e.sender,
                e.sender_domain,
                e.receiver,
                e.subject,
                m.first_spam,
                m.first_duration,
                m.first_reliability,
                m.second_spam,
                m.second_duration,
                m.second_reliability,
                m.human_verified_spam
            FROM 
                emails e
            LEFT JOIN 
                {model_name} m ON e.id = m.id
            """)

        # 모든 모델 결과를 통합하는 뷰 생성
        logging.info("통합 결과 뷰 (all_results_view) 생성")
        cursor.execute("DROP VIEW IF EXISTS all_results_view")
        cursor.execute("""
        CREATE VIEW IF NOT EXISTS all_results_view AS
        SELECT 
            e.id, e.sender, e.sender_domain, e.receiver, e.subject, 
            r.model, r.first_spam, r.first_duration, r.first_reliability, 
            r.second_spam, r.second_duration, r.second_reliability, r.human_verified_spam
        FROM 
            emails e
        JOIN 
            all_results r ON e.id = r.id
        """)
        
        # 모델별로 그룹화된 통계 뷰 생성
        logging.info("모델별 통계 뷰 (model_stats_view) 생성")
        cursor.execute("DROP VIEW IF EXISTS model_stats_view")
        cursor.execute("""
        CREATE VIEW IF NOT EXISTS model_stats_view AS
        SELECT
            model,
            'first' as analysis_type,
            COUNT(id) as total_emails,
            AVG(first_reliability) as avg_reliability,
            AVG(first_duration) as avg_duration,
            MIN(first_reliability) as min_reliability,
            MAX(first_reliability) as max_reliability
        FROM
            all_results
        WHERE first_reliability IS NOT NULL
        GROUP BY model
        UNION ALL
        SELECT
            model,
            'second' as analysis_type,
            COUNT(id) as total_emails,
            AVG(second_reliability) as avg_reliability,
            AVG(second_duration) as avg_duration,
            MIN(second_reliability) as min_reliability,
            MAX(second_reliability) as max_reliability
        FROM
            all_results
        WHERE second_reliability IS NOT NULL
        GROUP BY model
        """)
        
        conn.commit()
        conn.close()
        logging.info("뷰 생성 완료")
        
    except sqlite3.Error as e:
        logging.error(f"뷰 생성 중 오류 발생: {str(e)}")
        if 'conn' in locals() and conn:
            try:
                conn.close()
            except:
                pass
        raise


def main() -> int:
    """메인 프로그램 함수입니다.
    
    명령줄 인자를 파싱하고 전체 변환 프로세스를 실행합니다.
    
    Returns:
        int: 프로그램 종료 코드 (0: 성공, 1: 오류)
    """
    # 명령줄 인자 파싱
    args = parse_arguments()
    
    # 로깅 설정
    log_level = getattr(logging, args.log_level)
    setup_logging(args.log_file, log_level)
    
    logging.info("스팸 분석 결과 JSON → SQLite 변환 시작")
    logging.info(f"입력 디렉토리: {args.input_dir}")
    logging.info(f"출력 데이터베이스: {args.output_db}")
    
    try:
        # 기존 데이터베이스 파일 확인
        if os.path.exists(args.output_db) and not args.overwrite:
            logging.warning(f"데이터베이스 파일이 이미 존재합니다: {args.output_db}")
            logging.warning("계속하려면 --overwrite 옵션을 사용하세요.")
            logging.info("프로그램을 종료합니다.")
            return 1
        elif os.path.exists(args.output_db) and args.overwrite:
            logging.warning(f"기존 데이터베이스 파일을 덮어씁니다: {args.output_db}")
            os.remove(args.output_db)
        
        # 1. 파일 스캔 및 메타데이터 추출
        file_infos = scan_result_files(args.input_dir)
        
        if not file_infos:
            logging.warning("변환할 유효한 파일이 없습니다.")
            logging.info("프로그램을 종료합니다.")
            return 0
        
        # 2. 데이터베이스 스키마 생성
        create_database_schema(args.output_db, file_infos)
        
        # 3. 데이터 삽입
        success_count, error_count, total_emails = insert_all_data(args.output_db, file_infos)
        
        # 4. 뷰 생성
        create_views(args.output_db, file_infos)
        
        logging.info("변환 작업이 성공적으로 완료되었습니다.")
        logging.info(f"데이터베이스 파일: {args.output_db}")
        logging.info(f"처리 결과: {len(file_infos)}개 파일 중 {success_count}개 성공, {error_count}개 실패")
        logging.info(f"저장된 데이터: {total_emails}개 이메일")
        
        return 0
        
    except Exception as e:
        logging.error(f"변환 작업 중 오류가 발생했습니다: {str(e)}")
        logging.debug("상세 오류 정보:", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
