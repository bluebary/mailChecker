import json, os, chromadb, time, re, requests, logging, datetime, traceback
import db_sqlite_converter
from typing import List, Union
from pathlib import Path
from collections import defaultdict
from chromadb.utils import embedding_functions
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from ollama._types import ResponseError

"""
이메일 스팸 검사 시스템

이 모듈은 Ollama LLM을 사용하여 이메일의 스팸 여부를 분석하고 분류하는 시스템을 구현합니다.

특징:
    - Ollama LLM 모델을 사용한 이메일 분석
    - 사용 가능한 Ollama 모델 선택 기능
    - ChromaDB를 활용한 유사 이메일 검색
    - 분석 결과의 신뢰도 점수 제공
    - JSON 형식의 결과 저장

버전:
    RDB - ChromaDB를 사용한 검색 기능 강화 버전
"""

# 전역 변수 초기화
collection = None
client = None
embedding_function = None
llm = None
chain = None
logger = None

def setup_logging() -> None:
    """로깅 시스템을 설정합니다.
    
    로그 파일은 logs 디렉토리에 날짜별로 저장됩니다.
    오류 로그는 별도의 파일에 저장됩니다.
    """
    global logger
    
    # 로그 디렉토리 생성
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 현재 날짜와 시간을 파일명에 포함
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'mail_checker_{current_time}.log')
    error_log_file = os.path.join(log_dir, f'mail_checker_error_{current_time}.log')
    
    # 루트 로거 설정
    logger = logging.getLogger('mail_checker')
    logger.setLevel(logging.DEBUG)
    
    # 파일 핸들러 설정 (일반 로그)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # 파일 핸들러 설정 (오류 로그)
    error_file_handler = logging.FileHandler(error_log_file, encoding='utf-8')
    error_file_handler.setLevel(logging.ERROR)
    error_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n%(message)s\n')
    error_file_handler.setFormatter(error_format)
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(error_file_handler)
    logger.addHandler(console_handler)
    
    logger.info("로깅 시스템이 초기화되었습니다")
    logger.info(f"로그 파일 경로: {log_file}")
    logger.info(f"오류 로그 파일 경로: {error_log_file}")

def log_exception(e: Exception, context: str = ""):
    """예외를 로그에 기록합니다.
    
    Args:
        e (Exception): 발생한 예외
        context (str, optional): 예외가 발생한 컨텍스트 설명
    """
    error_msg = f"{context}: {str(e)}" if context else str(e)
    logger.error(error_msg)
    logger.error(traceback.format_exc())

def get_ollama_models() -> list:
    """Ollama 서버에서 설치된 모델 목록을 가져옵니다.

    Returns:
        list: 설치된 모델 정보를 담은 리스트. 각 모델은 name과 size 정보를 포함합니다.
    """
    try:
        logger.info("Ollama 서버에서 모델 목록을 가져오는 중...")
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = response.json().get('models', [])
            logger.info(f"{len(models)}개의 모델을 찾았습니다")
            return models
        else:
            error_msg = f"Ollama 서버 응답 오류: {response.status_code}"
            logger.error(error_msg)
            return []
    except Exception as e:
        log_exception(e, "Ollama 서버 연결 중 오류 발생")
        return []

def choose_ollama_model() -> str:
    """사용자가 사용할 Ollama 모델을 선택하도록 합니다.

    Returns:
        str: 선택된 모델의 이름. 연결 실패 시 기본값 'gemma3:12b'를 반환합니다.
    """
    models = get_ollama_models()
    
    if not models:
        logger.warning("모델을 찾을 수 없거나 Ollama에 연결할 수 없습니다. 기본 모델을 사용합니다.")
        return "gemma3:12b"
    elif len(models) > 1:
        print("\n=== 설치된 Ollama 모델 목록 ===")
        for i, model in enumerate(models, 1):
            model_name = model.get('name', 'Unknown')
            model_size = model.get('size', 0) // (1024*1024)  # Convert to MB
            print(f"{i}. {model_name} ({model_size} MB)")
    
        while True:
            try:
                choice = input("\n사용할 모델 번호를 선택하세요 (기본값: 1): ")
                if choice.strip() == "":
                    selected_idx = 0
                else:
                    selected_idx = int(choice) - 1
                    
                if 0 <= selected_idx < len(models):
                    selected_model = models[selected_idx]['name']
                    print(f"선택한 모델: {selected_model}")
                    logger.info(f"선택된 모델: {selected_model}")
                    return selected_model
                else:
                    logger.warning(f"잘못된 모델 번호 입력: {selected_idx + 1}")
                    print(f"잘못된 번호입니다. 1부터 {len(models)}까지의 번호를 입력하세요.")
            except ValueError as e:
                logger.warning(f"모델 선택 중 오류: {e}")
                print("숫자를 입력하세요.")
    elif len(models) == 1:
        selected_model = models[0]['name']
        return selected_model

def list_json_convert_dict(file_path: str) -> dict:
    """mail_json_list.json 파일을 읽어서 딕셔너리로 변환합니다.

    Args:
        file_path (str): mail_json_list.json 파일의 경로

    Returns:
        dict: 연도별 파일 목록과 총 파일 수를 포함하는 딕셔너리
    """
    logger.info(f"JSON 목록 파일 읽기: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            total_files = data.get('total_files', 0)
            logger.info(f"JSON 목록을 성공적으로 읽었습니다: 총 {total_files}개 파일")
            print(f"JSON 목록을 성공적으로 읽었습니다: {file_path}")
            print(f"총 파일 수: {total_files}")
            return data
    except FileNotFoundError as e:
        log_exception(e, f"파일을 찾을 수 없습니다: {file_path}")
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        log_exception(e, f"JSON 디코딩 오류 발생: {file_path}")
        print(f"JSON 디코딩 오류 발생: {file_path}: {e}")
        return {}
    except Exception as e:
        log_exception(e, f"JSON 파일 읽기 오류: {file_path}")
        print(f"JSON 파일 읽기 오류: {file_path}: {e}")
        return {}

def mail_json_convert_dict(file_path: str) -> dict:
    """JSON 파일을 읽어서 지정된 데이터 구조의 딕셔너리로 변환합니다.

    Args:
        file_path (str): 변환할 JSON 파일의 절대 경로

    Returns:
        dict: 표준화된 이메일 데이터 구조
            {
                "id": str,
                "spam": bool,
                "sender": str,
                "sender_domain": str,
                "receiver": str,
                "subject": str,
                "duration": float,
                "reliability": float
            }
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        # 이메일 주소에서 도메인 추출
        sender_domain = data.get('from', '').split('@')[-1] if '@' in data.get('from', '') else ''
        
        # 수신자 목록을 문자열로 변환
        receivers = data.get('to', [])
        receiver = ', '.join(receivers) if isinstance(receivers, list) else str(receivers)
        
        email_id = Path(file_path).stem
        logger.debug(f"이메일 파일 변환: {email_id}")
        
        return {
            "id": email_id,
            "spam": False,  # 기본값
            "sender": data.get('from', ''),
            "sender_domain": sender_domain,
            "receiver": receiver,
            "subject": data.get('subject', ''),
            "duration": 0.0,  # 기본값
            "reliability": 0.0  # 기본값
        }
    except Exception as e:
        log_exception(e, f"JSON 파일 변환 중 오류 발생: {file_path}")
        print(f"JSON 파일 변환 중 오류 발생: {e}")
        return {}

def ollama_low_analysis() -> dict:
    """Ollama 모델을 사용하여 모든 이메일의 스팸 여부를 분석합니다.
    
    mail_json_list.json 파일에서 이메일 목록을 읽어와 각 이메일을 분석하고,
    결과를 연도별로 정리하여 반환합니다.
    """
    result_data = defaultdict(list)
    root_path = "/home/sound/mailChecker/mail_json_list.json"
        
    file_list = list_json_convert_dict(root_path)
    if not file_list:
        logger.error("이메일 목록을 읽을 수 없습니다. 분석을 중단합니다.")
        return result_data
    
    total_processed = 0
    total_files = sum(year_data.get('count', 0) for year_data in file_list.get('files_by_year', {}).values())
    logger.info(f"총 분석 대상 파일 수: {total_files}")
    
    for year, year_data in file_list['files_by_year'].items():
        # 임시로 2000년대만 처리
        if year == '2004':
            total_year_count = year_data['count']
            logger.info(f"{year}년 데이터 처리 시작 (총 {total_year_count}개 파일)")
            print(f"\n=== {year}년 처리 중 ===")
            print(f"파일 개수: {total_year_count}")
            
            processed_year_count = 0
            for i, file_path in enumerate(year_data['files'], 1):
                try:
                    # 파일 처리
                    file_data = mail_json_convert_dict(file_path)
                    if not file_data:
                        logger.warning(f"파일을 처리할 수 없습니다: {file_path}")
                        continue
                    
                    # 스팸 분석
                    start_time = time.time()
                    is_spam, reliability_score = check_spam(file_data)
                    end_time = time.time()
                    processing_time = round(end_time - start_time, 1)
                    
                    # 데이터 업데이트
                    file_data["spam"] = is_spam
                    file_data["duration"] = processing_time
                    file_data["reliability"] = reliability_score

                    # 결과 데이터 저장
                    result_data[year].append(file_data)
                    processed_year_count += 1
                    total_processed += 1
                    
                    # 진행 상황 표시
                    # progress = (i / total_year_count) * 100
                    # print(f"\r진행률: [{('=' * int(progress/2)).ljust(50)}] {progress:.1f}% ({i}/{total_year_count})", end='')
                
                except Exception as e:
                    log_exception(e, f"파일 처리 중 오류 발생: {file_path}")
                    continue
                
                logger.info(f"{year}년 데이터 처리 완료: {processed_year_count}/{total_year_count} 파일 처리됨, 처리시간: {processing_time}초")    
    
    # 처리 완료 메시지
    logger.info(f"모든 이메일 분석이 완료되었습니다. 총 {total_processed}/{total_files} 파일 처리됨")
    print(f"\n모든 이메일 분석이 완료되었습니다. 총 {total_processed}/{total_files} 파일 처리됨")
    return result_data

def get_similar_emails(query_text: str, k: int = 3) -> str:
    """ChromaDB에서 주어진 쿼리와 유사한 이메일을 검색합니다.

    Args:
        query_text (str): 검색할 이메일 제목
        k (int, optional): 반환할 유사 이메일의 수. 기본값은 3입니다.

    Returns:
        str: 유사 이메일들의 정보를 포함한 문자열
    """
    if not query_text:
        logger.warning("유사 이메일 검색을 위한 쿼리 텍스트가 비어 있습니다.")
        return ""
    
    try:
        logger.debug(f"유사 이메일 검색: '{query_text}'")
        results = collection.query(
            query_texts=[query_text],
            n_results=k
        )
        
        if not results["documents"][0]:
            logger.debug("유사한 이메일을 찾을 수 없습니다.")
            return "유사한 이메일을 찾을 수 없습니다."
        
        # 결과를 문자열로 변환
        context_str = ""
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            context_str += f"Email {i+1}:\n"
            context_str += f"Subject: {doc}\n"
            context_str += f"Sender: {metadata.get('sender', 'Unknown')}\n"
            context_str += f"Domain: {metadata.get('sender_domain', 'Unknown')}\n"
            context_str += f"Spam: {metadata.get('is_spam', False)}\n\n"
        
        logger.debug(f"{len(results['documents'][0])}개의 유사 이메일을 찾았습니다.")
        return context_str
    except Exception as e:
        log_exception(e, "유사 이메일 검색 중 오류 발생")
        return "유사 이메일 검색 중 오류가 발생했습니다."

def check_spam(data: dict) -> List[Union[bool, float]]:
    """이메일의 스팸 여부를 Ollama 모델을 사용하여 확인하고 파싱합니다.

    Args:
        data (dict): 분석할 이메일 데이터

    Returns:
        List[Union[bool, float]]: 스팸 여부(bool)와 신뢰도 점수(float)를 담은 리스트 (예: [True, 85.2]).
                                  실패 시 [False, 0.0]을 반환합니다.
    """
    global chain
    max_retries: int = 3
    retry_delay: int = 5  # 재시도 전 대기 시간(초)
    
    try:
        similar_emails = get_similar_emails(data['subject'])
        
        message = f"""Similar emails:
        {similar_emails}

        Target email to classify:
        Sender: {data['sender']}
        Receiver: {data['receiver']}  
        Subject: {data['subject']}
        """
        
        for attempt in range(max_retries):
            try:
                # 1. LLM 호출
                response_str = chain.invoke({"message": message})
                
                # 2. 응답 파싱
                cleaned_response = re.sub(r'<think>.*?</think>\n*', '', response_str, flags=re.DOTALL).strip()
                parts = cleaned_response.split(',')

                if len(parts) != 2:
                    raise ValueError(f"응답 형식이 잘못되었습니다: '{cleaned_response}'")

                is_spam_str = parts[0].strip().lower()
                if is_spam_str == "true":
                    is_spam = True
                elif is_spam_str == "false":
                    is_spam = False
                else:
                    raise ValueError(f"스팸 여부 파싱 실패: '{is_spam_str}'")
                reliability = float(parts[1].strip())

                return [is_spam, reliability]

            except Exception as e:
                is_ollama_500_error = isinstance(e, ResponseError) and "status code: 500" in str(e)
                
                if attempt < max_retries - 1:
                    error_type = "Ollama 500 오류" if is_ollama_500_error else "파싱 또는 API 오류"
                    logger.warning(
                        f"{error_type} 발생, {retry_delay}초 후 재시도... "
                        f"({attempt + 1}/{max_retries}) (ID: {data.get('id', 'N/A')}) - 오류: {e}"
                    )
                    print(
                        f"{error_type} 발생, {retry_delay}초 후 재시도... "
                        f"({attempt + 1}/{max_retries})"
                    )
                    time.sleep(retry_delay)
                else:
                    # 마지막 시도에서는 오류를 기록하고 루프를 중단
                    log_exception(e, f"최대 재시도({max_retries}) 후에도 스팸 확인 실패: {data.get('id', '알 수 없음')}")
                    print(f"최대 재시도 후에도 스팸 확인에 실패했습니다: {e}")
                    break  # 루프 중단
        
    except Exception as e:
        # get_similar_emails 등 루프 외부에서 발생한 예외 처리
        log_exception(e, f"스팸 확인 준비 중 오류 발생: {data.get('id', '알 수 없음')}")
        print(f"스팸 확인 준비 중 오류가 발생했습니다: {e}")

    # 모든 재시도 실패 또는 준비 과정에서 오류 발생 시 기본값 반환
    return [False, 0.0]

def save_result(result_data: dict, filename: str) -> None:
    """분석 결과를 JSON 파일로 저장합니다.

    Args:
        result_data (dict): 저장할 결과 데이터
        filename (str): 저장할 파일의 이름
    """
    try:
        current_dir = os.path.dirname(__file__)
        if os.path.exists(current_dir):
            # 결과 디렉토리 생성
            results_dir = os.path.join(current_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            
            # 타임스탬프 추가
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_with_timestamp = f"{os.path.splitext(filename)[0]}_{timestamp}{os.path.splitext(filename)[1]}"
            path = os.path.join(results_dir, filename_with_timestamp)
            
            # 결과 저장
            with open(path, "w", encoding="utf-8") as file: 
                json.dump(result_data, file, ensure_ascii=False, indent=4)
            
            logger.info(f"분석 결과가 저장되었습니다: {path}")
            print(f"결과가 저장되었습니다: {path}")
            
            # 스팸 통계 로깅
            spam_count = 0
            total_count = 0
            for year, emails in result_data.items():
                year_spam_count = sum(1 for email in emails if email.get('spam'))
                year_total = len(emails)
                spam_count += year_spam_count
                total_count += year_total
                logger.info(f"{year}년 스팸 통계: {year_spam_count}/{year_total} ({year_spam_count/year_total*100:.1f}%)")
            
            if total_count > 0:
                logger.info(f"전체 스팸 통계: {spam_count}/{total_count} ({spam_count/total_count*100:.1f}%)")
    except Exception as e:
        log_exception(e, "결과 저장 중 오류 발생")
        print(f"결과 저장 중 오류 발생: {e}")

def convert_all_results_to_db() -> None:
    """results 디렉토리의 모든 JSON 파일을 SQLite DB로 변환"""
    current_dir = os.path.dirname(__file__)
    results_dir = os.path.join(current_dir, "results")
    db_path = os.path.join(current_dir, "../email_analysis.db")
    
    if not os.path.exists(results_dir):
        logger.warning("results 디렉토리가 존재하지 않습니다.")
        return
    
    try:
        logger.info("JSON 파일을 SQLite DB로 변환 중...")
        success = db_sqlite_converter.convert_json_to_sqlite(
            input_dir=results_dir,
            output_db=db_path,
            overwrite=True
        )
        
        if success:
            logger.info(f"DB 변환 완료: {db_path}")
            print(f"SQLite 데이터베이스가 생성되었습니다: {db_path}")
        else:
            logger.error("DB 변환 실패")
            print("DB 변환 중 오류가 발생했습니다.")
            
    except Exception as e:
        log_exception(e, "DB 변환 중 오류 발생")
        print(f"DB 변환 중 오류가 발생했습니다: {e}")

def init_system() -> None:
    """시스템을 초기화하고, 설치된 모든 모델에 대해 분석을 실행합니다."""
    global collection, client, embedding_function, llm, chain
    
    try:
        # 로깅 시스템 설정
        setup_logging()
        start_time = time.time()
        logger.info("시스템 초기화 시작")
        
        # 임베딩 함수 초기화 (공통)
        logger.info("임베딩 함수 초기화")
        print("1. 임베딩 함수 초기화")
        try:
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="Linq-AI-Research/Linq-Embed-Mistral"
            )
            logger.info("임베딩 함수가 성공적으로 초기화되었습니다.")
        except Exception as e:
            log_exception(e, "임베딩 함수 초기화 실패")
            print(f"임베딩 함수 초기화 실패: {e}")
            raise
            
        # 설치된 모든 Ollama 모델 가져오기
        logger.info("Ollama 모델 목록을 가져오는 중...")
        models = get_ollama_models()
        if not models:
            logger.error("Ollama 모델을 찾을 수 없습니다. 분석을 중단합니다.")
            print("Ollama 모델을 찾을 수 없습니다. 분석을 중단합니다.")
            return

        # LLM 체인 템플릿 설정 (공통)
        template = """You are a spam email classifier. Analyze the provided similar emails and the target email to make your decision.

        Similar emails from database:
        {message}

        Rules:
        - If you find emails with identical/very similar subjects, follow the majority classification among those examples
        - Prioritize consistency for similar cases
        - **STRICTLY FOLLOW THE OUTPUT FORMAT - NO EXCEPTIONS**
        - **Do not include any explanations, reasoning, or additional text**
        - **Output ONLY the required format: classification,reliability**
        - Classification: True (spam) or False (not spam)  
        - Reliability: Confidence level as a percentage with one decimal place (e.g., 95.5%)

        **MANDATORY Result Format: True,85.2 or False,92.5**
        **Any deviation from this exact format will be considered incorrect**
        """
        prompt = PromptTemplate(template=template, input_variables=["message"])

        # 각 모델에 대해 분석 실행
        for model_info in models:
            selected_model = model_info['name']
            model_filename = selected_model.replace(":", "_").replace("/", "_")
            
            logger.info(f"===== 모델: {selected_model} 분석 시작 =====")
            print(f"\n===== 모델: {selected_model} 분석 시작 =====")

            try:
                # 모델별 ChromaDB 설정
                db_path = f"../vectorDB/chroma_db_{model_filename}"
                logger.info(f"ChromaDB 초기화 (경로: {db_path}")
                try:
                    client = chromadb.PersistentClient(path=db_path)
                    collection = client.get_or_create_collection(
                        name="email_data",
                        embedding_function=embedding_function
                    )
                    logger.info("ChromaDB 컬렉션이 성공적으로 초기화되었습니다.")
                except Exception as e:
                    log_exception(e, f"ChromaDB 초기화 실패 (모델: {selected_model})")
                    print(f"ChromaDB 초기화 실패: {e}")
                    raise

                # Ollama LLM 및 체인 초기화
                llm = OllamaLLM(model=selected_model)
                chain = prompt | llm
                logger.info(f"Ollama LLM 및 체인이 초기화되었습니다. 모델: {selected_model}")

                # 이메일 분석 실행
                logger.info(f"이메일 스팸 분석 1차 프로세스 시작 (모델: {selected_model})")
                
                start_time = time.time()
                result_data = ollama_low_analysis()
                end_time = time.time()
                
                execution_time = end_time - start_time
                hours, remainder = divmod(execution_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                time_str = f"{int(hours)}시간 {int(minutes)}분 {int(seconds)}초"
                logger.info(f"1차 분석 완료 (모델: {selected_model}). 소요 시간: {time_str}")
                print(f"\n1차 분석 완료 (모델: {selected_model}). 소요 시간: {time_str}")

                # 결과 저장 (모델 이름 포함)
                save_result(result_data, f"spam_analysis_{model_filename}_first_results.json")

                logger.info(f"이메일 스팸 분석 2차 프로세스 시작 (모델: {selected_model})")
                
                start_time = time.time()
                result_data = ollama_low_analysis()
                end_time = time.time()
                
                execution_time = end_time - start_time
                hours, remainder = divmod(execution_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                time_str = f"{int(hours)}시간 {int(minutes)}분 {int(seconds)}초"
                logger.info(f"2차 분석 완료 (모델: {selected_model}). 소요 시간: {time_str}")
                print(f"\n2차 분석 완료 (모델: {selected_model}). 소요 시간: {time_str}")

                # 결과 저장 (모델 이름 포함)
                save_result(result_data, f"spam_analysis_{model_filename}_second_results.json")

            except Exception as e:
                log_exception(e, f"모델 {selected_model} 처리 중 오류 발생")
                print(f"\n모델 {selected_model} 처리 중 오류가 발생했습니다. 다음 모델로 넘어갑니다.")
                continue
        
        end_time = time.time()
        total_execution_time = end_time - start_time
        hours, remainder = divmod(total_execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"모든 모델에 대한 분석이 완료되었습니다. 소요 시간: {hours}시간 {minutes}분 {seconds}초")
        print(f"\n모든 모델에 대한 분석이 완료되었습니다. 소요 시간: {hours}시간 {minutes}분 {seconds}초")

        # 모든 모델 분석 완료 후 DB 변환 실행
        try:
            logger.info("===== 모든 모델 분석 완료 =====")
            logger.info("SQLite 데이터베이스 변환을 시작합니다...")
            print(f"\n===== 모든 모델 분석 완료 =====")
            print("SQLite 데이터베이스 변환을 시작합니다...")
            
            convert_all_results_to_db()
            
            logger.info("전체 프로세스가 완료되었습니다.")
            print("전체 프로세스가 완료되었습니다.")
            
        except Exception as e:
            log_exception(e, "DB 변환 중 오류 발생")
            print("DB 변환 중 오류가 발생했지만 JSON 파일은 정상적으로 저장되었습니다.")

    except Exception as e:
        log_exception(e, "시스템 초기화 및 분석 중 치명적인 오류 발생")
        print(f"시스템 초기화 및 분석 중 치명적인 오류가 발생했습니다: {e}")
        raise

def main() -> None:
    """메인 함수: 시스템을 초기화하고 모든 모델에 대해 이메일 분석을 실행합니다."""
    try:
        # 시스템 초기화 및 전체 분석 실행
        init_system()
        
    except KeyboardInterrupt:
        logger.warning("사용자에 의해 프로그램이 중단되었습니다.")
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        # init_system에서 이미 로깅되었을 수 있지만, 만약을 위해 여기서도 로깅
        log_exception(e, "프로그램 실행 중 예기치 않은 오류 발생")
        print(f"\n\n프로그램 실행 중 오류가 발생했습니다.")
        print("자세한 오류 내용은 로그 파일을 확인하세요.")
    finally:
        # 종료 메시지
        logger.info("프로그램 종료")
        print("\n프로그램을 종료합니다.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(e, "메인 실행 중 오류 발생")
        print("오류가 발생하여 프로그램이 종료됩니다. 자세한 내용은 로그 파일을 확인하세요.")
