import json, hashlib, os, chromadb, argparse, chromadb, csv, time, itertools
import pandas as pd
import requests
from chromadb.utils import embedding_functions
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

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
    original - Ollama 모델을 사용한 기본 점검 버전
"""

collection, client, embedding_function, llm, chain = None, None, None, None, None
allData = dict()

# Get a list of installed models from Ollama
def get_ollama_models() -> list:
    """Ollama 서버에서 설치된 모델 목록을 가져옵니다.

    Returns:
        list: 설치된 모델 정보를 담은 리스트. 각 모델은 name과 size 정보를 포함합니다.
    """
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = response.json().get('models', [])
            return models
        else:
            print(f"Error fetching models: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return []

# Display models and let the user choose
def choose_ollama_model() -> str:
    """사용자가 사용할 Ollama 모델을 선택하도록 합니다.

    Returns:
        str: 선택된 모델의 이름. 연결 실패 시 기본값 'gemma3:12b'를 반환합니다.
    """
    models = get_ollama_models()
    
    if not models:
        print("No models found or unable to connect to Ollama. Using default model 'gemma3:12b'")
        return "gemma3:12b"
    
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
                return selected_model
            else:
                print(f"잘못된 번호입니다. 1부터 {len(models)}까지의 번호를 입력하세요.")
        except ValueError:
            print("숫자를 입력하세요.")

def read_mail_json(filename: str) -> dict:
    """JSON 파일을 읽어 데이터를 반환합니다.

    Args:
        filename (str): 읽을 JSON 파일의 경로

    Returns:
        dict: JSON 파일의 데이터를 담은 딕셔너리
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {filename}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON 디코딩 오류 발생: {e}")
        return None
    except Exception as e:
        print(f"파일 읽기 중 오류 발생: {e}")
        return None

# Read CSV analytics data, convert it to a dictionary, and save it to allData
def read_csv_convert_json(filename: str) -> None:
    """CSV 파일을 읽어 JSON 형식으로 변환하여 전역 변수에 저장합니다.

    Args:
        filename (str): 읽을 CSV 파일의 경로
    """
    global allData  
    csvData = read_file_csv(filename)
    allData = convert_to_dict(csvData)
    
def read_file_csv(filename: str) -> list:
    """CSV 파일을 읽어 데이터를 리스트로 반환합니다.

    Args:
        filename (str): 읽을 CSV 파일의 경로

    Returns:
        list: CSV 파일의 각 행을 리스트로 변환한 데이터
    """
    data = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data.append(row)
            return data
    except FileNotFoundError:
        print(f"File Not Found: {filename}")
        return None
    except Exception as e:
        print(f"CSV Reading Error: {e}")
        return None

# Convert the data to a dictionary format
def convert_to_dict(csv_data: list) -> dict:
    """CSV 데이터를 딕셔너리 형식으로 변환합니다.

    Args:
        csv_data (list): 변환할 CSV 데이터 리스트

    Returns:
        dict: 다음과 같은 구조의 데이터를 포함하는 딕셔너리:
            {
                "spam": bool,
                "sender": str,
                "sender_domain": str,
                "receiver": str,
                "subject": str,
                "duration": float,
                "reliability": float
            }
    """
    rtn_dict = dict()
    """Data Structure
    {
        "spam": True/False,
        "sender": Mail Sender,
        "sender_domain": Mail Sender Domain,
        "receiver": Mail Receiver,
        "subject": Mail Subject,
        "duration": Duration of the llm processing time
        "reliability": Reliability score of the llm response
    }
    """
    for data in csv_data:
        key = f"spam-{hashlib.sha256(data[3].strip().encode()).hexdigest()}"
        rtn_data = {
            "spam": False,
            "sender": data[0],
            "sender_domain": data[1],
            "receiver": data[2],
            "subject": data[3].strip(),
            "duration": 0,
            "reliability": 0.0
        }
        rtn_dict[key] = rtn_data
    # 임시로 500개 데이터만
    # return dict(itertools.islice(rtn_dict.items(), 500))
    return rtn_dict

# Using the Ollama model to First check if the email is spam
def ollama_Low_analysis() -> None:
    """Ollama 모델을 사용하여 모든 이메일의 스팸 여부를 분석합니다.
    
    분석 결과는 전역 변수 allData에 저장되며, 진행 상황이 프로그레스 바로 표시됩니다.
    """
    root_path = "/home/sound/mailChecker/saveData"
    
    # 모든 파일 목록 생성
    all_files = []
    for year in sorted(os.listdir(root_path)):
        year_path = os.path.join(root_path, year)
        if not os.path.isdir(year_path):
            continue
            
        for month in sorted(os.listdir(year_path)):
            month_path = os.path.join(year_path, month)
            if not os.path.isdir(month_path):
                continue
                
            json_files = [f for f in os.listdir(month_path) if f.endswith('.json')]
            for json_file in json_files:
                file_path = os.path.join(month_path, json_file)
                all_files.append((year, month, file_path))
    
    total_files = len(all_files)
    if total_files == 0:
        print("처리할 파일이 없습니다.")
        return
        
    print(f"\n총 {total_files}개의 파일을 처리합니다.")
    
    # 파일 처리
    current_year = None
    year_data = {}
    
    for idx, (year, month, file_path) in enumerate(all_files, 1):
        # 연도가 바뀌면 이전 연도 데이터 처리
        if current_year != year:
            if current_year is not None:
                allData = year_data
                print(f"\n{current_year}년도 데이터 처리 완료: {len(year_data)}개 항목")
            current_year = year
            year_data = {}
            print(f"\n처리 중인 연도: {year}")
        
        # 프로그레스 바 표시
        percent = idx / total_files * 100
        bar_length = 40
        filled_length = int(bar_length * idx // total_files)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"\r[{bar}] {idx}/{total_files} ({percent:.1f}%) - {os.path.basename(file_path)}", end='', flush=True)
        
        # 파일 처리
        file_data = read_json_file(file_path)
        year_data.update(file_data)
        
        for key, item in allData.items():
            chkCnt = chkCnt + 1
            start_time = time.time()
            rst_tmp = check_spam(item)
            rst = rst_tmp.strip().split(',')[0]
            end_time = time.time()
            item["spam"] = rst
            item["duration"] = round(end_time - start_time, 1)
            item["reliability"] = float(rst_tmp.strip().split(',')) if len(rst_tmp.strip().split(',')) > 1 else 0.0

    
    # 마지막 연도 데이터 처리
    if current_year is not None:
        allData = year_data
        print(f"\n{current_year}년도 데이터 처리 완료: {len(year_data)}개 항목")

def get_similar_emails(query_text: str, k: int = 3) -> str:
    """ChromaDB에서 주어진 쿼리와 유사한 이메일을 검색합니다.

    Args:
        query_text (str): 검색할 이메일 제목
        k (int, optional): 반환할 유사 이메일의 수. 기본값은 3입니다.

    Returns:
        str: 유사 이메일들의 정보를 포함한 문자열
    """
    # ChromaDB 컬렉션에서 직접 쿼리
    results = collection.query(
        query_texts=[query_text],
        n_results=k
    )
    
    # 결과를 문자열로 변환
    context_str = ""
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        context_str += f"Email {i+1}:\n"
        context_str += f"Subject: {doc}\n"
        context_str += f"Sender: {metadata.get('sender', 'Unknown')}\n"
        context_str += f"Domain: {metadata.get('sender_domain', 'Unknown')}\n"
        context_str += f"Spam: {metadata.get('is_spam', False)}\n\n"
    
    return context_str

# ollama model query to check if the email is spam
def check_spam(data: dict) -> str:
    """이메일의 스팸 여부를 Ollama 모델을 사용하여 확인합니다.

    Args:
        data (dict): 분석할 이메일 데이터

    Returns:
        str: 스팸 여부와 신뢰도 점수를 포함한 문자열
    """
    global chain
    
    similar_emails = get_similar_emails(data['subject'])
    message = f"""Similar emails:
    {similar_emails}

    Target email to classify:
    Sender: {data['sender']}
    Receiver: {data['receiver']}  
    Subject: {data['subject']}
    """
    
    try:
        # LangChain's chain.run is deprecated, so use chain.invoke
        response = chain.invoke({"message": message})
        return response
    except Exception as e:
        print(f"Error in check_spam: {e}")
        return False   

# Save the result to a JSON file
def save_to_result(filename: str) -> None:
    """분석 결과를 JSON 파일로 저장합니다.

    Args:
        filename (str): 저장할 파일의 이름
    """
    global allData
    current_dir = os.path.dirname(__file__)
    if os.path.exists(current_dir):
        path = os.path.join(current_dir, filename)
        with open(path, "w", encoding="utf-8") as file: 
            json.dump(allData, file, ensure_ascii=False, indent=4)

# Find JSON files in the saveData directory
def find_json_files(base_dir: str = "./saveData") -> list:
    """지정된 디렉토리에서 JSON 파일들을 찾습니다.

    Args:
        base_dir (str, optional): 검색할 기본 디렉토리. 기본값은 "./saveData"입니다.

    Returns:
        list: 찾은 JSON 파일들의 경로 리스트
    """
    """
    saveData 폴더에서 JSON 파일을 찾아 경로 목록을 반환합니다.
    파일은 연도/월 구조로 저장되어 있습니다.
    """
    json_files = []
    
    try:
        # 절대 경로로 변환
        base_dir = os.path.abspath(base_dir)
        print(f"Searching for JSON files in: {base_dir}")
        
        # 디렉토리가 존재하는지 확인
        if not os.path.exists(base_dir):
            print(f"Error: Directory '{base_dir}' does not exist")
            return json_files
        
        # 디렉토리를 탐색하며 JSON 파일 찾기
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(root, file)
                    json_files.append(json_path)
                    print(f"Found JSON file: {json_path}")
                    
        if not json_files:
            print("No JSON files found in the directory")
        else:
            print(f"Found {len(json_files)} JSON files")
            
        return json_files
    
    except Exception as e:
        print(f"Error finding JSON files: {e}")
        return json_files

# Read JSON data from a file
def read_json_file(file_path: str) -> dict:
    """JSON 파일을 읽어 데이터를 반환합니다.

    Args:
        file_path (str): 읽을 JSON 파일의 경로

    Returns:
        dict: JSON 파일의 데이터를 담은 딕셔너리
    """
    """
    JSON 파일을 읽어서 데이터를 반환합니다.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print(f"Successfully read JSON data from: {file_path}")
            print(f"Data contains {len(data)} entries")
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"JSON decode error in {file_path}: {e}")
        return {}
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return {}

# Test function to process a JSON file
def test_process_json() -> None:
    """saveData 폴더의 첫 번째 JSON 파일을 찾아 처리하는 테스트 함수입니다."""
    global allData
    
    # JSON 파일 찾기
    json_files = find_json_files()
    
    if not json_files:
        print("No JSON files found to process")
        return
    
    # 첫 번째 파일 처리
    first_file = json_files[0]
    print(f"\n처리할 파일: {first_file}")
    
    # JSON 파일 읽기
    json_data = read_json_file(first_file)
    
    if not json_data:
        print("No data found in the JSON file")
        return
    
    # allData에 데이터 저장
    allData = json_data
    
    # 데이터 샘플 출력
    sample_keys = list(json_data.keys())[:3]  # 처음 3개 키만 출력
    print("\n데이터 샘플:")
    for key in sample_keys:
        print(f"Key: {key}")
        print(f"Data: {json_data[key]}")
        print("-" * 50)
    
    print(f"총 {len(json_data)}개의 데이터가 처리되었습니다.")

def mainProcess():
    global allData
    print("3. Fist SLM Proccessing")
    ollama_Low_analysis()
    save_to_result("result_First.json")
       
    print("4. Second SLM Proccessing")
    ollama_Low_analysis()  
    save_to_result("result_Second.json")

def init():
    global collection, client, embedding_function, llm, chain   
    # Initialize ChromaDB client and embedding function, Model is Linq-AI-Research/Linq-Embed-Mistral(https://huggingface.co/Linq-AI-Research/Linq-Embed-Mistral)
    print("1. Initialize ChromaDB client and embedding function")
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="Linq-AI-Research/Linq-Embed-Mistral"
    )
    
    # chromadb settings, path is default "chroma_db" and persist is default True
    client = chromadb.PersistentClient(path="./chroma_db_origin")
    # Check if the collection exists
    existing_collections = [col.name for col in client.list_collections()]
    if "email_data" in existing_collections:
        while True:
            user_input = input("The 'email_data' collection already exists. Do you want to reset (delete and recreate) it? (y/N): ").strip().lower()
            if user_input in ("y", "yes"):
                client.delete_collection("email_data")
                collection = client.get_or_create_collection(name="email_data")
                print("Collection 'email_data' has been reset.")
                break
            elif user_input in ("n", "no", ""):
                collection = client.get_or_create_collection(name="email_data")
                print("Using existing 'email_data' collection.")
                break
            else:
                print("Please enter 'y' or 'n'.")
    else:
        collection = client.get_or_create_collection(name="email_data")
    
    # Let user choose Ollama model from installed models
    print("2. 사용할 Ollama 모델 선택")
    selected_model = choose_ollama_model()
    
    # Initialize Ollama LLM with the selected model
    llm = OllamaLLM(model=selected_model)           
    
    # Default Template for the LLMChain
    template = """You are a spam email classifier. Analyze the provided similar emails and the target email to make your decision.

    Similar emails from database:
    {message}

    Rules:
    - If you find emails with identical/very similar subjects, follow the majority classification among those examples
    - Prioritize consistency for similar cases
    - Respond with: classification,reliability
    - Classification: True (spam) or False (not spam)
    - Reliability: confidence level as percentage (0-100)

    Format: True,85 or False,92"""
    
    prompt = PromptTemplate(template=template, input_variables=["message"])
    chain = prompt | llm
    
# Default arguments for the script
def parse_arguments():
    """
    Parse command line arguments.
    """
    # ArgumentParser 객체 생성
    parser = argparse.ArgumentParser(description='LLM-based analysis scripts for spam checking')
    
    # 선택적 인자 추가 (필수 아님)
    parser.add_argument('-f', '--filename', help='Filename to analyze (optional, defaults to searching in saveData folder)')
    
    # 인자 파싱
    args = parser.parse_args()
    return args

if __name__ == "__main__":   
    init()
    
    args = parse_arguments()
    
    # 파일명이 지정된 경우 CSV 처리
    if args.filename and args.filename.endswith('.csv'):
        print(f"\nCSV 파일 처리: {args.filename}")
        read_file_convert_json(args.filename)
    else:
        # 파일명이 지정되지 않은 경우 saveData 폴더에서 JSON 파일 찾기
        print("\n테스트 모드: saveData 폴더에서 첫 번째 JSON 파일 처리")
        test_process_json()
    
    # 메인 프로세스 실행
    if allData:
        print("\n메인 프로세스 실행")
        mainProcess()
    else:
        print("\n처리할 데이터가 없습니다. 프로그램을 종료합니다.")

