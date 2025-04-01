import json, hashlib, os, ollama, chromadb, tempfile, shutil, paramiko
import pandas as pd
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import ollama
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from pathlib import Path

def read_file_with_tabs():
    filtered_list = []
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, "Data")
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                # with open(file_path, 'r', encoding="utf-8", errors="ignore") as file:
                with open(file_path, 'r', encoding="utf-8", errors="ignore") as file:
                    lines = file.readlines()
                    lines = [line.strip() for line in lines]
                    for element in lines:
                        element = element.split("\t")
                        if len(element) > 5 and element[0].split(" ")[1] == "SUCCESS":
                            element[0] = element[0].replace(" SUCCESS", "")
                            element[6] = element[6].lstrip()
                            filtered_list.append(element)

    return filtered_list

def ssh_connect_and_merge_files(remote_dir_path, local_output_path):
    """
    .env 파일에서 SSH 접속 정보(IP, ID, PW)를 가져와 SSH 접속 후
    원격 서버의 지정된 경로에 있는 모든 파일을 하나의 파일로 통합하여 로컬에 저장합니다.
    
    Args:
        remote_dir_path (str): 원격 서버에서 통합할 파일들이 있는 디렉토리 경로
        local_output_path (str): 통합된 파일을 저장할 로컬 경로
    
    Returns:
        bool: 성공 여부
    """
    # .env 파일에서 환경 변수 로드
    load_dotenv()
    
    # SSH 접속 정보 가져오기
    ssh_host = os.getenv('SSH_HOST')
    ssh_username = os.getenv('SSH_USERNAME')
    ssh_password = os.getenv('SSH_PASSWORD')
    
    # 필수 환경 변수 확인
    if not all([ssh_host, ssh_username, ssh_password]):
        print("ERROR: SSH_HOST, SSH_USERNAME, SSH_PASSWORD 환경 변수가 모두 필요합니다.")
        return False
    
    # 임시 디렉토리 생성
    temp_dir = tempfile.mkdtemp()
    try:
        # SSH 클라이언트 생성 및 접속
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        print(f"{ssh_username}@{ssh_host}로 SSH 접속 시도 중...")
        ssh_client.connect(
            hostname=ssh_host,
            username=ssh_username,
            password=ssh_password
        )
        print("SSH 접속 성공!")
        
        # SFTP 세션 생성
        sftp = ssh_client.open_sftp()
        
        # 원격 디렉토리 내 파일 목록 가져오기
        try:
            remote_files = sftp.listdir(remote_dir_path)
        except IOError as e:
            print(f"ERROR: 원격 디렉토리 접근 실패: {e}")
            return False
        
        if not remote_files:
            print(f"WARNING: 원격 디렉토리 '{remote_dir_path}'에 파일이 없습니다.")
            return False
        
        print(f"총 {len(remote_files)}개 파일을 찾았습니다. 파일 다운로드 중...")
        
        # 임시 디렉토리에 파일 다운로드
        downloaded_files = []
        for filename in remote_files:
            remote_file_path = os.path.join(remote_dir_path, filename)
            local_file_path = os.path.join(temp_dir, filename)
            
            # 디렉토리인지 확인
            try:
                if stat_info := sftp.stat(remote_file_path):
                    if stat_info.st_mode & 0o40000:  # 디렉토리인 경우 건너뛰기
                        print(f"'{filename}'은 디렉토리이므로 건너뜁니다.")
                        continue
            except:
                print(f"'{filename}' 상태 확인 중 오류가 발생했습니다. 건너뜁니다.")
                continue
            
            try:
                sftp.get(remote_file_path, local_file_path)
                downloaded_files.append(local_file_path)
                print(f"'{filename}' 다운로드 완료")
            except Exception as e:
                print(f"'{filename}' 다운로드 실패: {e}")
        
        # 파일 통합
        if not downloaded_files:
            print("다운로드된 파일이 없습니다.")
            return False
        
        print(f"총 {len(downloaded_files)}개 파일 통합 중...")
        
        # 출력 파일 디렉토리 확인 및 생성
        output_dir = os.path.dirname(local_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 파일 통합 (파일명 순으로 정렬)
        with open(local_output_path, 'wb') as outfile:
            for file_path in sorted(downloaded_files):
                with open(file_path, 'rb') as infile:
                    shutil.copyfileobj(infile, outfile)
                outfile.write(b'\n')  # 각 파일 사이에 줄바꿈 추가
        
        print(f"파일 통합 완료. 저장 위치: {local_output_path}")
        return True
    
    except Exception as e:
        print(f"오류 발생: {e}")
        return False
    
    finally:
        # 연결 종료 및 임시 파일 정리
        if 'sftp' in locals():
            sftp.close()
        if 'ssh_client' in locals():
            ssh_client.close()
        shutil.rmtree(temp_dir)

# dataframe 데이터를 통계 기반으로 분석
def statistics_FirstData(df):
    # "spam"이 True인 데이터만 필터링
    spam_df = df[df['spam'] == True]

    # 1. 동일한 sender를 가진 스팸 메일 그룹화
    sender_groups = spam_df.groupby('sender')
    # 각 sender별로 2개 이상의 메일이 있는 경우만 필터링
    sender_multiple = sender_groups.filter(lambda x: len(x) >= 2)
    # 결과 확인 (sender별로 정렬)
    sender_result = sender_multiple.sort_values('sender')

    # 2. 동일한 sender_domain을 가진 스팸 메일 그룹화
    domain_groups = spam_df.groupby('sender_domain')
    # 각 domain별로 2개 이상의 메일이 있는 경우만 필터링
    domain_multiple = domain_groups.filter(lambda x: len(x) >= 2)
    # 결과 확인 (domain별로 정렬)
    domain_result = domain_multiple.sort_values('sender_domain')

    # 3. 동일한 subject를 가진 스팸 메일 그룹화
    subject_groups = spam_df.groupby('subject')
    # 각 subject별로 2개 이상의 메일이 있는 경우만 필터링 (비어있지 않은 subject만)
    subject_multiple = subject_groups.filter(lambda x: len(x) >= 2 and x['subject'].iloc[0] != '')
    # 결과 확인 (subject별로 정렬)
    subject_result = subject_multiple.sort_values('subject')

    # 통계 출력
    print(f"총 스팸 메일: {len(spam_df)}")
    print(f"동일 발신자가 보낸 스팸 메일: {len(sender_multiple)}")
    print(f"동일 도메인에서 온 스팸 메일: {len(domain_multiple)}")
    print(f"동일 제목의 스팸 메일: {len(subject_multiple)}")

    # 각 그룹별 상위 발신자/도메인/제목 확인
    print("\n가장 많은 스팸을 보낸 발신자:")
    print(sender_groups.size().sort_values(ascending=False).head(10))

    print("\n가장 많은 스팸이 발생한 도메인:")
    print(domain_groups.size().sort_values(ascending=False).head(10))

    print("\n가장 많이 사용된 스팸 제목:")
    print(subject_groups.size().sort_values(ascending=False).head(10))

    # 특정 발신자의 모든 메일 확인 예시
    # specific_sender = "example@domain.com"
    # print(f"\n발신자 {specific_sender}의 모든 스팸 메일:")
    # print(spam_df[spam_df['sender'] == specific_sender])

# ollama 모델을 사용하여 스팸 여부 확인
def check_spam(sender, recipient, subject):
    # Ollama 모델과 메시지 설정
    # testMSG = f'Check if this email is spam and reply with True or False, Sender: {sender}, Recipient: {recipient}, Subject: {subject}'
    # print(subject)
    response = ollama.chat(
        model='gemma3:12b',
        messages=[
            {'role': 'system', 'content': 'You are an AI designed to detect spam emails.'},
            {'role': 'user', 'content': f'Check if this email is spam and reply with only True or False, Sender: {sender}, Recipient: {recipient}, Subject: {subject}'}
        ]
    )
    
    # Ollama 응답을 기반으로 스팸 여부 확인
    
    data = {
        "spam": response['message']['content'].strip(),
        "sender": sender,
        "sender_domain": sender.split("@")[1],
        "receiver": recipient,
        "subject": subject,
        "duration": response['total_duration'] / 1_000_000_000
    }
    return data

def main():
    data = read_file_with_tabs()
    allData = dict()
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, "result.json")
    for txt in data:
        answer = check_spam(txt[4], txt[5], txt[6])
        key = hashlib.sha256(txt[4].encode()).hexdigest()
        allData[key] = answer

    data = json.loads(allData)
    with open(path, "w", encoding="utf-8") as file: 
        json.dump(allData, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()

