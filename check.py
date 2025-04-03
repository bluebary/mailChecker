import json, hashlib, os, ollama, chromadb, tempfile, shutil, paramiko
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from langchain_ollama import OllamaEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

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
    response = OllamaLLM.chat(
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

