import json, hashlib, os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

collection, client, embedding_function, llm, chain, allData = None, None, None, None, None, None

def read_file_with_tabs():
    global allData
    print("2. Read file with tabs")
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
                            key = hashlib.sha256(element[4].encode()).hexdigest()
                            allData[key] = convert_to_dict(element)
                            # element[0] = element[0].replace(" SUCCESS", "")
                            # element[6] = element[6].lstrip()
                            # filtered_list.append(element)

    return filtered_list

def convert_to_dict(data):
    rtn_data = {
        "spam": False,
        "sender": data[4],
        "sender_domain": data[4].split("@")[1],
        "receiver": data[5],
        "subject": data[6].strip(),
        "duration": 0
    }
    return rtn_data

# def loadJsonFile():
#     current_dir = os.path.dirname(__file__)
#     path = os.path.join(current_dir, "result.json")
#     with open(path, "r", encoding="utf-8") as file:
#         data = json.load(file)
#     return data

def subject_group(df):
    global collection
    # 1. 동일한 subject를 가진 스팸 메일 그룹화
    subject_groups = df.groupby('subject')
    # 각 subject별로 2개 이상의 메일이 있는 경우만 필터링 (비어있지 않은 subject만)
    subject_multiple = subject_groups.filter(lambda x: len(x) >= 2 and x['subject'].iloc[0] != '')
    # 결과 확인 (subject별로 정렬)
    subject_result = subject_multiple.sort_values('subject')
    for idx, row in subject_result.iterrows():
        collection.add(
            documents=[row['subject']],
            metadatas=[{
                "sender": row['sender'], 
                "sender_domain": row['sender_domain'],
                "is_spam": True}],
            ids=[f"spam-{hashlib.sha256(row['subject'].encode()).hexdigest()}"]
        )

def sender_group(df):
    global collection
    # 1. 동일한 sender를 가진 스팸 메일 그룹화
    sender_groups = df.groupby('sender')
    # 각 sender별로 2개 이상의 메일이 있는 경우만 필터링
    sender_multiple = sender_groups.filter(lambda x: len(x) >= 2)
    # 결과 확인 (sender별로 정렬)
    sender_result = sender_multiple.sort_values('sender')
    print(len(sender_result))
    for idx, row in sender_result.iterrows():
        collection.add(
            documents=[row['subject']],
            metadatas=[{
                "sender": row['sender'], 
                "sender_domain": row['sender_domain'],
                "is_spam": True}],
            ids=[f"spam-{hashlib.sha256(row['subject'].encode()).hexdigest()}"]
        )
    
def sender_domain_group(df):
    global collection
    # 2. 동일한 sender_domain을 가진 스팸 메일 그룹화
    domain_groups = df.groupby('sender_domain')
    # 각 domain별로 2개 이상의 메일이 있는 경우만 필터링
    domain_multiple = domain_groups.filter(lambda x: len(x) >= 2)
    # 결과 확인 (domain별로 정렬)
    domain_result = domain_multiple.sort_values('sender_domain')
    for idx, row in domain_result.iterrows():
        collection.add(
            documents=[row['subject']],
            metadatas=[{
                "sender": row['sender'], 
                "sender_domain": row['sender_domain'],
                "is_spam": True}],
            ids=[f"spam-{hashlib.sha256(row['subject'].encode()).hexdigest()}"]
        )

def ollama_Low_analysis():
    global allData
    for key, item in allData.items():
        item["spam"] = check_spam(item)

def statistics_Data_Analysis():
    global allData
    df = pd.DataFrame.from_dict(allData, orient='index')
    # sender_groups = df.groupby('sender')
    sender_group(df)
    sender_domain_group(df)
    subject_group(df)

def mainProcess():
    global allData
    print("3. Fist SLM Proccessing")
    ollama_Low_analysis()
    df = pd.DataFrame.from_dict(allData, orient='index')
    # sender_groups = df.groupby('sender')
    print("4. Statistics Analysis")
    sender_group(df)
    sender_domain_group(df)
    subject_group(df)
    print("5. Second SLM Proccessing")
    ollama_Low_analysis(allData)
    data = json.loads(allData)

    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, "result.json")
    with open(path, "w", encoding="utf-8") as file: 
        print("6. Save result.json")
        json.dump(allData, file, ensure_ascii=False, indent=4)

def check_spam(data):
    global chain
    # Ollama 응답을 기반으로 스팸 여부 확인
    message = f"Check if this email is spam and reply with only True or False, Sender: {data['sender']}, Recipient: {data['receiver']}, Subject: {data['subject']}"
    response =  chain.run(message)
    chain = LLMChain(llm=llm, prompt=message)
    return response['message']['content'].strip()

    # data = {
    #     "spam": response['message']['content'].strip(),
    #     "sender": sender,
    #     "sender_domain": sender.split("@")[1],
    #     "receiver": recipient,
    #     "subject": subject,
    #     "duration": response['total_duration'] / 1_000_000_000
    # }
    # return data

def init():
    global collection, client, embedding_function, llm, chain, allData
    # Initialize ChromaDB client and embedding function
    print("1. Initialize ChromaDB client and embedding function")
    allData = dict()
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="Linq-AI-Research/Linq-Embed-Mistral"
    )
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="email_data")
    llm = Ollama(model="gemma3:12b")
    template = "You are an AI designed to detect spam emails."
    prompt = PromptTemplate(template=template, input_variables=["message"])
    chain = LLMChain(llm=llm, prompt=prompt)


if __name__ == "__main__":
    init()
    read_file_with_tabs()
    mainProcess()
    
    