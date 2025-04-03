import json, hashlib, os, chromadb, argparse, chromadb, csv, time, itertools
import pandas as pd
from chromadb.utils import embedding_functions
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


collection, client, embedding_function, llm, chain = None, None, None, None, None
allData = dict()

# Read CSV analytics data, convert it to a dictionary, and save it to allData
def read_file_convert_json(filename):
    global allData  
    csvData = read_file_csv(filename)
    allData = convert_to_dict(csvData)
    
def read_file_csv(filename):
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
def convert_to_dict(csv):
    rtn_dict = dict()
    """Data Structure
    {
        "spam": True/False,
        "sender": Mail Sender,
        "sender_domain": Mail Sender Domain,
        "receiver": Mail Receiver,
        "subject": Mail Subject,
        "duration": Duration of the llm processing time
    }
    """
    for data in csv:
        key = f"spam-{hashlib.sha256(data[3].strip().encode()).hexdigest()}"
        rtn_data = {
            "spam": False,
            "sender": data[0],
            "sender_domain": data[1],
            "receiver": data[2],
            "subject": data[3].strip(),
            "duration": 0
        }
        rtn_dict[key] = rtn_data
    # 임시로 500개 데이터만
    return dict(itertools.islice(rtn_dict.items(), 500))
    # return rtn_dict

# Mail Subject based spam detection
def subject_group(df):
    global collection
    # 1. Grouping spam mail with the same subject
    subject_groups = df.groupby('subject')
    
    # Filter only if there is more than one mail for each subject (only non-empty subjects)
    subject_multiple = subject_groups.filter(lambda x: len(x) >= 2 and x['subject'].iloc[0] != '')
    
    # View results (sorted by subject)
    subject_result = subject_multiple.sort_values('subject')
    for row in subject_result.iterrows():
        collection.add(
            documents=[row['subject']],
            metadatas=[{
                "sender": row['sender'], 
                "sender_domain": row['sender_domain'],
                "is_spam": True}],
            # hash the subject to create a unique ID for each spam email
            ids=[f"spam-{hashlib.sha256(row['subject'].encode()).hexdigest()}"]
        )

# Mail Sender based spam detection
def sender_group(df):
    global collection
    # Grouping spam mail with the same sender
    sender_groups = df.groupby('sender')
    
    # Filter only if there is more than one mail for each sender
    sender_multiple = sender_groups.filter(lambda x: len(x) >= 2)
    
    # View results (sorted by sender)
    sender_result = sender_multiple.sort_values('sender')
    # print(len(sender_result))
    for idx, row in sender_result.iterrows():
        collection.add(
            documents=[row['subject']],
            metadatas=[{
                "sender": row['sender'], 
                "sender_domain": row['sender_domain'],
                "is_spam": True}],
            # hash the subject to create a unique ID for each spam email
            ids=[f"spam-{hashlib.sha256(row['subject'].encode()).hexdigest()}"]
        )
    
# Mail Sender Domain based spam detection
def sender_domain_group(df):
    global collection
    # Grouping spam mail with the same sender_domain
    domain_groups = df.groupby('sender_domain')
    
    # Filter only if there is more than one mail for each domain
    domain_multiple = domain_groups.filter(lambda x: len(x) >= 2)
    
    # View results (sorted by domain)
    domain_result = domain_multiple.sort_values('sender_domain')
    for idx, row in domain_result.iterrows():
        collection.add(
            documents=[row['subject']],
            metadatas=[{
                "sender": row['sender'], 
                "sender_domain": row['sender_domain'],
                "is_spam": True}],
            # hash the subject to create a unique ID for each spam email
            ids=[f"spam-{hashlib.sha256(row['subject'].encode()).hexdigest()}"]
        )

# Using the Ollama model to First check if the email is spam
def ollama_Low_analysis():
    global allData
    allCount = len(allData)
    chkCnt = 0
    for key, item in allData.items():
        chkCnt = chkCnt + 1
        start_time = time.time()
        rst = check_spam(item)
        end_time = time.time()
        item["spam"] = rst
        item["duration"] = round(end_time - start_time, 1)
        percent = chkCnt / allCount * 100
        bar_length = 40
        filled_length = int(bar_length * chkCnt // allCount)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"\r[{bar}] {chkCnt}/{allCount} ({percent:.1f}%)", end='', flush=True)

# SpamMail Low level analysis by statistics and Save the ChromaDB
def statistics_Data_Analysis():
    global allData
    df = pd.DataFrame.from_dict(allData, orient='index')
    # sender_groups = df.groupby('sender')
    sender_group(df)
    sender_domain_group(df)
    subject_group(df)

# ollama model query to check if the email is spam
def check_spam(data):
    global chain
    message = f"Check if this email is spam and reply with only True or False, Sender: {data['sender']}, Recipient: {data['receiver']}, Subject: {data['subject']}"
    
    try:
        # LangChain's chain.run is deprecated, so use chain.invoke
        # response = chain.run(message=message)
        response = chain.invoke({"message": message})
               
        # The response is assumed to be a True/False string, but it can be incorrect, so filter it with the
        response = response.strip()
        return True if "true" in response.lower() else False
    except Exception as e:
        print(f"Error in check_spam: {e}")
        return False

# Save the result to a JSON file
def save_to_result(filename):
    global allData
    current_dir = os.path.dirname(__file__)
    if os.path.exists(current_dir):
        path = os.path.join(current_dir, filename)
        with open(path, "w", encoding="utf-8") as file: 
            json.dump(allData, file, ensure_ascii=False, indent=4)

def mainProcess():
    global allData
    print("3. Fist SLM Proccessing")
    ollama_Low_analysis()
    save_to_result("result_First.json")
    
    print("4. Statistics Analysis")
    statistics_Data_Analysis()
    
    print("5. Second SLM Proccessing")
    ollama_Low_analysis()  
    save_to_result("result_Second.json")

def init():
    global collection, client, embedding_function, llm, chain, allData   
    # Initialize ChromaDB client and embedding function, Model is Linq-AI-Research/Linq-Embed-Mistral(https://huggingface.co/Linq-AI-Research/Linq-Embed-Mistral)
    print("1. Initialize ChromaDB client and embedding function")
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="Linq-AI-Research/Linq-Embed-Mistral"
    )
    
    # chromadb settings, path is default "chroma_db" and persist is default True
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="email_data")
    
    # llm Model is Ollama gemma3:12b (https://ollama.com/models/gemma3)
    llm = OllamaLLM(model="gemma3:12b")
    
    # Default Template for spam detection
    template = """You are an AI designed to detect spam emails.
    
    {message}
    
    Is this email spam? Reply with only True or False."""
    prompt = PromptTemplate(template=template, input_variables=["message"])
    
    # Chain deprecation warning
    # chain = LLMChain(llm=llm, prompt=prompt)
    chain = prompt | llm
    
# Default arguments for the script
def parse_arguments():
    """
    Parse command line arguments.
    """
    # ArgumentParser 객체 생성
    parser = argparse.ArgumentParser(description='LLM-based analysis scripts for spam checking')
    
    # 위치 인자 추가 (필수)
    parser.add_argument('-f', '--filename', help='Filename to analyze', required=True)
    
    # 인자 파싱
    args = parser.parse_args()
    return args

if __name__ == "__main__":   
    read_file_convert_json('sampleData.csv')
    init()
    
    # 실제 오픈시 사용
    # args = parse_arguments()
    # read_file_convert_json(args.filename)
    mainProcess()
    
    