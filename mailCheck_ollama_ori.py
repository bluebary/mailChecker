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
    # return dict(itertools.islice(rtn_dict.items(), 500))
    return rtn_dict

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

def get_similar_emails(query_text, k=3):
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
def check_spam(data):
    global chain
    
    similar_emails = get_similar_emails(data['subject'])
    message = f"""Use the following context from previous spam emails:
    {similar_emails}

    Determine whether the following email is spam:
    Sender: {data['sender']}, Receiver: {data['receiver']}, Subject: {data['subject']}

    If it is spam, respond with True; otherwise, respond with False.
    """
    
    try:
        # LangChain's chain.run is deprecated, so use chain.invoke
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
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="email_data")
    
    # llm Model is Ollama gemma3:12b (https://ollama.com/models/gemma3)
    llm = OllamaLLM(model="gemma3:12b")           
    
    # Default Template for the LLMChain
    template = """You are an AI designed to detect spam emails.
    
    The following are similar emails from our database:
    {message}
    
    When making your decision, if you find identical or very similar email subjects in the retrieved examples, base your judgment on the most frequent classification (spam/not spam) among those similar examples. Always prioritize consistency for similar cases.
    
    Is this email spam? Reply with only True or False."""
    
    prompt = PromptTemplate(template=template, input_variables=["message"])   
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
    init()
    read_file_convert_json('sampleData.csv')
    mainProcess()
    
    