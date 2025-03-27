from ollama import chat
import json, hashlib, os

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

def check_spam(sender, recipient, subject):
    # Ollama 모델과 메시지 설정
    # testMSG = f'Check if this email is spam and reply with True or False, Sender: {sender}, Recipient: {recipient}, Subject: {subject}'
    # print(subject)
    response = chat(
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

if __name__ == "__main__":
    data = read_file_with_tabs()
    allData = dict()
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, "result.json")
    for txt in data:
        answer = check_spam(txt[4], txt[5], txt[6])
        key = hashlib.sha256(txt[4].encode()).hexdigest()
        allData[key] = answer
    with open(path, "w", encoding="utf-8") as file: 
        json.dump(allData, file, ensure_ascii=False, indent=4)