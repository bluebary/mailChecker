from ollama import chat
import os, ipaddress, tempfile, shutil, paramiko, argparse
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv

"""
DataStructure
[0] = True/False (Success/Fail)
[1] = Date
[2] = Time
[3] = IP
[4] = Account
"""
networks = []
single_ips = []
model="Gemma3:12b"

# 날짜 리스트 생성
def generate_date_list(start_date, end_date):
    """
    "YYYYMMDD" 형식의 시작 날짜와 종료 날짜를 입력받아 
    그 사이의 모든 날짜를 "YYYYMMDD" 형식으로 리스트에 저장합니다.
    
    Args:
        start_date (str): "YYYYMMDD" 형식의 시작 날짜
        end_date (str): "YYYYMMDD" 형식의 종료 날짜
        
    Returns:
        list: "YYYYMMDD" 형식의 날짜 리스트
    
    Raises:
        ValueError: 날짜 형식이 올바르지 않은 경우
        ValueError: 시작 날짜가 종료 날짜보다 늦은 경우
    """
    # 날짜 형식 검증
    try:
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
    except ValueError:
        raise ValueError("날짜 형식이 올바르지 않습니다. 'YYYYMMDD' 형식으로 입력해주세요.")
    
    # 시작 날짜가 종료 날짜보다 늦은 경우
    if start > end:
        raise ValueError("시작 날짜는 종료 날짜보다 이전이어야 합니다.")
    
    # 날짜 리스트 생성
    date_list = []
    current_date = start
    
    while current_date <= end:
        date_list.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)
    
    return date_list

# SSH 접속 후 원격 서버의 여러 경로에 있는 모든 파일을 하나의 파일로 통합하여 로컬에 저장하는 함수
def ssh_connect_and_merge_files(base_remote_dir, sub_dirs, base_local_output_dir):
    """
    .env 파일에서 SSH 접속 정보(IP, ID, PW)를 가져와 SSH 접속 후
    원격 서버의 기본 경로 아래 여러 하위 경로에 있는 모든 파일을 하나의 파일로 통합하여 로컬에 저장합니다.
    
    Args:
        base_remote_dir (str): 원격 서버의 기본 디렉토리 경로 (리눅스 형식)
        sub_dirs (str or list): 기본 경로 아래의 하위 디렉토리(들)
                               단일 경로(문자열) 또는 경로 목록(리스트)
        local_output_path (str, optional): 통합된 파일을 저장할 특정 로컬 경로. 지정하면 모든 파일을 하나로 통합
        base_local_output_dir (str, optional): 통합된 파일을 저장할 로컬 기본 경로 (윈도우 형식)
                                           None인 경우 현재 작업 디렉토리 사용. local_output_path가 지정되면 무시됨.
        filename_suffix (str, optional): 파일명에 추가할 접미사 (기본값: None). local_output_path가 지정되면 무시됨.
    
    Returns:
        list: 생성된 로컬 파일 경로 목록. local_output_path가 지정된 경우 단일 요소 리스트 반환.
    """
    # 단일 경로를 리스트로 변환
    if isinstance(sub_dirs, str):
        sub_dirs = [sub_dirs]
        
    # 전체 원격 경로 생성 (리눅스 스타일 경로)
    remote_dir_paths = [f"{base_remote_dir}/{sub_dir}" for sub_dir in sub_dirs]
    
    # 기본 로컬 경로가 지정되지 않은 경우 현재 작업 디렉토리 사용
    if base_local_output_dir is None:
        base_local_output_dir = os.getcwd()
    
    # 생성된 로컬 파일 경로 목록
    created_files = []
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
        
        # 임시 디렉토리에 파일 다운로드
        total_files_found = 0
        
        # 각 원격 디렉토리에서 파일 처리
        for dir_index, remote_dir_path in enumerate(remote_dir_paths):
            downloaded_files = []
            print(f"\n[{dir_index + 1}/{len(remote_dir_paths)}] '{remote_dir_path}' 경로 처리 중...")
            
            # 원격 디렉토리 내 파일 목록 가져오기
            try:
                remote_files = sftp.listdir(remote_dir_path)
            except IOError as e:
                print(f"WARNING: 원격 디렉토리 '{remote_dir_path}' 접근 실패: {e}")
                continue
            
            if not remote_files:
                print(f"INFO: 원격 디렉토리 '{remote_dir_path}'에 파일이 없습니다.")
                continue
            
            print(f"총 {len(remote_files)}개 파일을 찾았습니다. 파일 다운로드 중...")
            total_files_found += len(remote_files)
            
            # 디렉토리 구분을 위한 접두사 생성 (파일 이름 충돌 방지)
            dir_prefix = f"dir{dir_index}_"
            
            for filename in remote_files:
                remote_file_path = f"{remote_dir_path}/{filename}"
                # 파일 이름 충돌 방지를 위해 접두사 추가
                local_filename = dir_prefix + filename
                local_file_path = os.path.join(temp_dir, local_filename)
                
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
                    print(f"'{filename}' 다운로드 완료 -> '{local_filename}'")
                except Exception as e:
                    print(f"'{filename}' 다운로드 실패: {e}")
               
            # 파일 통합
            if not downloaded_files:
                print("다운로드된 파일이 없습니다.")
                return False
        
            print(f"총 {len(downloaded_files)}개 파일 통합 중...")
        
            # 출력 파일 디렉토리 확인 및 생성
            output_dir = os.path.dirname(base_local_output_dir)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 파일 통합 (파일명 순으로 정렬)
            output_filename = os.path.join(base_local_output_dir, f"{sub_dirs[dir_index]}.log")
            with open(output_filename, 'wb') as outfile:
                for file_path in sorted(downloaded_files):
                    with open(file_path, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile)
                    outfile.write(b'\n')  # 각 파일 사이에 줄바꿈 추가
        
        print(f"파일 통합 완료. 저장 위치: {base_local_output_dir}")
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

#  로그 내 메일 정보를 지정 데이터 structure로 변환
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

# 화이트 리스트 로긷 함수
def load_whitelist(whitelist_file):
    """
    화이트리스트 파일을 읽고 네트워크 및 IP 목록을 구성합니다.
    
    Args:
        whitelist_file (str): 화이트리스트 파일 경로
    """
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, whitelist_file)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                whitelist_entries = [line.strip() for line in f if line.strip()]
        
            for entry in whitelist_entries:
                try:
                    # CIDR 표기법인 경우 (/가 포함된 경우)
                    if '/' in entry:
                        network = ipaddress.ip_network(entry)
                        networks.append(network)
                    # 단일 IP인 경우
                    else:
                        ip = ipaddress.ip_address(entry)
                        single_ips.append(ip)
                except ValueError as e:
                    # 잘못된 형식의 항목은 경고 후 무시
                    print(f"화이트리스트 항목 '{entry}' 처리 중 오류: {e}")
                    continue       
        except FileNotFoundError:
            print(f"화이트리스트 파일을 찾을 수 없음: {whitelist_file}")

# IP 주소가 화이트리스트에 포함되는지 확인하는 함수
def is_ip_in_whitelist(ip_to_check):
    """
    IP 주소가 화이트리스트에 포함되는지 확인합니다.
        
    Args:
        ip_to_check (str): 확인할 IP 주소 (예: '192.168.8.5')
         
    Returns:
        bool: IP 주소가 화이트리스트에 포함되면 True, 아니면 False
    """
    try:
        # 확인할 IP를 ipaddress 객체로 변환
        ip_obj = ipaddress.ip_address(ip_to_check)
            
        # 단일 IP 목록에서 확인
        if ip_obj in single_ips:
            return True
            
        # 네트워크 목록에서 확인
        for network in networks:
            if ip_obj in network:
                return True
            
        # 어느 목록에도 없음
        return False
            
    except ValueError as e:
        # IP 주소 형식이 잘못됨
        return False

# POP3 로그 파일 읽기
def read_file_POP3(start_date, end_date):
    filtered_list = []
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, "POP3")
    if os.path.exists(path):
        # start_date와 end_date를 list 객체로 변환하고 그 list를 저장
        date_list = generate_date_list(start_date, end_date)
        ssh_connect_and_merge_files(
            base_remote_dir="/Data1/mobigen/CrediMail/Data/LOG/POP3",
            sub_dirs=date_list,
            base_local_output_dir=path
            # base_local_output_dir=os.path.join(path, datetime.now().strftime("%Y%m%d") + ".log")
        )
        # for date in date_list:
        #     date_add = os.path.join(path, date + ".log")
        #     ssh_connect_and_merge_files(
        #         remote_dir_paths=f"/Data1/mobigen/CrediMail/Data/LOG/POP3/{date}",
        #         local_output_path=date_add
        #     )
            
        # 파일을 읽어와서 처리
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding="utf-8", errors="ignore") as file:
                    lines = file.readlines()
                    lines = [line.strip() for line in lines]
                    for element in lines:
                        element = element.split("\t")
                        if len(element) > 5 and is_ip_in_whitelist(element[3]) == False:
                            chkData = element[0].split(" ")[1]
                            if chkData == "SUCCESS":
                                rtData = preProcessData_File(element, 0, os.path.splitext(filename)[0], True)
                                if rtData != None:
                                    filtered_list.append(rtData)
                            elif chkData == "USERDB_FAILED" or chkData == "PASSWD_FAILED":
                                rtData = preProcessData_File(element, 1, os.path.splitext(filename)[0], False)
                                if rtData != None:
                                    filtered_list.append(rtData)
    return filtered_list

# IMAP 로그 파일 읽기
def read_file_with_tabs_IMAP(start_date, end_date):
    filtered_list = []
    current_dir = os.path.dirname(__file__)
    path = os.path.join(current_dir, "IMAP")
    if os.path.exists(path):
        
        # start_date와 end_date를 list 객체로 변환하고 그 list를 저장
        date_list = generate_date_list(start_date, end_date)
        ssh_connect_and_merge_files(
            base_remote_dir="/Data1/mobigen/CrediMail/Data/LOG/IMAP",
            sub_dirs=date_list,
            base_local_output_dir=path
            # local_output_path=os.path.join(path, datetime.now().strftime("%Y%m%d") + ".log")
        )
        
        # for date in date_list:
        #     ssh_connect_and_merge_files(
        #         remote_dir_path=f"/Data1/mobigen/CrediMail/Data/LOG/IMAP/{date}",
        #         local_output_path = os.path.join(path, date + ".log")
        #     )
            
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding="utf-8", errors="ignore") as file:
                    lines = file.readlines()
                    lines = [line.strip() for line in lines]
                    for element in lines:
                        element = element.split("\t")
                        if len(element) > 5  and is_ip_in_whitelist(element[3]) == False:
                            if element[0].split(" ")[1] == "IMAP_LOGIN_SUCCESS":
                                rtData = preProcessData_File(element, 1, os.path.splitext(filename)[0], True)
                                if rtData != None : filtered_list.append(rtData)
                            elif element[0].split(" ")[1] == "IMAP_PASSWD_FAIL":
                                rtData = preProcessData_File(element, 1, os.path.splitext(filename)[0], False)
                                if rtData != None : filtered_list.append(rtData)
    return filtered_list

# 파일에서 읽어온 데이터를 list로 반환, type은 0: POP3, 1: IMAP
def preProcessData_File(data, type, date, loginCHK):
    if type == 0:
        rtnData = []
        rtnData.append(loginCHK)
        rtnData.append(date)
        rtnData.append(data[0].split(" ")[0])
        rtnData.append(data[3])
        rtnData.append(data[2].split("@")[0])
        return rtnData
    elif type == 1:
        rtnData = []
        rtnData.append(loginCHK)
        rtnData.append(date)
        rtnData.append(data[0].split(" ")[0])
        rtnData.append(data[3])
        rtnData.append(data[2].split("@")[0])
        return rtnData
    else:
        return None

# DataFrame으로 변환
def preProcessData_DF(data):
    df = pd.DataFrame(data, columns=["success", "date", "time", "ip", "user"])
    return df

# DataFrame내에 추가적으로 필요한 전처리 데이터 추가
def preProcessData_LVL1(df):
    df['datetime_str'] = df['date'] + ' ' + df['time']
    df['timestamp'] = pd.to_datetime(df['datetime_str'], format='%Y%m%d %H:%M:%S')
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=월요일, 6=일요일
    df['weekend'] = df['day_of_week'].isin([5, 6])  # 주말 여부
    df['business_hours'] = df['hour'].between(9, 18) # 업무시간 여부
    return df
# 3. 기본 통계 분석
def basic_statistics(df):
    """
    기본적인 통계 정보를 계산하는 함수
    """
    stats = {}
    
    # 전체 로그인 시도 횟수
    stats['total_attempts'] = len(df)
    
    # 성공/실패 비율
    if 'success' in df.columns:
        stats['success_rate'] = df['success'].mean() * 100
        stats['success_count'] = df['success'].sum()
        stats['failure_count'] = len(df) - df['success'].sum()
    
    # 고유 IP 수
    if 'ip' in df.columns:
        stats['unique_ips'] = df['ip'].nunique()
    
    # 고유 사용자 수
    if 'user' in df.columns:
        stats['unique_users'] = df['user'].nunique()
    
    return stats

# 4. 시간별 분석
def time_analysis(df):
    """
    시간대별 로그인 패턴을 분석하는 함수
    """
    if 'hour' not in df.columns:
        raise ValueError("시간 정보가 전처리되지 않았습니다.")
    
    # 시간대별 로그인 시도 횟수
    hourly_attempts = df.groupby('hour').size()
    
    # 시간대별 성공/실패 비율
    if 'success' in df.columns:
        hourly_success_rate = df.groupby('hour')['success'].mean() * 100
    
    # 요일별 로그인 시도 횟수
    if 'weekday' in df.columns:
        weekday_attempts = df.groupby('weekday').size()
    
    return {
        'hourly_attempts': hourly_attempts,
        'hourly_success_rate': hourly_success_rate,
        'weekday_attempts': weekday_attempts if 'weekday' in df.columns else None
    }

def get_subnet(ip, prefix_len=24):
    try:
        return str(ipaddress.ip_network(f"{ip}/{prefix_len}", strict=False))
    except:
        return None

# 5. IP 분석
def ip_analysis(df):
    """
    IP 주소 관련 분석을 수행하는 함수
    """
    if 'ip' not in df.columns:
        raise ValueError("IP 정보가 없습니다.")
    
    # IP별 로그인 시도 횟수
    ip_attempts = df.groupby('ip').size().sort_values(ascending=False)
    
    # IP별 성공/실패 비율
    if 'success' in df.columns:
        ip_success_rate = df.groupby('ip')['success'].mean() * 100
    
    # IP 서브넷 분석   
    df['subnet'] = df['ip'].apply(lambda x: get_subnet(x))
    subnet_attempts = df.groupby('subnet').size().sort_values(ascending=False)
    
    return {
        'ip_attempts': ip_attempts,
        'ip_success_rate': ip_success_rate,
        'subnet_attempts': subnet_attempts
    }

# 6. 사용자 분석
def user_analysis(df):
    """
    사용자 계정 관련 분석을 수행하는 함수
    """
    if 'user' not in df.columns:
        raise ValueError("사용자 정보가 없습니다.")
    
    # 사용자별 로그인 시도 횟수
    user_attempts = df.groupby('user').size().sort_values(ascending=False)
    
    # 사용자별 성공/실패 비율
    # if 'success' in df.columns:
    #     user_success_rate = df.groupby('user')['success'].mean() * 100
    if 'success' in df.columns:
        success_df = df[df['success'] == True]  # 또는 df[~df['success']]
        user_success_rate = success_df.groupby('user').size().sort_values(ascending=False)
    
    # 사용자별 로그인 IP 다양성
    user_ip_diversity = df.groupby('user')['ip'].nunique().sort_values(ascending=False)
    
    # 실패 로그인이 많은 사용자
    if 'success' in df.columns:
        failed_df = df[df['success'] == False]
        user_failures = failed_df.groupby('user').size().sort_values(ascending=False)
    
    return {
        'user_attempts': user_attempts,
        'user_success_rate': user_success_rate,
        'user_ip_diversity': user_ip_diversity,
        'user_failures': user_failures if 'success' in df.columns else None
    }

# 7. 이상 탐지 분석
def anomaly_detection(df):
    """
    이상 로그인 행동을 탐지하는 함수
    """
    anomalies = {}
    
    # 동일 IP에서 다수의 실패 로그인
    if 'success' in df.columns and 'ip' in df.columns:
        failed_df = df[df['success'] == False]
        ip_failure_count = failed_df.groupby('ip').size()
        suspicious_ips = ip_failure_count[ip_failure_count > 5].sort_values(ascending=False)
        anomalies['suspicious_ips'] = suspicious_ips
    
    # 짧은 시간 내 다수의 로그인 시도
    if 'timestamp' in df.columns and 'ip' in df.columns:
        df_sorted = df.sort_values('timestamp')
        ip_timestamps = defaultdict(list)
        
        for _, row in df_sorted.iterrows():
            ip_timestamps[row['ip']].append(row['timestamp'])
        
        rapid_attempts = {}
        for ip, timestamps in ip_timestamps.items():
            if len(timestamps) < 5:
                continue
                
            for i in range(len(timestamps) - 4):
                time_diff = (timestamps[i+4] - timestamps[i]).total_seconds()
                if time_diff < 300:  # 5분 이내 5회 이상 시도
                    rapid_attempts[ip] = time_diff
                    break
        
        anomalies['rapid_attempts'] = rapid_attempts
    
    # 비정상 시간대 로그인
    if 'hour' in df.columns and 'success' in df.columns and 'user' in df.columns:
        # 사용자별 평상시 로그인 시간대 파악 (성공한 로그인만)
        success_df = df[df['success'] == 1]
        
        user_hour_patterns = {}
        for user, group in success_df.groupby('user'):
            hour_counts = group['hour'].value_counts()
            if len(hour_counts) > 0:
                user_hour_patterns[user] = set(hour_counts.index)
        
        # 비정상 시간대 로그인 탐지
        odd_hour_logins = []
        for _, row in df.iterrows():
            user = row['user']
            hour = row['hour']
            
            if user in user_hour_patterns and hour not in user_hour_patterns[user]:
                odd_hour_logins.append({
                    'user': user,
                    'timestamp': row['timestamp'],
                    'ip': row['ip'],
                    'status': 'SUCCESS' if row['success'] == 1 else 'FAILURE'
                })
        
        anomalies['odd_hour_logins'] = odd_hour_logins
    
    return anomalies

# 9. Ollama를 활용한 분석 함수
def analyze_with_ollama(df, anomalies):
    """
    Ollama를 사용하여 로그인 데이터와 이상 행동을 분석하는 함수
    """
    # 기본 통계 및 이상 행동 요약
    summary = f"""
    Login data analysis summary:
    - Total login attempts: {len(df)} 건
    - Successful logins: {df['success'].sum()} 건 ({df['success'].mean()*100:.2f}%)
    - Failed logins: {len(df) - df['success'].sum()} 건 ({(1-df['success'].mean())*100:.2f}%)
    - Unique IP addresses: {df['ip'].nunique()} 개
    - Unique user accounts: {df['user'].nunique()} 개
    
    Anomaly summary:
    - Suspicious IPs (multiple failed logins): {len(anomalies['suspicious_ips'])} 개
    - IPs with short time mass attempts: {len(anomalies['rapid_attempts'])} 개
    - Unusual time zone logins: {len(anomalies['odd_hour_logins'])} 건
    """
    
    prompt = f"""
    Based on the following login data analysis, please analyze it from a security perspective and provide recommendations in Korean:
    
    {summary}
    
    Top 5 suspicious IPs (ordered by number of failed logins):
    {anomalies['suspicious_ips'].head().to_string()}
    
    Examples of abnormal time of day logins (up to 5):
    {pd.DataFrame(anomalies['odd_hour_logins'][:5]).to_string() if anomalies['odd_hour_logins'] else "없음"}
    
    Please answer the following questions
    1. do you see any security threats in this login pattern?
    2. which IPs or accounts are most suspicious?
    """
    
    try:
        # Ollama 모델 호출
        response = chat(model=model, messages=[
            {'role': 'system', 'content': 'You\'re a cybersecurity expert. You analyze login data to detect security threats and provide recommendations.'},
            {'role': 'user', 'content': prompt}
        ])
        
        return response['message']['content']
    except Exception as e:
        return f"Ollama 분석 중 오류 발생: {str(e)}"

# 10. 세부 로그 분석 함수
def analyze_detailed_logs(df, suspicious_ips=None, suspicious_users=None):
    """
    의심스러운 IP나 사용자의 세부 로그를 분석하는 함수
    """
    detailed_logs = {}
    
    # 의심스러운 IP의 상세 로그 추출
    if suspicious_ips is not None and len(suspicious_ips) > 0:
        suspicious_ip_logs = {}
        for ip in suspicious_ips:
            ip_df = df[df['ip'] == ip].sort_values('timestamp')
            suspicious_ip_logs[ip] = ip_df
        detailed_logs['suspicious_ip_logs'] = suspicious_ip_logs

    # # 의심스러운 IP의 성공 로그 추출
    # if suspicious_ips is not None and len(suspicious_ips) > 0:
    #     suspicious_success_ip_logs = {}
    #     for ip in suspicious_ips:
    #         # ip_df = df[df['ip'] == ip].sort_values('timestamp')
    #         ip_successful_df = df[(df['ip'] == ip) & (df['success'] == True)].sort_values('timestamp')
    #         suspicious_success_ip_logs[ip] = ip_successful_df
    #     detailed_logs['suspicious_success_ip_logs'] = suspicious_success_ip_logs
    
    # 의심스러운 사용자의 상세 로그 추출
    if suspicious_users is not None and len(suspicious_users) > 0:
        suspicious_user_logs = {}
        for user in suspicious_users:
            user_df = df[df['user'] == user].sort_values('timestamp')
            suspicious_user_logs[user] = user_df
        detailed_logs['suspicious_user_logs'] = suspicious_user_logs
    
    # 실패 후 성공 패턴 분석 (브루트포스 공격 탐지)
    if 'success' in df.columns:
        brute_force_candidates = []
        for user, user_df in df.groupby('user'):
            user_df = user_df.sort_values('timestamp')
            if len(user_df) < 3:
                continue
                
            # 연속된 실패 후 성공 패턴 찾기
            statuses = user_df['success'].tolist()
            timestamps = user_df['timestamp'].tolist()
            ips = user_df['ip'].tolist()
            
            for i in range(len(statuses) - 3):
                # 3번 이상 연속 실패 후 성공 패턴 감지
                if statuses[i:i+3] == [0, 0, 0] and i+3 < len(statuses) and statuses[i+3] == 1:
                    # 시간 간격이 짧은 경우만 고려 (10분 이내)
                    time_diff = (timestamps[i+3] - timestamps[i]).total_seconds()
                    if time_diff < 600:
                        brute_force_candidates.append({
                            'user': user,
                            'start_time': timestamps[i],
                            'success_time': timestamps[i+3],
                            'time_diff_seconds': time_diff,
                            'ip': ips[i+3]
                        })
        
        detailed_logs['brute_force_candidates'] = brute_force_candidates
    
    return detailed_logs

def detailed_report(df, stats, time_data, ip_data, user_data, anomalies, detailed_logs, ollama_analysis, output_file):
# def detailed_report(df, stats, time_data, ip_data, user_data, anomalies, detailed_logs, output_file):
    """
    모든 분석 내용을 포함한 상세 보고서 생성
    
    Args:
        df: 분석 데이터프레임
        stats: 기본 통계 정보 딕셔너리
        time_data: 시간 분석 결과 딕셔너리
        ip_data: IP 분석 결과 딕셔너리
        user_data: 사용자 분석 결과 딕셔너리
        anomalies: 이상 탐지 결과 딕셔너리
        detailed_logs: 세부 로그 분석 결과 딕셔너리
        ollama_analysis: Ollama를 통한 분석 결과 문자열
        output_file: 출력 파일 경로 (기본값: "login_analysis_report.md")
        
    Returns:
        str: 보고서 내용 문자열
    """
    
    if df is None or df.empty:
        raise ValueError("데이터가 비어 있습니다. 먼저 데이터를 로드해야 합니다.")    
   
    # 보고서 생성
    report = [
        "# 로그인 분석 보고서",
        f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 요약",
        f"총 로그인 시도: {stats['total_attempts']}",
        f"성공한 로그인: {stats['success_count']} ({stats['success_rate']:.2f}%)",
        f"실패한 로그인: {stats['failure_count']} ({100 - stats['success_rate']:.2f}%)",
        f"고유 IP 주소 수: {stats['unique_ips']}",
        f"고유 사용자 계정 수: {stats['unique_users']}",
        "",
        "## 2. 시간별 분석",
        "### 시간대별 로그인 시도",
        "```",
        time_data['hourly_attempts'].to_string(),
        "```",
        "",
        "### 시간대별 성공률",
        "```",
        time_data['hourly_success_rate'].to_string(),
        "```",
        "",
    ]
    
    # 요일별 분석 (있는 경우)
    if 'weekday_attempts' in time_data and time_data['weekday_attempts'] is not None:
        report.extend([
            "### 요일별 로그인 시도",
            "```",
            time_data['weekday_attempts'].to_string(),
            "```",
            "",
        ])
    
    report.extend([
        "## 3. IP 분석",
        "### IP별 로그인 시도",
        "```",
        ip_data['ip_attempts'].head(20).to_string(),
        "```",
        "",
        "### IP별 성공률",
        "```",
        ip_data['ip_success_rate'].head(20).to_string(),
        "```",
        "",
        "### 서브넷별 로그인 시도",
        "```",
        ip_data['subnet_attempts'].head(20).to_string(),
        "```",
        "",
        "## 4. 사용자 분석",
        "### 사용자별 로그인 시도",
        "```",
        user_data['user_attempts'].head(20).to_string(),
        "```",
        "",
        "### 사용자별 성공률",
        "```",
        user_data['user_success_rate'].head(20).to_string(),
        "```",
        "",
        "### 사용자별 IP 다양성",
        "```",
        user_data['user_ip_diversity'].head(20).to_string(),
        "```",
        "",
    ])
    
    # 실패 로그인이 많은 사용자 (있는 경우)
    if 'user_failures' in user_data and user_data['user_failures'] is not None:
        report.extend([
            "### 실패 로그인이 많은 사용자",
            "```",
            user_data['user_failures'].head(20).to_string(),
            "```",
            "",
        ])
    
    report.extend([
        "## 5. 이상 탐지",
        "### 의심스러운 IP (다수 실패 로그인)",
        f"총 {len(anomalies['suspicious_ips'])}개의 의심스러운 IP가 탐지되었습니다.",
        "```",
        anomalies['suspicious_ips'].head(20).to_string() if len(anomalies['suspicious_ips']) > 0 else "없음",
        "```",
        "",
        "### 단시간 대량 시도 IP",
        f"총 {len(anomalies['rapid_attempts'])}개의 IP에서 단시간 내 다수의 시도가 탐지되었습니다.",
        "```",
    ])
    
    # 단시간 대량 시도 IP 정보
    if len(anomalies['rapid_attempts']) > 0:
        rapid_attempts_list = [f"IP: {ip}, 5회 시도 소요 시간: {seconds}초" for ip, seconds in list(anomalies['rapid_attempts'].items())[:10]]
        report.append("\n".join(rapid_attempts_list))
    else:
        report.append("없음")
    
    report.extend([
        "```",
        "",
        "### 비정상 시간대 로그인",
        f"총 {len(anomalies['odd_hour_logins'])}건의 비정상 시간대 로그인이 탐지되었습니다.",
        "```",
    ])
    
    # 비정상 시간대 로그인 정보
    if len(anomalies['odd_hour_logins']) > 0:
        odd_hour_df = pd.DataFrame(anomalies['odd_hour_logins'][:20])
        report.append(odd_hour_df.to_string())
    else:
        report.append("없음")
    
    report.extend([
        "```",
        "",
    ])
    
    # 세부 로그 분석 결과 추가
    report.append("## 6. 세부 로그 분석")
    
    # 브루트포스 공격 의심 패턴
    report.append("### 브루트포스 공격 의심 패턴")
    if 'brute_force_candidates' in detailed_logs and len(detailed_logs['brute_force_candidates']) > 0:
        report.append(f"총 {len(detailed_logs['brute_force_candidates'])}건의 브루트포스 의심 패턴이 탐지되었습니다.")
        report.append("```")
        brute_force_df = pd.DataFrame(detailed_logs['brute_force_candidates'][:20])
        report.append(brute_force_df.to_string())
        report.append("```")
    else:
        report.append("브루트포스 공격 의심 패턴이 탐지되지 않았습니다.")
    
    report.append("")
    
    # 의심스러운 IP의 상세 로그
    report.append("### 의심스러운 IP의 상세 로그")
    if 'suspicious_ip_logs' in detailed_logs and len(detailed_logs['suspicious_ip_logs']) > 0:
        report.append(f"총 {len(detailed_logs['suspicious_ip_logs'])}개의 의심스러운 IP에 대한 상세 로그입니다.")
        
        for i, (ip, ip_df) in enumerate(list(detailed_logs['suspicious_ip_logs'].items())[:5]):
            report.extend([
                f"#### IP: {ip}",
                "```",
                ip_df.to_string(),
                "```",
                ""
            ])
    
    for ip, ip_df in detailed_logs['suspicious_ip_logs'].items():
        successful_logins_by_suspicious_ips = {}
        # 해당 IP의 성공한 로그인만 필터링
        successful_df = ip_df[ip_df['success'] == True]  # 또는 ip_df[ip_df['success']]
    
        # 성공한 로그인이 있는 경우에만 결과에 추가
        if not successful_df.empty:
            successful_logins_by_suspicious_ips[ip] = successful_df

        # 결과 출력
        print(f"총 {len(successful_logins_by_suspicious_ips)}개의 의심스러운 IP에서 성공한 로그인이 발견되었습니다.")

        for ip, success_df in successful_logins_by_suspicious_ips.items():
            print(f"\nIP: {ip}에서 {len(success_df)}건의 성공한 로그인:")
            print(success_df)

        else:
            report.append("의심스러운 IP의 로그인 성공 로그가 없습니다.")
    
    report.append("")
    
    # 의심스러운 사용자의 상세 로그
    report.append("### 의심스러운 사용자의 상세 로그")
    if 'suspicious_user_logs' in detailed_logs and len(detailed_logs['suspicious_user_logs']) > 0:
        report.append(f"총 {len(detailed_logs['suspicious_user_logs'])}개의 의심스러운 사용자에 대한 상세 로그입니다.")
        
        for i, (user, user_df) in enumerate(list(detailed_logs['suspicious_user_logs'].items())[:20]):
            report.extend([
                f"#### 사용자: {user}",
                "```",
                user_df.to_string(),
                "```",
                ""
            ])
    else:
        report.append("의심스러운 사용자의 상세 로그가 없습니다.")
    
    report.append("")
    
    # Ollama 분석 결과 추가
    report.extend([
        "## 7. Ollama 분석 결과",
        "```",
        ollama_analysis,
        "```",
        "",
        "## 8. 결론 및 권장사항",
        "이 보고서는 로그인 데이터 분석 결과를 제공합니다. 위에서 탐지된 의심스러운 활동에 대해 다음과 같은 조치를 권장합니다:",
        "",
        "1. 의심스러운 IP에서의 접근을 차단하거나 모니터링을 강화하세요.",
        "2. 브루트포스 공격이 의심되는 계정에 대해 추가 인증 요소를 적용하세요.",
        "3. 비정상 시간대에 로그인하는 사용자에 대해 추가 확인을 수행하세요.",
        "4. 로그인 실패 횟수 제한 및 일시적 계정 잠금 정책을 구현하세요.",
        "5. 로그인 시도에 대한 실시간 모니터링 시스템을 구축하세요.",
        "",
        "더 자세한 분석이나 추가 질문이 있으시면 보안팀에 문의하세요.",
    ])
    
    # 보고서를 파일로 저장
    report_text = "\n".join(report)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"상세 보고서가 '{output_file}'에 저장되었습니다.")
    
    return report_text

def main(args):
    """
    메인 실행 함수
    """
    print("로그인 데이터 분석을 시작합니다...")
    load_whitelist("whitelist.conf")
    # 데이터 로드 및 전처리
    # df = load_data(file_path, file_type)
    # 시작일 종료일 설정
    startDate = args.st if args.st else None
    endDate = args.et if args.et else None
    
    # 시작일과 종료일이 주어지지 않은 경우, 기본값으로 어제 날짜부터 오늘까지로 설정
    if startDate is None:
        startDate = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    if endDate is None:
        endDate = datetime.now().strftime("%Y%m%d")
        
    listData = read_file_POP3(startDate, endDate) + read_file_with_tabs_IMAP(startDate, endDate)
    df = preProcessData_DF(listData)
    df = preProcessData_LVL1(df)
    
    # 통계 분석
    stats = basic_statistics(df)
    time_data = time_analysis(df)
    ip_data = ip_analysis(df)
    user_data = user_analysis(df)
    
    # 이상 탐지
    anomalies = anomaly_detection(df)
   
    # 의심스러운 IP와 사용자 추출
    suspicious_ips = anomalies['suspicious_ips'].head(10).index.tolist() if len(anomalies['suspicious_ips']) > 0 else []
    suspicious_users = []
    if 'user_failures' in user_data and user_data['user_failures'] is not None:
        suspicious_users = user_data['user_failures'].head(5).index.tolist()
    
    # 세부 로그 분석
    detailed_logs = analyze_detailed_logs(df, suspicious_ips, suspicious_users)
     
    # 결과 출력
    print("\n===== 기본 통계 =====")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n===== 이상 행동 탐지 =====")
    print(f"의심스러운 IP (다수 실패 로그인): {len(anomalies['suspicious_ips'])} 개")
    if len(anomalies['suspicious_ips']) > 0:
        print(anomalies['suspicious_ips'].head())
    
    print(f"\n단시간 대량 시도 IP: {len(anomalies['rapid_attempts'])} 개")
    for ip, seconds in list(anomalies['rapid_attempts'].items())[:5]:
        print(f"IP: {ip}, 5회 시도 소요 시간: {seconds}초")
    
    print(f"\n비정상 시간대 로그인: {len(anomalies['odd_hour_logins'])} 건")
    for login in anomalies['odd_hour_logins'][:5]:
        print(f"사용자: {login['user']}, 시간: {login['timestamp']}, IP: {login['ip']}, 상태: {login['status']}")
    
    # print(user_data['user_success_rate'].head(10).to_string())


    # Ollama 분석
    print("Ollama를 활용한 분석을 수행합니다...")
    ollama_analysis = analyze_with_ollama(df, anomalies)

    print("\n===== Ollama 분석 결과 =====")
    print(ollama_analysis)
    today = datetime.now().strftime("%Y%m%d")
    # 결과 내보내기
    detailed_report(df, stats, time_data, ip_data, user_data, anomalies, detailed_logs, ollama_analysis, output_file=f"login_analysis_report_{today}.md")
    # detailed_report(df, stats, time_data, ip_data, user_data, anomalies, detailed_logs, output_file=f"login_analysis_report.md")
    
    print("\n분석이 완료되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IMAP/POP3 Login Brute Force Detection')
    
    # 선택적 인자 추가
    parser.add_argument('--st', help='Start Date ex): YYYYMMDD')
    parser.add_argument('--et', help='End Date ex): YYYYMMDD')
    
    main(parser.parse_args())