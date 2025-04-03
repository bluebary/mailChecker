# MailCheck-Ollama

이메일 스팸 탐지를 위한 로컬 LLM 기반 분석 도구입니다. Ollama를 활용하여 이메일의 제목, 발신자, 도메인 등의 패턴을 분석하고 스팸 여부를 판별합니다.

## 주요 기능

- Ollama 기반 LLM을 활용한 이메일 스팸 탐지
- ChromaDB를 이용한 스팸 메일 패턴 저장 및 분석
- 이메일 제목, 발신자, 도메인 기반 그룹화 분석
- 다단계 분석 프로세스 (초기 분석 → 통계 분석 → 심화 분석)

## 사전 요구사항

- Python 3.8 이상
- [Ollama](https://ollama.com/) 설치 및 실행
- gemma3:12b 모델 설치 (`ollama pull gemma3:12b`)

## 설치 방법

1. 저장소 클론
   ```bash
   git clone https://github.com/your-username/mailcheck-ollama.git
   cd mailcheck-ollama
   ```

2. 필요한 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```

## 사용 방법

1. 분석할 이메일 데이터를 CSV 형식으로 준비합니다. 아래와 같은 포맷을 따라야 합니다:
   ```
   발신자이메일,발신자도메인,수신자이메일,제목
   example@domain.com,domain.com,recipient@example.com,이메일 제목
   ```

2. 스크립트 실행
   ```bash
   python mailCheck_ollama_ori.py -f 파일명.csv
   ```

3. 결과는 `result_First.json`과 `result_Second.json` 파일로 저장됩니다.

## 분석 프로세스

1. 초기화: ChromaDB 클라이언트 및 임베딩 함수 설정
2. 데이터 로드: CSV 데이터를 읽고 딕셔너리 형태로 변환
3. 1차 분석: Ollama LLM을 사용한 기본 스팸 탐지
4. 통계 분석: 제목, 발신자, 도메인 기반 그룹화 및 패턴 분석
5. 2차 분석: 패턴 분석 결과를 반영한 심화 스팸 탐지

## 데이터 구조

분석 결과는 다음과 같은 JSON 형식으로 저장됩니다:
```json
{
    "spam-해시값": {
        "spam": true/false,
        "sender": "발신자 이메일",
        "sender_domain": "발신자 도메인",
        "receiver": "수신자 이메일",
        "subject": "이메일 제목",
        "duration": 분석에 소요된 시간(초)
    }
}
```

## 임베딩 모델

기본 임베딩 모델로 [Linq-AI-Research/Linq-Embed-Mistral](https://huggingface.co/Linq-AI-Research/Linq-Embed-Mistral)을 사용합니다.

## 라이센스

[MIT 라이센스](LICENSE)