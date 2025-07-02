# 📧 이메일 분석 대시보드

Streamlit 기반의 이메일 스팸 분석 결과 시각화 및 관리 도구입니다. 다양한 LLM 모델의 이메일 스팸 탐지 결과를 한눈에 보고, 사용자가 직접 스팸 여부를 검증할 수 있는 웹 인터페이스를 제공합니다.

## 🌟 주요 기능

- **다양한 데이터 뷰**: 모든 결과, 모델별 통계, 특정 모델 결과 조회
- **데이터 시각화**: 스팸 분류 비교, 성능 지표 분석, Confusion Matrix
- **대화형 편집**: 실시간 데이터 편집 및 사용자 검증 결과 입력
- **검색 및 필터링**: 키워드 검색과 스팸 상태별 필터링

## 🛠️ 사전 요구사항

- Python 3.8 이상
- SQLite 데이터베이스 (email_analysis.db)

## 📦 설치 방법

### Linux/macOS
```bash
# 저장소 클론
git clone <repository-url>
cd mailChecker

# 가상환경 생성 (권장)
python3 -m venv venv
source venv/bin/activate

# 필수 패키지 설치
pip install -r requirements.txt
```

### Windows
```cmd
# 저장소 클론
git clone <repository-url>
cd mailChecker

# 가상환경 생성 (권장)
python -m venv venv
venv\Scripts\activate

# 필수 패키지 설치
pip install -r requirements.txt
```

## 🚀 실행 방법

### Streamlit 대시보드 실행

#### Linux/MacOS
```bash
streamlit run app.py
```

#### Windows
```cmd
streamlit run app.py
```

실행 후 웹 브라우저에서 `http://localhost:8501`로 접속하면 대시보드를 이용할 수 있습니다.

## 📋 사용 방법

### 1. 데이터 뷰 선택
- 좌측 사이드바에서 원하는 분석 뷰를 선택합니다
- **모든 결과**: 전체 데이터 overview
- **모델별 통계**: 모델 성능 비교 및 통계
- **[모델명] 결과**: 특정 모델의 상세 결과

### 2. 데이터 탐색
- **검색**: 키워드로 이메일 내용 검색
- **필터링**: 스팸/정상/미확인 상태별 필터링
- **페이지네이션**: 대용량 데이터를 페이지별로 탐색

### 3. 데이터 편집
- `human_verified_spam` 컬럼의 체크박스를 통해 사용자 검증 결과 입력
- 변경사항은 실시간으로 데이터베이스에 저장
- 수정 완료 시 성공 메시지 표시

### 4. 시각화 분석
- **모델별 통계** 뷰에서 다양한 차트와 그래프 확인
- Confusion Matrix로 모델 성능 평가
- 성능 지표 비교를 통한 최적 모델 선택

## 🗂️ 데이터베이스 구조

### 주요 테이블
- `emails`: 기본 이메일 정보 (ID, 발신자, 도메인, 수신자, 제목)
- `all_results`: 모든 모델의 통합 분석 결과
- `[model_name]`: 개별 모델별 분석 결과

### 주요 컬럼
- `first_spam`, `second_spam`: 1차/2차 분석 결과
- `first_reliability`, `second_reliability`: 분석 신뢰도
- `first_duration`, `second_duration`: 분석 소요 시간
- `human_verified_spam`: 사용자 검증 결과

## 📈 시각화 기능

### 모델별 통계 뷰
1. **분류 통계**: 모델별 평균 신뢰도, 분석 시간, 처리 건수
2. **Confusion Matrix**: 모델별 분류 성능 히트맵
3. **성능 지표 비교**: Accuracy, Precision, Recall, F1-Score 비교

### 일반 결과 뷰
1. **스팸 분류 비교**: 1차/2차/사용자 확인 결과 비교 차트
2. **분석 신뢰도/시간 분포**: 히스토그램, 박스플롯, 산점도
3. **발신자 분포**: 상위 발신자별 스팸 분류 현황

## 🔧 설정 및 로그

### 로그 파일
- `streamlit.log`: Streamlit 애플리케이션 로그
- `dbmanager.log`: 데이터베이스 관리 로그

### 로그 레벨
- INFO: 일반적인 실행 정보
- WARNING: 경고 메시지 (콘솔 출력)
- ERROR: 오류 메시지

## 🔒 데이터 안전성

- **트랜잭션 기반 업데이트**: 데이터 일관성 보장
- **자동 롤백**: 오류 발생 시 자동 복구
- **입력 검증**: 유효하지 않은 데이터 입력 방지

**참고**: 이 대시보드는 기존에 수집된 이메일 분석 데이터를 시각화하고 관리하는 도구입니다. 새로운 이메일 분석을 수행하려면 별도의 분석 스크립트를 실행해야 합니다.
