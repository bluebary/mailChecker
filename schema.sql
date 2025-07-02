-- 이메일 분석 DB 스키마 정의 (db_sqlite_converter.py 기반)

-- 1. 메인 이메일 테이블
CREATE TABLE IF NOT EXISTS emails (
    id TEXT PRIMARY KEY,
    sender TEXT,
    sender_domain TEXT,
    receiver TEXT,
    subject TEXT
);

-- 2. 모델별 결과 테이블 (모델명에 따라 동적으로 생성)
-- 예시: 모델명이 model1일 경우
CREATE TABLE IF NOT EXISTS model1 (
    id TEXT PRIMARY KEY,
    fisrt_spam BOOLEAN,
    first_duration REAL,
    first_reliability REAL,
    second_spam BOOLEAN,
    second_duration REAL,
    second_reliability REAL,
    human_verified_spam BOOLEAN,
    FOREIGN KEY (id) REFERENCES emails(id)
);

-- 3. 모든 모델의 결과를 통합하는 테이블
CREATE TABLE IF NOT EXISTS all_results (
    id TEXT,
    model TEXT,
    first_spam BOOLEAN,
    first_duration REAL,
    first_reliability REAL,
    second_spam BOOLEAN,
    second_duration REAL,
    second_reliability REAL,
    human_verified_spam BOOLEAN,
    PRIMARY KEY (id, model),
    FOREIGN KEY (id) REFERENCES emails(id)
);

-- 4. 모델별 first/second 결과 뷰 (모델명, 분석유형에 따라 동적으로 생성)
-- 예시: 모델명이 model1일 경우
CREATE VIEW IF NOT EXISTS model1_view AS
SELECT 
    e.id, e.sender, e.sender_domain, e.receiver, e.subject, r.first_spam, r.first_duration, r.first_reliability, 
    r.second_spam, r.second_duration, r.second_reliability, r.human_verified_spam
FROM 
    emails e
JOIN 
    model1_first r ON e.id = r.id;

-- 5. 통합 결과 뷰
CREATE VIEW IF NOT EXISTS all_results_view AS
SELECT 
    e.id, e.sender, e.sender_domain, e.receiver, e.subject, r.first_spam, r.first_duration, r.first_reliability, 
    r.second_spam, r.second_duration, r.second_reliability, r.human_verified_spam
FROM 
    emails e
JOIN 
    all_results r ON e.id = r.id;

-- 6. 모델별 통계 뷰
CREATE VIEW IF NOT EXISTS model_stats_view AS
SELECT 
    model, 
    analysis_type,
    COUNT(*) as total_emails,
    AVG(reliability) as avg_reliability,
    AVG(duration) as avg_duration,
    MIN(reliability) as min_reliability,
    MAX(reliability) as max_reliability
FROM 
    all_results
GROUP BY 
    model, analysis_type;

-- 실제 모델명에 따라 위 테이블/뷰는 반복 생성됨. 필요시 Python 코드에서 동적으로 생성하세요.
