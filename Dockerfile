# Python 3.11 기반 이미지 사용
FROM python:3.11-slim

# 작업 디렉터리 생성
WORKDIR /app

# 의존성 파일 복사
COPY requirements.txt .

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 포트 노출 (FastAPI용 8000, MLflow용 5000)
EXPOSE 8000 5000

# FastAPI + MLflow 서버 동시 실행
CMD ["bash", "-c", "mlflow server --host 0.0.0.0 --port 5000 & uvicorn main:app --host 0.0.0.0 --port 8000"]
