pip install -r requirements.txt

echo "MLflow 서버 시작 중..."
nohup mlflow server --host 0.0.0.0 > mlflow.log 2>&1 &
mlflow_pid=$!

# MLflow 서버가 시작될 때까지 대기
echo "MLflow 서버가 준비될 때까지 대기 중..."
sleep 10  # 서버 시작을 위한 기본 대기 시간

# MLflow 프로세스가 실행 중인지 확인
if ps -p $mlflow_pid > /dev/null; then
  echo "MLflow 서버가 성공적으로 시작되었습니다. (PID: $mlflow_pid)"
  
  # uvicorn 서버 시작
  echo "Uvicorn 서버 시작 중..."
  nohup uvicorn main:app --reload > uvicorn.log 2>&1 &
  uvicorn_pid=$!
  echo "Uvicorn 서버가 백그라운드에서 실행 중입니다. (PID: $uvicorn_pid)"
else
  echo "MLflow 서버 시작에 실패했습니다."
  exit 1
fi
