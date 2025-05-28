pip install -r requirements.txt
mlflow server --host 0.0.0.0
uvicorn main:app --reload
