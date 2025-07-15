# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, HTTPException, Header
from datetime import datetime
from typing import Optional
from pydantic import BaseModel
import jwt
import json
import base64

# 사용자 정의 모듈
from modules.memory_manager import memory_manager
from chains.router import ChainRouter

# 랭체인 트래킹
import mlflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")

mlflow.set_experiment("langchain")

# Enable MLflow tracing
# mlflow.langchain.autolog()  # Commented out due to compatibility issues

def log_to_mlflow(question: str, response: str, start_time: datetime):
    try:
        with mlflow.start_run():
            mlflow.set_tag("endpoint", "/async/chat")
            mlflow.log_param("question", question)
            mlflow.log_text(response, "response.txt")
            mlflow.log_metric("response_length", len(response))
            mlflow.log_metric("duration_seconds", (datetime.now() - start_time).total_seconds())
    except Exception as e:
        # 에러가 나도 main 흐름에는 영향 안 줌
        print(f"[MLflow Logging Error] {e}")

# API 키 정보 로드
load_dotenv()

# Initialize chain router
chain_router = ChainRouter(model="gpt-4o-mini")

# For backward compatibility, create a function that uses the new router
def extract_user_id_from_token(access_token: str) -> str:
    """JWT 토큰에서 userId를 추출합니다."""
    try:
        # JWT 토큰을 '.'로 분할하여 payload 부분 추출
        parts = access_token.split('.')
        if len(parts) != 3:
            raise ValueError("Invalid JWT token format")
        
        # payload 부분 (두 번째 부분) 디코딩
        payload_encoded = parts[1]
        # Base64 패딩 추가 (필요시)
        payload_encoded += '=' * (4 - len(payload_encoded) % 4)
        
        # Base64 디코딩
        payload_decoded = base64.b64decode(payload_encoded)
        payload_json = json.loads(payload_decoded)
        
        # userId 추출
        user_id = payload_json.get('userId')
        if user_id is None:
            raise ValueError("userId not found in token")
        
        return str(user_id)
    
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

def create_full_chain_with_memory(thread_id: str):
    """메모리가 포함된 전체 체인을 생성합니다."""
    return chain_router.create_full_chain_with_memory(thread_id)

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    response: str
    timestamp: datetime
    thread_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    access_token: Optional[str] = Header(None, alias="access_token")
):
    """채팅 엔드포인트 - POST 방식"""
    start_time = datetime.now()
    
    try:
        # JWT 토큰에서 userId 추출하여 thread_id로 사용
        if access_token:
            thread_id = extract_user_id_from_token(access_token)
        else:
            raise HTTPException(status_code=401, detail="Access token required")
        
        # 스레드별 체인 생성
        full_chain = create_full_chain_with_memory(thread_id)
        
        # 체인 실행
        response = await full_chain.ainvoke({"question": request.question})
        
        # 응답 텍스트 추출
        response_text = str(response)
        
        # 로깅은 백그라운드로 넘김
        background_tasks.add_task(log_to_mlflow, request.question, response_text, start_time)
        
        return ChatResponse(
            response=response_text,
            timestamp=datetime.now(),
            thread_id=thread_id
        )
        
    except Exception as e:
        print(f"[채팅 처리 오류] {e}")
        error_response = f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"
        raise HTTPException(status_code=500, detail=error_response)

# 세션 리셋 엔드포인트
@app.post("/reset")
async def reset_session(access_token: Optional[str] = Header(None, alias="access_token")):
    """대화 메모리를 리셋합니다."""
    try:
        # JWT 토큰에서 userId 추출하여 thread_id로 사용
        if access_token:
            thread_id = extract_user_id_from_token(access_token)
        else:
            raise HTTPException(status_code=401, detail="Access token required")
        
        memory_manager.cleanup_thread_memory(thread_id)
        
        return {"message": "Session reset successfully", "thread_id": thread_id}
        
    except Exception as e:
        print(f"[세션 리셋 오류] {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# 관리자용 엔드포인트들
@app.get("/admin/threads")
async def get_active_threads():
    """현재 활성화된 메모리 스레드 목록을 반환합니다. (관리자용)"""
    return memory_manager.get_all_active_threads()

@app.delete("/admin/thread/{thread_id}")
async def cleanup_thread(thread_id: str):
    """특정 스레드의 메모리를 정리합니다. (관리자용)"""
    memory_cleaned = memory_manager.cleanup_thread_memory(thread_id)
    
    return {
        "thread_id": thread_id,
        "memory_cleaned": memory_cleaned,
        "message": "Thread cleanup completed"
    }

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    memory_threads = memory_manager.get_all_active_threads()
    chain_info = chain_router.get_chain_info()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "milvus_connected": chain_info["finance"]["milvus_available"],
        "active_memory_threads": memory_threads["total_count"],
        "chain_info": chain_info
    }

