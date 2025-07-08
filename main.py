# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
from fastapi import FastAPI, Query, BackgroundTasks, HTTPException, Header
from datetime import datetime
from typing import Optional
from pydantic import BaseModel

# AI 기능을 위한 프레임워크
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_milvus import Milvus
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever

# 사용자 정의 모듈
from modules.thread_manager import thread_id_manager
from modules.memory_manager import memory_manager

# 랭체인 트래킹
import mlflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:5001")

mlflow.set_experiment("langchain")

# Enable MLflow tracing
mlflow.langchain.autolog()

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

# 대화 히스토리를 포함한 프롬프트 템플릿
classification_prompt = PromptTemplate.from_template(
    """주어진 사용자 질문과 대화 히스토리를 보고 `최신소식`, `전문지식`, `리셋`, 또는 `기타` 중 하나로 분류하세요. 
리셋 관련 키워드: "리셋", "초기화", "지워", "새로시작", "reset", "clear" 등
한 단어 이상으로 응답하지 마세요.

<chat_history>
{chat_history}
</chat_history>

<question>
{question}
</question>

Classification:"""
)

# 체인을 생성합니다.
chain = (
    classification_prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()  # 문자열 출력 파서를 사용합니다.
)

# 1. 웹서치
retriever = TavilySearchAPIRetriever(k=3)

news_prompt = PromptTemplate.from_template(
    """You are an HUMINT in news. \
Always answer questions starting with "최신 데이터에 따르면..". \
When you work with numbers, be mindful of units.\
If you don't know the answer, just say that you don't know\

Previous conversation:
{chat_history}

Respond to the following question based on the context and previous conversation:
Context: {context}
Question: {question}
Answer:"""
)

news_chain = (
    {"question": lambda x: x["question"],
     "context": lambda x: retriever.invoke(x["question"]),
     "chat_history": lambda x: x.get("chat_history", "")}
    | news_prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# 2. 벡터 DB 서치
embeddings = OpenAIEmbeddings()

# Milvus 연결 추가 (연결 실패 시 예외 처리)
try:
    import os
    milvus_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    print(f"Milvus 연결 시도: {milvus_uri}")
    
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={
            "uri": milvus_uri,
        },
        collection_name="coindesk_articles",
    )
    print("Milvus 연결 성공")
    use_milvus = True
except Exception as e:
    print(f"Milvus 연결 실패: {e}")
    print("주의: Milvus 연결 실패로 벡터 검색 기능을 사용할 수 없습니다.")
    use_milvus = False
    vectorstore = None

FINANCE_PROMPT_TEMPLATE = """You are an expert in finance. \
Always answer questions starting with "전문가에 따르면..". \

Previous conversation:
{chat_history}

Respond to the following question based the context, statistical information, and previous conversation when possible:
Context: {context}
Question: {question}
Answer:"""

finance_prompt = PromptTemplate(
    template=FINANCE_PROMPT_TEMPLATE, 
    input_variables=["context", "question", "chat_history"]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if use_milvus and vectorstore:
    retriever1 = vectorstore.as_retriever(search_kwargs={"k": 3})
    context_chain = RunnableLambda(lambda x: x["question"]) | retriever1 | format_docs
    
    finance_chain = (
        {"context": context_chain,
         "question": RunnablePassthrough(),
         "chat_history": lambda x: x.get("chat_history", "")}
        | finance_prompt
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )
else:
    # Milvus가 없는 경우 기본 체인
    finance_chain = (
        PromptTemplate.from_template(
            """전문가에 따르면, 현재 벡터 데이터베이스에 연결할 수 없어 전문 지식을 제공하기 어렵습니다. 
            시스템 관리자에게 문의해주세요.
            
            Previous conversation:
            {chat_history}
            
            Question: {question}
            """
        )
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )

# 3. 기타
general_chain = (
    PromptTemplate.from_template(
        """Previous conversation:
{chat_history}

Respond to the following question concisely:
If the question is not about expert knowledge or recent events, reply:

"도와드리지 못해서 죄송합니다. 저는 비트코인 관련 전문지식과 최신소식만 답변드릴 수 있습니다."

Only respond with factual, concise answers supported by the context when applicable.
Question: {question}
Answer:
"""
    )
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# 4. 리셋 체인
reset_chain = (
    PromptTemplate.from_template(
        """네, 대화 기록을 모두 지웠습니다. 새롭게 시작하겠습니다! 
        비트코인 관련 전문지식이나 최신소식에 대해 무엇이든 물어보세요."""
    )
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# 5. 경로 설정
def route(info):
    topic = info["topic"].strip()
    if topic == "최신소식":
        return news_chain
    elif topic == "전문지식":
        return finance_chain
    elif topic == "리셋":
        return reset_chain
    else:
        return general_chain

def create_full_chain_with_memory(thread_id: str):
    """메모리가 포함된 전체 체인을 생성합니다."""
    memory = memory_manager.get_memory(thread_id)
    
    def process_with_memory(inputs):
        question = inputs["question"]
        chat_history = memory_manager.get_chat_history_string(memory)
        
        # 분류 체인 실행
        classification_input = {"question": question, "chat_history": chat_history}
        topic = chain.invoke(classification_input)
        
        # 리셋 요청인 경우 메모리 초기화
        if topic.strip() == "리셋":
            memory_manager.reset_memory(thread_id)
            return reset_chain.invoke({"question": question})
        
        # 해당 토픽의 체인 선택 및 실행
        selected_chain = route({"topic": topic})
        chain_input = {"question": question, "chat_history": chat_history}
        response = selected_chain.invoke(chain_input)
        
        # 메모리에 대화 저장
        memory.save_context({"input": question}, {"output": response})
        
        return response
    
    return RunnableLambda(process_with_memory)

app = FastAPI()

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    user_id: int
    timestamp: datetime

class SessionResponse(BaseModel):
    thread_id: str
    user_id: int
    message: str

# JWT 토큰에서 스레드 ID 생성 엔드포인트
@app.post("/session/create", response_model=SessionResponse)
async def create_session(authorization: str = Header(..., description="Bearer JWT토큰")):
    """JWT 토큰으로부터 사용자 세션을 생성하고 스레드 ID를 반환합니다."""
    try:
        # Bearer 토큰에서 실제 토큰 추출
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header format")
        
        access_token = authorization.replace("Bearer ", "")
        
        # 스레드 ID 생성
        thread_id = thread_id_manager.get_or_create_thread_id(access_token)
        if thread_id is None:
            raise HTTPException(status_code=401, detail="Invalid or expired JWT token")
        
        # 사용자 ID 추출
        user_id = thread_id_manager.get_user_id_from_thread(thread_id)
        
        return SessionResponse(
            thread_id=thread_id,
            user_id=user_id,
            message="Session created successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[세션 생성 오류] {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# 채팅 엔드포인트 (인증 필요)
@app.get("/async/chat", response_model=ChatResponse)
async def async_chat(
    question: str = Query(..., min_length=1, max_length=500),
    authorization: str = Header(..., description="Bearer JWT토큰"),
    background_tasks: BackgroundTasks = None
):
    start_time = datetime.now()
    
    try:
        # Bearer 토큰에서 실제 토큰 추출
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header format")
        
        access_token = authorization.replace("Bearer ", "")
        
        # 스레드 ID 가져오기 (없으면 새로 생성)
        thread_id = thread_id_manager.get_or_create_thread_id(access_token)
        if thread_id is None:
            raise HTTPException(status_code=401, detail="Invalid or expired JWT token. Please create a session first.")
        
        # 사용자 ID 추출
        user_id = thread_id_manager.get_user_id_from_thread(thread_id)
        
        # 스레드별 체인 생성
        full_chain = create_full_chain_with_memory(thread_id)
        
        # 체인 실행
        response = await full_chain.ainvoke({"question": question})
        
        # AIMessage 객체에서 content 추출
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        # 로깅은 백그라운드로 넘김
        if background_tasks:
            background_tasks.add_task(log_to_mlflow, question, response_text, start_time)
        
        return ChatResponse(
            response=response_text,
            thread_id=thread_id,
            user_id=user_id,
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[채팅 처리 오류] {e}")
        error_response = f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"
        # 오류 시에도 적절한 응답 구조 유지
        raise HTTPException(status_code=500, detail=error_response)

# 세션 리셋 엔드포인트
@app.post("/session/reset")
async def reset_user_session(authorization: str = Header(..., description="Bearer JWT토큰")):
    """사용자의 세션과 메모리를 모두 리셋합니다."""
    try:
        # Bearer 토큰에서 실제 토큰 추출
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header format")
        
        access_token = authorization.replace("Bearer ", "")
        
        # 기존 스레드 ID 가져오기
        current_thread_id = thread_id_manager.get_or_create_thread_id(access_token)
        if current_thread_id:
            # 메모리 정리
            memory_manager.cleanup_thread_memory(current_thread_id)
        
        # 새로운 세션 생성
        success = thread_id_manager.reset_user_session(access_token)
        if not success:
            raise HTTPException(status_code=401, detail="Invalid or expired JWT token")
        
        # 새로운 스레드 ID 가져오기
        new_thread_id = thread_id_manager.get_or_create_thread_id(access_token)
        user_id = thread_id_manager.get_user_id_from_thread(new_thread_id)
        
        return SessionResponse(
            thread_id=new_thread_id,
            user_id=user_id,
            message="Session reset successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[세션 리셋 오류] {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# 관리자용 엔드포인트들
@app.get("/admin/sessions")
async def get_all_sessions():
    """모든 세션 정보를 반환합니다. (관리자용)"""
    thread_sessions = thread_id_manager.get_all_sessions()
    memory_threads = memory_manager.get_all_active_threads()
    
    return {
        "thread_sessions": thread_sessions,
        "memory_threads": memory_threads,
        "timestamp": datetime.now()
    }

@app.get("/admin/threads")
async def get_active_threads():
    """현재 활성화된 메모리 스레드 목록을 반환합니다. (관리자용)"""
    return memory_manager.get_all_active_threads()

@app.delete("/admin/session/{thread_id}")
async def cleanup_session(thread_id: str):
    """특정 스레드의 세션과 메모리를 정리합니다. (관리자용)"""
    thread_cleaned = thread_id_manager.cleanup_session(thread_id)
    memory_cleaned = memory_manager.cleanup_thread_memory(thread_id)
    
    return {
        "thread_id": thread_id,
        "thread_cleaned": thread_cleaned,
        "memory_cleaned": memory_cleaned,
        "message": "Session cleanup completed"
    }

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    thread_sessions = thread_id_manager.get_all_sessions()
    memory_threads = memory_manager.get_all_active_threads()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "milvus_connected": use_milvus,
        "active_thread_sessions": thread_sessions["total_sessions"],
        "active_memory_threads": memory_threads["total_count"]
    }

# 토큰 테스트용 엔드포인트 (개발용)
@app.post("/debug/parse-token")
async def debug_parse_token(authorization: str = Header(..., description="Bearer JWT토큰")):
    """JWT 토큰 파싱을 테스트합니다. (개발용)"""
    try:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header format")
        
        access_token = authorization.replace("Bearer ", "")
        user_id = thread_id_manager.parse_jwt_token(access_token)
        
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid JWT token")
        
        return {
            "user_id": user_id,
            "token_valid": True,
            "message": "Token parsed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[토큰 파싱 오류] {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
