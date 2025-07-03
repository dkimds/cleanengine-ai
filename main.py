# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
from fastapi import FastAPI, Query, BackgroundTasks
from datetime import datetime, timedelta
import threading
import time
from typing import Dict, List, Optional
from pydantic import BaseModel

# AI 기능을 위한 프레임워크
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_milvus import Milvus
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage

# 랭체인 트래킹
import mlflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

mlflow.set_experiment("langchain")

# Enable MLflow tracing
mlflow.langchain.autolog()

# 메모리 관리를 위한 클래스
class ThreadMemoryManager:
    def __init__(self):
        self.memories: Dict[str, ConversationBufferWindowMemory] = {}
        self.last_activity: Dict[str, datetime] = {}
        self.lock = threading.Lock()
        
        # 백그라운드 스레드로 30분마다 비활성 메모리 정리
        self.cleanup_thread = threading.Thread(target=self._cleanup_inactive_memories, daemon=True)
        self.cleanup_thread.start()
    
    def get_memory(self, thread_id: str) -> ConversationBufferWindowMemory:
        """스레드 ID에 해당하는 메모리를 가져오거나 생성합니다."""
        with self.lock:
            if thread_id not in self.memories:
                self.memories[thread_id] = ConversationBufferWindowMemory(
                    k=10,  # 최근 10개의 대화만 기억
                    return_messages=True
                )
            
            self.last_activity[thread_id] = datetime.now()
            return self.memories[thread_id]
    
    def reset_memory(self, thread_id: str) -> bool:
        """특정 스레드의 메모리를 초기화합니다."""
        with self.lock:
            if thread_id in self.memories:
                self.memories[thread_id].clear()
                self.last_activity[thread_id] = datetime.now()
                return True
            return False
    
    def _cleanup_inactive_memories(self):
        """30분 동안 비활성 상태인 메모리를 정리합니다."""
        while True:
            time.sleep(300)  # 5분마다 체크
            current_time = datetime.now()
            inactive_threshold = timedelta(minutes=30)
            
            with self.lock:
                inactive_threads = []
                for thread_id, last_time in self.last_activity.items():
                    if current_time - last_time > inactive_threshold:
                        inactive_threads.append(thread_id)
                
                for thread_id in inactive_threads:
                    print(f"[메모리 정리] 스레드 {thread_id} 비활성으로 인한 메모리 삭제")
                    del self.memories[thread_id]
                    del self.last_activity[thread_id]

# 전역 메모리 매니저 인스턴스
memory_manager = ThreadMemoryManager()

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
)

# 4. 리셋 체인
reset_chain = (
    PromptTemplate.from_template(
        """네, 대화 기록을 모두 지웠습니다. 새롭게 시작하겠습니다! 
        비트코인 관련 전문지식이나 최신소식에 대해 무엇이든 물어보세요."""
    )
    | ChatOpenAI(model="gpt-4o-mini")
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

def get_chat_history_string(memory: ConversationBufferWindowMemory) -> str:
    """메모리에서 대화 히스토리를 문자열로 변환합니다."""
    try:
        messages = memory.chat_memory.messages
        if not messages:
            return ""
        
        history_parts = []
        for message in messages[-6:]:  # 최근 6개 메시지만 사용
            if isinstance(message, HumanMessage):
                history_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                history_parts.append(f"Assistant: {message.content}")
        
        return "\n".join(history_parts)
    except Exception as e:
        print(f"대화 히스토리 변환 오류: {e}")
        return ""

from langchain_core.runnables import RunnableLambda

def create_full_chain_with_memory(thread_id: str):
    """메모리가 포함된 전체 체인을 생성합니다."""
    memory = memory_manager.get_memory(thread_id)
    
    def process_with_memory(inputs):
        question = inputs["question"]
        chat_history = get_chat_history_string(memory)
        
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
        
        # 메모리에 대화 저장 (response가 AIMessage 객체인 경우 content 추출)
        response_content = response.content if hasattr(response, 'content') else str(response)
        memory.save_context({"input": question}, {"output": response_content})
        
        return response_content
    
    return RunnableLambda(process_with_memory)

app = FastAPI()

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    timestamp: datetime

# 비동기 인보크
@app.get("/async/chat", response_model=ChatResponse)
async def async_chat(
    question: str = Query(..., min_length=1, max_length=500),
    thread_id: str = Query(default="default", description="대화 스레드 ID"),
    background_tasks: BackgroundTasks = None
):
    start_time = datetime.now()
    
    try:
        # 스레드별 체인 생성
        full_chain = create_full_chain_with_memory(thread_id)
        
        # 체인 실행
        response = await full_chain.ainvoke({"question": question})
        
        # response를 문자열로 변환 (AIMessage 객체인 경우 content 추출)
        if hasattr(response, 'content'):
            response_str = str(response.content)
        else:
            response_str = str(response)
        
        # 로깅은 백그라운드로 넘김
        if background_tasks:
            background_tasks.add_task(log_to_mlflow, question, response_str, start_time)
        
        return ChatResponse(
            response=response_str,
            thread_id=thread_id,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        error_response = f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"
        return ChatResponse(
            response=error_response,
            thread_id=thread_id,
            timestamp=datetime.now()
        )

# 메모리 리셋 엔드포인트
@app.post("/reset")
async def reset_thread_memory(thread_id: str = Query(default="default")):
    """특정 스레드의 메모리를 수동으로 리셋합니다."""
    success = memory_manager.reset_memory(thread_id)
    if success:
        return {"message": f"스레드 {thread_id}의 메모리가 초기화되었습니다.", "thread_id": thread_id}
    else:
        return {"message": f"스레드 {thread_id}의 메모리가 존재하지 않습니다.", "thread_id": thread_id}

# 활성 스레드 조회 엔드포인트
@app.get("/threads")
async def get_active_threads():
    """현재 활성화된 스레드 목록을 반환합니다."""
    with memory_manager.lock:
        active_threads = []
        current_time = datetime.now()
        
        for thread_id, last_activity in memory_manager.last_activity.items():
            time_since_activity = current_time - last_activity
            active_threads.append({
                "thread_id": thread_id,
                "last_activity": last_activity,
                "minutes_since_activity": int(time_since_activity.total_seconds() / 60),
                "messages_count": len(memory_manager.memories[thread_id].chat_memory.messages)
            })
        
        return {"active_threads": active_threads, "total_count": len(active_threads)}

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "milvus_connected": use_milvus,
        "active_threads": len(memory_manager.memories)
    }
