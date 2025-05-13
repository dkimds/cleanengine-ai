# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever

# API 키 정보 로드
load_dotenv()

prompt = PromptTemplate.from_template(
    """주어진 사용자 질문을 `최신소식`, `전문지식`, 또는 `기타` 중 하나로 분류하세요. 한 단어 이상으로 응답하지 마세요.

<question>
{question}
</question>

Classification:"""
)

# 체인을 생성합니다.
chain = (
    prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()  # 문자열 출력 파서를 사용합니다.
)
# 1. 웹서치
retriever = TavilySearchAPIRetriever(k=3)

news_chain = (
    {"question": lambda x: x["question"],
    "context": lambda x: retriever.invoke(x["question"])}
    | PromptTemplate.from_template(
        """You are an HUMINT in news. \
Always answer questions starting with "최신 데이터에 따르면..". \
When you work with numbers, be mindful of units.\
If you don't know the answer, just say that you don't know\
Respond to the following question based on the context provided:
Context: {context}
Question: {question}
Answer:"""
    )
    # OpenAI의 LLM을 사용합니다.
    | ChatOpenAI(model="gpt-4o-mini")
)

# 2. 벡터 DB 서치
embeddings = OpenAIEmbeddings()

vectorstore = Milvus(
    # documents=docs,
    embedding_function=embeddings,
    connection_args={
        "uri": "http://localhost:19530",
    },
    # drop_old=True,  # Drop the old Milvus collection if it exists
)
PROMPT_TEMPLATE = """You are an expert in finance. \
Always answer questions starting with "전문가에 따르면..". \
Respond to the following question based the context and statistical information when possible:
Context: {context}
Question: {question}
Answer:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)
retriever1 = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

context_chain = RunnableLambda(lambda x: x["question"]) | retriever | format_docs

finance_chain = (
    {"context": context_chain , 
    "question": RunnablePassthrough()}
    # OpenAI의 LLM을 사용합니다.
    | prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)


# 3. 기타
general_chain = (
    PromptTemplate.from_template(
        """Respond to the following question concisely:
If the question is not about expert knowledge or recent events, reply:

"도와드리지 못해서 죄송합니다. 저는 비트코인 관련 전문지식과 최신소식만 답변드릴 수 있습니다."

Only respond with factual, concise answers supported by the context when applicable.
Question: {question}
Answer:
"""
    )
    # OpenAI의 LLM을 사용합니다.
    | ChatOpenAI(model="gpt-4o-mini")
)

# 4. 경로 설정
def route(info):
    topic = info["topic"].strip()
    if topic == "최신소식":
        return news_chain
    elif topic == "전문지식":
        return finance_chain  # 원래 코드에선 'expertise_chain' → 오타 수정
    else:
        return general_chain


from langchain_core.runnables import RunnableLambda

full_chain = (
    {"topic": chain, "question": lambda x: x["question"]}
    | RunnableLambda(
        # 경로를 지정하는 함수를 인자로 전달합니다.
        route
    )
    | StrOutputParser()
)

app = FastAPI()

# 비동기 인보크
@app.get("/async/chat")
async def async_chat(question: str = Query(None, min_length=3, max_length=50)):
    response = await full_chain.ainvoke({"question": question})
    return response
# @app.get("/streaming_async/chat")
# async def streaming_async(query: str = Query(None, min_length=3, max_length=50)):
#     async def event_stream():
#         try:
#             async for chunk in full_chain.astream({"question": query}):
#                 if len(chunk) > 0:
#                     yield f"data: {chunk}\n\n"
#         except Exception as e:
#             yield f"data: {str(e)}\n\n"

#     return StreamingResponse(event_stream(), media_type="text/event-stream")
