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
