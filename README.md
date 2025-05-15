# cleanengine-ai
구름 프로펙트 풀스택 과정 2기 1조
## 개요
OpenAI API 기반으로 암호화폐 모의투자를 도와주는 챗봇(gpt-4o-mini) 
### 기능
- 코인 투자 최신소식 검색: Tavily Search API 활용
- 코인 관련 전문지식 검색: Milvus 벡터 데이터베이스 활용
- 투자와 무관한 질문 거절: 프롬프트 엔지니어링
### 코드 구조
```python
# API 키 정보 로드
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
...
# 2. 벡터 DB 서치
...
# 3. 질문 필터링
...
# 4. 라우팅
def route(info):
    topic = info["topic"].strip()
    if topic == "최신소식":
        return news_chain
    elif topic == "전문지식":
        return finance_chain  
    else:
        return general_chain

# 5. FastAPI 비동기 호출
```
## 사용법
본 문서는 macOS/Linux 로컬 실행을 기준으로 작성하였습니다.
### 벡터 DB 설치
아래 커맨드로 Milvus를 도커 콘테이너로 설치
```sh
bash standalone_embed.sh start
```
### 파이썬 환경 설정
가상환경에서 실행을 권장합니다.
```sh
pip install -r requirements.txt
```
### AI 서버 실행
코드 수정하면 바로 해당사항 적용
```sh
uvicorn main:app --reload
```
## 기타
`agent.py`: main.py 같은 기능의 LangGraph 프로토타입
