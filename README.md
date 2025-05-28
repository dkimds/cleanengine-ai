# cleanengine-ai
구름 프로펙트 풀스택 과정 2기 1조
## 개요
OpenAI API 기반으로 암호화폐 모의투자를 도와주는 챗봇(gpt-4o-mini) 
### 기능
- 코인 투자 최신소식 검색: Tavily Search API 활용
- 코인 관련 전문지식 검색: Milvus 벡터 데이터베이스 활용
- 투자와 무관한 질문 거절: 프롬프트 엔지니어링
### 코드 구조(main.py)
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
...
# 5. FastAPI 비동기 호출
```
## 사용법
본 문서는 콘테이너 실행을 기준으로 작성하였습니다.
### 벡터 DB 설치
아래 커맨드로 Milvus를 도커 콘테이너로 설치
```sh
bash standalone_embed.sh start
```
### 도커 이미지 당겨오기
AI 서비스용 Docker 이미지를 로컬에 다운로드
```sh
docker pull rlaqhguse/if-ai
```
### 도커 네트워크 설정(권장)
Milvus와 AI 서비스가 통신할 수 있도록 같은 사용자 정의 네트워크에 연결
```sh
docker network create milvus-net
docker network connect milvus-net milvus-standalone
```
### 도커 콘테이너 실행
AI 서비스 컨테이너를 Milvus와 같은 네트워크에 연결하여 실행
```sh
docker run -p 8000:8000 \
  --network milvus-net \
  --name if-ai rlaqhguse/if-ai
```
## 기타
`agent.py`: main.py 같은 기능의 LangGraph 프로토타입

`etl.py`: 외부 데이터 수집 스케줄링