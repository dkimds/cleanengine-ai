from prefect import flow, task
from dotenv import load_dotenv

import hashlib
import feedparser

from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

load_dotenv()
# --- Task 1: 데이터 수집 ---
@task
def extract():
    # Add User-Agent header to avoid 403 error
    feedparser.USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    
    feed = feedparser.parse("https://www.coindesk.com/arc/outboundfeeds/rss/")
    articles = []
    
    for entry in feed.entries:
        title = entry.get("title", "")
        description = entry.get("description", "")
        summary = entry.get("summary", "")
        
        # Try different content fields
        content = ""
        if hasattr(entry, 'content') and entry.content:
            content = entry.content[0].value if isinstance(entry.content, list) and len(entry.content) > 0 else ""
        elif entry.get("content_encoded"):
            content = entry.get("content_encoded")
        elif summary:
            content = summary
        elif description:
            content = description
        
        link = entry.get("link", "")
        author = entry.get("author", "") or entry.get("dc_creator", "")
        pub_date = entry.get("published", "")
        
        articles.append({
            "title": title,
            "content": content,
            "description": description,
            "link": link,
            "author": author,
            "pub_date": pub_date
        })
    return articles

# --- Task 2: 텍스트 변환 ---
@task
def transform(articles):
    results = []
    for article in articles:
        title = article.get("title", "")
        content = article.get("content", "")
        text = f"{title}\n\n{content}"
        results.append({
            "text": text,
            "metadata": {"source": "coindesk"}
        })
    return results

# --- Task 3: Milvus에 저장 ---
@task
def load(documents):
    connections.connect("default", host="localhost", port="19530")
    embedder = OpenAIEmbeddings()
    collection_name = "coindesk_articles"

    if collection_name not in utility.list_collections():
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        ]
        schema = CollectionSchema(fields=fields, description="Coindesk article embeddings")
        collection = Collection(name=collection_name, schema=schema)

        index_params = {
            "index_type": "IVF_FLAT",  # 예: IVF_FLAT, HNSW, etc.
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="vector", index_params=index_params)
    else:
        collection = Collection(collection_name)

    collection.load()

    new_ids, new_embeddings, new_texts = [], [], []

    for d in documents:
        doc = Document(page_content=d["text"], metadata=d.get("metadata", {}))
        text = doc.page_content
        uid = hashlib.md5(text.encode("utf-8")).hexdigest()

        expr = f'id == "{uid}"'
        if collection.query(expr, output_fields=["id"]):
            continue

        embedding = embedder.embed_query(text)
        new_ids.append(uid)
        new_embeddings.append(embedding)
        new_texts.append(text)

    if new_ids:
        collection.insert([new_ids, new_embeddings, new_texts])
        print(f"✅ {len(new_ids)} documents inserted.")
    else:
        print("⚠️ No new documents to insert.")

# --- Prefect Flow ---
@flow(name="Coindesk ETL to Milvus")
def etl_pipeline():
    articles = extract()
    docs = transform(articles)
    load(docs)

# 실행
if __name__ == "__main__":
    # 스케줄 만드려면 아래 코드 주석 해제하고 다음 줄 주석 설정
    # etl_pipeline.serve(name="coindesk_etl", cron="0 8 * * *")
    etl_pipeline()