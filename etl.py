from prefect import flow, task

import hashlib
import feedparser

from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

 
# --- Task 1: 데이터 수집 ---
@task
def extract():
    feed = feedparser.parse("https://www.coindesk.com/arc/outboundfeeds/rss/")
    articles = []
    for entry in feed.entries:
        title = entry.get("title", "")
        content = entry.get("content", [{}])[0].get("value", "")
        articles.append({"title": title, "content": content})
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
    embedder = OpenAIEmbeddings()
    connections.connect("default", host="localhost", port="19530")

    collection_name = "coindesk_articles"

    if collection_name not in utility.list_collections():
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        ]
        schema = CollectionSchema(fields, description="Coindesk article embeddings")
        Collection(name=collection_name, schema=schema)

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
    etl_pipeline()
