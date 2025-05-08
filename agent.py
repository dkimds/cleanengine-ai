import os
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import convert_to_messages

# Setup
load_dotenv()

def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")



# 1. Create worker agents
# Research agent

web_search = TavilySearch(max_results=3)
web_search_results = web_search.invoke("who is the mayor of NYC?")


research_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[web_search],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with economy-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
)

# vector DB agent
embeddings = OpenAIEmbeddings()

vectorstore = Milvus(
    # documents=docs,
    embedding_function=embeddings,
    connection_args={
        "uri": "http://localhost:19530",
    },
    # drop_old=True,  # Drop the old Milvus collection if it exists
)


retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@tool
def search_milvus(query: str) -> str:
    """Search Milvus vector DB and return relevant text chunks."""
    docs = retriever.invoke(query)
    return format_docs(docs)


vectordb_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[search_milvus],
    name="vectordb_agent",
)

# 3. Create supervisor with langgraph-supervisor
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model

supervisor = create_supervisor(
    model=init_chat_model("openai:gpt-4.1"),
    agents=[research_agent, vectordb_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign economy-related tasks to this agent\n"
        "- a vector DB agent. Assign AI-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()

async def finalinvoke(query):
    last_content = None

    for chunk in supervisor.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        },
    ):
        pretty_print_messages(chunk, last_message=True)
        # 각 chunk에서 메시지를 검사
    messages = chunk.get("supervisor", {}).get("messages", [])
    last_content = messages[-1].content  # 가장 마지막 supervisor 메시지 저장

    return last_content 

app = FastAPI()

# 비동기 인보크
@app.get("/async/chat")
async def async_chat(query: str = Query(None, min_length=3, max_length=50)):
    response = await finalinvoke(query)
    return response