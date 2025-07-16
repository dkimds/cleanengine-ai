"""
Finance chain for handling financial expert knowledge queries.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_milvus import Milvus
from typing import Dict, Any, Optional
import os


class FinanceChain:
    """
    Handles queries about financial expert knowledge using vector database search.
    """
    
    def __init__(self, model: str = "gpt-4o-mini", search_k: int = 3):
        """
        Initialize the finance chain.
        
        Args:
            model: OpenAI model to use for response generation
            search_k: Number of search results to retrieve from vector DB
        """
        self.model = model
        self.search_k = search_k
        self.use_milvus = False
        self.vectorstore = None
        self._setup_milvus()
        self._setup_chain()
    
    def _setup_milvus(self):
        """Set up Milvus vector database connection."""
        try:
            embeddings = OpenAIEmbeddings()
            milvus_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
            print(f"Milvus 연결 시도: {milvus_uri}")
            
            self.vectorstore = Milvus(
                embedding_function=embeddings,
                connection_args={
                    "uri": milvus_uri,
                },
                collection_name="coindesk_articles",
            )
            print("Milvus 연결 성공")
            self.use_milvus = True
        except Exception as e:
            print(f"Milvus 연결 실패: {e}")
            print("주의: Milvus 연결 실패로 벡터 검색 기능을 사용할 수 없습니다.")
            self.use_milvus = False
            self.vectorstore = None
    
    def _setup_chain(self):
        """Set up the finance chain with retriever, prompt and model."""
        
        # Finance prompt template
        self.prompt = PromptTemplate(
            template="""You are an expert in finance. \
Always answer questions starting with "전문가에 따르면..". \

Previous conversation:
{chat_history}

Respond to the following question based the context, statistical information, and previous conversation when possible:
Context: {context}
Question: {question}
Answer:""",
            input_variables=["context", "question", "chat_history"]
        )
        
        # Create the chain based on Milvus availability
        if self.use_milvus and self.vectorstore:
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.search_k})
            context_chain = RunnableLambda(lambda x: x["question"]) | self.retriever | self._format_docs
            
            self.chain = (
                {
                    "context": context_chain,
                    "question": RunnablePassthrough(),
                    "chat_history": lambda x: x.get("chat_history", "")
                }
                | self.prompt
                | ChatOpenAI(model=self.model)
                | StrOutputParser()
            )
        else:
            # Fallback chain when Milvus is not available
            fallback_prompt = PromptTemplate.from_template(
                """전문가에 따르면, 현재 벡터 데이터베이스에 연결할 수 없어 전문 지식을 제공하기 어렵습니다. 
시스템 관리자에게 문의해주세요.

Previous conversation:
{chat_history}

Question: {question}
"""
            )
            
            self.chain = (
                fallback_prompt
                | ChatOpenAI(model=self.model)
                | StrOutputParser()
            )
    
    def _format_docs(self, docs):
        """Format retrieved documents for use in the prompt."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def process(self, question: str, chat_history: str = "") -> str:
        """
        Process a finance-related question.
        
        Args:
            question: The user's question
            chat_history: Previous conversation history
            
        Returns:
            Response based on financial expert knowledge
        """
        return self.chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
    
    def invoke(self, inputs: Dict[str, Any]) -> str:
        """
        Invoke the finance chain with inputs.
        
        Args:
            inputs: Dictionary containing 'question' and optional 'chat_history'
            
        Returns:
            Finance-based response as string
        """
        return self.chain.invoke(inputs)
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> str:
        """
        Asynchronously invoke the finance chain.
        
        Args:
            inputs: Dictionary containing 'question' and optional 'chat_history'
            
        Returns:
            Finance-based response as string
        """
        return await self.chain.ainvoke(inputs)
    
    def get_retriever(self) -> Optional[Any]:
        """Get the vector database retriever if available."""
        return self.retriever if self.use_milvus else None
    
    def is_milvus_available(self) -> bool:
        """Check if Milvus is available for use."""
        return self.use_milvus