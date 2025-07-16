"""
News chain for handling latest news and current events queries.
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever
from typing import Dict, Any


class NewsChain:
    """
    Handles queries about latest news and current events using web search.
    """
    
    def __init__(self, model: str = "gpt-4o-mini", search_k: int = 3):
        """
        Initialize the news chain.
        
        Args:
            model: OpenAI model to use for response generation
            search_k: Number of search results to retrieve
        """
        self.model = model
        self.search_k = search_k
        self._setup_chain()
    
    def _setup_chain(self):
        """Set up the news chain with retriever, prompt and model."""
        
        # Web search retriever
        self.retriever = TavilySearchAPIRetriever(k=self.search_k)
        
        # News prompt template
        self.prompt = PromptTemplate.from_template(
            """You are a news analysis expert. You must respond in Korean.
Always answer questions starting with "최신 데이터에 따르면..".

IMPORTANT: When mentioning prices or monetary values, ALWAYS include the currency unit:
- For Bitcoin/crypto prices: include "달러" or "USD" (e.g., "117,525 달러")
- For Korean won: include "원" (e.g., "1,500,000 원")
- Never give numbers without currency units

If you don't know the answer, just say "죄송하지만 해당 정보를 찾을 수 없습니다."

Previous conversation:
{chat_history}

Respond to the following question in Korean based on the context and previous conversation:
Context: {context}
Question: {question}
Answer (in Korean):"""
        )
        
        # Create the chain
        self.chain = (
            {
                "question": lambda x: x["question"],
                "context": lambda x: self.retriever.invoke(x["question"]),
                "chat_history": lambda x: x.get("chat_history", "")
            }
            | self.prompt
            | ChatOpenAI(model=self.model)
            | StrOutputParser()
        )
    
    def process(self, question: str, chat_history: str = "") -> str:
        """
        Process a news-related question.
        
        Args:
            question: The user's question
            chat_history: Previous conversation history
            
        Returns:
            Response based on latest news data
        """
        return self.chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
    
    def invoke(self, inputs: Dict[str, Any]) -> str:
        """
        Invoke the news chain with inputs.
        
        Args:
            inputs: Dictionary containing 'question' and optional 'chat_history'
            
        Returns:
            News-based response as string
        """
        return self.chain.invoke(inputs)
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> str:
        """
        Asynchronously invoke the news chain.
        
        Args:
            inputs: Dictionary containing 'question' and optional 'chat_history'
            
        Returns:
            News-based response as string
        """
        return await self.chain.ainvoke(inputs)