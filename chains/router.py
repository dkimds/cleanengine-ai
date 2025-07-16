"""
Chain router for routing queries to appropriate chain handlers.
"""

from .classification import ClassificationChain
from .news import NewsChain
from .finance import FinanceChain
from .general import GeneralChain
from .reset import ResetChain
from modules.memory_manager import memory_manager
from langchain_core.runnables import RunnableLambda
from typing import Dict, Any, Optional


class ChainRouter:
    """
    Routes user queries to the appropriate chain based on classification.
    """
    
    def __init__(self, model: str = "Qwen/Qwen2-0.5B-Instruct"):
        """
        Initialize the chain router with all chain types.
        
        Args:
            model: vLLM model to use for all chains
        """
        self.model = model
        self._setup_chains()
    
    def _setup_chains(self):
        """Initialize all the chain handlers."""
        # Use vLLM for classification, general, and reset chains
        self.classification_chain = ClassificationChain(self.model)
        self.general_chain = GeneralChain(self.model)
        self.reset_chain = ResetChain(self.model)
        
        # Use OpenAI for news and finance chains (better for real-time data and complex reasoning)
        openai_model = "gpt-4o-mini"
        self.news_chain = NewsChain(openai_model)
        self.finance_chain = FinanceChain(openai_model)
        
        # Chain routing map
        self.chain_map = {
            "최신소식": self.news_chain,
            "전문지식": self.finance_chain,
            "리셋": self.reset_chain,
            "기타": self.general_chain
        }
    
    def route(self, topic: str):
        """
        Route to the appropriate chain based on topic.
        
        Args:
            topic: Classification result from classification chain
            
        Returns:
            Appropriate chain handler
        """
        topic = topic.strip()
        return self.chain_map.get(topic, self.general_chain)
    
    def process_query(self, question: str, chat_history: str = "") -> str:
        """
        Process a user query by classifying and routing to appropriate chain.
        
        Args:
            question: The user's question
            chat_history: Previous conversation history
            
        Returns:
            Response from the appropriate chain
        """
        # Classify the question
        classification_input = {
            "question": question,
            "chat_history": chat_history
        }
        topic = self.classification_chain.invoke(classification_input)
        
        # Route to appropriate chain
        selected_chain = self.route(topic)
        
        # Process with the selected chain
        chain_input = {
            "question": question,
            "chat_history": chat_history
        }
        
        return selected_chain.invoke(chain_input)
    
    def create_full_chain_with_memory(self, thread_id: str):
        """
        Create a complete chain with memory management.
        
        Args:
            thread_id: Thread ID for memory management
            
        Returns:
            RunnableLambda that processes queries with memory
        """
        def process_with_memory(inputs):
            question = inputs["question"]
            memory = memory_manager.get_memory(thread_id)
            chat_history = memory_manager.get_chat_history_string(memory)
            
            # Classify the question
            classification_input = {
                "question": question,
                "chat_history": chat_history
            }
            topic = self.classification_chain.invoke(classification_input)
            
            # Handle reset requests
            if topic.strip() == "리셋":
                memory_manager.reset_memory(thread_id)
                return self.reset_chain.invoke({"question": question})
            
            # Route to appropriate chain
            selected_chain = self.route(topic)
            chain_input = {
                "question": question,
                "chat_history": chat_history
            }
            response = selected_chain.invoke(chain_input)
            
            # Save to memory
            memory.save_context({"input": question}, {"output": response})
            
            return response
        
        return RunnableLambda(process_with_memory)
    
    async def aprocess_query(self, question: str, chat_history: str = "") -> str:
        """
        Asynchronously process a user query.
        
        Args:
            question: The user's question
            chat_history: Previous conversation history
            
        Returns:
            Response from the appropriate chain
        """
        # Classify the question
        classification_input = {
            "question": question,
            "chat_history": chat_history
        }
        topic = await self.classification_chain.ainvoke(classification_input)
        
        # Route to appropriate chain
        selected_chain = self.route(topic)
        
        # Process with the selected chain
        chain_input = {
            "question": question,
            "chat_history": chat_history
        }
        
        return await selected_chain.ainvoke(chain_input)
    
    def get_chain_info(self) -> Dict[str, Any]:
        """
        Get information about available chains.
        
        Returns:
            Dictionary with chain information
        """
        return {
            "classification": {
                "class": self.classification_chain.__class__.__name__,
                "model": self.model
            },
            "news": {
                "class": self.news_chain.__class__.__name__,
                "model": self.model,
                "search_k": self.news_chain.search_k
            },
            "finance": {
                "class": self.finance_chain.__class__.__name__,
                "model": self.model,
                "search_k": self.finance_chain.search_k,
                "milvus_available": self.finance_chain.is_milvus_available()
            },
            "general": {
                "class": self.general_chain.__class__.__name__,
                "model": self.model
            },
            "reset": {
                "class": self.reset_chain.__class__.__name__,
                "model": self.model
            }
        }