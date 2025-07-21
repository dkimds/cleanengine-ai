from config import DEFAULT_MODEL
"""
Chain router for routing queries to appropriate chain handlers.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    
    def __init__(self, model: str = DEFAULT_MODEL):
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
        
        # OpenAI model for news and finance chains
        self.openai_model = "gpt-4o-mini"
        
        # Chain routing map (chains will be created with thread_id when needed)
        self.chain_map = {
            "최신소식": "news",
            "전문지식": "finance",
            "리셋": self.reset_chain,
            "기타": self.general_chain
        }
    
    def route(self, topic: str, thread_id: Optional[str] = None):
        """
        Route to the appropriate chain based on topic.
        
        Args:
            topic: Classification result from classification chain
            thread_id: Thread ID for memory management
            
        Returns:
            Appropriate chain handler
        """
        topic = topic.strip()
        chain_type = self.chain_map.get(topic, self.general_chain)
        
        # Create news/finance chains with thread_id
        if chain_type == "news":
            return NewsChain(self.openai_model, thread_id=thread_id)
        elif chain_type == "finance":
            return FinanceChain(self.openai_model, thread_id=thread_id)
        else:
            return chain_type
    
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
        async def process_with_memory_async(inputs):
            """진짜 비동기 메모리 처리 함수"""
            question = inputs["question"]
            memory = memory_manager.get_memory(thread_id)
            chat_history = memory_manager.get_chat_history_string(memory)
            
            # 비동기로 분류
            classification_input = {
                "question": question,
                "chat_history": chat_history
            }
            topic = await self.classification_chain.ainvoke(classification_input)
            
            # Reset 요청 처리
            if topic.strip() == "리셋":
                memory_manager.reset_memory(thread_id)
                return self.reset_chain.invoke({"question": question})
            
            # 적절한 체인으로 라우팅
            selected_chain = self.route(topic, thread_id)
            chain_input = {
                "question": question,
                "chat_history": chat_history
            }
            
            # 체인에 따라 비동기/동기 처리
            if hasattr(selected_chain, 'ainvoke'):
                response = await selected_chain.ainvoke(chain_input)
            else:
                response = selected_chain.invoke(chain_input)
            
            # 메모리에 저장
            memory.save_context({"input": question}, {"output": response})
            
            return response
        
        def process_with_memory_sync(inputs):
            """동기 버전 (호환성용)"""
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
            selected_chain = self.route(topic, thread_id)
            chain_input = {
                "question": question,
                "chat_history": chat_history
            }
            response = selected_chain.invoke(chain_input)
            
            # Save to memory
            memory.save_context({"input": question}, {"output": response})
            
            return response
        
        # RunnableLambda 생성
        lambda_chain = RunnableLambda(process_with_memory_sync)
        
        # 진짜 비동기 메서드 추가
        lambda_chain.ainvoke = process_with_memory_async
        
        return lambda_chain
    
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
                "class": "NewsChain",
                "model": self.openai_model,
                "search_k": 3
            },
            "finance": {
                "class": "FinanceChain", 
                "model": self.openai_model,
                "search_k": 3
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