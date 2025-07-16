"""
General chain for handling miscellaneous queries.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any
from .vllm_singleton import vllm_singleton


class GeneralChain:
    """
    Handles general queries that don't fit into specific categories.
    """
    
    def __init__(self, model: str = "Qwen/Qwen2-0.5B-Instruct"):
        """
        Initialize the general chain.
        
        Args:
            model: vLLM model to use for response generation
        """
        self.model = model
        self.llm = vllm_singleton.get_llm(model)
        self.sampling_params = vllm_singleton.create_sampling_params(temperature=0.7, max_tokens=512)
        self._setup_chain()
    
    def _setup_chain(self):
        """Set up the general chain with prompt and model."""
        
        # General prompt template
        self.prompt = PromptTemplate.from_template(
            """You are a Bitcoin chatbot. You must respond in Korean only.

Previous conversation:
{chat_history}

For questions NOT related to Bitcoin/cryptocurrency, always respond exactly with:
"도와드리지 못해서 죄송합니다. 저는 코인 관련 전문지식과 최신소식만 답변드릴 수 있습니다."

Question: {question}

Korean response:"""
        )
        
        # Setup vLLM-based processing
    
    def process(self, question: str, chat_history: str = "") -> str:
        """
        Process a general question.
        
        Args:
            question: The user's question
            chat_history: Previous conversation history
            
        Returns:
            General response or polite refusal
        """
        # For non-crypto questions, just return the standard Korean message
        return "도와드리지 못해서 죄송합니다. 저는 비트코인 관련 전문지식과 최신소식만 답변드릴 수 있습니다."
    
    def invoke(self, inputs: Dict[str, Any]) -> str:
        """
        Invoke the general chain with inputs.
        
        Args:
            inputs: Dictionary containing 'question' and optional 'chat_history'
            
        Returns:
            General response as string
        """
        return self.process(
            inputs.get("question", ""),
            inputs.get("chat_history", "")
        )
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> str:
        """
        Asynchronously invoke the general chain.
        
        Args:
            inputs: Dictionary containing 'question' and optional 'chat_history'
            
        Returns:
            General response as string
        """
        # vLLM doesn't have native async support, so we use sync method
        return self.invoke(inputs)
    
    def get_chain(self):
        """Get the underlying vLLM object."""
        return self.llm