"""
Reset chain for handling conversation reset requests.
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any


class ResetChain:
    """
    Handles conversation reset requests.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the reset chain.
        
        Args:
            model: OpenAI model to use for response generation
        """
        self.model = model
        self._setup_chain()
    
    def _setup_chain(self):
        """Set up the reset chain with prompt and model."""
        
        # Reset prompt template
        self.prompt = PromptTemplate.from_template(
            """네, 대화 기록을 모두 지웠습니다. 새롭게 시작하겠습니다! 
        비트코인 관련 전문지식이나 최신소식에 대해 무엇이든 물어보세요."""
        )
        
        # Create the chain
        self.chain = (
            self.prompt
            | ChatOpenAI(model=self.model)
            | StrOutputParser()
        )
    
    def process(self, question: str = "", chat_history: str = "") -> str:
        """
        Process a reset request.
        
        Args:
            question: The user's question (not used for reset)
            chat_history: Previous conversation history (not used for reset)
            
        Returns:
            Reset confirmation message
        """
        return self.chain.invoke({"question": question})
    
    def invoke(self, inputs: Dict[str, Any]) -> str:
        """
        Invoke the reset chain with inputs.
        
        Args:
            inputs: Dictionary containing 'question' (not used for reset)
            
        Returns:
            Reset confirmation message as string
        """
        return self.chain.invoke(inputs)
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> str:
        """
        Asynchronously invoke the reset chain.
        
        Args:
            inputs: Dictionary containing 'question' (not used for reset)
            
        Returns:
            Reset confirmation message as string
        """
        return await self.chain.ainvoke(inputs)
    
    def get_chain(self):
        """Get the underlying chain object."""
        return self.chain