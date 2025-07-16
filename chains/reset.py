"""
Reset chain for handling conversation reset requests.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any
from .vllm_singleton import vllm_singleton


class ResetChain:
    """
    Handles conversation reset requests.
    """
    
    def __init__(self, model: str = "Qwen/Qwen2-0.5B-Instruct"):
        """
        Initialize the reset chain.
        
        Args:
            model: vLLM model to use for response generation
        """
        self.model = model
        self.llm = vllm_singleton.get_llm(model)
        self.sampling_params = vllm_singleton.create_sampling_params(temperature=0.7, max_tokens=512)
        self._setup_chain()
    
    def _setup_chain(self):
        """Set up the reset chain with prompt and model."""
        
        # Reset prompt template
        self.prompt = PromptTemplate.from_template(
            """네, 대화 기록을 모두 지웠습니다. 새롭게 시작하겠습니다! 
        비트코인 관련 전문지식이나 최신소식에 대해 무엇이든 물어보세요."""
        )
        
        # Setup vLLM-based processing
    
    def process(self, question: str = "", chat_history: str = "") -> str:
        """
        Process a reset request.
        
        Args:
            question: The user's question (not used for reset)
            chat_history: Previous conversation history (not used for reset)
            
        Returns:
            Reset confirmation message
        """
        prompt_text = self.prompt.format()
        outputs = self.llm.generate([prompt_text], self.sampling_params)
        return outputs[0].outputs[0].text.strip()
    
    def invoke(self, inputs: Dict[str, Any]) -> str:
        """
        Invoke the reset chain with inputs.
        
        Args:
            inputs: Dictionary containing 'question' (not used for reset)
            
        Returns:
            Reset confirmation message as string
        """
        return self.process(inputs.get("question", ""))
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> str:
        """
        Asynchronously invoke the reset chain.
        
        Args:
            inputs: Dictionary containing 'question' (not used for reset)
            
        Returns:
            Reset confirmation message as string
        """
        # vLLM doesn't have native async support, so we use sync method
        return self.invoke(inputs)
    
    def get_chain(self):
        """Get the underlying vLLM object."""
        return self.llm