from config import DEFAULT_MODEL
"""
Singleton vLLM instance to be shared across all chains.
"""

from vllm import LLM, SamplingParams
from typing import Optional
from config import DEFAULT_MODEL


class VLLMSingleton:
    """Singleton class to manage a single vLLM instance across all chains."""
    
    _instance: Optional['VLLMSingleton'] = None
    _llm: Optional[LLM] = None
    _model: str = ""
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VLLMSingleton, cls).__new__(cls)
        return cls._instance
    
    def get_llm(self, model: str = DEFAULT_MODEL) -> LLM:
        """Get the vLLM instance, creating it if necessary."""
        if self._llm is None or self._model != model:
            print(f"Initializing vLLM with model: {model}")
            self._llm = LLM(model=model)
            self._model = model
        return self._llm
    
    def create_sampling_params(self, temperature: float = 0.7, max_tokens: int = 512) -> SamplingParams:
        """Create sampling parameters for text generation."""
        return SamplingParams(temperature=temperature, max_tokens=max_tokens)


# Global singleton instance
vllm_singleton = VLLMSingleton()