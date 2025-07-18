from config import DEFAULT_MODEL
"""
Reset chain for handling conversation reset requests.
"""

from typing import Dict, Any


class ResetChain:
    """
    Handles conversation reset requests.
    """
    
    def __init__(self, model: str = DEFAULT_MODEL):
        pass
    
    
    def process(self) -> str:
        return "네, 대화 기록을 모두 지웠습니다. 새롭게 시작하겠습니다! 비트코인 관련 전문지식이나 최신소식에 대해 무엇이든 물어보세요."
    
    def invoke(self, inputs: Dict[str, Any]) -> str:
        return self.process()
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> str:
        return self.invoke(inputs)
    
