"""
General chain for handling miscellaneous queries.
"""

from typing import Dict, Any


class GeneralChain:
    """
    Handles general queries that don't fit into specific categories.
    """
    
    def __init__(self, model: str = "Qwen/Qwen2-0.5B-Instruct"):
        pass
    
    
    def process(self, question: str, chat_history: str = "") -> str:
        return "도와드리지 못해서 죄송합니다. 저는 코인 관련 전문지식과 최신소식만 답변드릴 수 있습니다."
    
    def invoke(self, inputs: Dict[str, Any]) -> str:
        return self.process(inputs.get("question", ""), inputs.get("chat_history", ""))
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> str:
        return self.invoke(inputs)
    
