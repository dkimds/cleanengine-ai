"""
Classification chain for determining the type of user query.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from typing import Dict, Any
from .vllm_singleton import vllm_singleton


class ClassificationChain:
    """
    Classifies user questions into categories: 최신소식, 전문지식, 리셋, or 기타
    """
    
    def __init__(self, model: str = "Qwen/Qwen2-0.5B-Instruct"):
        """
        Initialize the classification chain.
        
        Args:
            model: vLLM model to use for classification
        """
        self.model = model
        self.llm = vllm_singleton.get_llm(model)
        self.sampling_params = vllm_singleton.create_sampling_params(temperature=0.0, max_tokens=20)
        self._setup_chain()
    
    def _setup_chain(self):
        """Set up the classification chain with prompt and model."""
        
        # Classification prompt template
        self.prompt = PromptTemplate.from_template(
            """Classify this question into exactly ONE category:

Categories:
1. 최신소식 - Bitcoin/crypto prices, latest news, recent updates
   Examples: "비트코인 가격은?", "최신 비트코인 뉴스", "현재 이더리움 시세"

2. 전문지식 - Technical analysis, investment strategies, crypto education
   Examples: "비트코인 투자 전략", "블록체인 기술 설명", "차트 분석 방법"

3. 리셋 - Reset/clear conversation requests
   Examples: "리셋", "초기화", "지워줘", "reset", "clear"

4. 기타 - Everything else (food, weather, general chat, non-crypto topics)
   Examples: "점심 뭐야", "날씨 어때", "안녕하세요", "영화 추천"

Question: {question}

Answer with only the category name:"""
        )
        
        # Setup vLLM-based processing
    
    def classify(self, question: str, chat_history: str = "") -> str:
        """
        Classify a user question.
        
        Args:
            question: The user's question
            chat_history: Previous conversation history
            
        Returns:
            Classification result as string
        """
        prompt_text = self.prompt.format(
            question=question,
            chat_history=chat_history
        )
        outputs = self.llm.generate([prompt_text], self.sampling_params)
        result = outputs[0].outputs[0].text.strip()
        
        # Extract just the classification category
        valid_categories = ["최신소식", "전문지식", "리셋", "기타"]
        
        # Try to extract category from vLLM output, but be careful about false matches
        question_lower = question.lower()
        
        # First, do content-based classification as primary method
        # Check if question contains Bitcoin/crypto keywords
        crypto_keywords = ["비트코인", "bitcoin", "btc", "이더리움", "ethereum", "암호화폐", "코인", "crypto"]
        has_crypto = any(keyword in question_lower for keyword in crypto_keywords)
        
        if any(word in question_lower for word in ["리셋", "초기화", "지워", "reset", "clear"]):
            return "리셋"
        elif not has_crypto:
            # Non-crypto questions go to 기타
            return "기타"
        elif any(word in question_lower for word in ["가격", "시세", "얼마", "달러", "원", "뉴스", "최신"]):
            return "최신소식"
        elif any(word in question_lower for word in ["전략", "분석", "방법", "기술", "블록체인", "투자법"]):
            return "전문지식"
        else:
            # Default crypto questions to 최신소식
            return "최신소식"
    
    def invoke(self, inputs: Dict[str, Any]) -> str:
        """
        Invoke the classification chain with inputs.
        
        Args:
            inputs: Dictionary containing 'question' and 'chat_history'
            
        Returns:
            Classification result as string
        """
        return self.classify(
            inputs.get("question", ""),
            inputs.get("chat_history", "")
        )
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> str:
        """
        Asynchronously invoke the classification chain.
        
        Args:
            inputs: Dictionary containing 'question' and 'chat_history'
            
        Returns:
            Classification result as string
        """
        # vLLM doesn't have native async support, so we use sync method
        return self.invoke(inputs)