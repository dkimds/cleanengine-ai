from config import DEFAULT_MODEL, CRYPTO_KEYWORDS
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
    
    def __init__(self, model: str = DEFAULT_MODEL):
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
        
        # Classification prompt template - focused on crypto distinction
        self.prompt = PromptTemplate.from_template(
"""Classify this cryptocurrency question into exactly ONE category:

Categories:
1. 최신소식 - Questions about CURRENT events, prices, or recent happenings
   - Key indicators: 오늘, 현재, 지금, 최근, 갑자기, 방금, 어제, 이번주
   - Examples: "비트코인 현재 가격", "오늘 급락 원인", "최근 ETF 뉴스"
   - Focus: What is happening NOW or recently happened

2. 전문지식 - Questions asking HOW TO do something or EXPLAIN concepts
   - Key indicators: 방법, 전략, 원리, 설명, 어떻게, 계산, 분석
   - Examples: "투자 전략", "차트 분석 방법", "DeFi 원리", "수익률 계산법"
   - Focus: Learning, understanding, or step-by-step guidance

CRITICAL RULE:
- If the question contains time words (오늘, 현재, 지금, 최근, 갑자기) → 최신소식
- If the question asks "how to" or "explain" → 전문지식

Question: {question}

Category:"""
        )
        
        # Setup vLLM-based processing
    
    def classify(self, question: str, chat_history: str = "") -> str:
        """
        Classify a user question using true hybrid approach.
        Rules for obvious cases, LLM for crypto ambiguity.
        
        Args:
            question: The user's question
            chat_history: Previous conversation history
            
        Returns:
            Classification result as string
        """
        question_lower = question.lower()
        
        # 1. RULES FIRST - Fast path for obvious cases
        
        # Reset requests (highest priority)
        if any(word in question_lower for word in ["리셋", "초기화", "지워", "reset", "clear"]):
            return "리셋"
        
        # Non-crypto questions → 기타
        has_crypto = any(keyword in question_lower for keyword in CRYPTO_KEYWORDS)
        
        if not has_crypto:
            return "기타"

        # 2. LLM for ambiguous crypto questions
        try:
            prompt_text = self.prompt.format(
                question=question,
                chat_history=chat_history
            )
            outputs = self.llm.generate([prompt_text], self.sampling_params)
            result = outputs[0].outputs[0].text.strip()
            
            # Extract category from LLM output
            if "전문지식" in result:
                return "전문지식"
            elif "최신소식" in result:
                return "최신소식"
            else:
                # Default for crypto questions
                return "최신소식"
                
        except Exception as e:
            return "최신소식"  # Safe fallback for crypto questions
    
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