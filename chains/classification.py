from config import DEFAULT_MODEL, CRYPTO_KEYWORDS
"""
Classification chain for determining the type of user query.
"""

import asyncio
import concurrent.futures
import torch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from typing import Dict, Any, List
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
        
        # GPU 사용 여부 확인
        self.is_gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
        self.batch_enabled = self.is_gpu_available
        
        print(f"🔍 Classification Chain GPU Status: {'GPU' if self.is_gpu_available else 'CPU'}")
        print(f"🚀 Batch Processing: {'Enabled' if self.batch_enabled else 'Disabled'}")
        
        # 비동기 처리를 위한 ThreadPoolExecutor
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, 
            thread_name_prefix="classification_async"
        )
    
    def _setup_chain(self):
        """Set up the classification chain with prompt and model."""
        
        # Classification prompt template - focused on crypto distinction
        self.prompt = PromptTemplate.from_template(
            """Classify this cryptocurrency question into exactly ONE category:

Categories:
1. 최신소식 - Current prices, recent news, market updates, real-time data
   Examples: "비트코인 현재 가격", "이더리움 급락 이유", "오늘 암호화폐 뉴스", "최근 ETF 승인"

2. 전문지식 - Investment strategies, technical analysis, educational content, how-to guides
   Examples: "투자 전략", "차트 분석 방법", "스테이킹 수익률 계산", "DeFi 원리 설명"

Question: {question}

Answer with only the category name (최신소식 or 전문지식):"""
        )
    
    def _classify_sync(self, question: str, chat_history: str = "") -> str:
        """
        동기적으로 분류를 수행하는 내부 메서드.
        
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
            print(f"[Classification Error] {e}")
            return "최신소식"  # Safe fallback for crypto questions
    
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
        return self._classify_sync(question, chat_history)
    
    async def classify_async(self, question: str, chat_history: str = "") -> str:
        """
        비동기적으로 분류를 수행합니다.
        
        Args:
            question: The user's question
            chat_history: Previous conversation history
            
        Returns:
            Classification result as string
        """
        # ThreadPoolExecutor를 사용하여 동기 작업을 비동기로 실행
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, 
            self._classify_sync, 
            question, 
            chat_history
        )
        return result
    
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
        진짜 비동기로 classification chain을 실행합니다.
        
        Args:
            inputs: Dictionary containing 'question' and 'chat_history'
            
        Returns:
            Classification result as string
        """
        return await self.classify_async(
            inputs.get("question", ""),
            inputs.get("chat_history", "")
        )
    
    async def classify_batch_async(self, questions: List[str], chat_histories: List[str] = None) -> List[str]:
        """
        여러 질문을 배치로 비동기 처리합니다. (GPU에서만 진짜 배치 처리)
        
        Args:
            questions: List of questions to classify
            chat_histories: List of chat histories (optional)
            
        Returns:
            List of classification results
        """
        if chat_histories is None:
            chat_histories = [""] * len(questions)
        
        # GPU가 없거나 질문이 1개면 개별 처리
        if not self.batch_enabled or len(questions) <= 1:
            print(f"📝 Individual processing: GPU={self.is_gpu_available}, Questions={len(questions)}")
            tasks = [
                self.classify_async(question, chat_history)
                for question, chat_history in zip(questions, chat_histories)
            ]
            return await asyncio.gather(*tasks)
        
        # GPU 환경에서 진짜 배치 처리
        print(f"🚀 GPU Batch processing: {len(questions)} questions")
        
        # 빠른 규칙 기반 처리
        fast_results = []
        llm_questions = []
        llm_histories = []
        llm_indices = []
        
        for i, (question, chat_history) in enumerate(zip(questions, chat_histories)):
            question_lower = question.lower()
            
            # 규칙 기반으로 즉시 처리 가능한 경우
            if any(word in question_lower for word in ["리셋", "초기화", "지워", "reset", "clear"]):
                fast_results.append((i, "리셋"))
            elif not any(keyword in question_lower for keyword in CRYPTO_KEYWORDS):
                fast_results.append((i, "기타"))
            else:
                # LLM 처리가 필요한 경우
                llm_questions.append(question)
                llm_histories.append(chat_history)
                llm_indices.append(i)
        
        # LLM 배치 처리 (GPU에서만)
        llm_results = []
        if llm_questions:
            prompts = [
                self.prompt.format(question=q, chat_history=h) 
                for q, h in zip(llm_questions, llm_histories)
            ]
            
            # ThreadPoolExecutor로 GPU 배치 처리
            loop = asyncio.get_event_loop()
            batch_outputs = await loop.run_in_executor(
                self.executor,
                self._batch_generate_sync,
                prompts
            )
            
            # 결과 파싱
            for output_text in batch_outputs:
                if "전문지식" in output_text:
                    llm_results.append("전문지식")
                elif "최신소식" in output_text:
                    llm_results.append("최신소식")
                else:
                    llm_results.append("최신소식")  # 기본값
        
        # 최종 결과 조합
        final_results = [""] * len(questions)
        
        # 빠른 처리 결과 적용
        for idx, result in fast_results:
            final_results[idx] = result
        
        # LLM 처리 결과 적용
        for idx, result in zip(llm_indices, llm_results):
            final_results[idx] = result
        
        print(f"✅ Batch completed: {len(fast_results)} rules, {len(llm_results)} GPU batch")
        return final_results
    
    def _batch_generate_sync(self, prompts: List[str]) -> List[str]:
        """GPU에서 동기 배치 생성 (ThreadPoolExecutor에서 실행)"""
        try:
            # GPU 환경에서 진짜 배치 처리
            outputs = self.llm.generate(prompts, self.sampling_params)
            
            results = []
            for output in outputs:
                if output.outputs:
                    results.append(output.outputs[0].text.strip())
                else:
                    results.append("")
            
            print(f"🎯 GPU Batch processed {len(prompts)} prompts successfully")
            return results
            
        except Exception as e:
            print(f"❌ GPU Batch classification error: {e}")
            return ["최신소식"] * len(prompts)  # 안전한 기본값
    
    def get_batch_info(self) -> Dict[str, Any]:
        """배치 처리 정보 반환"""
        return {
            "gpu_available": self.is_gpu_available,
            "batch_enabled": self.batch_enabled,
            "device": "GPU" if self.is_gpu_available else "CPU",
            "batch_mode": "True GPU Batch" if self.batch_enabled else "Individual Processing"
        }
    
    def __del__(self):
        """클린업: ThreadPoolExecutor 종료"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)