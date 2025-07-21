from config import DEFAULT_MODEL
"""
Singleton vLLM instance to be shared across all chains.
"""

from vllm import LLM, SamplingParams
from typing import Optional, Dict, Any
import os
import torch
import psutil
from config import DEFAULT_MODEL


class VLLMSingleton:
    """Singleton class to manage a single vLLM instance across all chains."""
    
    _instance: Optional['VLLMSingleton'] = None
    _llm: Optional[LLM] = None
    _model: str = ""
    _gpu_config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VLLMSingleton, cls).__new__(cls)
        return cls._instance
    
    def _get_optimal_gpu_config(self) -> Dict[str, Any]:
        """GPU 환경에 맞는 최적 설정을 자동으로 계산합니다."""
        config = {}
        
        # GPU 사용 가능 여부 확인
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, using CPU mode")
            return {"device": "cpu"}
        
        # GPU 메모리 정보 수집
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            return {"device": "cpu"}
            
        # 첫 번째 GPU 메모리 정보
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print(f"Detected {gpu_count} GPU(s), GPU 0 memory: {gpu_memory_gb:.1f}GB")
        
        # GPU 메모리에 따른 설정 최적화
        if gpu_memory_gb >= 24:  # RTX 4090, A100 등
            config.update({
                "gpu_memory_utilization": 0.85,  # 85% 사용
                "max_num_seqs": 256,             # 높은 동시성
                "max_model_len": 4096,           # 긴 컨텍스트
            })
        elif gpu_memory_gb >= 16:  # RTX 4080, V100 등  
            config.update({
                "gpu_memory_utilization": 0.8,   # 80% 사용
                "max_num_seqs": 128,             # 중간 동시성
                "max_model_len": 2048,           # 중간 컨텍스트
            })
        elif gpu_memory_gb >= 8:   # RTX 3070, 4060Ti 등
            config.update({
                "gpu_memory_utilization": 0.75,  # 75% 사용  
                "max_num_seqs": 64,              # 낮은 동시성
                "max_model_len": 1024,           # 짧은 컨텍스트
            })
        else:  # 8GB 미만
            config.update({
                "gpu_memory_utilization": 0.7,   # 70% 사용 (안전)
                "max_num_seqs": 32,              # 최소 동시성
                "max_model_len": 512,            # 최소 컨텍스트
            })
        
        # 멀티 GPU 설정
        if gpu_count > 1:
            config["tensor_parallel_size"] = min(gpu_count, 2)  # 최대 2개 GPU 사용
            print(f"Using tensor parallelism with {config['tensor_parallel_size']} GPUs")
        
        # 추가 최적화 설정
        config.update({
            "swap_space": 4,                 # 4GB 스왑 공간 (CPU 메모리)
            "disable_log_stats": False,      # 성능 통계 활성화
            "block_size": 16,                # 메모리 블록 크기 최적화
        })
        
        return config
    
    def _get_env_gpu_config(self) -> Dict[str, Any]:
        """환경변수로부터 GPU 설정을 읽어옵니다."""
        config = {}
        
        # 환경변수 우선순위로 설정 오버라이드
        if os.getenv("VLLM_GPU_MEMORY_UTILIZATION"):
            config["gpu_memory_utilization"] = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION"))
            
        if os.getenv("VLLM_MAX_NUM_SEQS"):
            config["max_num_seqs"] = int(os.getenv("VLLM_MAX_NUM_SEQS"))
            
        if os.getenv("VLLM_MAX_MODEL_LEN"):
            config["max_model_len"] = int(os.getenv("VLLM_MAX_MODEL_LEN"))
            
        if os.getenv("VLLM_TENSOR_PARALLEL_SIZE"):
            config["tensor_parallel_size"] = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE"))
        
        return config
    
    def get_llm(self, model: str = DEFAULT_MODEL) -> LLM:
        """Get the vLLM instance, creating it if necessary."""
        if self._llm is None or self._model != model:
            print(f"Initializing vLLM with model: {model}")
            
            # GPU 설정 계산
            auto_config = self._get_optimal_gpu_config()
            env_config = self._get_env_gpu_config()
            
            # 환경변수가 자동 설정을 오버라이드
            final_config = {**auto_config, **env_config}
            self._gpu_config = final_config
            
            print("vLLM GPU Configuration:")
            for key, value in final_config.items():
                print(f"  {key}: {value}")
            
            try:
                # vLLM 인스턴스 생성
                self._llm = LLM(model=model, **final_config)
                self._model = model
                
                print("✅ vLLM initialized successfully")
                self._print_memory_usage()
                
            except Exception as e:
                print(f"❌ vLLM initialization failed: {e}")
                
                # 실패 시 더 보수적인 설정으로 재시도
                fallback_config = {
                    "gpu_memory_utilization": 0.6,
                    "max_num_seqs": 16,
                    "max_model_len": 512,
                }
                print("Retrying with fallback configuration...")
                print(f"Fallback config: {fallback_config}")
                
                self._llm = LLM(model=model, **fallback_config)
                self._model = model
                self._gpu_config = fallback_config
                
                print("✅ vLLM initialized with fallback config")
                
        return self._llm
    
    def _print_memory_usage(self):
        """현재 메모리 사용량을 출력합니다."""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                
                print(f"GPU {i} Memory Usage:")
                print(f"  Allocated: {allocated:.2f}GB")
                print(f"  Reserved: {reserved:.2f}GB") 
                print(f"  Total: {total:.2f}GB")
                print(f"  Utilization: {reserved/total*100:.1f}%")
        
        # CPU 메모리도 확인
        cpu_memory = psutil.virtual_memory()
        print(f"CPU Memory: {cpu_memory.used/(1024**3):.2f}GB / {cpu_memory.total/(1024**3):.2f}GB ({cpu_memory.percent:.1f}%)")
    
    def get_gpu_config(self) -> Dict[str, Any]:
        """현재 GPU 설정을 반환합니다."""
        return self._gpu_config.copy()
    
    def create_sampling_params(self, temperature: float = 0.7, max_tokens: int = 512) -> SamplingParams:
        """Create sampling parameters for text generation."""
        return SamplingParams(temperature=temperature, max_tokens=max_tokens)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """현재 메모리 사용 통계를 반환합니다."""
        stats = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                
                stats[f"gpu_{i}"] = {
                    "allocated_gb": round(allocated, 2),
                    "reserved_gb": round(reserved, 2),
                    "total_gb": round(total, 2),
                    "utilization_percent": round(reserved/total*100, 1)
                }
        
        cpu_memory = psutil.virtual_memory()
        stats["cpu"] = {
            "used_gb": round(cpu_memory.used/(1024**3), 2),
            "total_gb": round(cpu_memory.total/(1024**3), 2),
            "utilization_percent": round(cpu_memory.percent, 1)
        }
        
        return stats


# Global singleton instance
vllm_singleton = VLLMSingleton()