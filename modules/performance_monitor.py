# 간단한 성능 모니터링 클래스
import time
from datetime import datetime
from typing import Dict, Any


class SimpleMonitor:
    def __init__(self):
        self.stats = {
            "requests": 0, 
            "errors": 0, 
            "avg_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "total_time": 0.0,
            "start_time": datetime.now()
        }
        self.chain_stats = {
            "classification": {"count": 0, "total_time": 0.0},
            "news": {"count": 0, "total_time": 0.0},
            "finance": {"count": 0, "total_time": 0.0},
            "general": {"count": 0, "total_time": 0.0},
            "reset": {"count": 0, "total_time": 0.0}
        }
    
    def record_request(self, duration: float, success: bool = True, chain_type: str = "unknown"):
        """요청 성능 기록"""
        self.stats["requests"] += 1
        self.stats["total_time"] += duration
        
        if not success:
            self.stats["errors"] += 1
        
        # 평균, 최소, 최대 시간 업데이트
        self.stats["avg_time"] = self.stats["total_time"] / self.stats["requests"]
        self.stats["min_time"] = min(self.stats["min_time"], duration)
        self.stats["max_time"] = max(self.stats["max_time"], duration)
        
        # 체인별 통계
        if chain_type in self.chain_stats:
            self.chain_stats[chain_type]["count"] += 1
            self.chain_stats[chain_type]["total_time"] += duration
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        uptime = (datetime.now() - self.stats["start_time"]).total_seconds()
        
        # 체인별 평균 시간 계산
        chain_averages = {}
        for chain, data in self.chain_stats.items():
            if data["count"] > 0:
                chain_averages[chain] = {
                    "count": data["count"],
                    "avg_time": data["total_time"] / data["count"],
                    "usage_percent": (data["count"] / max(self.stats["requests"], 1)) * 100
                }
        
        return {
            "total_requests": self.stats["requests"],
            "total_errors": self.stats["errors"],
            "error_rate": (self.stats["errors"] / max(self.stats["requests"], 1)) * 100,
            "avg_response_time": round(self.stats["avg_time"], 3),
            "min_response_time": round(self.stats["min_time"] if self.stats["min_time"] != float('inf') else 0, 3),
            "max_response_time": round(self.stats["max_time"], 3),
            "uptime_seconds": round(uptime, 1),
            "requests_per_minute": round((self.stats["requests"] / max(uptime / 60, 1)), 2),
            "chain_performance": chain_averages,
            "timestamp": datetime.now().isoformat()
        }
    
    def reset_stats(self):
        """통계 초기화"""
        self.__init__()


# 전역 모니터 인스턴스
monitor = SimpleMonitor()
