from datetime import datetime, timedelta
import threading
import time
from typing import Dict
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage


class ThreadMemoryManager:
    """스레드별 대화 메모리를 관리하는 클래스"""
    
    def __init__(self):
        self.memories: Dict[str, ConversationBufferWindowMemory] = {}
        self.last_activity: Dict[str, datetime] = {}
        self.lock = threading.Lock()
        
        # 백그라운드 스레드로 30분마다 비활성 메모리 정리
        self.cleanup_thread = threading.Thread(target=self._cleanup_inactive_memories, daemon=True)
        self.cleanup_thread.start()
    
    def get_memory(self, thread_id: str) -> ConversationBufferWindowMemory:
        """스레드 ID에 해당하는 메모리를 가져오거나 생성합니다."""
        with self.lock:
            if thread_id not in self.memories:
                self.memories[thread_id] = ConversationBufferWindowMemory(
                    k=10,  # 최근 10개의 대화만 기억
                    return_messages=True
                )
            
            self.last_activity[thread_id] = datetime.now()
            return self.memories[thread_id]
    
    def reset_memory(self, thread_id: str) -> bool:
        """특정 스레드의 메모리를 초기화합니다."""
        with self.lock:
            if thread_id in self.memories:
                self.memories[thread_id].clear()
                self.last_activity[thread_id] = datetime.now()
                return True
            return False
    
    def cleanup_thread_memory(self, thread_id: str) -> bool:
        """특정 스레드의 메모리를 완전히 삭제합니다."""
        with self.lock:
            deleted = False
            if thread_id in self.memories:
                del self.memories[thread_id]
                deleted = True
            if thread_id in self.last_activity:
                del self.last_activity[thread_id]
                deleted = True
            
            if deleted:
                print(f"[메모리 정리] 스레드 {thread_id} 메모리 삭제 완료")
            return deleted
    
    def get_chat_history_string(self, memory: ConversationBufferWindowMemory) -> str:
        """메모리에서 대화 히스토리를 문자열로 변환합니다."""
        try:
            messages = memory.chat_memory.messages
            if not messages:
                return ""
            
            history_parts = []
            for message in messages[-6:]:  # 최근 6개 메시지만 사용
                if isinstance(message, HumanMessage):
                    history_parts.append(f"Human: {message.content}")
                elif isinstance(message, AIMessage):
                    history_parts.append(f"Assistant: {message.content}")
            
            return "\n".join(history_parts)
        except Exception as e:
            print(f"대화 히스토리 변환 오류: {e}")
            return ""
    
    def get_all_active_threads(self) -> Dict:
        """모든 활성 스레드 정보를 반환합니다."""
        with self.lock:
            active_threads = []
            current_time = datetime.now()
            
            for thread_id, last_activity in self.last_activity.items():
                time_since_activity = current_time - last_activity
                active_threads.append({
                    "thread_id": thread_id,
                    "last_activity": last_activity,
                    "minutes_since_activity": int(time_since_activity.total_seconds() / 60),
                    "messages_count": len(self.memories[thread_id].chat_memory.messages) if thread_id in self.memories else 0
                })
            
            return {"active_threads": active_threads, "total_count": len(active_threads)}
    
    def _cleanup_inactive_memories(self):
        """30분 동안 비활성 상태인 메모리를 정리합니다."""
        while True:
            time.sleep(300)  # 5분마다 체크
            current_time = datetime.now()
            inactive_threshold = timedelta(minutes=30)
            
            with self.lock:
                inactive_threads = []
                for thread_id, last_time in self.last_activity.items():
                    if current_time - last_time > inactive_threshold:
                        inactive_threads.append(thread_id)
                
                for thread_id in inactive_threads:
                    print(f"[메모리 정리] 스레드 {thread_id} 비활성으로 인한 메모리 삭제")
                    if thread_id in self.memories:
                        del self.memories[thread_id]
                    del self.last_activity[thread_id]


# 전역 메모리 매니저 인스턴스
memory_manager = ThreadMemoryManager()
