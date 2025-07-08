import base64
import json
import uuid
from datetime import datetime
from typing import Dict, Optional
import threading


class ThreadIDManager:
    """JWT 토큰을 파싱하여 사용자 ID를 추출하고 스레드 ID를 생성/관리하는 클래스"""
    
    def __init__(self):
        self.user_sessions: Dict[int, str] = {}  # user_id -> thread_id 매핑
        self.thread_users: Dict[str, int] = {}   # thread_id -> user_id 매핑
        self.lock = threading.Lock()
    
    def parse_jwt_token(self, access_token: str) -> Optional[int]:
        """
        JWT 토큰에서 userId를 추출합니다.
        
        Args:
            access_token: JWT 토큰 문자열
            
        Returns:
            userId (int) 또는 None (파싱 실패시)
        """
        try:
            # JWT는 header.payload.signature 형태
            parts = access_token.split('.')
            if len(parts) != 3:
                print(f"[JWT 파싱 오류] 잘못된 JWT 형식: {len(parts)}개 부분")
                return None
            
            # payload 부분을 base64 디코딩
            payload = parts[1]
            
            # Base64 패딩 추가 (필요한 경우)
            padding = len(payload) % 4
            if padding:
                payload += '=' * (4 - padding)
            
            # Base64 디코딩
            decoded_bytes = base64.b64decode(payload)
            decoded_str = decoded_bytes.decode('utf-8')
            
            # JSON 파싱
            payload_data = json.loads(decoded_str)
            
            # userId 추출
            user_id = payload_data.get('userId')
            if user_id is None:
                print(f"[JWT 파싱 오류] userId가 payload에 없습니다: {payload_data}")
                return None
                
            print(f"[JWT 파싱 성공] User ID: {user_id}")
            return int(user_id)
            
        except Exception as e:
            print(f"[JWT 파싱 오류] {e}")
            return None
    
    def get_or_create_thread_id(self, access_token: str) -> Optional[str]:
        """
        JWT 토큰에서 사용자 ID를 추출하고, 해당 사용자의 스레드 ID를 반환하거나 새로 생성합니다.
        
        Args:
            access_token: JWT 토큰
            
        Returns:
            thread_id (str) 또는 None (토큰이 유효하지 않은 경우)
        """
        user_id = self.parse_jwt_token(access_token)
        if user_id is None:
            return None
        
        with self.lock:
            # 기존 세션이 있는지 확인
            if user_id in self.user_sessions:
                thread_id = self.user_sessions[user_id]
                print(f"[스레드 ID] 기존 세션 사용 - User: {user_id}, Thread: {thread_id}")
                return thread_id
            
            # 새로운 스레드 ID 생성
            thread_id = f"user_{user_id}_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
            
            # 매핑 저장
            self.user_sessions[user_id] = thread_id
            self.thread_users[thread_id] = user_id
            
            print(f"[스레드 ID] 새 세션 생성 - User: {user_id}, Thread: {thread_id}")
            return thread_id
    
    def get_user_id_from_thread(self, thread_id: str) -> Optional[int]:
        """스레드 ID로부터 사용자 ID를 조회합니다."""
        with self.lock:
            return self.thread_users.get(thread_id)
    
    def reset_user_session(self, access_token: str) -> bool:
        """사용자의 세션을 리셋하고 새로운 스레드 ID를 생성합니다."""
        user_id = self.parse_jwt_token(access_token)
        if user_id is None:
            return False
        
        with self.lock:
            # 기존 세션 정리
            if user_id in self.user_sessions:
                old_thread_id = self.user_sessions[user_id]
                del self.thread_users[old_thread_id]
                del self.user_sessions[user_id]
            
            # 새로운 스레드 ID 생성
            new_thread_id = f"user_{user_id}_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
            self.user_sessions[user_id] = new_thread_id
            self.thread_users[new_thread_id] = user_id
            
            print(f"[세션 리셋] User: {user_id}, New Thread: {new_thread_id}")
            return True
    
    def get_all_sessions(self) -> Dict:
        """모든 활성 세션 정보를 반환합니다."""
        with self.lock:
            return {
                "user_sessions": dict(self.user_sessions),
                "thread_users": dict(self.thread_users),
                "total_sessions": len(self.user_sessions)
            }
    
    def cleanup_session(self, thread_id: str) -> bool:
        """특정 스레드 ID의 세션을 정리합니다."""
        with self.lock:
            if thread_id in self.thread_users:
                user_id = self.thread_users[thread_id]
                del self.thread_users[thread_id]
                if user_id in self.user_sessions:
                    del self.user_sessions[user_id]
                print(f"[세션 정리] Thread: {thread_id}, User: {user_id}")
                return True
            return False


# 전역 스레드 ID 매니저 인스턴스
thread_id_manager = ThreadIDManager()
