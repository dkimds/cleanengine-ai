import base64
import json
import uuid
from datetime import datetime
from typing import Dict, Optional
import threading
import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError, InvalidSignatureError
import os
from dotenv import load_dotenv

load_dotenv()


class ThreadIDManager:
    """JWT 토큰을 파싱하여 사용자 ID를 추출하고 스레드 ID를 생성/관리하는 클래스"""
    
    def __init__(self):
        self.user_sessions: Dict[int, str] = {}  # user_id -> thread_id 매핑
        self.thread_users: Dict[str, int] = {}   # thread_id -> user_id 매핑
        self.lock = threading.Lock()
        
        # JWT 설정
        self.jwt_secret = os.getenv('JWT_SECRET', 'your-secret-key-here')
        self.jwt_algorithm = os.getenv('JWT_ALGORITHM', 'HS256')
        self.jwt_verify_exp = os.getenv('JWT_VERIFY_EXP', 'True').lower() == 'true'
        
        if self.jwt_secret == 'your-secret-key-here':
            print("[보안 경고] JWT_SECRET 환경 변수가 설정되지 않았습니다. 기본값을 사용합니다.")
    
    def parse_jwt_token(self, access_token: str) -> Optional[int]:
        """
        JWT 토큰에서 userId를 추출합니다.
        PyJWT 라이브러리를 사용하여 토큰을 안전하게 검증하고 파싱합니다.
        
        Args:
            access_token: JWT 토큰 문자열
            
        Returns:
            userId (int) 또는 None (파싱 실패시)
        """
        try:
            # Bearer 토큰 형식인 경우 "Bearer " 제거
            if access_token.startswith('Bearer '):
                access_token = access_token[7:]
            
            # JWT 토큰 디코딩 및 검증
            payload = jwt.decode(
                access_token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm],
                options={
                    'verify_signature': True,
                    'verify_exp': self.jwt_verify_exp,
                    'verify_iat': True,
                    'verify_nbf': True,
                    'verify_aud': False,  # audience 검증은 필요에 따라 설정
                    'verify_iss': False,  # issuer 검증은 필요에 따라 설정
                }
            )
            
            # userId 추출
            user_id = payload.get('userId') or payload.get('user_id') or payload.get('sub')
            if user_id is None:
                print(f"[JWT 파싱 오류] userId가 payload에 없습니다: {list(payload.keys())}")
                return None
            
            # 사용자 ID를 정수로 변환
            try:
                user_id = int(user_id)
            except (ValueError, TypeError):
                print(f"[JWT 파싱 오류] userId가 유효한 정수가 아닙니다: {user_id}")
                return None
                
            print(f"[JWT 파싱 성공] User ID: {user_id}")
            return user_id
            
        except ExpiredSignatureError:
            print("[JWT 파싱 오류] 토큰이 만료되었습니다")
            return None
        except InvalidSignatureError:
            print("[JWT 파싱 오류] 토큰 서명이 유효하지 않습니다")
            return None
        except InvalidTokenError as e:
            print(f"[JWT 파싱 오류] 유효하지 않은 토큰: {e}")
            return None
        except Exception as e:
            print(f"[JWT 파싱 오류] 예기치 않은 오류: {e}")
            return None
    
    def validate_jwt_token(self, access_token: str) -> bool:
        """
        JWT 토큰의 유효성을 검증합니다.
        
        Args:
            access_token: JWT 토큰 문자열
            
        Returns:
            bool: 토큰이 유효한 경우 True, 그렇지 않으면 False
        """
        return self.parse_jwt_token(access_token) is not None
    
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
