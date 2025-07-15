#!/usr/bin/env python3
"""
JWT validation test script for thread_manager.py
"""
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.thread_manager import thread_id_manager
import jwt
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_jwt_validation():
    """Test JWT token validation functionality"""
    
    # Get JWT secret from environment
    jwt_secret = os.getenv('JWT_SECRET', 'your-secure-jwt-secret-key-change-this-in-production')
    jwt_algorithm = os.getenv('JWT_ALGORITHM', 'HS256')
    
    print("=== JWT 검증 테스트 시작 ===")
    print(f"JWT Secret: {jwt_secret}")
    print(f"JWT Algorithm: {jwt_algorithm}")
    print()
    
    # Test 1: Valid JWT token
    print("1. 유효한 JWT 토큰 테스트")
    valid_payload = {
        'userId': 12345,
        'iat': datetime.now(timezone.utc),
        'exp': datetime.now(timezone.utc) + timedelta(hours=1)
    }
    
    valid_token = jwt.encode(valid_payload, jwt_secret, algorithm=jwt_algorithm)
    print(f"생성된 토큰: {valid_token}")
    
    # Test parsing
    user_id = thread_id_manager.parse_jwt_token(valid_token)
    print(f"파싱된 User ID: {user_id}")
    
    # Test validation
    is_valid = thread_id_manager.validate_jwt_token(valid_token)
    print(f"토큰 유효성: {is_valid}")
    
    # Test thread ID generation
    thread_id = thread_id_manager.get_or_create_thread_id(valid_token)
    print(f"생성된 Thread ID: {thread_id}")
    print()
    
    # Test 2: Bearer token format
    print("2. Bearer 토큰 형식 테스트")
    bearer_token = f"Bearer {valid_token}"
    user_id_bearer = thread_id_manager.parse_jwt_token(bearer_token)
    print(f"Bearer 토큰에서 파싱된 User ID: {user_id_bearer}")
    print()
    
    # Test 3: Expired JWT token
    print("3. 만료된 JWT 토큰 테스트")
    expired_payload = {
        'userId': 67890,
        'iat': datetime.now(timezone.utc) - timedelta(hours=2),
        'exp': datetime.now(timezone.utc) - timedelta(hours=1)  # 1시간 전에 만료
    }
    
    expired_token = jwt.encode(expired_payload, jwt_secret, algorithm=jwt_algorithm)
    print(f"만료된 토큰: {expired_token}")
    
    user_id_expired = thread_id_manager.parse_jwt_token(expired_token)
    print(f"만료된 토큰에서 파싱된 User ID: {user_id_expired}")
    print()
    
    # Test 4: Invalid signature
    print("4. 잘못된 서명 테스트")
    invalid_signature_token = jwt.encode(valid_payload, "wrong-secret", algorithm=jwt_algorithm)
    print(f"잘못된 서명 토큰: {invalid_signature_token}")
    
    user_id_invalid = thread_id_manager.parse_jwt_token(invalid_signature_token)
    print(f"잘못된 서명 토큰에서 파싱된 User ID: {user_id_invalid}")
    print()
    
    # Test 5: Malformed token
    print("5. 잘못된 형식 토큰 테스트")
    malformed_token = "not.a.valid.jwt.token"
    user_id_malformed = thread_id_manager.parse_jwt_token(malformed_token)
    print(f"잘못된 형식 토큰에서 파싱된 User ID: {user_id_malformed}")
    print()
    
    # Test 6: Token without userId
    print("6. userId가 없는 토큰 테스트")
    no_userid_payload = {
        'username': 'testuser',
        'iat': datetime.now(timezone.utc),
        'exp': datetime.now(timezone.utc) + timedelta(hours=1)
    }
    
    no_userid_token = jwt.encode(no_userid_payload, jwt_secret, algorithm=jwt_algorithm)
    print(f"userId가 없는 토큰: {no_userid_token}")
    
    user_id_no_userid = thread_id_manager.parse_jwt_token(no_userid_token)
    print(f"userId가 없는 토큰에서 파싱된 User ID: {user_id_no_userid}")
    print()
    
    # Test 7: Alternative user ID fields
    print("7. 대체 사용자 ID 필드 테스트")
    alt_payload = {
        'user_id': 11111,  # alternative field name
        'iat': datetime.now(timezone.utc),
        'exp': datetime.now(timezone.utc) + timedelta(hours=1)
    }
    
    alt_token = jwt.encode(alt_payload, jwt_secret, algorithm=jwt_algorithm)
    print(f"user_id 필드 토큰: {alt_token}")
    
    user_id_alt = thread_id_manager.parse_jwt_token(alt_token)
    print(f"user_id 필드에서 파싱된 User ID: {user_id_alt}")
    print()
    
    # Test 8: Session management
    print("8. 세션 관리 테스트")
    print("현재 활성 세션:")
    sessions = thread_id_manager.get_all_sessions()
    print(f"총 세션 수: {sessions['total_sessions']}")
    print(f"사용자 세션: {sessions['user_sessions']}")
    print(f"스레드 사용자: {sessions['thread_users']}")
    print()
    
    print("=== JWT 검증 테스트 완료 ===")

if __name__ == "__main__":
    test_jwt_validation()