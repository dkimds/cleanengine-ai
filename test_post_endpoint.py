#!/usr/bin/env python3
"""
Test script for POST endpoint functionality.
"""
import requests
import json

def test_post_endpoint():
    """Test the POST chat endpoint"""
    
    # Test data
    test_cases = [
        {
            "question": "최신 비트코인 뉴스를 알려주세요",
            "expected_prefix": "최신 데이터에 따르면"
        },
        {
            "question": "비트코인 투자 전략은?",
            "expected_prefix": "전문가에 따르면"
        },
        {
            "question": "안녕하세요",
            "expected_prefix": "도와드리지 못해서 죄송합니다"
        },
        {
            "question": "리셋해주세요",
            "expected_prefix": "네, 대화 기록을 모두 지웠습니다"
        }
    ]
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing POST endpoints...")
    print("=" * 50)
    
    # Test health endpoint first
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed")
            print(f"   Milvus connected: {health_data.get('milvus_connected', 'Unknown')}")
            print(f"   Available chains: {list(health_data.get('chain_info', {}).keys())}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return
    
    # Test POST chat endpoint
    print("\n2. Testing POST chat endpoint...")
    for i, test_case in enumerate(test_cases, 1):
        try:
            print(f"\n   Test {i}: {test_case['question']}")
            
            # Make POST request
            response = requests.post(
                f"{base_url}/chat",
                json={"question": test_case["question"]},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "")
                
                # Check if response starts with expected prefix
                if response_text.startswith(test_case["expected_prefix"]):
                    print(f"   ✅ Correct routing: {response_text[:50]}...")
                else:
                    print(f"   ⚠️  Unexpected response: {response_text[:50]}...")
                    
                print(f"   📅 Timestamp: {data.get('timestamp', 'N/A')}")
                
            else:
                print(f"   ❌ Request failed: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Test error: {e}")
    
    # Test reset endpoint
    print("\n3. Testing reset endpoint...")
    try:
        response = requests.post(f"{base_url}/reset")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Reset successful: {data.get('message', 'No message')}")
        else:
            print(f"❌ Reset failed: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Reset error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Test completed!")

if __name__ == "__main__":
    print("Make sure to start the server first:")
    print("uvicorn main:app --reload")
    print("")
    
    test_post_endpoint()