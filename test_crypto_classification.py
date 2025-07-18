#!/usr/bin/env python3
"""
Test script for crypto news vs expertise classification.
Tests the classification chain with the provided crypto queries.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chains.classification import ClassificationChain

def test_crypto_classification():
    """Test classification of crypto news vs expertise queries."""
    
    # Initialize classification chain
    classifier = ClassificationChain()
    
    # Test queries with expected classifications
    test_cases = [
        # 최신소식 (현재 상황/실시간 데이터)
        {
            "query": "어제 밤부터 비트코인 가격이 계속 하락하고 있는데 현재 정확한 시세가 얼마나 되는지 알고 싶어요",
            "expected": "최신소식",
            "category": "News - Current Price"
        },
        {
            "query": "오늘 아침에 암호화폐 시장 전체가 갑자기 10% 이상 급락했다고 들었는데 정확한 원인과 현재 상황이 어떻게 되고 있나요?",
            "expected": "최신소식",
            "category": "News - Market Update"
        },
        {
            "query": "지난주부터 이더리움 가격이 급상승하고 있다는 소식을 들었는데 현재까지 얼마나 올랐고 이런 상승세가 계속될 것으로 보이는지 궁금해요",
            "expected": "최신소식",
            "category": "News - Price Trend"
        },
        {
            "query": "오늘 코인마켓캡에서 가장 상위권에 있는 암호화폐 중에서 24시간 동안 가장 많이 상승한 코인이 무엇인지 알려주세요",
            "expected": "최신소식",
            "category": "News - Market Rankings"
        },
        {
            "query": "최근 미국에서 비트코인 ETF가 승인되었다는 뉴스를 봤는데 이것이 정확히 언제 발표된 것이고 시장 반응은 어떤지 현재 상황을 알려주세요",
            "expected": "최신소식",
            "category": "News - ETF Approval"
        },
        
        # 전문지식 (분석/전략/원리)
        {
            "query": "암호화폐 투자를 처음 시작하는 초보자에게 적합한 포트폴리오 구성 전략과 리스크 관리 방법에 대해 자세히 설명해주세요",
            "expected": "전문지식",
            "category": "Expertise - Investment Strategy"
        },
        {
            "query": "블록체인 기술이 어떤 방식으로 작동하는지 해시, 블록, 체인의 개념부터 합의 알고리즘까지 전체적인 원리를 이해하기 쉽게 설명해주세요",
            "expected": "전문지식",
            "category": "Expertise - Blockchain Technology"
        },
        {
            "query": "암호화폐 차트에서 볼링거 밴드, RSI, MACD 같은 기술적 지표들을 활용해서 매매 시점을 판단하는 구체적인 분석 방법을 알려주세요",
            "expected": "전문지식",
            "category": "Expertise - Technical Analysis"
        },
        {
            "query": "탈중앙화 금융(DeFi) 프로토콜이 기존 전통 금융 시스템과 어떤 근본적인 차이점이 있고 각각의 장단점은 무엇인지 비교 분석해주세요",
            "expected": "전문지식",
            "category": "Expertise - DeFi Analysis"
        },
        {
            "query": "이더리움 네트워크에서 스테이킹을 통해 얻을 수 있는 연간 수익률을 계산하는 방법과 관련된 리스크 요소들에 대해 상세히 가르쳐주세요",
            "expected": "전문지식",
            "category": "Expertise - Staking Calculation"
        }
    ]
    
    print("=== Crypto Classification Test ===\n")
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected = test_case["expected"]
        category = test_case["category"]
        
        # Get classification result
        result = classifier.classify(query)
        
        # Check if prediction is correct
        is_correct = result == expected
        if is_correct:
            correct_predictions += 1
            
        # Print results
        status = "✓" if is_correct else "✗"
        print(f"{status} Test {i}: {category}")
        print(f"  Query: {query[:80]}...")
        print(f"  Expected: {expected}")
        print(f"  Got: {result}")
        print(f"  Correct: {is_correct}")
        print()
    
    # Print summary
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"=== Results Summary ===")
    print(f"Total Tests: {total_predictions}")
    print(f"Correct: {correct_predictions}")
    print(f"Incorrect: {total_predictions - correct_predictions}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    return accuracy >= 80.0  # Return True if accuracy is 80% or higher

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    
    classifier = ClassificationChain()
    
    edge_cases = [
        # Ambiguous cases
        {
            "query": "비트코인 투자 전략과 현재 시장 상황을 모두 알려주세요",
            "description": "Mixed news and expertise"
        },
        {
            "query": "이더리움 가격 분석을 위한 차트 읽는 방법을 알려주세요",
            "description": "Analysis method (expertise) for price (news context)"
        },
        # Non-crypto queries
        {
            "query": "주식 투자 방법을 알려주세요",
            "expected": "기타",
            "description": "Non-crypto investment question"
        },
        {
            "query": "날씨가 어떤가요?",
            "expected": "기타",
            "description": "Completely unrelated question"
        },
        # Reset queries
        {
            "query": "대화를 리셋해주세요",
            "expected": "리셋",
            "description": "Reset request"
        }
    ]
    
    print("=== Edge Cases Test ===\n")
    
    for i, test_case in enumerate(edge_cases, 1):
        query = test_case["query"]
        description = test_case["description"]
        expected = test_case.get("expected", "Unknown")
        
        result = classifier.classify(query)
        
        print(f"Test {i}: {description}")
        print(f"  Query: {query}")
        print(f"  Result: {result}")
        if expected != "Unknown":
            is_correct = result == expected
            print(f"  Expected: {expected}")
            print(f"  Correct: {'✓' if is_correct else '✗'}")
        print()

if __name__ == "__main__":
    print("Starting Crypto Classification Tests...\n")
    
    try:
        # Run main classification test
        main_success = test_crypto_classification()
        
        print("\n" + "="*50 + "\n")
        
        # Run edge cases test
        test_edge_cases()
        
        print("\n" + "="*50)
        print("Test completed successfully!" if main_success else "Test completed with issues.")
        
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)