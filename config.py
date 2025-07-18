"""
Global configuration for the application.
"""

# Default model configuration
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Cryptocurrency keywords for classification
CRYPTO_KEYWORDS = [
    # Bitcoin
    "비트코인", "bitcoin", "btc",
    # Ethereum
    "이더리움", "ethereum", "eth", "ether",
    # Popular altcoins
    "리플", "ripple", "xrp",
    "라이트코인", "litecoin", "ltc",
    "에이다", "cardano", "ada",
    "폴카닷", "polkadot", "dot",
    "체인링크", "chainlink", "link",
    "솔라나", "solana", "sol",
    # General crypto terms
    "암호화폐", "cryptocurrency", "crypto",
    "코인", "coin", "토큰", "token",
    "알트코인", "altcoin",
    "디파이", "defi", "디코인", "블록체인", "blockchain",
    "채굴", "mining", "마이닝",
    "지갑", "wallet", "거래소", "exchange",
    "스테이킹", "staking", "스테이킹", "스테이크"
]