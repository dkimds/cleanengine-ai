#!/bin/bash

# AI 서비스용 Docker 이미지 다운로드
echo "✅ Docker 이미지 다운로드 중..."
docker pull rlaqhguse/if-ai

# 사용자 정의 네트워크 생성 (이미 존재하면 무시)
echo "✅ milvus-net 네트워크 생성 시도..."
docker network create milvus-net 2>/dev/null || echo "이미 milvus-net 네트워크가 존재합니다."

# milvus-standalone 컨테이너가 네트워크에 연결되었는지 확인하고 연결
if ! docker network inspect milvus-net | grep -q "milvus-standalone"; then
  echo "✅ milvus-standalone을 milvus-net에 연결 중..."
  docker network connect milvus-net milvus-standalone
else
  echo "✅ milvus-standalone은 이미 milvus-net에 연결되어 있습니다."
fi

# if-ai 컨테이너 실행
echo "✅ if-ai 컨테이너 실행 중..."
docker run -p 8000:8000 \
  --network milvus-net \
  --name if-ai \
  rlaqhguse/if-ai
