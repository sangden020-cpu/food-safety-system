FROM python:3.10-slim

# 시스템 라이브러리 설치 (OpenCV 구동용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉터리 설정
WORKDIR /app

# 파이썬 의존성 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# Hugging Face Spaces 기본 포트 7860 오픈
EXPOSE 7860
ENV PORT=7860

# FastAPI 서버 실행
CMD ["python", "server.py"]
