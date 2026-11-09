FROM python:3.11-slim

ARG BUILD_WITH_MODELS=false

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai \
    HF_ENDPOINT=https://hf-mirror.com \
    LINGUARELAY_PORT=8765 \
    LINGUARELAY_UDP_PORT=8766 \
    LINGUARELAY_HOST_IP="" \
    LINGUARELAY_EPG_URL="https://epg.112114.xyz/pp.xml.gz,https://live.fanmingming.com/e.xml" \
    LINGUARELAY_LLM_BASE_URL="https://api.minimaxi.com/v1" \
    LINGUARELAY_LLM_MODEL="MiniMax-M3.1-Flash"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    tzdata \
    curl \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY scripts /app/scripts
COPY server /app/server
COPY dist /app/dist
COPY assets /app/assets

# Pre-package default 337-channel IPTV playlist & 100% EPG archive into /app/seed_data
# On first boot, scripts/download_models.py automatically copies these into mounted /app/data if empty.
RUN mkdir -p /app/models /app/data /app/seed_data
COPY data/playlist_source.txt /app/seed_data/playlist_source.txt
COPY data/epg.xml.gz /app/seed_data/epg.xml.gz

RUN if [ "$BUILD_WITH_MODELS" = "true" ]; then \
      HF_ENDPOINT=https://huggingface.co python scripts/download_models.py; \
    fi

VOLUME ["/app/models", "/app/data"]

EXPOSE 8765/tcp
EXPOSE 8766/udp

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${LINGUARELAY_PORT}/api/status" >/dev/null || exit 1

CMD ["sh", "-c", "python scripts/download_models.py && uvicorn server.main:app --host 0.0.0.0 --port ${LINGUARELAY_PORT}"]
