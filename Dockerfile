# Scalper bot (paper). Code baked in at build time; on every start the entrypoint also tries to fetch the
# latest code from GitHub and uses it only if its self-test passes (remote "restart" = update).
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 TZ=UTC
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
RUN chmod +x /app/docker-entrypoint.sh && python test_engine.py && mkdir -p /data
WORKDIR /data
VOLUME /data
EXPOSE 8088
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8088/healthz',timeout=4).status==200 else 1)"
ENTRYPOINT ["/app/docker-entrypoint.sh"]
