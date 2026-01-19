FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir aiohttp watchdog requests
COPY boxsync.py /app/boxsync.py
EXPOSE 8080
CMD ["python", "-u", "/app/boxsync.py"]
