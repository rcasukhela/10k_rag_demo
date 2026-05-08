FROM python:3.11-slim

WORKDIR /code

COPY pyproject.toml .
COPY src ./src
COPY config ./config

RUN pip install --no-cache-dir -e .

CMD ["uvicorn", "sec_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]