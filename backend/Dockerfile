FROM python:3.12-slim

WORKDIR /workspace/medical_rag_agent

COPY requirements.txt /workspace/medical_rag_agent/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /workspace/medical_rag_agent/requirements.txt

COPY backend /workspace/medical_rag_agent/backend

WORKDIR /workspace/medical_rag_agent/backend

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
