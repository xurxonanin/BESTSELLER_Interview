FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY ml ./ml
COPY data ./data

RUN python ./ml/prediction_model.py

EXPOSE 8000


CMD ["uvicorn", "app.endpoints:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]