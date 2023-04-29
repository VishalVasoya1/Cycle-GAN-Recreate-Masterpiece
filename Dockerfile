FROM python:3.10

WORKDIR /app 

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./Deployment-FastAPI/ /app/Deployment-FastAPI/
COPY ./Models/ /app/Models/

EXPOSE 8000

CMD ["uvicorn", "Deployment-FastAPI.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "0"]