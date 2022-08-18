FROM python:3.10.6

WORKDIR ~/Algoritmo_Faltante_de_Anaquel_Chedraui/ 

COPY main.py 
COPY requirements.txt 
COPY models
COPY ms 

RUN pip install -r requirements.txt 

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
