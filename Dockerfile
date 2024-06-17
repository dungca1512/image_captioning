FROM python:3.9

RUN mkdir /app

COPY src /app/src
COPY data /app/data
COPY models /app/models
COPY image /app/image
COPY csv /app/csv
COPY requirements.txt /app/
COPY pretrain_weights.pt /app/

WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/src

CMD ["python", "infer.py", "--image", "../image/picnic.png"]
