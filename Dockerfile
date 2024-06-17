FROM python:3.9

RUN mkdir /app

COPY . /app/

WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/src

# CMD ["python", "infer.py", "--image", "../image/picnic.png"]
