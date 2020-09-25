FROM alpine:latest

RUN apt-get install -y python-pip python-dev build-essential

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python", "app.py"]