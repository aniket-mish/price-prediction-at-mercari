FROM python:3

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 4242

CMD ["python", "app.py"]