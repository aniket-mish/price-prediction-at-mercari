FROM python:3.8-slim-buster

# set "app" as the working directory from which CMD, RUN, ADD references
WORKDIR /app

# now copy all the files in this directory to /code
COPY . /app

# pip install the local requirements.txt
RUN pip install -r requirements.txt

# Define our command to be run when launching the container
CMD ["flask", "run"]