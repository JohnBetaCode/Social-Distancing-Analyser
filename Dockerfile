FROM python:3.6

WORKDIR /code

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY app ./app
COPY detection ./detection

ARG MODEL_URL
RUN mkdir models && \
    wget -O models/model.tar.gz $MODEL_URL && \
    cd models && \
    tar -xzvf model.tar.gz

WORKDIR /code