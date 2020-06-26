FROM frolvlad/alpine-python-machinelearning:latest

WORKDIR /app

COPY . /app

RUN apk add build-base && \
    apk add --no-cache --virtual .build-deps g++ python3-dev libffi-dev openssl-dev && \
    pip install -r requirements.txt && \
    python -m nltk.downloader punkt

EXPOSE 4000

ENTRYPOINT ["python"]

CMD ["manage.py"]
