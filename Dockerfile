ARG ARCH=amd64

FROM ubuntu:latest as builder

WORKDIR /leapfrogai

RUN apt update
RUN apt-get install -y python3.11
RUN apt-get install -y python3-venv
RUN python3 -m venv .venv
ENV PATH="/leapfrogai/.venv/bin:$PATH"
# create virtual environment for light-weight portability and minimal libraries
RUN python3 -m pip install -U uv
COPY requirements.txt .
COPY overrides.txt .
RUN uv pip install -r requirements.txt --override overrides.txt
RUN uv pip install -U huggingface_hub[cli,hf_transfer]

# download model
ARG REPO_ID=TheBloke/Synthia-7B-v3.0-AWQ
ARG REVISION=main
COPY scripts/model_download.py scripts/model_download.py
RUN REPO_ID=${REPO_ID} FILENAME=${FILENAME} REVISION=${REVISION} python3 scripts/model_download.py

# hardened and slim python image
FROM ghcr.io/defenseunicorns/leapfrogai/python:3.11-${ARCH}

WORKDIR /leapfrogai

ENV PATH="/leapfrogai/.venv/bin:$PATH"
ENV QUANTIZATION=awq

COPY --from=builder /leapfrogai/.venv/ /leapfrogai/.venv/
COPY --from=builder /leapfrogai/.model/ /leapfrogai/.model/

COPY main.py .
COPY config.yaml .

EXPOSE 50051:50051

ENTRYPOINT ["leapfrogai", "--app-dir=.", "main:Model"]