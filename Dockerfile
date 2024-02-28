ARG ARCH=amd64

FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 as builder

WORKDIR /leapfrogai

ENV DEBIAN_FRONTEND=noninteractive

# grab necesary python dependencies
RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe \
    && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y update

# get deps for vllm compilation and model download
RUN apt-get -y install python3.11-full git python3-venv

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

ENV QUANTIZATION=awq

COPY main.py .
COPY config.yaml .

EXPOSE 50051:50051

ENTRYPOINT ["leapfrogai", "--app-dir=.", "main:Model"]