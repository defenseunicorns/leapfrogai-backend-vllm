ARG ARCH=amd64

FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 as builder

ENV DEBIAN_FRONTEND=noninteractive

USER root

RUN groupadd -g 65532 vglusers && \
    useradd -ms /bin/bash nonroot -u 65532 -g 65532 && \
    usermod -a -G video,sudo nonroot

WORKDIR /home/leapfrogai

# grab necesary python dependencies
RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe \
    && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y update

# get deps for vllm compilation and model download
RUN apt-get -y install python3.11-full git python3-venv

RUN chown -R nonroot /home/leapfrogai/
USER nonroot

RUN python3 -m venv .venv
ENV PATH="/home/leapfrogai/.venv/bin:$PATH"
# create virtual environment for light-weight portability and minimal libraries
RUN python3 -m pip install -U uv
COPY requirements.txt .
COPY overrides.txt .
RUN uv pip install -r requirements.txt --override overrides.txt
RUN uv pip install -U huggingface_hub[cli,hf_transfer]

# download model
ENV HF_HOME=/home/leapfrogai/.cache/huggingface
COPY . .
RUN mv config.example.yaml config.yaml

RUN python3 src/model_download.py

ENV QUANTIZATION=gptq


EXPOSE 50051:50051

ENTRYPOINT ["leapfrogai", "--app-dir=src/", "main:Model"]