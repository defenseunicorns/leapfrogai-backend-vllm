ARG ARCH=amd64

# hardened and slim python w/ developer tools image
FROM ghcr.io/defenseunicorns/leapfrogai/python:3.11-dev-${ARCH} as builder

WORKDIR /leapfrogai

# download model
RUN python -m pip install -U huggingface_hub[cli,hf_transfer]
ARG REPO_ID=TheBloke/Synthia-7B-v3.0-AWQ
ARG REVISION=main
COPY scripts/model_download.py scripts/model_download.py
RUN REPO_ID=${REPO_ID} FILENAME=${FILENAME} REVISION=${REVISION} python3.11 scripts/model_download.py

# create virtual environment for light-weight portability and minimal libraries
RUN python -m pip install -U uv
RUN uv venv
ENV PATH="/leapfrogai/.venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

# hardened and slim python image
FROM ghcr.io/defenseunicorns/leapfrogai/python:3.11-${ARCH}

ENV PATH="/leapfrogai/.venv/bin:$PATH"
ENV QUANTIZATION=awq

WORKDIR /leapfrogai

COPY --from=builder /leapfrogai/.venv/ /leapfrogai/.venv/
COPY --from=builder /leapfrogai/.model/ /leapfrogai/.model/

COPY main.py .
COPY config.yaml .

EXPOSE 50051:50051

ENTRYPOINT ["leapfrogai", "--app-dir=.", "main:Model"]