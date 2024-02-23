# LeapfrogAI VLLM Backend

## Description

A LeapfrogAI API-compatible [VLLM](https://github.com/vllm-project/vllm) wrapper for quantized and un-quantized model inferencing across GPU infrastructures.

## Usage

See [instructions](#instructions) to get the backend up and running. Then, use the [LeapfrogAI API server](https://github.com/defenseunicorns/leapfrogai-api) to interact with the backend.

## Instructions

The instructions in this section assume the following:

1. Properly installed and configured Python 3.11.x, to include its development tools
2. The LeapfrogAI API server is deployed and running

The following are additional assumptions for GPU inferencing:

3. You have properly installed one or more NVIDIA GPUs and GPU drivers
4. You have properly installed and configured the [cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)

### Model Selection

The default model that comes with this backend in this repository's officially released images is a [4-bit quantization of the Synthia-7b model](TheBloke/Synthia-7B-v3.0-AWQ).

Other models can be loaded into this backend by modifying or supplying the [model_download.py](./scripts/model_download.py) arguments during image creation or Makefile command execution. The arguments must point to a single quantized model file, else you will need to use the [autoawq](https://docs.vllm.ai/en/latest/quantization/auto_awq.html) converter on an un-quantized model. Please see the Dockerfile or Makefile for further details.

### Run Locally

```bash
# install with GPU compilation and deps
make requirements-dev
```

```bash
# Setup Virtual Environment
make create-venv
source .venv/bin/activate
make requirements-dev

# Clone Model
# Supply a REPO_ID, FILENAME and REVISION if a different model is desired
make fetch-model

# Copy the environment variable file, change this if different params are needed
cp .env.example .env

# Start Model Backend
make dev
```

### Run in Docker

#### Local Image Build and Run

For local image building and running.

```bash
# Supply a REPO_ID, FILENAME and REVISION if a different model is desired
make docker-build
make docker-run
```

#### Remote Image Build and Run

For pulling a tagged image from the main release repository.

Where `<IMAGE_TAG>` is the released packages found [here](https://github.com/orgs/defenseunicorns/packages/container/package/leapfrogai%2Fvllm).

```bash
docker build -t ghcr.io/defenseunicorns/leapfrogai/vllm:<IMAGE_TAG> .
docker run --gpus device=0 -e GPU_ENABLED=true -p 50051:50051 -d --name vllm ghcr.io/defenseunicorns/leapfrogai/vllm:<IMAGE_TAG>
```

### VLLM Specific Packaging

VLLM requires access to host system GPU drivers in order to operate when compiled specifically for GPU inferencing. Even if no layers are offloaded to the GPU at runtime, VLLM will throw an unrecoverable exception.

Zarf package creation:

```bash
zarf package create --set IMAGE_REPOSITORY=ghcr.io/defenseunicorns/leapfrogai/vllm --set IMAGE_VERSION=<IMAGE_TAG> --set NAME=vllm --insecure
zarf package publish zarf-package-vllm-amd64-<IMAGE_TAG>.tar.zst oci://ghcr.io/defenseunicorns/packages/leapfrogai
```
