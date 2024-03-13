# LeapfrogAI VLLM Backend

## Description

A LeapfrogAI API-compatible [VLLM](https://github.com/vllm-project/vllm) wrapper for quantized and un-quantized model inferencing across GPU infrastructures.

## Usage

See [instructions](#instructions) to get the backend up and running. Then, use the [LeapfrogAI API server](https://github.com/defenseunicorns/leapfrogai-api) to interact with the backend.

## Instructions

The instructions in this section assume the following:

1. Properly installed and configured Python 3.11.x, to include its development tools, and [uv](https://github.com/astral-sh/uv)
2. The LeapfrogAI API server is deployed and running

The following are additional assumptions for GPU inferencing:

3. You have properly installed one or more NVIDIA GPUs and GPU drivers
4. You have properly installed and configured the [cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)

### Model Selection

The default model that comes with this backend in this repository's officially released images is an [AWQ quantization of the Synthia-7b model](https://huggingface.co/TheBloke/SynthIA-7B-v2.0-AWQ).

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

# Copy the config file, change this if different params are needed
cp config.example.yaml config.yaml

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
# device=0 means it will use the first slotted GPU
docker run --gpus device=0 -e GPU_ENABLED=true -p 50051:50051 -d --name vllm ghcr.io/defenseunicorns/leapfrogai/vllm:<IMAGE_TAG>
```

### VLLM Specific Packaging

VLLM requires access to host system GPU drivers in order to operate when compiled specifically for GPU inferencing. Even if no layers are offloaded to the GPU at runtime, VLLM will throw an unrecoverable exception.

Zarf package creation:

```bash
zarf package create --set IMAGE_REPOSITORY=ghcr.io/defenseunicorns/leapfrogai/vllm --set IMAGE_VERSION=<IMAGE_TAG> --set NAME=vllm --insecure
zarf package publish zarf-package-vllm-amd64-<IMAGE_TAG>.tar.zst oci://ghcr.io/defenseunicorns/packages/leapfrogai
```

### Package Naming for Production Deployment

To change the name of the model's Zarf package and Docker image being produced for installation into a cluster and exposure to the end user, do the following:

```bash
# Create the Docker image
# ASSUMPTION: localized registry is up and running, 
#   see Docker documentation for more details
export model_name="synthia-7b-awq" # name of the final package and image (no longer than 63 characters)
export version="0.0.1" # desired image and package version
export model_repo_id="TheBloke/Synthia-7B-v2.0-AWQ"
export model_revision="main"
export repository=localhost:5000/defenseunicorns/leapfrogai/$model_name

# build and push to local registry in 1-shot
docker build --push --build-arg REPO_ID=$model_repo_id --build-arg REVISION=$model_revision -t $repository:$version .

# Create Zarf package
# See Zarf documentation for more details
zarf package create \
    --set image_version=$version \
    --set name=$model_name \
    --set image_repository=$repository \
    --confirm

# Deploy the target Zarf package 
#   Change the resource limits as required by the model size
zarf package deploy \
    --set GPU_ENABLED=true \
    --set LIMITS_GPU=1 \
    --set REQUESTS_GPU=1 \
    --set LIMITS_CPU=4 \
    --set REQUESTS_CPU=4 \
    --set LIMITS_MEMORY="25Gi" \
    --set REQUESTS_MEMORY="10Gi" \
    zarf-package-synthia-7b-awq-amd64-0.0.1.tar.zst
```
