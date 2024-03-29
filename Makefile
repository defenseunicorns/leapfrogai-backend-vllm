VERSION ?= $(shell git describe --abbrev=0 --tags | sed -e 's/^v//')
ifeq ($(VERSION),)
  VERSION := latest
endif

DEVICE ?= 0

.PHONY: all

create-venv:
	pip install uv
	uv venv

requirements-dev:
	pip install uv
	uv pip install -r requirements-dev.txt --override overrides.txt

requirements:
	pip install uv
	uv pip sync requirements.txt
	uv pip sync requirements-dev.txt

build-requirements:
	pip install uv
	uv pip compile pyproject.toml -o requirements.txt --override overrides.txt --generate-hashes

build-requirements-dev:
	pip install uv
	uv pip compile pyproject.toml -o requirements-dev.txt --override overrides.txt --extra dev --generate-hashes

fetch-model:
	python3 src/model_download.py

test:
	pytest **/*.py

dev:
	python -u src/main.py

lint:
	ruff check . --fix
	ruff format .

docker-build:
	docker build -t ghcr.io/defenseunicorns/leapfrogai/vllm:${VERSION} .

docker-build-local-registry:
	make docker-build
	docker tag ghcr.io/defenseunicorns/leapfrogai/vllm:${VERSION} localhost:5000/defenseunicorns/leapfrogai/vllm:${VERSION}
	docker push localhost:5000/defenseunicorns/leapfrogai/vllm:${VERSION}

docker-run:
	docker run -it --gpus device=0 -p 50051:50051 ghcr.io/defenseunicorns/leapfrogai/vllm:${VERSION}

docker-push:
	docker push ghcr.io/defenseunicorns/leapfrogai/vllm:${VERSION}

zarf-create:
	zarf package create . --confirm

zarf-create-local-registry:
	zarf package create . --confirm --registry-override ghcr.io=localhost:5000 --set IMG=defenseunicorns/leapfrogai/vllm:${VERSION}

zarf-deploy:
	zarf package deploy --confirm zarf-package-*.tar.zst --set GPU_ENABLED=true --set REQUESTS_GPU=1 --set LIMITS_GPU=1 --set REQUESTS_CPU=0 --set LIMITS_CPU=0 --set LIMITS_MEMORY=0 --set REQUEST_MEMORY=0

zarf-publish:
	zarf package publish zarf-*.tar.zst oci://ghcr.io/defenseunicorns/leapfrogai/packages/
