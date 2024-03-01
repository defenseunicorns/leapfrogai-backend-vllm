VERSION ?= $(shell git describe --abbrev=0 --tags | sed -e 's/^v//')
ifeq ($(VERSION),)
  VERSION := latest
endif

DEVICE ?= 0

.PHONY: all

create-venv:
	uv venv

requirements-dev:
	uv pip install -r requirements-dev.txt --override overrides.txt

requirements:
	uv pip sync requirements.txt
	uv pip sync requirements-dev.txt

build-requirements:
	uv pip compile pyproject.toml -o requirements.txt --override overrides.txt

build-requirements-dev:
	pip install uv
	uv pip compile pyproject.toml -o requirements-dev.txt --override overrides.txt --extra dev

fetch-model:
	python scripts/model_download.py

test:
	pytest **/*.py

dev:
	leapfrogai --app-dir=. main:Model

lint:
	ruff check . --fix
	ruff format .

docker-build:
	docker build -t ghcr.io/defenseunicorns/leapfrogai/vllm:${VERSION} .

docker-build-local-registry:
	docker build -t ghcr.io/defenseunicorns/leapfrogai/vllm:${VERSION} .
	docker tag ghcr.io/defenseunicorns/leapfrogai/vllm:${VERSION} localhost:5000/defenseunicorns/leapfrogai/vllm:${VERSION}
	docker push localhost:5000/defenseunicorns/leapfrogai/vllm:${VERSION}

docker-run:
	docker run -it ghcr.io/defenseunicorns/leapfrogai/vllm:${VERSION}

docker-push:
	docker push ghcr.io/defenseunicorns/leapfrogai/vllm:${VERSION}

zarf-create:
	zarf package create . --confirm

zarf-create-local-registry:
	zarf package create . --confirm --registry-override ghcr.io=localhost:5000 --set IMG=defenseunicorns/leapfrogai/vllm:${VERSION}

zarf-deploy:
	zarf package deploy --confirm zarf-package-*.tar.zst --set GPU_ENABLED=true --set REQUESTS_GPU=1 --set LIMITS_GPU=1

zarf-publish:
	zarf package publish zarf-*.tar.zst oci://ghcr.io/defenseunicorns/leapfrogai/packages/