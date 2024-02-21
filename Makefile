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
	echo "NotImplementedError"

test:
	pytest **/*.py

dev:
	leapfrogai --app-dir=. main:Model

lint:
	ruff check . --fix
	ruff format .

docker-build:
	docker build -t ghcr.io/defenseunicorns/leapfrogai/vllm:${VERSION} .

docker-run:
	docker run -it ghcr.io/defenseunicorns/leapfrogai/vllm:${VERSION}

docker-push:
	docker push ghcr.io/defenseunicorns/leapfrogai/vllm:${VERSION}

zarf-create:
	zarf package create . --confirm

zarf-deploy:
	zarf package deploy --confirm zarf-package-*.tar.zst

zarf-publish:
	zarf package publish zarf-*.tar.zst oci://ghcr.io/defenseunicorns/leapfrogai/packages/