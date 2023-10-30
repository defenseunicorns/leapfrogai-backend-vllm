# before building me, build the vllm base image with
# docker build -f Dockerfile-vllm --tag leapfrogai/vllm:0.1.7
# then build me with
# docker build --tag leapfrogai/mpt-7b-8k-chat .
FROM leapfrogai/vllm:0.1.4

WORKDIR app

COPY --chown=user:user mpt-7b-8k-chat mpt-7b-8k-chat

# copy in extra requirements and install
COPY --chown=user:user requirements.txt .
RUN pip install -r requirements.txt

# copy in local copy of leapfrogai
COPY --chown=user:user leapfrogai-0.3.4a0-py3-none-any.whl .
RUN pip install leapfrogai-0.3.4a0-py3-none-any.whl

# Move the rest of the python files (most likely place layer cache will be invalidated)
COPY --chown=user:user *.py .

# Publish port
EXPOSE 50051:50051

# Enjoy
ENTRYPOINT ["python3", "model.py"]