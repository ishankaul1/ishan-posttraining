FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

WORKDIR /app

RUN pip install --no-cache-dir \
    trl \
    transformers \
    accelerate \
    peft \
    datasets \
    wandb \
    pyyaml \
    loguru

# Install gcloud CLI for gsutil/gcloud storage rsync
RUN apt-get update && apt-get install -y curl apt-transport-https gnupg && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb https://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && apt-get install -y google-cloud-cli && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . /app/

ENTRYPOINT ["bash", "train.sh"]
