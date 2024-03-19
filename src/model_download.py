import os
from huggingface_hub import snapshot_download
from config import AppConfig

REPO_ID = AppConfig().backend_options.repo_id
REVISION = AppConfig().backend_options.revision
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = AppConfig().backend_options.hf_hub_enable_hf_transfer

print(f"Downloading model from {REPO_ID} at revision {REVISION}...")

snapshot_download(
    repo_id=REPO_ID,
    local_dir=".model",
    local_dir_use_symlinks=False,
    revision=REVISION,
)
