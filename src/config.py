from typing import Literal

from confz import BaseConfig, FileSource
from pydantic import Field


class ConfigOptions(BaseConfig):
    quantization: Literal[None, "awq", "gptq", "squeezellm"] = Field(
        default=None,
        description="type of quantization, for un-quantized models omit this field"
    )
    hf_hub_enable_hf_transfer: Literal["0", "1"] = Field(
        description="Option (0 - Disable, 1 - Enable) for faster transfers, tradeoff stability for faster speeds"
    )
    repo_id: str = Field(
        description="HuggingFace repo id (e.g., TheBloke/Synthia-7B-v2.0-GPTQ)"
    )
    revision: str = Field(
        description="The branch to use (e.g., main, gptq-4bit-64g-actorder_True, etc.)"
    )


class AppConfig(BaseConfig):
    options: ConfigOptions
    CONFIG_SOURCES = FileSource(file='config.yaml')
