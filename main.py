import logging
from typing import Any, Generator, AsyncIterable, AsyncGenerator

from leapfrogai import BackendConfig
from leapfrogai.llm import GenerationConfig, LLM
from starlette.requests import Request
from starlette.requests import Scope
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import CompletionRequest
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion


@LLM
class Model:
    backend_config = BackendConfig()
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    model = "migtissera/Synthia-7B-v3.0"

    async def create_response(self, prompt, config: GenerationConfig) -> AsyncIterable:
        engine_args = AsyncEngineArgs(model=self.model,
                                      trust_remote_code=True,
                                      max_model_len=4096)
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        openai_serving_completion = OpenAIServingCompletion(engine=engine, served_model=self.model)

        do_sample: bool
        return_full_text: bool
        truncate: int
        typical_p: float
        watermark: bool
        seed: int

        chat_completion_request = CompletionRequest(
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            n=config.n,
            stop=config.stop,
            repetition_penalty=config.repetition_penalty,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            best_of=config.best_of,
            logit_bias=config.logit_bias,
            model=self.model,
            prompt=prompt,
            stream=True)
        dummy_request: Request = Request(Scope())
        generator: AsyncGenerator[str, None] = await openai_serving_completion.create_completion(
            request=chat_completion_request,
            raw_request=dummy_request)

        try:
            async for response in generator:
                yield response
        except (GeneratorExit, KeyError):
            logging.info(f"cancelation complete: chat_stream generator id {dummy_request.id}")

    def generate(
            self, prompt: str, config: GenerationConfig
    ) -> Generator[str, Any, Any]:
        yield self.create_response(prompt, config)
