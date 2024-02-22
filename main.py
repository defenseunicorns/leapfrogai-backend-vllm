import asyncio
import logging
import queue
import sys
import threading
import time
from typing import Any, Generator, Dict

from leapfrogai import BackendConfig
from leapfrogai.llm import GenerationConfig, LLM
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid


def clamp(n: float | int, smallest: float | int, largest: float | int):
    return max(smallest, min(n, largest))

@LLM
class Model:
    done_by_id: Dict[str, bool] = {}
    delta_queue_by_id: Dict[str, queue] = {}

    backend_config = BackendConfig()
    model = "TheBloke/Synthia-7B-v3.0-GPTQ"
    engine_args = AsyncEngineArgs(engine_use_ray=True,
                                  model=model,
                                  trust_remote_code=True,
                                  quantization="gptq",
                                  max_context_len_to_capture=backend_config.max_context_length
                                  )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def create_response(self, request_id: str, prompt: str, config: GenerationConfig):
        sampling_params = SamplingParams(temperature=config.temperature,
                                         top_p=clamp(config.top_p,
                                                     0.0 + sys.float_info.epsilon, 1.0 - sys.float_info.epsilon),
                                         top_k=config.top_k if config.top_k >= 1 else -1,
                                         stop=self.backend_config.stop_tokens,
                                         max_tokens=config.max_new_tokens)
        logging.debug(sampling_params)
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        gen_iter = self.engine.generate(prompt, sampling_params, request_id)
        t0 = time.time()
        index, num_tokens = 0, 0
        async for output in gen_iter:
            if (
                    output.outputs[0].text
                    and "\ufffd" == output.outputs[0].text[-1]
            ):
                continue
            text_delta = output.outputs[0].text[index:]
            index = len(output.outputs[0].text)
            num_tokens = len(output.outputs[0].token_ids)
            self.delta_queue_by_id[request_id].append(text_delta)

        logging.info(f"Generated {num_tokens} tokens in {time.time() - t0:.2f}s")

    async def generate_session(self, session: str, prompt: str, config: GenerationConfig):
        if self.delta_queue_by_id.get(session) is None:
            self.delta_queue_by_id[session] = []

        await self.create_response(session, prompt, config)
        self.done_by_id[session] = True

    def is_queue_empty(self, request_id) -> bool:
        cur_request_queue = self.delta_queue_by_id.get(request_id)
        return cur_request_queue is None or len(cur_request_queue) == 0

    def generate(
            self, prompt: str, config: GenerationConfig
    ) -> Generator[str, Any, Any]:
        request_id = random_uuid()
        self.done_by_id[request_id] = False
        _thread = threading.Thread(target=asyncio.run, args=(self.generate_session(request_id, prompt, config),))
        _thread.start()

        while not self.done_by_id.get(request_id) or not self.is_queue_empty(request_id):
            if not self.is_queue_empty(request_id):
                result = self.delta_queue_by_id.get(request_id).pop()
                yield result
