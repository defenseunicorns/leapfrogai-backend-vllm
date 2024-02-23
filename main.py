import asyncio
import logging
import os
import queue
import sys
import threading
import time
from typing import Any, Generator, Dict, AsyncIterator

import aiostream.stream
from leapfrogai import BackendConfig
from leapfrogai.llm import GenerationConfig, LLM
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid
from vllm.outputs import RequestOutput
from dotenv import load_dotenv

load_dotenv()


def clamp(n: float | int, smallest: float | int, largest: float | int):
    return max(smallest, min(n, largest))


@LLM
class Model:
    done_by_id: Dict[str, bool] = {}
    delta_queue_by_id: Dict[str, queue.Queue] = {}
    result_by_id: Dict[str, RequestOutput] = {}
    iterables: AsyncIterator[RequestOutput] = None
    all_iterators: list = []

    def __init__(self):
        logging.getLogger().setLevel(logging.DEBUG)
        _thread = threading.Thread(target=asyncio.run, args=(self.iterate_outputs(),))
        _thread.start()

        # Load the local config.yaml
        self.backend_config = BackendConfig()
        self.model = self.backend_config.model.source
        self.engine_args = AsyncEngineArgs(engine_use_ray=True,
                                           model=self.model,
                                           trust_remote_code=True,
                                           quantization=os.environ["QUANTIZATION"] or "awq",
                                           max_context_len_to_capture=self.backend_config.max_context_length,
                                           worker_use_ray=True)
        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)

    # Iterate over all the output AsyncIterables which are produced whenever a new request comes in
    async def iterate_outputs(self):
        t0_by_id: dict[str, float] = {}
        index_by_id: dict[str, int] = {}
        num_tokens_by_id: dict[str, int] = {}

        while True:
            if self.iterables is not None:
                async for request_output in self.iterables:
                    request_id = request_output.request_id

                    # Signal that the generator can stop waiting for additional inputs
                    if request_output.finished:
                        logging.info(
                            f"Generated {num_tokens_by_id[request_id]} tokens in {time.time() - t0_by_id[request_id]:.2f}s")
                        self.done_by_id[request_id] = True
                    else:
                        # Initialize dictionary entries
                        if t0_by_id.get(request_id) is None:
                            t0_by_id[request_id] = time.time()

                        if index_by_id.get(request_id) is None:
                            index_by_id[request_id] = 0

                        if num_tokens_by_id.get(request_id) is None:
                            num_tokens_by_id[request_id] = 0

                    if request_output.outputs[0].text and "\ufffd" == request_output.outputs[0].text[-1]:
                        continue

                    text_delta = request_output.outputs[0].text[index_by_id[request_id]:]
                    index_by_id[request_id] = len(request_output.outputs[0].text)
                    num_tokens_by_id[request_id] = len(request_output.outputs[0].token_ids)
                    self.delta_queue_by_id[request_id].put(text_delta)
            time.sleep(0)

    async def create_response(self, request_id: str, prompt: str, config: GenerationConfig):
        sampling_params = SamplingParams(temperature=config.temperature,
                                         # Clamp top_p value to prevent float errors
                                         top_p=clamp(config.top_p,
                                                     0.0 + sys.float_info.epsilon, 1.0 - sys.float_info.epsilon),
                                         # Restrict top_k to valid values, -1 disables top_k
                                         top_k=config.top_k if config.top_k >= 1 else -1,
                                         stop=self.backend_config.stop_tokens,
                                         stop_token_ids=[2],
                                         max_tokens=config.max_new_tokens,
                                         skip_special_tokens=False)
        logging.debug(sampling_params)
        logging.info(f"Begin generation for request {request_id}")
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        gen_iter = self.engine.generate(prompt, sampling_params, request_id)
        logging.info(f"Begin iteration for request {request_id}")
        if self.iterables is None:
            self.iterables = gen_iter
        else:
            self.iterables = aiostream.stream.merge(self.iterables, gen_iter)

    async def generate_session(self, session: str, prompt: str, config: GenerationConfig):
        if self.delta_queue_by_id.get(session) is None:
            self.delta_queue_by_id[session] = queue.Queue()

        await self.create_response(session, prompt, config)

    def is_queue_empty(self, request_id) -> bool:
        cur_request_queue = self.delta_queue_by_id.get(request_id)
        return cur_request_queue is None or cur_request_queue.empty()

    def generate(
            self, prompt: str, config: GenerationConfig
    ) -> Generator[str, Any, Any]:
        request_id = random_uuid()
        self.done_by_id[request_id] = False
        _thread = threading.Thread(target=asyncio.run, args=(self.generate_session(request_id, prompt, config),))
        _thread.start()
        logging.info(f"Begin reading the output for request {request_id}")

        while not self.done_by_id.get(request_id) or not self.is_queue_empty(request_id):
            result = ""
            if not self.is_queue_empty(request_id):
                result = self.delta_queue_by_id.get(request_id).get()
            yield result

        logging.info(f"Finished request {request_id}")
