import asyncio
import logging
import os

from typing import AsyncGenerator, Dict, List, Union

from google.protobuf.internal.containers import RepeatedCompositeFieldContainer

from prometheus_client import start_http_server, Counter, Gauge

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.grpc import GrpcAioInstrumentorServer
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

import transformers.utils.logging

from vllm import AsyncLLMEngine, RequestOutput, SamplingParams, AsyncEngineArgs

# completion
from leapfrogai import (
    CompletionRequest,
    CompletionChoice,
    CompletionUsage,
    CompletionResponse,
    CompletionServiceServicer,
    CompletionStreamServiceServicer,
    CompletionLogProbs,
    CompletionTopLogProb,
)

# chat
from leapfrogai import (
    ChatCompletionRequest,
    ChatCompletionChoice,
    ChatItem,
    ChatRole,
    ChatCompletionResponse,
    ChatCompletionServiceServicer,
    ChatCompletionStreamServiceServicer,
    ChatCompletionLogProbs,
    ChatCompletionTopLogProb,
    ChatCompletionUsage,
    ChatCancellationRequest,
    ChatCancellationResponse,
    ChatCancellationServiceServicer,
)

# general
from leapfrogai import (
    GrpcContext,
    serve,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
transformers.utils.logging.disable_progress_bar()
transformers.utils.logging.set_verbosity_debug()

class RandomSpanIdGenerator(RandomIdGenerator):
    """A custom ID generator that generates random span IDs
    
    We need to ensure these spanIDs that are cryptographically random (not pseudorandom)
    to ensure that across application restarts we don't get colliding spanIDs that
    would break the use of these IDs as database foreign keys downstream"""
    def generate_span_id(self) -> int:
        return int.from_bytes(os.urandom(8), byteorder="big")

# Constants
MODEL_ID = "Synthia-7B-v1.3"
PROMPT_TOKENS = "tokens.prompt"
COMPLETION_TOKENS = "tokens.completion"
REQUEST_ID = "inference_request.id"

# Prometheus metrics
running_requests = Gauge("running_requests", "Number of running inference requests")
swapped_requests = Gauge("swapped_requests", "Number of swapped inference requests")
waiting_requests = Gauge(
    "waiting_requests", "Number of inference requests waiting in the queue"
)
requests_counter = Counter("requests_counter", "Total number of gRPC requests received")
prompt_tokens_counter = Counter("prompt_tokens", "Total number of prompt tokens received")
completion_tokens_counter = Counter("completion_tokens", "Total number of completion tokens generated")

# OpenTelemetry Resource Identification so we can uniquely identify multiple instances of the same model
replica_id = os.getenv("MODEL_REPLICA_ID", "0")
service_prefix = f"mpt-7b-8k-chat-{replica_id}"
service_name = f"{service_prefix}.tracer"
resource = Resource(attributes={SERVICE_NAME: service_name})
trace.set_tracer_provider(TracerProvider(resource=resource, id_generator=RandomSpanIdGenerator()))

# Creates a tracer from the global tracer provider
tracer = trace.get_tracer(service_name)

# Prompt Templates
SYSTEM_FORMAT = "SYSTEM: {}\n"
USER_FORMAT = "USER: {}\n"
ASSISTANT_FORMAT = "ASSISTANT: {}\n"
# what gets appended to the end of the prompt to open the assistant's part of the conversation
RESPONSE_PREFIX = ASSISTANT_FORMAT.split("{}")[0]


def chat_items_to_prompt(chat_items: RepeatedCompositeFieldContainer[ChatItem]) -> str:
    """Converts a repeated ChatItem from a ChatCompletionRequest proto into a string

    This is the actual string that gets fed into the model to generate the outputs
    """
    prompt = ""
    for item in chat_items:
        if item.role == ChatRole.SYSTEM:  # type: ignore
            prompt += SYSTEM_FORMAT.format(item.content)
        elif item.role == ChatRole.ASSISTANT:  # type: ignore
            prompt += ASSISTANT_FORMAT.format(item.content)
        elif item.role == ChatRole.USER:  # type: ignore
            prompt += USER_FORMAT.format(item.content)
        elif item.role == ChatRole.FUNCTION:  # type: ignore
            logging.warning(
                "ChatRole FUNCTION is not implemented for this model and this ChatItem will be ignored."
            )
    # add the response prefix to start the model's reponse
    prompt += RESPONSE_PREFIX
    return prompt


def check_finish_reason(finish_reason: str) -> Union[str, None]:
    """Validates a finish_reason is one of the valid strings, returning None if invalid"""
    if finish_reason == "stop":
        return finish_reason
    elif finish_reason == "length":
        return finish_reason
    else:
        return None


class MPTChat(
    ChatCompletionServiceServicer,
    ChatCompletionStreamServiceServicer,
    CompletionServiceServicer,
    CompletionStreamServiceServicer,
    ChatCancellationServiceServicer,
):
    # Model configuration arguments
    engine_args = AsyncEngineArgs(
        model=MODEL_ID,
        trust_remote_code=True,
    )
    
    engine_args.max_model_len = 4096

    # create the async engine from the model config
    async_engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = async_engine.engine.tokenizer  # type:ignore

    def chat_logprobs(
        self,
        token_ids: List[int],
        id_logprobs: List[Dict[int, float]],
        initial_text_offset: int = 0,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style completion logprobs. If this method fails (raises some error type I don't know), we'll return a grpc.StatusCode.INTERNAL"""

        last_token_len = 0
        tokens = []
        token_logprobs = []
        text_offset = []
        top_logprobs = []
        for token_id, id_logprob in zip(token_ids, id_logprobs):
            token = self.tokenizer.convert_ids_to_tokens(token_id)  # type:ignore
            tokens.append(token)
            token_logprobs.append(id_logprob[token_id])
            if len(text_offset) == 0:
                text_offset.append(initial_text_offset)
            else:
                text_offset.append(text_offset[-1] + last_token_len)
            last_token_len = len(token)

            top_logprobs.append(
                {
                    self.tokenizer.convert_ids_to_tokens(i): p
                    for i, p in id_logprob.items()  # type:ignore
                }
            )
        top_logprobs = [ChatCompletionTopLogProb(log_probs=i) for i in top_logprobs]
        return ChatCompletionLogProbs(
            text_offset=text_offset,
            token_logprobs=token_logprobs,
            tokens=tokens,
            top_logprobs=top_logprobs,
        )

    def completion_logprobs(
        self,
        token_ids: List[int],
        id_logprobs: List[Dict[int, float]],
        initial_text_offset: int = 0,
    ) -> CompletionLogProbs:
        """Create OpenAI-style completion logprobs. If this method fails (raises some error type I don't know), we'll return a grpc.StatusCode.INTERNAL"""
        last_token_len = 0
        tokens = []
        token_logprobs = []
        text_offset = []
        top_logprobs = []
        for token_id, id_logprob in zip(token_ids, id_logprobs):
            token = self.tokenizer.convert_ids_to_tokens(token_id)  # type:ignore
            tokens.append(token)
            token_logprobs.append(id_logprob[token_id])
            if len(text_offset) == 0:
                text_offset.append(initial_text_offset)
            else:
                text_offset.append(text_offset[-1] + last_token_len)
            last_token_len = len(token)

            top_logprobs.append(
                {
                    self.tokenizer.convert_ids_to_tokens(i): p
                    for i, p in id_logprob.items()
                }  # type:ignore
            )
        top_logprobs = [CompletionTopLogProb(log_probs=i) for i in top_logprobs]
        return CompletionLogProbs(
            text_offset=text_offset,
            token_logprobs=token_logprobs,
            tokens=tokens,
            top_logprobs=top_logprobs,
        )

    def count_tokens(self, prompt: str) -> int:
        """Count the number of tokens in a string using the model's tokenizer"""
        return len(
            self.async_engine.engine.tokenizer(prompt)["input_ids"]  # type:ignore
        )
    
    def update_metrics(self, prompt_tokens: int, completion_tokens: int):
        # token counts
        prompt_tokens_counter.inc(prompt_tokens)
        completion_tokens_counter.inc(completion_tokens)
        
        # increment total requests
        requests_counter.inc()

        # pull VLLM engine stats
        running_requests.set(
            len(self.async_engine.engine.scheduler.running)  # type:ignore
        )
        swapped_requests.set(
            len(self.async_engine.engine.scheduler.swapped)  # type:ignore
        )
        waiting_requests.set(
            len(self.async_engine.engine.scheduler.waiting)  # type:ignore
        )

    async def chat_stream(
        self, request: ChatCompletionRequest,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Implements the logic for the chat endpoint of the MPT model
        
        request_id: 8 byte spanID for this request as a hex string. This allows us to join requests that are logged in a database to the user activity data from tracing"""
        # convert chat items to prompt
        prompt = chat_items_to_prompt(request.chat_items)

        # create text streamer and validate parameters
        max_new_tokens = 1536 if request.max_new_tokens == 0 else request.max_new_tokens
        temperature = 0.1 if request.temperature == 0.0 else request.temperature
        top_p = 1.0 if request.top_p == 0.0 else request.top_p
        top_k = -1 if request.top_k == 0.0 else int(request.top_k)
        logprobs = request.logprobs if request.logprobs != 0 else None
        n = request.n if request.n > 0 else 1

        sampling_params = SamplingParams(
            n=n,
            logprobs=logprobs,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=["</s>"],
        )

        # set the id of the request to something that was set client side so the client can cancel it
        response_generator = self.async_engine.generate(
            prompt, sampling_params, request_id=request.id
        )

        # Each response that gets yielded will be contain the information that was in the
        # previous response so we're building our way up to yielding a final, complete response
        # instead of only yielding chunks each time. This is super convenient for things like
        # logprobs
        try:
            async for response in response_generator:  # type: ignore
                yield response
        except (GeneratorExit, KeyError):
            logging.info(f"cancelation complete: chat_stream generator id {request.id}")

    # @tracer.start_as_current_span("ChatCompletee")
    async def ChatComplete(
        self, request: ChatCompletionRequest, context: GrpcContext
    ) -> ChatCompletionResponse:
        """Implementation for the stream=False ChatCompletion endpoint"""
        with tracer.start_as_current_span("ChatComplete") as span:
            n = request.n if request.n > 0 else 1
            chat_stream = self.chat_stream(request)
            full_texts = [""] * n
            full_response = ChatCompletionResponse()
            async for response in chat_stream:
                full_texts = [output.text for output in response.outputs]
                full_response = response

            # construct the list of Chat Completion Choices for this response
            choices = []
            for i, text in enumerate(full_texts):
                # create choice elements
                item = ChatItem(role=ChatRole.ASSISTANT, content=text)  # type: ignore
                finish_reason = check_finish_reason(
                    full_response.outputs[i].finish_reason  # type:ignore
                )
                logprobs = self.chat_logprobs(
                    full_response.outputs[i].token_ids,  # type:ignore
                    full_response.outputs[i].logprobs,  # type:ignore
                )

                # ...and combine them into the choice
                choice = ChatCompletionChoice(
                    index=i,
                    chat_item=item,
                    finish_reason=finish_reason,
                    logprobs=logprobs,  # type:ignore
                )
                choices.append(choice)
            # calculate token usage
            prompt_tokens = self.request_token_count_chat(request)
            completion_tokens = sum([self.count_tokens(t) for t in full_texts])
            usage = ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            self.update_metrics(prompt_tokens, completion_tokens)

            return ChatCompletionResponse(choices=choices, usage=usage)


    async def ChatCompleteStream(
        self, request: ChatCompletionRequest, context: GrpcContext
    ) -> AsyncGenerator[ChatCompletionResponse, None]:
        """Streaming implementation for Chat Completion
        
        TODO: sometimes when you cancel a request, this will throw a value error (and keep going)
        ValueError: <Token var=<ContextVar name='current_context' default={} at 0x7ff861ce7e20> at 0x7ff86d8fb1c0> was created in a different Context"""
        with tracer.start_as_current_span("ChatCompleteStream") as span:
            result_generator = self.chat_stream(request)
            n = request.n if request.n > 0 else 1
            previous_texts = [""] * n  # future place to add n generations
            previous_num_tokens = [0] * n
            prompt_tokens = self.request_token_count_chat(request)
            completion_tokens = 0
            try:
                async for res in result_generator:
                    # just get the new piece of text
                    new_texts = [res.outputs[i].text[len(previous_texts[i]):]
                        for i in range(n)
                    ]
                    # construct the list of Chat Completion Choices for this response
                    choices = []
                    for i, text in enumerate(new_texts):
                        # create choice elements
                        item = ChatItem(role=ChatRole.ASSISTANT, content=text)  # type: ignore
                        if request.logprobs > 0:
                            logprobs = self.chat_logprobs(
                                res.outputs[i].token_ids[previous_num_tokens[i]:],  # type:ignore
                                res.outputs[i].logprobs[previous_num_tokens[i]:],  # type:ignore
                                len(previous_texts[i])
                            )
                        else:
                            logprobs = None
                        # ...and combine them into the choice
                        choice = ChatCompletionChoice(
                            index=i,
                            chat_item=item,
                            logprobs=logprobs,  # type:ignore
                        )
                        choices.append(choice)
                        previous_num_tokens[i] = len(res.outputs[i].token_ids)
                    
                    # usage
                    completion_tokens += sum([self.count_tokens(t) for t in new_texts])
                    usage = ChatCompletionUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens
                    )
                    yield ChatCompletionResponse(choices=choices, usage=usage)
                    previous_texts = [output.text for output in res.outputs]
            except (GeneratorExit, ValueError, RuntimeError):
                self.update_metrics(prompt_tokens, completion_tokens)
                print("CAUGHT THE GENERATOREXIT")
            

    def request_token_count_chat(self, request: ChatCompletionRequest) -> int:
        """Returns the number of tokens in the prompt

        Converts the list of dictionaries of chat items into a prompt and calls
        count_tokens to get the token count for the prompt"""
        prompt = chat_items_to_prompt(request.chat_items)
        return self.count_tokens(prompt)

    # TODO: this should probably be structured a bit differently to not assume the user is providing all the prompt structure tokens
    async def completion_stream(
        self, request: CompletionRequest
    ) -> AsyncGenerator[RequestOutput, None]:
        """Completion stream for use by Complete and CompleteStream"""
        # inputs = self.tokenizer(, return_tensors="pt").to("cuda")

        # create text streamer and validate parameters
        max_new_tokens = 1536 if request.max_new_tokens == 0 else request.max_new_tokens
        temperature = 0.1 if request.temperature == 0.0 else request.temperature
        top_p = 1.0 if request.top_p == 0.0 else request.top_p
        top_k = -1 if request.top_k == 0.0 else request.top_k

        sampling_params = SamplingParams(
            n=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=["<|end|>"],
        )

        response_generator = self.async_engine.generate(
            request.prompt, sampling_params, request_id
        )

        full_responses = [""] * 1  # future place to add n generations
        async for new_responses in response_generator:  # type:ignore
            # just get the new piece of text
            new_text = [
                new.text.replace(old, "")
                for new, old in zip(new_responses.outputs, full_responses)
            ]
            full_responses = [output.text for output in new_responses.outputs]
            yield new_text[0]  # only works for n = 1

    async def Complete(
        self, request: CompletionRequest, context: GrpcContext
    ) -> CompletionResponse:
        completion_stream = self.completion_stream(request)
        response = ""
        async for text_chunk in completion_stream:
            response += text_chunk

        choice = CompletionChoice(text=response, index=0)

        # add inference attributes to the span
        current_span = trace.get_current_span()
        input_tokens = self.request_token_count_complete(request)
        current_span.set_attribute(PROMPT_TOKENS, input_tokens)
        generated_tokens = self.count_tokens(response)
        current_span.set_attribute(COMPLETION_TOKENS, generated_tokens)

        # increment total requests
        requests_counter.inc()

        return CompletionResponse(choices=[choice])

    async def CompleteStream(
        self, request: CompletionRequest, context
    ) -> AsyncGenerator[CompletionResponse, None]:
        completion_stream = self.completion_stream(request)
        response = ""
        async for text_chunk in completion_stream:
            response += text_chunk
            choice = CompletionChoice(text=text_chunk, index=0)
            yield CompletionResponse(choices=[choice])

        # add inference attributes to the span
        current_span = trace.get_current_span()
        input_tokens = self.request_token_count_complete(request)
        current_span.set_attribute(PROMPT_TOKENS, input_tokens)
        generated_tokens = self.count_tokens(response)
        current_span.set_attribute(COMPLETION_TOKENS, generated_tokens)

        # increment total requests
        requests_counter.inc()

    def request_token_count_complete(self, request: CompletionRequest) -> int:
        return self.count_tokens(request.prompt)

    async def ChatCancel(self, request: ChatCancellationRequest, context: GrpcContext) -> ChatCancellationResponse:
        await self.async_engine.abort(request.id)
        for i in range(20):
            print(f"ABORTED {request.id}")
        return ChatCancellationResponse(id=request.id)


if __name__ == "__main__":
    OTEL_COLLECTOR_ADDR = os.getenv(
        "OTEL_COLLECTOR_ADDR"
    )  # should be a string like "tempo:4317"
    # if the variable isn't set...then fall through to console tracing
    if OTEL_COLLECTOR_ADDR is None:
        trace.get_tracer_provider().add_span_processor(  # type: ignore
            SimpleSpanProcessor(ConsoleSpanExporter())
        )
    # if the variable is set...then use the remote grpc exporter
    else:
        otlp_exporter = OTLPSpanExporter(
            endpoint=f"grpc://{OTEL_COLLECTOR_ADDR}", insecure=True
        )
        trace.get_tracer_provider().add_span_processor(  # type: ignore
            BatchSpanProcessor(otlp_exporter)
        )

    # add otel grpc instrumentation
    # grpc_server_instrumentor = GrpcAioInstrumentorServer()
    # grpc_server_instrumentor.instrument()

    # start prometheus metrics exporter at 9999
    start_http_server(9999)

    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(MPTChat()))
