"""Define LLM used in RAG."""

import os

from utils import get_project_dir

_LLM = {}
_MODEL = {}

_PROJECT_PATH = get_project_dir()


def get_reranker(name, reranker_configs):
    if name not in _MODEL:
        from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

        model_ = FlagEmbeddingReranker(
            model=os.path.join(_PROJECT_PATH, reranker_configs['relative_path']),
            top_n=reranker_configs.get('top_n', 10)
        )

        _MODEL[name] = model_

    return _MODEL[name]


def get_dense_embedder(name, embedder_configs):
    if name not in _MODEL:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        model_ = HuggingFaceEmbedding(
            model_name=os.path.join(_PROJECT_PATH, embedder_configs['relative_path'])
        )

        _MODEL[name] = model_

    return _MODEL[name]


def get_sparse_embedder(name, embedder_configs):
    if name+'_sparse' not in _MODEL:
        embedder = get_dense_embedder(name, embedder_configs)

        from llama_index.vector_stores.milvus.utils import BGEM3SparseEmbeddingFunction

        class _BGEM3SparseEmbeddingFunction(BGEM3SparseEmbeddingFunction):
            def __init__(self):
                from FlagEmbedding import BGEM3FlagModel
                self.model = BGEM3FlagModel(embedder.model_name, use_fp16=False)

        _MODEL[name+'_sparse'] = _BGEM3SparseEmbeddingFunction()

    return _MODEL[name+'_sparse']


def get_llm(name, llm_config):
    if name not in _LLM:
        if llm_config['type'] == 'dashscope':
            model_ = _get_dashscope_llm(**llm_config)
        elif llm_config['type'] == 'ollama':
            model_ = _get_ollama_server_llm(**llm_config)
        elif llm_config['type'] == 'llama_cpp':
            model_ = _get_llama_cpp_llm(**llm_config)
        elif llm_config['type'] == 'mlx':
            model_ = _get_mlx_server_llm(**llm_config)
        else:
            raise ValueError(f"{llm_config['type']} not supported!")

        _LLM[name] = model_

    return _LLM[name]


def _get_dashscope_llm(
    temperature=0.7, max_tokens=256, top_k=50, top_p=0.95, presence_penalty=1.2, seed=29, **kwargs
):
    from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
    return DashScope(
        model_name=DashScopeGenerationModels.QWEN_TURBO,
        temperature=temperature,
        max_tokens=max_tokens,
        top_k=top_k,
        top_p=top_p,
        presence_penalty=presence_penalty,
        is_function_calling_model=False,
        seed=seed,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        enable_search=False
    )


def _get_llama_cpp_llm(
    model_path, temperature=0.7, max_new_tokens=256, seed=29, context_window=1024*4, n_batch=128,
    use_mmap=True, use_mlock=False, system_prompt='You are a helpful assistant.', **kwargs
):
    from llama_index.llms.llama_cpp import LlamaCPP
    from utils import completion_to_prompt, messages_to_prompt

    return LlamaCPP(
        model_path=model_path,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        context_window=context_window,
        model_kwargs={
            'use_mmap': use_mmap,
            'seed': seed,
            'use_mlock':  use_mlock,
            'n_batch': n_batch,
            'n_ctx': context_window,
            'flash_attn': True
        },
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        system_prompt=system_prompt
    )


def _get_ollama_server_llm(
    model_name, temperature=0.7, max_tokens=256, presence_penalty=1.2, seed=29,
    url='http://127.0.0.1:11434/v1/', **kwargs
):
    from llama_index.llms.openai_like import OpenAILike

    return OpenAILike(
        model=model_name,  #: name of the OpenAI model to use.
        is_chat_model=True,
        temperature=temperature,  #: a float from 0 to 1 controlling randomness in generation;
        max_tokens=max_tokens,  #: the maximum number of tokens to generate.
        additional_kwargs={
            'presence_penalty': presence_penalty,
            'seed': seed,
        },  #: Add additional parameters to OpenAI request body.
        max_retries=3,  #: How many times to retry the API call if it fails.
        timeout=360,  #: How long to wait, in seconds, for an API call before failing.
        reuse_client=False,  #: Reuse the OpenAI client between requests.
        # When doing anything with large volumes of async API calls, setting this to false can improve stability.
        api_key='ollama',  #: Your OpenAI api key
        api_base=url,  #: The base URL of the API to call
    )


def _get_mlx_server_llm(temperature=0.7, max_tokens=256, url='http://127.0.0.1:8889/v1/', **kwargs):
    from llama_index.llms.openai_like import OpenAILike

    return OpenAILike(
        model='default_model',
        is_chat_model=True,
        temperature=temperature,  #: a float from 0 to 1 controlling randomness in generation;
        max_tokens=max_tokens,  #: the maximum number of tokens to generate.
        additional_kwargs={},  #: Add additional parameters to OpenAI request body.
        max_retries=3,  #: How many times to retry the API call if it fails.
        timeout=360,  #: How long to wait, in seconds, for an API call before failing.
        reuse_client=False,  #: Reuse the OpenAI client between requests.
        # When doing anything with large volumes of async API calls, setting this to false can improve stability.
        api_key='mlx',  #: Your OpenAI api key
        api_base=url,  #: The base URL of the API to call
    )


# if __name__ == '__main__':
#     pass
    # from llama_index.core.base.llms.types import ChatMessage

    # model_ = _get_dashscope_llm(name='test_llm', temperature=0.9)
    # print(model_.chat([ChatMessage('你好！')]))
    # print(_LLM)

    # model_ = _get_llama_cpp_llm(
    #     name='test_llm',
    #     model_path='/Users/Zhiyuan/Project/rag_projects/my-web-novel/models/Qwen/Qwen2.5-7B-Instruct-Q8_0.gguf',
    #     temperature=0.9
    # )
    # print('\n'*5)
    # print(model_.chat([ChatMessage('你好！')]))
    # print('\n'*5)
    # print(model_.complete('你好！你擅长什么?'))
    # print(_LLM)

    # model_ = _get_ollama_server_llm(name='test_model', model_name='qwen_local_7b_q8:latest')
    # print(model_.chat([ChatMessage('你好！')]))
    # print(model_.complete('你好！你擅长什么?'))
    # print(_LLM)

    # model_ = _get_mlx_server_llm(name='test_model')

    # print(model_.complete('你好！你擅长什么?'))
    # print()
    # print(model_.chat([ChatMessage('你好！')]))
    # print()
    # for response in model_.stream_chat([ChatMessage('你好！')]):
    #     print(response.delta, flush=True, end="")
    # print()

    # for response in model_.stream_complete('你好！你擅长什么?'):
    #     print(response.delta, flush=True, end="")
    # print()
    # print(_LLM)
