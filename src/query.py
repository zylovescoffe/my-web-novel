"""Define query flows."""

import copy
import json
import logging
import sys
import nest_asyncio

from llama_index.core import VectorStoreIndex
from llama_index.core import PromptTemplate
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from utils import (
    get_project_dir, get_config_file, get_milvus_vec_store, completion_to_prompt, display_eval_result
)
from prompts import (
    QA_PROMPT_TEMPLATE, LLM_EVAL_PROMPT_TEMPLATE, parse_llm_eval_result_for_score
)
from models import get_dense_embedder, get_reranker, get_llm, get_sparse_embedder

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

nest_asyncio.apply()

PROJECT_DIR = get_project_dir()
CONFIGS = get_config_file()
QUERY_CONFIG = CONFIGS['query_pipeline']


class _QueryEnginewithEval(CustomQueryEngine):
    """RAG Query Engine with evaluation incorporated."""

    retriever: BaseRetriever
    llm: OpenAI
    embedder: HuggingFaceEmbedding
    qa_prompt: PromptTemplate
    postprocess_list: list

    def custom_query(self, query_str: str, enable_eval=True, eval_ground_truth=None):
        if enable_eval is True:
            assert eval_ground_truth is not None, '`eval_ground_truth` must be provided for evaluation.'

        response_dict = {}
        query_bundle = QueryBundle(query_str=query_str, embedding=self.embedder.get_query_embedding(query_str))

        nodes = self.retriever.retrieve(query_bundle)
        if enable_eval is True:
            response_dict['eval_retrieve'] = self._eval_retrieve(nodes, eval_ground_truth)

        for ix, pprocess in enumerate(self.postprocess_list):
            nodes = pprocess.postprocess_nodes(nodes=nodes, query_bundle=query_bundle)
            if enable_eval is True:
                response_dict[f'eval_postprocess_{ix}'] = \
                    self._eval_preprocess(nodes, eval_ground_truth, response_dict['eval_retrieve'])

        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        prompt_ = completion_to_prompt(QA_PROMPT_TEMPLATE.format(context_str=context_str, query_str=query_str))

        if enable_eval is True:
            responses = []
            for _ in range(CONFIGS['llm_evaluation']['llm_runs']):
                responses.append(self.llm.complete(prompt_))

            response_dict['eval_generation'] = self._eval_generation(
                responses, eval_ground_truth,
                prompt_=QA_PROMPT_TEMPLATE.format(context_str=context_str, query_str=query_str),
                use_qwen=CONFIGS['llm_evaluation']['use_qwen']
            )
            response_dict['response'] = responses[0]

        else:
            response_text = ""
            resp = self.llm.stream_complete(query_str)
            for r in resp:
                print(r.delta, end="", flush=True)
                response_text += r.delta

            response_dict['response'] = response_text

        response_dict['nodes'] = nodes
        return response_dict

    def _eval_retrieve(self, nodes, ground_truth):
        nodes_retrieved = set([e.node.id_ for e in nodes])
        nodes_gt = set(ground_truth['nodes'])
        return {
            "recall": list(nodes_gt & nodes_retrieved),
            "miss": list(nodes_gt - nodes_retrieved),
            "scores": [e.score for e in nodes],
            "recall_percentage": round(len(nodes_gt & nodes_retrieved)/len(nodes_gt), 5) if nodes_gt else 0.
        }

    def _eval_preprocess(self, nodes, ground_truth, eval_retrieve):
        nodes_rerank = set([e.node.id_ for e in nodes])
        nodes_gt = set(ground_truth['nodes']) & set(eval_retrieve['recall'])
        return {
            "recall": list(nodes_gt & nodes_rerank),
            "miss": list(nodes_gt - nodes_rerank),
            "scores": [e.score for e in nodes],
            "recall_percentage": round(len(nodes_gt & nodes_rerank)/len(nodes_gt), 5) if nodes_gt else 0.
        }

    def _eval_generation(self, responses, ground_truth, prompt_, use_qwen=False):
        """
        1. Use multiple LLM generations -> check embeddings are similar or not.
        2. Use golden answer for embedding similarity.
        3. Use more advanced LLM - Dashscope/Qwen.
        """
        embeddings = self.embedder._model.encode([e.text for e in responses])
        similarity_in_group = embeddings @ embeddings.T

        embedding_gt = self.embedder._model.encode(ground_truth['answer'])
        similarity_with_gt = embedding_gt @ embeddings[0].T

        if use_qwen is True:
            eval_llm = get_llm('llm_evaluation', CONFIGS['llm_evaluation'])
            eval_llm_res = eval_llm.complete(
                LLM_EVAL_PROMPT_TEMPLATE.format(
                    instruction=prompt_,
                    response=responses[0].text,
                    reference_answer=ground_truth['answer']
                ))

        return {
            'similarity_within_group_avg': round(similarity_in_group[similarity_in_group != 1.].mean().item(), 5),
            'similarity_within_group_min': round(similarity_in_group.min().item(), 5),
            'similarity_with_gt': similarity_with_gt.item(),
            'LLM_evaluation': eval_llm_res.text if use_qwen is True else None,
            'LLM_evaluation_score': parse_llm_eval_result_for_score(eval_llm_res) if use_qwen is True else -1
        }


def get_index_query_engine(index=None, streaming=False, collection_name=None, eval_enable=True):
    if index is None:
        assert collection_name is not None, '`collection_name` must be provided if `index` is None.'
        vector_store = get_milvus_vec_store(
            collection_name=collection_name,
            similarity_metric='COSINE',
            enable_sparse=True,
            sparse_embedding_function=get_sparse_embedder(
                name='sentence_embedder', embedder_configs=CONFIGS['sentence_embedder']
            ),
            hybrid_ranker=CONFIGS['sentence_embedder']['hybrid_ranker']
        )

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=get_dense_embedder('sentence_embedder', CONFIGS['sentence_embedder'])
        )

    retriever = index.as_retriever(
        similarity_top_k=QUERY_CONFIG['similarity_top_k'],
        vector_store_query_mode='hybrid'
    )
    return _QueryEnginewithEval(
        retriever=retriever,
        llm=get_llm('llm_generation', CONFIGS['llm_generation']),
        qa_prompt=QA_PROMPT_TEMPLATE,
        embedder=get_dense_embedder('sentence_embedder', CONFIGS['sentence_embedder']),
        postprocess_list=[
            get_reranker('text_reranker', CONFIGS['text_reranker'])
        ]
    )


def query_stream(question, collection_name=None, query_engine=None):
    if query_engine is None:
        assert type(collection_name) is str, '`colelction_name` must be provided if `query_engine` is not provided.'
        query_engine = get_index_query_engine(collection_name=collection_name, streaming=True)

    return query_engine.custom_query(question, enable_eval=False)


def query_with_eval_dataset(query_engine, eval_json_list, eval_json_save_path, display_result=True):
    assert type(query_engine) is _QueryEnginewithEval, 'Only support engine from `get_index_query_engine`!'

    eval_results = []
    for ix, json_e in enumerate(eval_json_list):
        print(f"--> Processing question {ix+1}/{len(eval_json_list)}: {json_e['question']}")

        res_i = query_engine.custom_query(query_str=json_e['question'], enable_eval=True, eval_ground_truth=json_e)
        json_r = json_e.copy()
        json_r['response'] = res_i['response'].text
        json_r['evaluation'] = {kk: vv for kk, vv in res_i.items() if kk.startswith('eval_')}

        eval_results.append(copy.deepcopy(json_r))

    with open(eval_json_save_path, 'w') as fw:
        for res_line in eval_results:
            fw.write(json.dumps({
                'question': res_line['question'],
                'answer': res_line['answer'],
                'response': res_line['response'],
                'evaluation': res_line['evaluation']
            }, ensure_ascii=False)+'\n')

    if display_result is True:
        display_eval_result(eval_results, save_figure_path=eval_json_save_path+'.png')

    return eval_results


# if __name__ == "__main__":
#     response = query_stream('韩立红颜知己有谁?', collection_name='create_eval_dataset_collections_hybrid')
    # print(response['response'])

    # dataset_path = "../data/eval/eval_dataset_text_basic.jsonl"
    # with open(dataset_path, 'r') as fh:
    #     jsons_ = [json.loads(e) for e in fh.readlines()]

    # query_engine_eval = get_index_query_engine(collection_name="create_eval_dataset_collections_hybrid")
    # query_with_eval_dataset(
    #     query_engine=query_engine_eval,
    #     eval_json_list=jsons_[:2],
    #     eval_json_save_path=dataset_path.rstrip('.jsonl')+'.eval'
    # )

#     dataset_path = "../data/eval/eval_dataset_text_adv.jsonl"
#     with open(dataset_path, 'r') as fh:
#         jsons_ = [json.loads(e) for e in fh.readlines()]

#     query_engine_eval = get_index_query_engine(collection_name="create_eval_dataset_collections_hybrid_512")
#     query_with_eval_dataset(
#         query_engine=query_engine_eval,
#         eval_json_list=jsons_[:5],
#         eval_json_save_path=dataset_path.rstrip('.jsonl')+'.eval'
#     )
