"""Define data index pipeline."""

import asyncio
import os
import hashlib
import logging
import sys
import nest_asyncio

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

from utils import get_project_dir, get_config_file, get_milvus_vec_store
from models import get_dense_embedder, get_sparse_embedder

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

nest_asyncio.apply()

PROJECT_DIR = get_project_dir()
CONFIGS = get_config_file()
INDEX_CONFIG = CONFIGS['index_pipeline']


def _id_generator(i, doc):
    return hashlib.sha1(doc.metadata['file_name'].encode('utf-8')).hexdigest()[:8] + str(i).zfill(8)


def index_document_w_chunk_embed(
    raw_data_dir, ingest_data_name, pipeline_cache_path=os.path.join(PROJECT_DIR, INDEX_CONFIG['pipeline_cache_path'])
):
    """Define a most basic index pipeline with chunking and embedding."""
    reader = SimpleDirectoryReader(raw_data_dir)
    documents = reader.load_data(show_progress=True)

    vector_store = get_milvus_vec_store(
        collection_name=ingest_data_name,
        dim=CONFIGS['sentence_embedder']['dim'],
        overwrite=True,
        similarity_metric='COSINE',
        enable_sparse=True,
        sparse_embedding_function=get_sparse_embedder(
            name='sentence_embedder', embedder_configs=CONFIGS['sentence_embedder']
        ),
        hybrid_ranker=CONFIGS['sentence_embedder']['hybrid_ranker']
    )

    dense_embedder = get_dense_embedder(name='sentence_embedder', embedder_configs=CONFIGS['sentence_embedder'])
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=INDEX_CONFIG['chunk_size'],
                chunk_overlap=INDEX_CONFIG['chunk_overlap'],
                id_func=_id_generator
            ),
            dense_embedder,
        ],
        vector_store=vector_store
    )

    pipeline_cache_path = pipeline_cache_path + '_' + str(ingest_data_name)
    if os.path.exists(pipeline_cache_path):
        print('Loaded existed pipeline.')
        pipeline.load(pipeline_cache_path)
        asyncio.run(pipeline.arun(documents=documents, show_progress=True))
    else:
        print('Run new pipeline from scratch.')
        asyncio.run(pipeline.arun(documents=documents, show_progress=True))
        pipeline.persist(pipeline_cache_path)

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=dense_embedder)
    if vector_store.client.has_collection(collection_name=ingest_data_name):
        print(f'{ingest_data_name} CREATED!')
        print(vector_store.client.describe_collection(collection_name=ingest_data_name))
        return index

    raise ValueError(f'{ingest_data_name} NOT created!')


# if __name__ == '__main__':
#     # pass
#     ingest_collection_name = "create_eval_dataset_collections_hybrid_512"
#     index = index_document_w_chunk_embed(
#         raw_data_dir='/Users/Zhiyuan/Project/rag_projects/my-web-novel/data/filter',
#         ingest_data_name=ingest_collection_name
#     )
