"""Define utility function here."""

import os

from pathlib import Path

from llama_index.vector_stores.milvus import MilvusVectorStore


def get_project_dir():
    return Path(__file__).parent.parent


def get_config_file():
    import yaml
    with open(os.path.join(get_project_dir(), 'configs', 'config.yaml')) as fh:
        try:
            return yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc
        return {}


def completion_to_prompt(completion):
    return f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n"


def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
        elif message.role == "user":
            prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        elif message.role == "assistant":
            prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"

    if not prompt.startswith("<|im_start|>system"):
        prompt = "<|im_start|>system\n" + prompt

    prompt = prompt + "<|im_start|>assistant\n"
    return prompt


def get_milvus_vec_store(collection_name, **kwargs):
    configs = get_config_file()['db_setup']
    if configs.get('use_local', None):
        return MilvusVectorStore(
            uri=configs['use_local'], collection_name=collection_name, **kwargs
        )
    else:
        return MilvusVectorStore(
            uri=f"http://{configs['db_url']}:{configs['db_port']}", collection_name=collection_name, **kwargs
        )


def display_eval_result(eval_results, save_figure_path=None):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('tkagg')

    eval_stats = [e['evaluation'] for e in eval_results]
    display_data = {
        k: [e[k]['recall_percentage'] for e in eval_stats]
        for k in eval_stats[0].keys() if k not in ['eval_retrieve', 'eval_generation']
    }

    display_data.update({
        'eval_retrieve': [e['eval_retrieve']['recall_percentage'] for e in eval_stats],
        'eval_generation_llm': [e['eval_generation']['LLM_evaluation_score'] for e in eval_stats],
        'eval_generation_gt_similarity': [e['eval_generation']['similarity_with_gt'] for e in eval_stats],
        'eval_generation_consistency': [e['eval_generation']['similarity_within_group_avg'] for e in eval_stats],
    })

    num_plots = len(display_data)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 8), sharex=True, sharey=False)

    for i, (kk, vv) in enumerate(display_data.items()):
        axes[i].bar(x=[e for e in range(len(vv))], height=vv, alpha=0.8)
        axes[i].axhline(y=sum(vv)/len(vv), color='red', linestyle='dashed')
        axes[i].set_title(f"{kk}: (average: {round(sum(vv)/len(vv), 5)})")
        axes[i].axes.get_xaxis().set_visible(False)

    plt.tight_layout()

    if save_figure_path:
        plt.savefig(save_figure_path)

    plt.show(block=False)


# if __name__ == '__main__':
#     # print(get_milvus_vec_store('test_milvus_data_1212_collection'))
#     import json
#     with open("../data/eval/eval_test.jsonl", "r") as fr:
#         data = [json.loads(e) for e in fr.readlines()]
#     display_eval_result(data, save_figure_path="../data/eval/eval_test.png")
