"""Define Prompts here."""

from llama_index.core import PromptTemplate

_qa_prompt_tmpl_str = (
    "提供的文章段落如下.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "请根据所提供的文章段落，理解并汇总后，简洁明了地回答问题, 不要无中生有。\n"
    "Query: {query_str}\n"
    "Answer: "
)

QA_PROMPT_TEMPLATE = PromptTemplate(_qa_prompt_tmpl_str)

_eval_prompt_tmpl_str = (
    "###Task Description:\n"
    "An instruction (might include an Input inside it), a response to evaluate, "
    "a reference answer that gets a score of 5, and a score rubric representing "
    "a evaluation criteria are given.\n"
    "1. Write a detailed feedback that assess the quality of the response strictly "
    "based on the given score rubric, not evaluating in general.\n"
    "2. After the feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.\n"
    "3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] "
    "{{an integer number between 1 and 5}}\n"
    "4. Do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.\n\n"
    "###The instruction to evaluate:\n{instruction}\n\n"
    "###Response to evaluate:\n{response}\n\n"
    "###Reference Answer (Score 5):\n{reference_answer}\n\n"
    "###Score Rubrics:\n[Is the response correct, accurate, and factual based on the reference answer?]\n"
    "Score 1: The response is completely incorrect, inaccurate, and/or not factual.\n"
    "Score 2: The response is mostly incorrect, inaccurate, and/or not factual.\n"
    "Score 3: The response is somewhat correct, accurate, and/or factual.\n"
    "Score 4: The response is mostly correct, accurate, and factual.\n"
    "Score 5: The response is completely correct, accurate, and factual.\n\n"
    "###Feedback:"
)

LLM_EVAL_PROMPT_TEMPLATE = PromptTemplate(_eval_prompt_tmpl_str)


def parse_llm_eval_result_for_score(response):
    try:
        return int(response.text.split('[RESULT]')[-1])
    except Exception:
        print(response.text)
        print('Need LLM retry..')
        return -1
