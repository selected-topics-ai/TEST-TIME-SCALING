import numpy as np
import torch.nn as nn

from typing import List, Literal
from dataclasses import dataclass
from scores import aggregate_scores
from vllm import LLM, SamplingParams

@dataclass
class BestOfNConfig:
    system_prompt: str
    n_completions: int
    temperature: float
    max_tokens: int
    top_p: float
    agg_strategy: Literal["min", "prod", "last"]

def best_of_n(prompts: List[str], llm: LLM, reward_model: nn.Module, config: BestOfNConfig):

    conversations = [
        [
            {"role": "system", "content": config.system_prompt},
            {"role": "user",   "content": prompt              },
        ]
        for prompt in prompts
    ]

    tokenizer = llm.get_tokenizer()
    templated_conversations = tokenizer.apply(conversations, tokenize=False, add_generation_prompt=True)

    templated_conversations = [c for conversation in templated_conversations for c in
                               [conversation] * config.n_completions]

    completions = [[] for _ in range(len(prompts))]
    completion_tokens = [[] for _ in range(len(prompts))]

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        n=1,
    )

    responses = llm.generate(
        templated_conversations,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    if len(responses) != len(prompts) * config.n_completions:
        raise ValueError(
            f"Generated {len(responses)} responses instead of {len(prompts * config.n_completions)}"
        )

    for i in range(len(completions)):
        completions[i] = [
            output.text
            for r in responses[i * config.n_completions: (i + 1) * config.n_completions]
            for output in r.outputs
        ]
        completion_tokens[i] = [
            len(output.token_ids)
            for r in responses[i * config.n_completions: (i + 1) * config.n_completions]
            for output in r.outputs
        ]

    for c in completions:
        if len(c) != config.n_completions:
            raise ValueError(f"Generated {len(c)} completions instead of {config.n_completions}")

    scores = reward_model.score(prompts, completions)
    agg_scores = [
        [aggregate_scores(s, config.agg_strategy) for s in score] for score in scores
    ]

    pred = [completion[np.argmax(s)] for completion, s in zip(completions, agg_scores)]

    return {
        "completions": completions,
        "scores": scores,
        "pred": pred,
        "completion_tokens": completion_tokens,
    }

