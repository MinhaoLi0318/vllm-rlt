"""Rollout logprobs: the FP32 log-probability of each sampled token.

Scoring consumes no RNG. Raw logprobs score the model distribution. Processed
logprobs score ``Sampler.processed_logits``, the logits ``Sampler.probabilities``
normalizes, so they score the distribution the token was actually drawn from.
This module imports ``worker/sampler.py``, never the reverse.
"""

from vllm_rlt.worker.sampler import Sampler

LOGPROBS_MODES = ("raw_logprobs", "processed_logprobs")


def check_logprobs_mode(mode):
    if mode not in LOGPROBS_MODES:
        raise ValueError(
            f"logprobs_mode must be one of {LOGPROBS_MODES}, got {mode!r}; "
            "raw_logits and processed_logits are not supported"
        )


def token_logprob(logits, params, token_id, mode):
    """0-dim FP32 log-probability of ``token_id`` (0-dim long tensor) for one logits row.

    raw_logprobs uses the model distribution. processed_logprobs uses the
    distribution actually sampled from; greedy rows have no temperature (vLLM).
    Consumes no RNG and keeps the result on the logits device.
    """
    if mode == "processed_logprobs" and params.temperature != 0:
        logits = Sampler.processed_logits(logits, params)
    return Sampler.raw_logprobs(logits[None], token_id.view(1))[0]
