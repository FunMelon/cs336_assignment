try:
    from importlib.metadata import version

    from .util import (
        tokenize_prompt_and_output,
        compute_entropy,
        get_response_log_probs,
        masked_normalize,
        sft_microbatch_train_step,
    )

    from .rl_util import (
        compute_group_normalized_rewards,
        compute_naive_policy_gradient_loss,
        compute_grpo_clip_loss,
        compute_policy_gradient_loss,
        masked_mean,
        grpo_microbatch_train_step,
    )

    __version__ = version("cs336-alignment")
    __all__ = [
        "tokenize_prompt_and_output",
        "compute_entropy",
        "get_response_log_probs",
        "masked_normalize",
        "sft_microbatch_train_step",
        "compute_group_normalized_rewards",
        "compute_naive_policy_gradient_loss",
        "compute_grpo_clip_loss",
        "compute_policy_gradient_loss",
        "masked_mean",
        "grpo_microbatch_train_step",
    ]

except Exception:
    # 开发环境下的版本号
    __version__ = "0.1.0-dev"
