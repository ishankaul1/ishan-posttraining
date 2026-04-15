"""
Dispatcher: loads a YAML config and routes to the appropriate trainer.
Usage: accelerate launch train.py --config configs/sft_qwen_1.5b.yaml
"""

import argparse
import yaml

# Fields that must be specific types regardless of how YAML parses them
# (YAML parses scientific notation like 2e-5 as a string)
_FLOAT_FIELDS = {"learning_rate", "warmup_ratio", "weight_decay", "adam_epsilon", "max_grad_norm"}
_INT_FIELDS = {
    "num_train_epochs", "batch_size", "gradient_accumulation_steps", "logging_steps",
    "save_steps", "save_total_limit", "max_steps", "max_length",
    "num_generations", "max_completion_length",
}


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in _FLOAT_FIELDS:
        if key in cfg:
            cfg[key] = float(cfg[key])
    for key in _INT_FIELDS:
        if key in cfg:
            cfg[key] = int(cfg[key])
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args, extra = parser.parse_known_args()

    cfg = load_config(args.config)

    # Extra CLI args override config values (--key=value or --key value)
    i = 0
    while i < len(extra):
        key = extra[i].lstrip("-").replace("-", "_")
        if "=" in key:
            k, v = key.split("=", 1)
            cfg[k] = v
            i += 1
        else:
            cfg[key] = extra[i + 1]
            i += 2

    trainer_name = cfg.get("trainer")
    if not trainer_name:
        raise ValueError("Config must specify a 'trainer' field (e.g. trainer: sft)")

    if trainer_name == "sft":
        from trainers.sft import run
    elif trainer_name == "grpo":
        from trainers.grpo import run
    else:
        raise ValueError(f"Unknown trainer: {trainer_name!r}. Valid options: sft, grpo")

    run(cfg)


if __name__ == "__main__":
    main()
