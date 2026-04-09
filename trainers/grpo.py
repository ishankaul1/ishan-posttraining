from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from rewards import resolve_reward


def run(cfg: dict):
    logger.info(f"Starting GRPO run | model={cfg['model']} dataset={cfg['dataset']}")

    reward_fns = [resolve_reward(name) for name in cfg.get("reward_functions", [])]
    if not reward_fns:
        raise ValueError("GRPO requires at least one entry in 'reward_functions'")
    logger.info(f"Reward functions: {cfg['reward_functions']}")

    config = GRPOConfig(
        output_dir="checkpts/",
        num_train_epochs=cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=cfg.get("batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=cfg.get("learning_rate", 1e-6),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        bf16=cfg.get("bf16", True),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        save_total_limit=cfg.get("save_total_limit", 3),
        report_to=cfg.get("report_to", "wandb"),
        run_name=cfg.get("run_name", None),
        num_generations=cfg.get("num_generations", 8),
        max_prompt_length=cfg.get("max_prompt_length", 512),
        max_completion_length=cfg.get("max_completion_length", 512),
    )

    logger.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(cfg["model"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])

    logger.info(f"Loading dataset: {cfg['dataset']}")
    dataset = load_dataset(cfg["dataset"])

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=reward_fns,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
    )

    logger.info("Training...")
    trainer.train()
    logger.info("GRPO run complete.")
