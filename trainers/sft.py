from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig


def run(cfg: dict):
    logger.info(f"Starting SFT run | model={cfg['model']} dataset={cfg['dataset']}")

    config = SFTConfig(
        output_dir="checkpts/",
        num_train_epochs=cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=cfg.get("batch_size", 4),
        per_device_eval_batch_size=cfg.get("batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=cfg.get("learning_rate", 2e-5),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        bf16=cfg.get("bf16", True),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        save_total_limit=cfg.get("save_total_limit", 3),
        report_to=cfg.get("report_to", "wandb"),
        run_name=cfg.get("run_name", None),
        max_seq_length=cfg.get("max_seq_length", 2048),
    )

    logger.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(cfg["model"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])

    logger.info(f"Loading dataset: {cfg['dataset']}")
    dataset = load_dataset(cfg["dataset"])

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", dataset.get("test", None)),
        processing_class=tokenizer,
    )

    logger.info("Training...")
    trainer.train()
    logger.info("SFT run complete.")
