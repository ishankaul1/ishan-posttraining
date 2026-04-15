from loguru import logger


def _extract(completion) -> str:
    """Handle both standard (str) and conversational (list of dicts) formats."""
    if isinstance(completion, list):
        return completion[0]["content"].strip()
    return completion.strip()


def exact_match(completions, solution, **kwargs) -> list[float]:
    """Reward 1.0 if completion exactly matches the solution, else 0.0."""
    rewards = [float(_extract(c) == str(s).strip()) for c, s in zip(completions, solution)]
    logger.debug(f"exact_match: {rewards}")
    return rewards
