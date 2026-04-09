from loguru import logger


def exact_match(completions: list[str], **kwargs) -> list[float]:
    """Reward 1.0 if completion exactly matches the solution, else 0.0."""
    solutions = kwargs.get("solution", [])
    rewards = []
    for completion, solution in zip(completions, solutions):
        match = float(completion.strip() == str(solution).strip())
        rewards.append(match)
    logger.debug(f"exact_match: {rewards}")
    return rewards
