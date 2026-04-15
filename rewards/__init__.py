from rewards.exact import exact_match
from trl.rewards import accuracy_reward

REGISTRY = {
    "exact_match": exact_match,
    "accuracy_reward": accuracy_reward,  # TRL built-in: checks boxed math answers
}


def resolve_reward(name: str):
    if name in REGISTRY:
        return REGISTRY[name]
    print(f"[rewards] '{name}' not in registry, passing as HF model ID to TRL")
    return name
