from rewards.exact import exact_match

REGISTRY = {
    "exact_match": exact_match,
}


def resolve_reward(name: str):
    if name in REGISTRY:
        return REGISTRY[name]
    print(f"[rewards] '{name}' not in registry, passing as HF model ID to TRL")
    return name
