def invariant(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Invariant violated: {message}")
