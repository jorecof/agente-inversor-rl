import numpy as np


def policy(name, seed):
    rng = np.random.default_rng(seed)
    if name == "buy_hold":
        return lambda obs: 2
    if name == "cash":
        return lambda obs: 0
    if name == "random":
        return lambda obs: int(rng.integers(3))
    raise ValueError(name)
