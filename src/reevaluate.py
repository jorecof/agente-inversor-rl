"""Reevaluación sin entrenamiento ni modificación de la selección congelada."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN
from .data import features, load_prices, FEATURES
from .environment import InvestmentEnv
from .agent import TabularQ, action_for
from .baselines import policy
from .evaluate import metrics, rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="results/main")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run, out = Path(args.run), Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    cfg = json.loads((run/"config.json").read_text())
    manifest = json.loads((run/"manifest.json").read_text())
    selected = json.loads((run/"selection_frozen.json").read_text())
    if hashlib.sha256(Path(cfg["data"]).read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("El CSV no coincide con la instantánea original")
    df = features(load_prices(cfg["data"]))
    with np.load(run/"scaler.npz") as scaler:
        x = np.clip((df[FEATURES].to_numpy()-scaler["mean"])/scaler["std"], -10, 10).astype(np.float32)
    start, end = manifest["splits"]["test"]["indices"]
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    rows = []

    def env(reward):
        return InvestmentEnv(df, x, start, end, initial_capital=cfg["initial_capital"],
                             cost_rate=cfg["cost_rate"], reward_mode=reward, risk_lambda=cfg["risk_lambda"])

    for kind in ["qlearning", "dqn"]:
        for reward in ["log", "risk"]:
            for seed in cfg["seeds"]:
                name = f"{kind}_{reward}_{seed}"
                prefix = run/"models"/name
                if kind == "qlearning":
                    agent = TabularQ(seed)
                    agent.q = np.load(str(prefix)+".npy", allow_pickle=False)
                else:
                    agent = DQN.load(str(prefix)+".zip", device="cpu")
                curve = rollout(env(reward), lambda o, a=agent: action_for(a, o), seed)
                curve.to_csv(out/f"test_{name}.csv", index=False)
                rows.append(dict(method=kind, reward=reward, seed=seed,
                                 selected=kind == selected["method"] and reward == selected["reward"],
                                 **metrics(curve, cfg["initial_capital"])))
    for kind in ["buy_hold", "random", "cash"]:
        for seed in (cfg["seeds"] if kind == "random" else [cfg["seeds"][0]]):
            curve = rollout(env("log"), policy(kind, seed), seed)
            curve.to_csv(out/f"test_{kind}_{seed}.csv", index=False)
            rows.append(dict(method=kind, reward="baseline", seed=seed, selected=False,
                             **metrics(curve, cfg["initial_capital"])))
    results = pd.DataFrame(rows).sort_values(["method", "reward", "seed"]).reset_index(drop=True)
    results.to_csv(out/"test_metrics.csv", index=False)
    original = pd.read_csv(run/"test_metrics.csv").sort_values(["method", "reward", "seed"]).reset_index(drop=True)
    cols = ["return", "max_drawdown", "costs", "final_equity", "turnover", "trades"]
    np.testing.assert_allclose(results[cols], original[cols], rtol=1e-9, atol=1e-7)
    print("Reevaluación completa: métricas coincidentes con la ejecución guardada.")


if __name__ == "__main__":
    main()
