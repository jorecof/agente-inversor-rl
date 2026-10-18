"""Entrena candidatos, selecciona en validación y luego evalúa una vez en prueba."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.env_checker import check_env
from .data import prepare, FEATURES
from .environment import InvestmentEnv
from .agent import train_agent, action_for
from .baselines import policy
from .evaluate import rollout, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", default="results/main")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("Elige una carpeta nueva: no se sobrescriben experimentos.")
    out.mkdir(parents=True)
    (out/"models").mkdir()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    df, x, splits, mu, sigma = prepare(cfg["data"], cfg)
    train_frame = df.iloc[:splits["train"][1]+1]
    train_frame[["close", "volume"]+FEATURES].describe().to_csv(out/"eda_train_summary.csv") if "volume" in df else train_frame[["close"]+FEATURES].describe().to_csv(out/"eda_train_summary.csv")
    np.savez(out/"scaler.npz", mean=mu, std=sigma, features=FEATURES)
    (out/"config.json").write_text(json.dumps(cfg, indent=2))
    manifest = {"dataset": cfg["data"], "sha256": hashlib.sha256(Path(cfg["data"]).read_bytes()).hexdigest(),
                "python": sys.version, "platform": platform.platform(),
                "versions": {n: importlib.metadata.version(n) for n in
                             ["numpy", "pandas", "torch", "gymnasium", "stable-baselines3", "matplotlib"]},
                "splits": {s: {"indices": list(v), "start": str(df.timestamp.iloc[v[0]]),
                                "end": str(df.timestamp.iloc[v[1]]), "transitions": v[1]-v[0]}
                           for s, v in splits.items()},
                "selection_rule": "mean(validation_return - 0.5*abs(validation_max_drawdown))",
                "test_policy": "Congelar selección antes de evaluar; prueba no selecciona candidatos."}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2))

    def env(split, reward):
        return InvestmentEnv(df, x, *splits[split], initial_capital=cfg["initial_capital"],
                             cost_rate=cfg["cost_rate"], reward_mode=reward, risk_lambda=cfg["risk_lambda"])

    check_env(env("train", "log"), warn=True)
    candidates, trained = [], {}
    # Entrenamiento y validación: no hay rollout del periodo final en este bucle.
    for kind in ["qlearning", "dqn"]:
        for reward in ["log", "risk"]:
            for seed in cfg["seeds"]:
                name = f"{kind}_{reward}_{seed}"
                print("Entrenando", name, flush=True)
                model, history = train_agent(kind, env("train", reward), cfg, seed, out/"models"/name)
                trained[name] = model
                if history:
                    pd.DataFrame(history).to_csv(out/f"learning_{name}.csv", index=False)
                curve = rollout(env("validation", reward), lambda o, m=model: action_for(m, o), seed)
                m = metrics(curve, cfg["initial_capital"])
                score = m["return"]-cfg["selection_drawdown_penalty"]*abs(m["max_drawdown"])
                candidates.append(dict(method=kind, reward=reward, seed=seed, score=score, **m))
                curve.to_csv(out/f"validation_{name}.csv", index=False)
    validation = pd.DataFrame(candidates)
    validation.to_csv(out/"validation_metrics.csv", index=False)
    ranking = validation.groupby(["method", "reward"]).score.mean().sort_values(ascending=False)
    winner_kind, winner_reward = ranking.index[0]
    # Las semillas son réplicas, no hiperparámetros: no seleccionar la mejor semilla.
    selected = {"method": winner_kind, "reward": winner_reward,
                "seeds": cfg["seeds"], "validation_score": float(ranking.iloc[0]),
                "ranking": {"/".join(k): float(v) for k, v in ranking.items()}}
    (out/"selection_frozen.json").write_text(json.dumps(selected, indent=2))
    print("Selección congelada:", selected["method"], selected["reward"], flush=True)
    # Todos los candidatos se muestran como auditoría, sin cambiar la selección.
    rows = []
    for (kind, reward), group in validation.groupby(["method", "reward"]):
        for seed in cfg["seeds"]:
            name = f"{kind}_{reward}_{seed}"
            curve = rollout(env("test", reward), lambda o, m=trained[name]: action_for(m, o), seed)
            curve.to_csv(out/f"test_{name}.csv", index=False)
            rows.append(dict(method=kind, reward=reward, seed=seed,
                             selected=kind == winner_kind and reward == winner_reward,
                             **metrics(curve, cfg["initial_capital"])))
    for kind in ["buy_hold", "random", "cash"]:
        seeds = cfg["seeds"] if kind == "random" else [cfg["seeds"][0]]
        for seed in seeds:
            curve = rollout(env("test", "log"), policy(kind, seed), seed)
            curve.to_csv(out/f"test_{kind}_{seed}.csv", index=False)
            rows.append(dict(method=kind, reward="baseline", seed=seed, selected=False,
                             **metrics(curve, cfg["initial_capital"])))
    results = pd.DataFrame(rows)
    results.to_csv(out/"test_metrics.csv", index=False)
    cols = ["return", "max_drawdown", "costs", "final_equity", "trades", "cash_fraction", "mean_target_weight"]
    results.groupby(["method", "reward"])[cols].agg(["mean", "std"]).to_csv(out/"test_summary.csv")
    from .plots import create_plots
    create_plots(df, out, selected)
    from .report import write_report
    write_report(out, manifest, cfg, validation, results, selected)
    print(results[["method", "reward", "seed", "return", "max_drawdown", "costs"]].to_string(index=False))


if __name__ == "__main__":
    main()
