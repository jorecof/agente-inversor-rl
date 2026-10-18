"""Métricas sobre patrimonio monetario, independientes de la recompensa."""
import numpy as np
import pandas as pd


def rollout(env, act, seed):
    obs, _ = env.reset(seed=seed)
    rows = [{"timestamp": str(env.frame.timestamp.iloc[env.start]),
             "equity": env.initial_capital, "cost": 0., "cost_total": 0.,
             "drawdown": 0., "turnover": 0.}]
    while True:
        obs, _, term, trunc, info = env.step(act(obs))
        rows.append(info)
        if term or trunc:
            return pd.DataFrame(rows)


def metrics(curve, capital):
    steps = curve.iloc[1:]
    return {"return": float(curve.equity.iloc[-1]/capital-1),
            "max_drawdown": float((curve.equity/curve.equity.cummax()-1).min()),
            "costs": float(steps.cost.sum()), "final_equity": float(curve.equity.iloc[-1]),
            "turnover": float(steps.turnover.sum()),
            "trades": int((steps.cost > 1e-7).sum()),
            "cash_fraction": float((steps.target_weight == 0).mean()),
            "mean_target_weight": float(steps.target_weight.mean()),
            "reward_sum": float(steps.reward.sum())}
