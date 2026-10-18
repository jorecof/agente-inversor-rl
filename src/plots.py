import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rl_inversion_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_plots(df, out, selected):
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0, 0].plot(df.timestamp, df.close, lw=1)
    ax[0, 0].set(title="BTC/USDT: precio diario", ylabel="USDT")
    ax[0, 1].hist(df.return_1, bins=65, color="#376b98")
    ax[0, 1].set(title="Distribución de retornos diarios", xlabel="Retorno simple")
    ax[1, 0].plot(df.timestamp, df.volatility_20, lw=1)
    ax[1, 0].set(title="Volatilidad móvil, 20 días", ylabel="Desviación diaria")
    if "volume" in df:
        ax[1, 1].plot(df.timestamp, df.volume, lw=.7)
        ax[1, 1].set(title="Volumen diario", ylabel="BTC")
    else:
        ax[1, 1].plot(df.timestamp, df.momentum_20, lw=.7)
        ax[1, 1].set(title="Momentum, 20 observaciones")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out/"eda.png")
    plt.close(fig)
    names = [(f"{selected['method']}_{selected['reward']}_{s}", f"Agente, semilla {s}") for s in selected["seeds"]]
    names += [("buy_hold_7", "Comprar y mantener"), ("random_7", "Aleatorio, semilla 7"), ("cash_7", "Efectivo")]
    # Los nombres de las semillas de baselines se obtienen del experimento.
    seed0 = selected["seeds"][0]
    names[-3:] = [(f"buy_hold_{seed0}", "Comprar y mantener"),
                   (f"random_{seed0}", f"Aleatorio, semilla {seed0}"), (f"cash_{seed0}", "Efectivo")]
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for name, label in names:
        curve = pd.read_csv(out/f"test_{name}.csv")
        dates = pd.to_datetime(curve.timestamp)
        ax[0].plot(dates, curve.equity, label=label, lw=1.2)
        ax[1].plot(dates, curve.drawdown*100, lw=1.1)
    ax[0].set(title="Evaluación final: selección congelada en validación", ylabel="Capital (USDT)")
    ax[0].legend(fontsize=8, ncol=2)
    ax[1].set(ylabel="Drawdown (%)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out/"equity_drawdown.png")
    plt.close(fig)
    name = f"{selected['method']}_{selected['reward']}_{seed0}"
    curve = pd.read_csv(out/f"test_{name}.csv").iloc[1:]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.step(pd.to_datetime(curve.decision_timestamp), curve.target_weight, where="post")
    ax.set(title=f"Posiciones objetivo: {name}", ylabel="Fracción invertida", yticks=[0, .5, 1])
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out/"positions.png")
    plt.close(fig)
    val = pd.read_csv(out/"validation_metrics.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    groups = list(val.groupby(["method", "reward"]))
    means = [g.score.mean() for _, g in groups]
    errors = [g.score.std(ddof=1) for _, g in groups]
    ax.bar([" / ".join(k) for k, _ in groups], means, yerr=errors, capsize=5, color="#376b98")
    ax.set(title="Comparación de recompensas en validación", ylabel="Retorno - 0,5 × |drawdown máximo|")
    ax.axhline(0, color="black", lw=.7)
    fig.tight_layout()
    fig.savefig(out/"reward_comparison.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for path in sorted(out.glob("learning_*.csv")):
        history = pd.read_csv(path)
        axes[0].plot(history.episode, history.reward.rolling(10, min_periods=1).mean(),
                     label=path.stem[len("learning_"):].replace("qlearning", "Q"), lw=.8)
    for path in sorted((out/"models").glob("*_monitor.csv")):
        history = pd.read_csv(path, skiprows=1)
        axes[1].plot(history.l.cumsum(), history.r.rolling(3, min_periods=1).mean(),
                     label=path.stem.replace("_monitor", ""), lw=.8)
    axes[0].set(title="Q-learning: media móvil de 10 episodios", xlabel="Episodio", ylabel="Recompensa")
    axes[1].set(title="DQN: media móvil de 3 episodios", xlabel="Pasos", ylabel="Recompensa")
    for ax in axes:
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out/"learning_curves.png")
    plt.close(fig)
