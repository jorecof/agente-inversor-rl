"""Simulador long-only: órdenes al cierre y acciones como pesos tras pagar costos."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class InvestmentEnv(gym.Env):
    metadata = {"render_modes": []}
    weights = (0.0, 0.5, 1.0)

    def __init__(self, frame, x, start, end, initial_capital=100000.,
                 cost_rate=0.001, reward_mode="log", risk_lambda=10.):
        super().__init__()
        if not 0 <= start < end < len(frame):
            raise ValueError("Intervalo inválido")
        if not 0 <= cost_rate < 1 or initial_capital <= 0:
            raise ValueError("Capital o costo inválido")
        if reward_mode not in ("log", "risk"):
            raise ValueError("Recompensa desconocida")
        self.frame, self.x = frame, np.asarray(x, dtype=np.float32)
        self.prices = frame.close.to_numpy(dtype=float)
        self.start, self.end = int(start), int(end)
        self.initial_capital, self.cost_rate = float(initial_capital), float(cost_rate)
        self.reward_mode, self.risk_lambda = reward_mode, risk_lambda
        self.action_space = spaces.Discrete(3)
        # Cinco indicadores, peso actual y drawdown: todo observable hasta t.
        self.observation_space = spaces.Box(
            low=np.array([-10.]*5+[0., -1.], dtype=np.float32),
            high=np.array([10.]*5+[1., 0.], dtype=np.float32))

    def _obs(self):
        return np.concatenate([self.x[self.t], np.array([
            self.risky/self.equity, self.equity/self.peak-1], dtype=np.float32)]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.start
        self.equity = self.peak = self.initial_capital
        self.risky = self.cost_total = 0.
        self.done = False
        return self._obs(), {"equity": self.equity}

    def step(self, action):
        if self.done:
            raise RuntimeError("El episodio terminó; llama reset().")
        if not self.action_space.contains(action):
            raise ValueError("Acción fuera de {0,1,2}")
        t, before, old_risky = self.t, self.equity, self.risky
        w, c = self.weights[int(action)], self.cost_rate
        # Resolver X = w*(V-c*abs(X-B)) evita financiar comisiones con deuda.
        if w * before >= old_risky:
            target = w * (before+c*old_risky)/(1+w*c)
        else:
            target = w * (before-c*old_risky)/(1-w*c)
        traded = abs(target-old_risky)
        cost = c * traded
        cash = before-cost-target
        market_return = self.prices[t+1]/self.prices[t]-1
        self.risky = target * (1+market_return)
        self.equity = cash+self.risky
        net_return = self.equity/before-1
        log_return = np.log(self.equity/before)
        penalty = self.risk_lambda * min(net_return, 0.)**2 if self.reward_mode == "risk" else 0.
        # Escala fija para facilitar la optimización; no altera el orden de recompensas.
        reward = 100. * (log_return-penalty)
        self.cost_total += cost
        self.peak = max(self.peak, self.equity)
        self.t += 1
        self.done = self.t >= self.end
        info = {"timestamp": str(self.frame.timestamp.iloc[self.t]),
                "decision_timestamp": str(self.frame.timestamp.iloc[t]),
                "equity": self.equity, "action": int(action), "target_weight": w,
                "actual_weight": self.risky/self.equity, "net_return": net_return,
                "market_return": market_return, "cost": cost, "cost_total": self.cost_total,
                "turnover": traded/before, "drawdown": self.equity/self.peak-1,
                "reward": float(reward), "risk_penalty": penalty}
        return self._obs(), float(reward), self.done, False, info
