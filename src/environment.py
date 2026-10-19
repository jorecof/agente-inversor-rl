"""Simulador long-only: órdenes al cierre y acciones como pesos tras pagar costos."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class InvestmentEnv(gym.Env):
    metadata = {"render_modes": []}
    weights = (0.0, 0.5, 1.0)

    def __init__(self, frame, x, start, end, initial_capital=100000.,
                 cost_rate=0.001, reward_mode="log", risk_lambda=10.,
                 turnover_kappa=0.005, episode_length=None):
        super().__init__()
        if not 0 <= start < end < len(frame):
            raise ValueError("Intervalo inválido")
        if not 0 <= cost_rate < 1 or initial_capital <= 0:
            raise ValueError("Capital o costo inválido")
        if reward_mode not in ("log", "risk", "turnover"):
            raise ValueError("Recompensa desconocida")
        if episode_length is not None and not 1 <= episode_length <= end-start:
            raise ValueError("episode_length debe caber dentro del intervalo")
        self.frame, self.x = frame, np.asarray(x, dtype=np.float32)
        self.prices = frame.close.to_numpy(dtype=float)
        self.start, self.end = int(start), int(end)
        self.initial_capital, self.cost_rate = float(initial_capital), float(cost_rate)
        self.reward_mode, self.risk_lambda = reward_mode, risk_lambda
        self.turnover_kappa = float(turnover_kappa)
        # v2: en entrenamiento cada episodio es una ventana aleatoria dentro de
        # [start, end]; en validación/prueba (None) se recorre el tramo completo.
        self.episode_length = episode_length
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
        if self.episode_length is None:
            self.t, self.episode_end = self.start, self.end
        else:
            # Inicio uniforme entre start y end-episode_length (ambos incluidos);
            # nunca sale del intervalo, así que no cruza a otra partición.
            self.t = self.start + int(self.np_random.integers(
                0, self.end-self.start-self.episode_length+1))
            self.episode_end = self.t + self.episode_length
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
        turnover = traded/before
        if self.reward_mode == "risk":
            penalty = self.risk_lambda * min(net_return, 0.)**2
        elif self.reward_mode == "turnover":
            # v2: castiga la fracción del patrimonio negociada, además del costo
            # real que ya está dentro de log_return. Ataca la sobreoperación.
            penalty = self.turnover_kappa * turnover
        else:
            penalty = 0.
        # Escala fija para facilitar la optimización; no altera el orden de recompensas.
        reward = 100. * (log_return-penalty)
        self.cost_total += cost
        self.peak = max(self.peak, self.equity)
        self.t += 1
        self.done = self.t >= self.episode_end
        info = {"timestamp": str(self.frame.timestamp.iloc[self.t]),
                "decision_timestamp": str(self.frame.timestamp.iloc[t]),
                "equity": self.equity, "action": int(action), "target_weight": w,
                "actual_weight": self.risky/self.equity, "net_return": net_return,
                "market_return": market_return, "cost": cost, "cost_total": self.cost_total,
                "turnover": turnover, "drawdown": self.equity/self.peak-1,
                "reward": float(reward), "risk_penalty": penalty}
        return self._obs(), float(reward), self.done, False, info
