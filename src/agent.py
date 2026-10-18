"""Q-learning tabular y DQN de Stable-Baselines3."""
import json
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor


class TabularQ:
    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)
        # Momentum de 5 y 20 sesiones, volatilidad, posición y drawdown.
        self.q = np.zeros((3, 3, 3, 3, 3, 3), dtype=float)

    def state(self, obs):
        return (int(np.digitize(obs[1], [-.5, .5])),
                int(np.digitize(obs[2], [-.5, .5])),
                int(np.digitize(obs[4], [-.5, .5])),
                int(np.digitize(obs[5], [.25, .75])),
                int(np.digitize(obs[6], [-.15, -.05])))

    def act(self, obs):
        values = self.q[self.state(obs)]
        return int(np.argmax(values))

    def fit(self, env, episodes, seed):
        history = []
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed+episode)
            eps = 1.-.95*min(episode/max(1, int(.8*episodes)), 1.)
            total = 0.
            while True:
                s = self.state(obs)
                action = int(self.rng.integers(3)) if self.rng.random() < eps else self.act(obs)
                nxt, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                target = reward + (0. if done else .99 * np.max(self.q[self.state(nxt)]))
                self.q[s+(action,)] += .1*(target-self.q[s+(action,)])
                total += reward
                obs = nxt
                if done:
                    break
            history.append({"episode": episode+1, "reward": total, "epsilon": eps})
        return history


def train_agent(kind, env, config, seed, prefix):
    if kind == "qlearning":
        agent = TabularQ(seed)
        history = agent.fit(env, config["q_episodes"], seed)
        np.save(str(prefix)+".npy", agent.q)
        return agent, history
    model = DQN("MlpPolicy", Monitor(env, filename=str(prefix)+"_monitor.csv"),
                seed=seed, device="cpu", verbose=0, **config["dqn"])
    model.learn(total_timesteps=config["timesteps"])
    model.save(str(prefix))
    return model, []


def action_for(agent, obs):
    if isinstance(agent, TabularQ):
        return agent.act(obs)
    action, _ = agent.predict(obs, deterministic=True)
    return int(action)
