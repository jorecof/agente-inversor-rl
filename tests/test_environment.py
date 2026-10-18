import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env
from src.environment import InvestmentEnv
from src.data import features, prepare, FEATURES
from src.evaluate import metrics, rollout


def make_env(prices=(100, 110, 99), cost=.001, reward="log"):
    frame = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=len(prices), tz="UTC"), "close": prices})
    return InvestmentEnv(frame, np.zeros((len(prices), 5)), 0, len(prices)-1,
                         initial_capital=100000, cost_rate=cost, reward_mode=reward)


class EnvironmentTests(unittest.TestCase):
    def test_gym_contract(self):
        check_env(make_env(), warn=True)

    def test_cash_has_no_cost_or_return(self):
        env = make_env()
        obs, _ = env.reset(seed=7)
        _, r, _, _, info = env.step(0)
        self.assertEqual(r, 0)
        self.assertEqual(info["equity"], 100000)
        self.assertEqual(info["cost"], 0)

    def test_full_investment_cannot_borrow_commission(self):
        env = make_env()
        env.reset()
        _, _, _, _, info = env.step(2)
        invested = 100000/1.001
        self.assertAlmostEqual(info["cost"], invested*.001, places=7)
        self.assertAlmostEqual(info["equity"], invested*1.1, places=7)
        self.assertAlmostEqual(info["actual_weight"], 1.)

    def test_buy_hold_pays_entry_only(self):
        env = make_env()
        curve = rollout(env, lambda obs: 2, 7)
        self.assertAlmostEqual(curve.cost.iloc[-1], 0.)
        self.assertAlmostEqual(curve.equity.iloc[-1], 100000/1.001*.99)

    def test_sell_charges_existing_risky_value(self):
        env = make_env()
        env.reset()
        env.step(2)
        old = env.risky
        _, _, _, _, info = env.step(0)
        self.assertAlmostEqual(info["cost"], old*.001)
        self.assertAlmostEqual(info["equity"], old*.999)
        self.assertEqual(info["actual_weight"], 0)

    def test_repeating_half_weight_can_require_trade(self):
        env = make_env()
        env.reset()
        env.step(1)
        self.assertGreater(env.risky/env.equity, .5)
        _, _, _, _, info = env.step(1)
        self.assertGreater(info["cost"], 0)

    def test_log_rewards_telescope(self):
        env = make_env()
        curve = rollout(env, lambda obs: 2, 7)
        self.assertAlmostEqual(curve.reward.sum()/100, np.log(curve.equity.iloc[-1]/100000))

    def test_risk_changes_reward_but_not_accounting(self):
        a, b = make_env(reward="log"), make_env(reward="risk")
        ca, cb = rollout(a, lambda obs: 2, 7), rollout(b, lambda obs: 2, 7)
        np.testing.assert_allclose(ca.equity, cb.equity)
        self.assertLess(cb.reward.sum(), ca.reward.sum())

    def test_initial_capital_is_part_of_drawdown(self):
        env = make_env((100, 90), cost=0)
        curve = rollout(env, lambda obs: 2, 7)
        self.assertAlmostEqual(metrics(curve, 100000)["max_drawdown"], -.1)

    def test_cannot_step_after_done(self):
        env = make_env((100, 110))
        env.reset()
        env.step(0)
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_future_prices_do_not_change_past_features(self):
        prices = 100*np.exp(np.cumsum(np.random.default_rng(3).normal(0, .01, 200)))
        a = pd.DataFrame({"close": prices})
        b = a.copy()
        b.loc[130:, "close"] *= 10
        np.testing.assert_allclose(features(a)[FEATURES].iloc[:110], features(b)[FEATURES].iloc[:110])

    def test_test_data_do_not_affect_training_scaler(self):
        cfg = {"train_fraction": .6, "validation_fraction": .2}
        df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=200),
                           "close": 100+np.arange(200)+np.sin(np.arange(200))})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"prices.csv"
            df.to_csv(p, index=False)
            _, x1, splits, mu1, sigma1 = prepare(p, cfg)
            df.loc[175:, "close"] *= 10
            df.to_csv(p, index=False)
            _, x2, _, mu2, sigma2 = prepare(p, cfg)
            np.testing.assert_allclose(mu1, mu2)
            np.testing.assert_allclose(sigma1, sigma2)
            np.testing.assert_allclose(x1[:108], x2[:108])
            self.assertLess(splits["train"][1], splits["validation"][0])
            self.assertLess(splits["validation"][1], splits["test"][0])


if __name__ == "__main__":
    unittest.main()
