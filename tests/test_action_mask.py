import pytest
import numpy as np

from stable_baselines import A2C, PPO1, PPO2, TRPO, HER, GAIL, SAC, ACER, ACKTR
from stable_baselines.common.action_mask_env import DummyActionMaskEnv
from stable_baselines.common.vec_env import DummyVecEnv

MODEL_LIST = [
    A2C,
    ACER,
    ACKTR,
    PPO2
]


@pytest.mark.slow
@pytest.mark.parametrize("model_class", MODEL_LIST)
def test_action_mask(model_class):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    with a multidiscrete action space

    :param model_class: (BaseRLModel) A RL Model
    """
    env = DummyVecEnv([lambda: DummyActionMaskEnv()] * 4)

    model = model_class("MlpPolicy", env)

    model.learn(total_timesteps=1000, seed=0)

