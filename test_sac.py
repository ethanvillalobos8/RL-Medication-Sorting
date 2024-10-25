import pybullet as p
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from panda_env import PandaEnvSAC


def test_model():
    test_env = PandaEnvSAC(render_mode=p.GUI)
    test_env = DummyVecEnv([lambda: test_env])
    test_env = VecNormalize.load("vec_normalize.pkl", test_env)

    test_env.training = False  # Don't update stats at test time
    test_env.norm_reward = False  # Don't normalize rewards during evaluation

    model = SAC.load("sac_panda_best_model/best_model", env=test_env)

    obs = test_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        test_env.render()
        if done:
            obs = test_env.reset()

    test_env.close()


if __name__ == "__main__":
    test_model()