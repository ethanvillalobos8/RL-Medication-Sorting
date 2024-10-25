from stable_baselines3 import SAC
from panda_env import PandaEnvSAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import pybullet as p
import os

def main():
    os.makedirs("./sac_panda_tensorboard/", exist_ok=True)
    os.makedirs("./sac_panda_checkpoints/", exist_ok=True)
    os.makedirs("./sac_panda_best_model/", exist_ok=True)
    os.makedirs("./sac_panda_eval_logs/", exist_ok=True)

    train_env = PandaEnvSAC(render_mode=p.DIRECT)

    train_env = DummyVecEnv([lambda: train_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.)

    hyperparameters = {
        "learning_rate": 4e-4,        # SAC's default learning rate
        "buffer_size": int(1e6),      # Large replay buffer size
        "batch_size": 32,            # Batch size for training
        "tau": 0.005,                 # Soft update coefficient for target networks
        "gamma": 0.99,                # Discount factor for rewards
        "total_timesteps": 2000000,  # Total number of timesteps to train the agent
        "verbose": 1                  # Verbosity level (0: no output, 1: info)
    }

    policy_kwargs = dict(net_arch=[256, 256])

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=hyperparameters['learning_rate'],
        buffer_size=hyperparameters['buffer_size'],
        batch_size=hyperparameters['batch_size'],
        tau=hyperparameters['tau'],
        gamma=hyperparameters['gamma'],
        policy_kwargs=policy_kwargs,
        verbose=hyperparameters['verbose'],
        tensorboard_log="./sac_panda_tensorboard/"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./sac_panda_checkpoints/",
        name_prefix="sac_panda_model"
    )

    eval_callback = EvalCallback(
        train_env,
        best_model_save_path="./sac_panda_best_model/",
        log_path="./sac_panda_eval_logs/",
        eval_freq=500,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    callbacks = [checkpoint_callback, eval_callback]

    # Train the agent
    model.learn(
        total_timesteps=hyperparameters['total_timesteps'],
        callback=callbacks,
        log_interval=10
    )

    # Save the trained model and normalization stats
    model.save("sac_panda_model")
    train_env.save("vec_normalize.pkl")

    train_env.close()

if __name__ == "__main__":
    main()