import numpy as np
import random
from IPython.display import clear_output
import gym
from stable_baselines3.common.env_checker import check_env
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter


# Antonio Ruiz Molero, 2022.
# Most of the code is taken from different guides from stable baselines3, which
# can be found in https://stable-baselines3.readthedocs.io/

# This class allows to save the best model using the training reward.
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True

# Definition of moving average, which is used to smooth the output graph.


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve', save=False, file=""):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    if save:
        plt.savefig(file, dpi=300)
    plt.show()


# Create the env using our custom environment
env = gym.make('Ataxx-v0')
env.reset()
# Use the function check_env to check that the environment is compatible
# with the library.
check_env(env)


# Create log dir
log_dir = "tmp/ppo5x5/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
# Logs will be saved in log_dir/monitor.csv
env = Monitor(env, log_dir)
env.reset()

np.random.seed(3)
# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
# Create RL model
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
# If we want to load the agent, use:
# model.load("tmp/ppo4x4/ataxx_ppo")
model.learn(total_timesteps=int(10e5), callback=callback, log_interval=4)
model.save("tmp/ppo5x5/ataxx_ppo")


# Plot the graphics.
results_plotter.plot_results(
    [log_dir], 1e5, results_plotter.X_TIMESTEPS, "ppo ataxx")

plot_results(log_dir, save=True, file="ppo1.pdf")
