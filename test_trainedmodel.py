import numpy as np
import os
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.results_plotter import load_results, ts2xy

import imageio
import gymnasium as gym
import matplotlib.pyplot as plt
import ale_py

gym.register_envs(ale_py)  # unnecessary but helpful for IDEs

os.chdir(os.path.abspath(os.path.dirname(__file__)))

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    min_x, max_x = min(x), max(x)

    fig = plt.figure(title)
    plt.scatter(x, y, s=1,label="Actual Rewards")

    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    plt.plot(x, y, color="red",label="Average Rewards")
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.legend(loc="best")
    plt.xlim(min_x, max_x)
    plt.show()


# from huggingface_sb3 import load_from_hub
# checkpoint = load_from_hub(
# 	repo_id="sb3/ppo-PongNoFrameskip-v4",
# 	filename="ppo-PongNoFrameskip-v4.zip",
# )

env_id = "PongNoFrameskip-v4"

# vec_env = make_vec_env(env_id=env_id,n_envs=1)
vec_env = make_atari_env(env_id, n_envs=1, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)
model = PPO.load("PongNoFrameskip-v4_1/PongNoFrameskip-v4.zip",env=vec_env)

images = []
obs = vec_env.reset()
terminated = False
while not terminated:
    action, _ = model.predict(obs,None, None, True)
    obs, reward, terminated ,_ = vec_env.step(action)
    img = vec_env.render(mode="rgb_array")
    images.append(img)

imageio.mimsave("pong_ppo.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=30)

plot_results("PongNoFrameskip-v4_1/","PongNoFrameskip-v4 LC")