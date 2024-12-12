import numpy as np
import os
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
import imageio
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)  # unnecessary but helpful for IDEs

os.chdir(os.path.abspath(os.path.dirname(__file__)))

from huggingface_sb3 import load_from_hub
checkpoint = load_from_hub(
	repo_id="ThomasSimonini/ppo-PongNoFrameskip-v4",
	filename="ppo-PongNoFrameskip-v4.zip",
)
env_id = "PongNoFrameskip-v4"

# vec_env = make_vec_env(env_id=env_id,n_envs=1)
vec_env = make_atari_env(env_id, n_envs=1, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)
model = PPO.load(checkpoint,env=vec_env)

images = []
obs = vec_env.reset()
terminated = False
while not terminated:
    action, _ = model.predict(obs,None, None, True)
    obs, reward, terminated ,_ = vec_env.step(action)
    img = vec_env.render(mode="rgb_array")
    images.append(img)

imageio.mimsave("pong_ppo.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=30)
