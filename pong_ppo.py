from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO

import ale_py
import gymnasium as gym
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw
import numpy as np

gym.register_envs(ale_py)  # unnecessary but helpful for IDEs

""" PPO() defaults
 learning_rate=0.0003
 n_steps=2048
 batch_size=64
 n_epochs=10
 gamma=0.99
 gae_lambda=0.95
 clip_range=0.2
 clip_range_vf=None
 normalize_advantage=True
 ent_coef=0.0
 vf_coef=0.5
 max_grad_norm=0.5
 use_sde=False
 sde_sample_freq=-1
 rollout_buffer_class=None
 rollout_buffer_kwargs=None 
 target_kl=None
 stats_window_size=100
 tensorboard_log=None
 policy_kwargs=None
 verbose=0
 seed=None
 device='auto'
 _init_setup_model=True)
 """

def _add_text_info(frame, episode_num, step, reward,env_id):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/18,im.size[1]/16), 
                f"{env_id}\nEpisode: {episode_num+1}\nStep: {step}\nCurrent reward: {reward}", 
                fill=text_color)
    return im

def saveGif(filename="pong_ppo.gif",n_ep=5,trained_model="ppo_pong"):
    env_id = "PongNoFrameskip-v4"

    # vec_env = make_vec_env(env_id=env_id,n_envs=1)
    vec_env = make_atari_env(env_id, n_envs=1, seed=0)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    model = PPO.load(trained_model,env=vec_env)

    images = []
    for i in range(n_ep):
        R = 0
        t = 0
        obs = model.env.reset()
        terminated = False
        while not terminated:
            action, _ = model.predict(obs,None, None, True)
            obs, reward, terminated ,_ = model.env.step(action)
            R += reward
            t += 1
            img = model.env.render(mode="rgb_array")
            images.append(_add_text_info(img, episode_num=i, step=t, reward=R,env_id=env_id))

    imageio.mimsave(filename, [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=30)

def PongTrainer(save_model=False):
    # atari:
    # env_wrapper:
    # - stable_baselines3.common.atari_wrappers.AtariWrapper
    # frame_stack: 4
    # policy: 'CnnPolicy'
    # n_envs: 8
    # n_steps: 128
    # n_epochs: 4
    # batch_size: 256
    # n_timesteps: !!float 1e7
    # learning_rate: lin_2.5e-4
    # clip_range: lin_0.1
    # vf_coef: 0.5
    # ent_coef: 0.01

    ## Hyperparameters
    policy = "CnnPolicy"
    n_env = 8  # 16
    n_steps = 128
    batch_size = 256
    n_epochs = 4
    # gamma = 0.99     # default
    # gae_gamma = 0.95 # default
    # ent_coef = 0.0   # default 
    n_timesteps = int(10e6) 
    learning_rate = 5e-4
    clip_range = 0.1
    vf_coef = 0.5
    ent_coef = 0.01

    verbose = 0

    # environment generator that will make and wrap atari environments correctly.
    env = make_atari_env("PongNoFrameskip-v4", n_envs=n_env, seed=0)
    # Stack 4 frames
    env = VecFrameStack(env, n_stack=4)

    model = PPO(policy=policy,env=env,verbose=verbose,n_steps=n_steps,batch_size=batch_size,
                n_epochs=n_epochs,learning_rate=learning_rate,clip_range=clip_range,
                vf_coef=vf_coef,ent_coef=ent_coef)
    model.learn(total_timesteps=n_timesteps,progress_bar=True)

    if save_model:
        model.save("ppo_pong")


if __name__=="__main__":
    PongTrainer(save_model=True)
    # saveGif()

