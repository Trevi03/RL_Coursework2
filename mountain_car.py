import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import EvalCallback


import os
update_dir = 1   # 0 to disable
if update_dir:
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    

## Train model
def MountainCarTrainer(save_model=False,get_LC=False):
    # MountainCar-v0:
    # normalize: true
    # n_envs: 16
    # n_timesteps: !!float 1e6
    # policy: 'MlpPolicy'
    # n_steps: 16
    # gae_lambda: 0.98
    # gamma: 0.99
    # n_epochs: 4
    # ent_coef: 0.0

    ## Hyperparameters
    policy = "MlpPolicy"
    n_env = 16  # 16
    n_steps = 16
    batch_size = 128
    n_epochs = 10
    gamma = 0.99
    gae_lambda = 0.98
    ent_coef = 0.0
    n_timesteps = int(3e6) #1e6
    normalize = True

    verbose = 0

    if get_LC:
        # Create folder dir
        log_dir = "MountainCar/lc_logs/"
        os.makedirs(log_dir, exist_ok=True)

        env = make_vec_env("MountainCar-v0", n_envs=n_env, monitor_dir=log_dir) # Parallel environments

        # Initialise the PPO model
        model = PPO(policy=policy,env=env,verbose=verbose,n_steps=n_steps,normalize_advantage=normalize,
                    n_epochs=n_epochs,gamma=gamma,gae_lambda=gae_lambda,ent_coef=ent_coef,batch_size=batch_size)
        
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

        # Train the model
        model.learn(total_timesteps=n_timesteps,progress_bar=True, callback=callback)

    else:
        env = make_vec_env("MountainCar-v0", n_envs=n_env) # Parallel environments

        # Initialise the PPO model
        model = PPO(policy=policy,env=env,verbose=verbose,n_steps=n_steps,normalize_advantage=normalize,
                    n_epochs=n_epochs,gamma=gamma,gae_lambda=gae_lambda,ent_coef=ent_coef)

        # Train the model
        model.learn(total_timesteps=n_timesteps,progress_bar=True)

    # Save trained model as a zip file
    if save_model:
        model.save("ppo_mountcar")

## Plotting learning curves
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

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

## Saving episode animations
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

def saveGif(filename="mountcar_ppo.gif",n_ep=5,trained_model="ppo_mountcar"):
    env_id = "MountainCar-v0"

    # vec_env = make_vec_env(env_id=env_id,n_envs=1)
    vec_env = gym.make(env_id, render_mode="rgb_array")
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



if __name__=="__main__":
    # MountainCarTrainer(get_LC=True,save_model=True)
    # plot_results("MountainCar/lc_logs/","MountainCar-v0 Learning Curve")
    saveGif(filename="mountcar_ppo.gif",trained_model="ppo_mountcar",n_ep=5)