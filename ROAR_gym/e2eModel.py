import warnings
import logging
from typing import Optional, Dict

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("numpy").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
import os
import sys
from pathlib import Path

sys.path.append(Path(os.getcwd()).parent.as_posix())
import gym
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.agent_module.agent import Agent
from ROAR.agent_module.rl_e2e_ppo_agent import RLe2ePPOAgent##
from ROAR.agent_module.forward_only_agent import ForwardOnlyAgent   ##testing stuff
import torch as th
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.ppo.policies import CnnPolicy

from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps, CallbackList
from ppo_util import find_latest_model, CustomMaxPoolCNN,CustomMaxPoolCNN_no_map, CustomMaxPoolCNN_combine, Atari_PPO_Adapted_CNN

try:
    from ROAR_Gym.envs.roar_env import LoggingCallback
except:
    from ROAR_Gym.ROAR_Gym.envs.roar_env import LoggingCallback
from datetime import datetime

CUDA_VISIBLE_DEVICES=1

def main(pass_num):
    # Set gym-carla environment
    agent_config = AgentConfig.parse_file(Path("configurations/agent_configuration.json"))
    carla_config = CarlaConfig.parse_file(Path("configurations/carla_configuration.json"))

    params = {
        "agent_config": agent_config,
        "carla_config": carla_config,
        "ego_agent_class": RLe2ePPOAgent
    }

    model_dir_path = Path("./output/PPOe2e")

    env = gym.make('roar-e2e-ppo-v0', params=params)

    if env.mode=='no_map':
        policy_kwargs = dict(
            features_extractor_class=CustomMaxPoolCNN_no_map,
            features_extractor_kwargs=dict(features_dim=256)
        )
    elif env.mode=='combine':
        policy_kwargs = dict(
            features_extractor_class=CustomMaxPoolCNN_combine,
            features_extractor_kwargs=dict(features_dim=256)
        )
    elif env.mode=='baseline':
        policy_kwargs = dict(
            features_extractor_class=Atari_PPO_Adapted_CNN,
            features_extractor_kwargs=dict(features_dim=256)
        )
    else:
        policy_kwargs = dict(
            features_extractor_class=CustomMaxPoolCNN,
            features_extractor_kwargs=dict(features_dim=256)
        )

    run_fps=8

    training_kwargs = dict(
        learning_rate=0.00001,# be smaller 2.5e-4
        batch_size=64, # mini_batch_size = 256?
        gamma=0.99, # rec range .9 - .99
        ent_coef=.00,# rec range .0 - .01
        seed=1,
        device=th.device('cuda:0' if th.cuda.is_available() else 'cpu'),
        verbose=1,
        tensorboard_log=(Path(model_dir_path) / "tensorboard").as_posix(),
        # use_sde=True,
        # sde_sample_freq=5,
        n_steps=60*run_fps
    )


    latest_model_path = find_latest_model(model_dir_path)
    if latest_model_path is None:
        model = PPO(CnnPolicy, env=env, policy_kwargs=policy_kwargs, **training_kwargs)
    else:
        model = PPO.load(latest_model_path, env=env, policy_kwargs=policy_kwargs, **training_kwargs)
    print("Model Loaded Successfully")
    logging_callback = LoggingCallback(model=model)
    checkpoint_callback = CheckpointCallback(save_freq=3000*run_fps, verbose=2, save_path=(model_dir_path/"logs").as_posix())
    event_callback = EveryNTimesteps(n_steps=600*run_fps, callback=checkpoint_callback)
    callbacks = CallbackList([checkpoint_callback, event_callback, logging_callback])
    model = model.learn(total_timesteps=int(1e10), callback=callbacks, reset_num_timesteps=False)
    model.save(model_dir_path / f"roar_e2e_model_{pass_num}")
    print("Successful Save!")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt="%H:%M:%S", level=logging.INFO)
    logging.getLogger("Controller").setLevel(logging.ERROR)
    logging.getLogger("SimplePathFollowingLocalPlanner").setLevel(logging.ERROR)
    i=0
    while True:
        main(i)
        i += 1





#NOTE:
# need to add the following line:
# self._last_obs = np.nan_to_num(self._last_obs)
#
# to the following file:
#ROAR\venv\Lib\site-packages\stable_baselines3\common\on_policy_algorithm.py
#
#add this line after line 167 such that:
# with th.no_grad():
#     # Convert to pytorch tensor or to TensorDict
#     self._last_obs = np.nan_to_num(self._last_obs)
#     obs_tensor = obs_as_tensor(self._last_obs, self.device)
#     actions, values, log_probs = self.policy.forward(obs_tensor)


#change for on_policy_algorithm.py
# in function collect_rollouts
# add self.env.reset() before while loop while n_steps < n_rollout_steps: