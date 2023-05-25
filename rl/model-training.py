

from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import torch
#from stable_baselines.gail import generate_expert_traj
#from simulation import *
from simulation_data import *
#from simulation_data_queue import *

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

class MeanRewardCallback(BaseCallback):
	"""
	Callback for logging the mean reward of a SubprocVecEnv during training.
	"""

	def __init__(self, check_freq: int, verbose=0):
		"""
		:param check_freq: The frequency at which to log the mean reward.
		:param verbose: The level of verbosity for logging.
		"""
		super().__init__(verbose=verbose)
		self.check_freq = check_freq
		self.episode_rewards = []

	def _on_step(self) -> bool:
		"""
		This function is called at each step of the training.

		:return: Whether or not to continue training.
		"""
		if self.n_calls % self.check_freq == 0:
			self.logger.record("train/mean_rTime", np.mean(self.training_env.get_attr('responseTime')))
		return True
from torch import nn
import torch as th
from typing import Dict, List, Tuple, Type, Union
from gym import spaces


from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.type_aliases import TensorDict
class CombinedExtractor_own(BaseFeaturesExtractor):
	def __init__(
		self,
		observation_space: spaces.Dict,
		cnn_output_dim: int = 256,
		normalized_image: bool = False,
) -> None:
		# TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
		super().__init__(observation_space, features_dim=1)

		extractors: Dict[str, nn.Module] = {}

		total_concat_size = 0
		for key, subspace in observation_space.spaces.items():
			# The observation key is a vector, flatten it if needed
			#extractors[key] = nn.conv1d(1, 1, 10)(nn.Flatten()) 
			extractors[key] = nn.ReLU()
			total_concat_size += get_flattened_obs_dim(subspace)

		self.extractors = nn.ModuleDict(extractors)
		print(observation_space)
		# Update the features dim manually
		self._features_dim = total_concat_size
		kSize = 9
		self.cnn = nn.Sequential(
			nn.Conv1d(1, 1, kernel_size=kSize, stride=1, dilation=1, padding=4),
			#nn.Conv1d()
			nn.Flatten()
		)


	def forward(self, observations: TensorDict) -> th.Tensor:
		encoded_tensor_list = []

		for key, extractor in self.extractors.items():
			encoded_tensor_list.append(extractor(observations[key]))
		catted = th.cat(encoded_tensor_list, dim=1)
		catted = catted.unsqueeze(1)
		return self.cnn(catted)

def lowest_dist_expert(_obs):
	print(_obs)
	print(np.argmin(_obs))
	return np.argmin(_obs)


def possible_expert(_obs):
	a = np.random.choice(np.where(_obs["ambulance"] > 0)[0])
	i = np.random.choice(np.where(_obs["incident"] > 0)[0])
	print(a,i)
	return np.array([a,i])

modelLogDir = "./logs/"
tbLog = "./tLog/"

checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path=modelLogDir,
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)
callback = MeanRewardCallback(check_freq=1000)

from sb3_contrib import RecurrentPPO

def proposed_model(env):
	#model = MaskablePPO(MaskableMultiInputActorCriticPolicy, env, n_steps=2048, batch_size=128, verbose=1, learning_rate=0.01, device="cuda", tensorboard_log=tbLog, policy_kwargs={"net_arch" : [500,500,500], "activation_fn" : torch.nn.modules.activation.ReLU}) # policy_kwargs={"activation_fn" : torch.nn.modules.activation.ReLU}
	model =  MaskablePPO(MaskableMultiInputActorCriticPolicy, env, verbose=1, learning_rate=0.001, device="cuda", tensorboard_log=tbLog, policy_kwargs={"net_arch" : [100,100, 100], "activation_fn" : torch.nn.modules.activation.ReLU}) # policy_kwargs={"activation_fn" : torch.nn.modules.activation.ReLU}
	return model

def proposed_model_2(env):
	kkwarg = {"num_layers": 3, "n_units_l0": 175, "n_units_l1": 586, "n_units_l2": 246}

	#pKwarg = {"net_arch" : [175,586, 246],}
	pKwarg = {"net_arch" : [329,479],  "activation_fn" : torch.nn.modules.activation.ReLU}
	#model = MaskablePPO(MaskableMultiInputActorCriticPolicy, env, n_steps=2048, batch_size=128, verbose=1, learning_rate=0.01, device="cuda", tensorboard_log=tbLog, policy_kwargs={"net_arch" : [500,500,500], "activation_fn" : torch.nn.modules.activation.ReLU}) # policy_kwargs={"activation_fn" : torch.nn.modules.activation.ReLU}
	#kkwarg = {"n_steps": 1179.3617453020818, "batch_size": 72.37355967360466, "gamma": 0.9694239917150556, "learning_rate": 1.7387766344685753e-05, "ent_coef": 0.008206484040362031, "clip_range": 0.3422863211201763, "n_epochs": 12.106045793550255, "max_grad_norm": 0.3006030699455088, "gae_lambda": 0.9349867958852747, "normalize_advantage": False}
	
	kkwarg = {"n_steps": 11461.352007863052, "batch_size": 61.708757261184715, "gamma": 0.903019711717352, "learning_rate": 1.9772368158750766e-05, "ent_coef": 0.00010733154497861273, "clip_range": 0.3695860019310334, "n_epochs": 11.338206342606721, "max_grad_norm": 0.3544303911415594, "gae_lambda": 0.9416955466680422, "normalize_advantage": True}

	for f in ["n_epochs", "n_steps", "batch_size"]:
		kkwarg[f] = int(kkwarg[f])
	model =  MaskablePPO(MaskableMultiInputActorCriticPolicy, env,verbose=1, device="cuda", tensorboard_log=tbLog, policy_kwargs=pKwarg, **kkwarg) # policy_kwargs={"activation_fn" : torch.nn.modules.activation.ReLU}
	return model

def proposed_model_multi(env):
	#model = MaskablePPO(MaskableMultiInputActorCriticPolicy, env, n_steps=2048,  verbose=1, learning_rate=0.01, device="cuda", tensorboard_log=tbLog, policy_kwargs={"net_arch" : [500,500,500], "activation_fn" : torch.nn.modules.activation.ReLU}) # policy_kwargs={"activation_fn" : torch.nn.modules.activation.ReLU}
	#model =  RecurrentPPO(MaskableMultiInputActorCriticPolicy, env, verbose=1, learning_rate=0.001, device="cuda", tensorboard_log=tbLog, policy_kwargs={"net_arch" : [100,100, 100], "activation_fn" : torch.nn.modules.activation.ReLU}) # policy_kwargs={"activation_fn" : torch.nn.modules.activation.ReLU}
	model = RecurrentPPO("MultiInputLstmPolicy", env,verbose=1, n_steps=2048, batch_size=128, learning_rate=0.001, device="cuda", tensorboard_log=tbLog, policy_kwargs={"net_arch" : [256,256, 256], "activation_fn" : torch.nn.modules.activation.ReLU})
	return model


def proposed_env(training=True):
	env = dispatchEnv(lamb=0.21, LaLo=[59.910986, 10.752496], dist=5, training=False, fromData=False, waitTimes=True, surFunc=True)
	#env = ActionMasker(env, mask_fn)
	return env

def get_model_env(training=True):
	env = proposed_env(training)
	model = proposed_model(env)
	return model, env

def get_model_envs(n_jobs):
	env = SubprocVecEnv([make_env(0, i) for i in range(n_jobs)])
	return env
from stable_baselines3.common.vec_env import SubprocVecEnv



import multiprocessing as mp
def get_model_env_multi(n_jobs=10, rec=False):
	env = get_model_envs(n_jobs)
	#env = Monitor(env)
	#time.sleep(60)
	if not rec:
		#model = proposed_model(env)
		model = proposed_model_2(env)
	else:
		model = proposed_model_multi(env)
	return model, env

#a = simulator()
#a.simulate()

def lr_sched(x):
	return 0.01


def iterative_train(model, x=1000):
	fd = False
	while True:
		model.learn(total_timesteps=x, callback=[checkpoint_callback, callback], tb_log_name="tb", log_interval=1)
		model.env.set_attr("fromData", fd)
		fd = bool(1-fd)
import copy
import pickle # 0.6565853493142557


if __name__ == "__main__":
	TRAINING = True
	model, env = get_model_env_multi(rec=False, n_jobs=10)
	#generate_expert_traj(possible_expert, "lowest_dist", env, n_episodes=1)
	model.set_parameters("./logs/rl_model_740000_steps.zip")
	#iterative_train(model)

	model.learn(total_timesteps=1000000000000000000000000000, callback=[checkpoint_callback, callback], tb_log_name="tb", log_interval=1)

	"""
	eval_env = get_model_envs(1)

	eval_callback = EvalCallback(eval_env, best_model_save_path='./BestLogs/',
								log_path='./logs/', eval_freq=1000,
								deterministic=True, render=False)
	"""
	
	#env = SubprocVecEnv[]
	#policy  = MaskableMultiInputActorCriticPolicy(env.observation_space, env.action_space, lr_schedule=lr_sched, activation_fn=torch.nn.modules.activation.ReLU)

	#eval_env = copy.deepcopy(model.get_env())
	#eval_callback = EvalCallback(eval_env, best_model_save_path='./logsBest/', eval_freq=500, log_path="./logsBestlog/",
	#                         deterministic=True, render=False)
	

	#model = model.load("./logs/rl_model_46000_steps.zip")
	#model.set_env(env)
	#model.set_parameters("./logs/rl_model_1760000_steps.zip")

#model = A2C('MultiInputPolicy', env).learn(total_timesteps=1000000000000000000000000000)


