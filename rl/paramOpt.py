
import optuna
import gym
import numpy as np


from model import *

class CombinedExtractor_c(BaseFeaturesExtractor):
	def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256, lstm_hidden_size: int = 128):
		super().__init__(observation_space, features_dim=lstm_hidden_size)

		self.observation_space = observation_space
		cnn_extractors = []
		cnn_output_sizes = []
		for key, subspace in observation_space.spaces.items():
			# The observation key is a vector, flatten it if needed
			subspace_dim = get_flattened_obs_dim(subspace)

			# CNN extractor for the current input space
			cnn_extractor = nn.Sequential(
				nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.Conv1d(in_channels=128, out_channels=cnn_output_dim, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.Flatten()
			)
			cnn_extractors.append(cnn_extractor)
			cnn_output_sizes.append(cnn_output_dim)

		self._features_dim = sum(cnn_output_sizes)

		# LSTM extractor for the concatenated CNN outputs
		self.lstm_extractor = nn.LSTM(input_size=sum(cnn_output_sizes), hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
		self.linear = nn.Linear(cnn_output_dim, lstm_hidden_size)

		self.cnn_extractors = nn.ModuleList(cnn_extractors)
		self.cnn_output_sizes = cnn_output_sizes
		self.concatenated_size = sum(cnn_output_sizes)

	def forward(self, observations):
		cnn_outputs = []
		for i, (key, subspace) in enumerate(self.observation_space.spaces.items()):
			cnn_output = self.cnn_extractors[i](observations[key].unsqueeze(1))
			cnn_outputs.append(cnn_output)

		cnn_output_cat = th.cat(cnn_outputs, dim=-1)
		cnn_output_flat = cnn_output_cat.view(cnn_output_cat.size(0), -1)
		cnn_output_emb = self.linear(cnn_output_flat)
		lstm_input = cnn_output_emb.unsqueeze(0)
		lstm_output, _ = self.lstm_extractor(lstm_input)

		return lstm_output.squeeze(0)

#from stable_baselines3.common.torch_layers import Mlp

class GetLSTMOutput(nn.Module):
    def forward(self, x):
        out, _ = x
        return out
    
class CombinedExtractor_own(BaseFeaturesExtractor):
	def __init__(
		self,
		observation_space: spaces.Dict,
		cnn_output_dim: int = 256,
		normalized_image: bool = False,
) -> None:
		# TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
		super().__init__(observation_space, features_dim=69*3)

		extractors: Dict[str, nn.Module] = {}

		total_concat_size = 0
		for key, subspace in observation_space.spaces.items():
			# The observation key is a vector, flatten it if needed
			#extractors[key] = nn.conv1d(1, 1, 10)(nn.Flatten()) 
			extractors[key] = nn.Sequential(nn.Embedding(69,1), nn.Flatten(), nn.LSTM(69, 69), GetLSTMOutput())
			total_concat_size += 69
			#total_concat_size += get_flattened_obs_dim(subspace)

		self.extractors = nn.ModuleDict(extractors)
		#print(observation_space)
		# Update the features dim manually
		self._features_dim = total_concat_size
		kSize = 9
		self.cnn = nn.Sequential(
			nn.Conv1d(1, 1, kernel_size=kSize, stride=1, dilation=1, padding=4),
			#nn.Conv1d()
			nn.Flatten()
		)

	def convert_sparse(self, tensor):
		dense_array = np.array(tensor)#.numpy()
		nonzero_indices = np.nonzero(dense_array)
		nonzero_values = dense_array[nonzero_indices]

		# create a sparse tensor from the non-zero indices and values
		sparse_tensor = torch.sparse_coo_tensor(torch.LongTensor(nonzero_indices),
												torch.FloatTensor(nonzero_values),
												torch.Size(dense_array.shape))
		# convert the sparse tensor to a LongTensor
		indices = sparse_tensor.to(torch.long)

		return indices.to_dense()
	def convert_to_sparse(self, tensor):
		print(tensor)
		tensor = tensor.flatten()
		indices = tensor.nonzero().t()
		values = tensor[tensor.nonzero()]
		sparse_tensor = torch.sparse_coo_tensor(indices, values, tensor.size())
		return sparse_tensor

	def forward(self, observations: TensorDict) -> th.Tensor:
		encoded_tensor_list = []

		for key, extractor in self.extractors.items():
			data = observations[key]
			data = self.convert_sparse(data)
			encoded_tensor_list.append(extractor(data))
		#print(encoded_tensor_list)
		catted = th.cat(encoded_tensor_list, dim=1)
		#print(catted.shape)
		#catted = catted.unsqueeze(1)
		return catted
		#return self.cnn(catted)
	
		

def optimize_ppo2(trial):
	""" Learning hyperparamters we want to optimise"""
	return {
		'n_steps': int(trial.suggest_loguniform('n_steps', 1028, 58987//5)),
		'batch_size' : int(trial.suggest_loguniform('batch_size', 32, 128)),
		'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
		'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
		'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
		'clip_range': trial.suggest_uniform('cliprange', 0.1, 0.4),
		'n_epochs' : int(trial.suggest_loguniform("n_epochs", 5, 15)),
		'max_grad_norm' : trial.suggest_uniform('max_grad_norm', 0.2, 0.7),
		'gae_lambda' : trial.suggest_uniform('gae_lambda', 0.9, 0.95),
		'normalize_advantage' : trial.suggest_categorical("normalize_advantage", [True, False]),
	}
def create_model(trial):
	num_layers = trial.suggest_int("num_layers", 1, 5)
	#use_fx = trial.suggest_categorical("use_fx", [True, False])
	use_fx = False
	arch = []
	for i in range(num_layers):
		num_hidden = trial.suggest_int("n_units_l{}".format(i), 100, 1000, log=True)
		arch.append(num_hidden)
	if not use_fx:
		return {"net_arch" : arch, "activation_fn" : torch.nn.modules.activation.ReLU}
	else:
		f_ex = CombinedExtractor_own

		return {"net_arch" : arch, "activation_fn" : torch.nn.modules.activation.ReLU, "features_extractor_class" : f_ex}


class surVCallback(BaseCallback):
	"""
	Callback for logging the mean reward of a SubprocVecEnv during training.
	"""

	def __init__(self, check_freq, trial, verbose=0):
		"""
		:param check_freq: The frequency at which to log the mean reward.
		:param verbose: The level of verbosity for logging.
		"""
		super().__init__(verbose=verbose)
		self.check_freq = check_freq
		self.trial = trial
		self.episode_rewards = []

	def _on_step(self) -> bool:
		"""
		This function is called at each step of the training.

		:return: Whether or not to continue training.
		"""
		if self.n_calls % self.check_freq == 0:
			v = np.mean(self.training_env.get_attr('surVals'))
			print(self.n_calls, v)
			self.trial.report(self.n_calls, v)
			if self.trial.should_prune():
				raise optuna.TrialPruned()

			#self.logger.record("train/mean_rTime", )
		return True

def eval_agent(model, iter, n=1):
	global testEnv
	rewards = []
	for z in range(n):
		z = iter
		testEnv.dataReader.iter = np.random.randint(0,len(testEnv.dataReader.train.index)-500)
		state = testEnv.reset()
	
		while z > 0:
			acM = np.concatenate([state["ambulance"] > 0, state["incident"] > 0], axis=0)
			pred = model.predict(state, deterministic=True, action_masks=acM)[0]

			state, reward, term = testEnv.step(pred)[:3]
			if term:
				testEnv.reset()
			rewards.append(reward)
			z -= 1
	print(len(rewards))
	return np.mean(rewards)
import json
def save_best_params(study, trial):
	if study.best_trial == trial:
		with open('best_params.json', 'w') as f:
			json.dump(trial.params, f)
		
def optimize_agent(trial):
	""" Train the model and optimize
		Optuna maximises the negative log likelihood, so we
		need to negate the reward here
	"""
	global env
	model_params = optimize_ppo2(trial)
	model_params["device"] = "cuda:0"
	print(model_params)
	pKwarg = create_model(trial)
	callback = surVCallback(1000, trial)

	model = MaskablePPO(MaskableMultiInputActorCriticPolicy, env, policy_kwargs=pKwarg, verbose=0, **model_params)

	model.learn(5000, callback=callback)

	print("test")
	rew = eval_agent(model,100, n=100)
	print(rew)
	return rew

import warnings

if __name__ == '__main__':
	warnings.filterwarnings("ignore", category=DeprecationWarning)
	warnings.filterwarnings("ignore", category=FutureWarning) 

	testEnv = dispatchEnv(lamb=0.21, LaLo=[59.910986, 10.752496], dist=5, fromData=True, waitTimes=True, surFunc=True)
	testEnv = ActionMasker(testEnv, mask_fn)

	model, env = get_model_env_multi(rec=False, n_jobs=10)
	print("envs done")
	study = optuna.create_study(direction="maximize")
	try:
		study.optimize(optimize_agent, n_trials=10000, n_jobs=1, callbacks=[save_best_params])
	except KeyboardInterrupt:
		print('Interrupted by keyboard.')