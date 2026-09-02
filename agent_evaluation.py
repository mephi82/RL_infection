from tf_agents.policies import random_py_policy
import numpy as np
import multiprocessing
from tf_agents.trajectories import policy_step

class ConstantPolicy(random_py_policy.RandomPyPolicy):
    def __init__(self, time_step_spec, action_spec, constant_action, policy_state_spec=()):
        self._time_step_spec = time_step_spec
        self._action_spec = action_spec
        self._constant_action = np.array(constant_action)#, dtype = np.int32)
        
    def action(self, time_step, policy_state=()):
        return policy_step.PolicyStep(action=self._constant_action)
        # return policy_step.PolicyStep(action=(np.random.sample(len(self._constant_action))<self._constant_action).astype(int))

class ConstantProbPolicy(random_py_policy.RandomPyPolicy):
    def __init__(self, time_step_spec, action_spec, constant_prob, policy_state_spec=()):
        self._time_step_spec = time_step_spec
        self._action_spec = action_spec
        self._constant_prob = np.array(constant_prob)#, dtype = np.int32)
        
    def action(self, time_step, policy_state=()):
        # return policy_step.PolicyStep(action=self._constant_action)
        return policy_step.PolicyStep(action=(np.random.sample(len(self._constant_prob))<self._constant_prob).astype(int))


def run_episode(seed, environment, policy):
  np.random.seed(seed)
  time_step = environment.reset()
  episode_return = 0.0

  I = 1
  while not time_step.is_last():
      # print(environment.t)      
    action_step = policy.action(time_step)
    time_step = environment.step(action_step.action)
    episode_return += I*time_step.reward
    I *= time_step.discount
  return(episode_return)

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0

  for e in range(num_episodes):
    
    total_return += run_episode(e, environment, policy)

  avg_return = total_return / num_episodes
  return avg_return#.numpy()[0]
    
  # for _ in range(num_episodes):

  #   time_step = environment.reset()
  #   episode_return = 0.0

  #   while not time_step.is_last():
  #     # print(environment.t)      
  #     action_step = policy.action(time_step)
  #     time_step = environment.step(action_step.action)
  #     episode_return += time_step.reward
  #   total_return += episode_return

  # avg_return = total_return / num_episodes
  # return avg_return#.numpy()[0]    


def compute_avg_return_parallel(environment, policy, num_episodes=10):

  total_return = 0.0

  with multiprocessing.Pool(processes=num_episodes) as pool:

    returns = pool.starmap(run_episode, [(seed, environment, policy) for seed in range(num_episodes)])

  # avg_return = np.array(returns).sum() / num_episodes
  return np.array(returns)#.numpy()[0]    