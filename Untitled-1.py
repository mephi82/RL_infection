# %%
import pickle
from episim import *
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm
# %%
#     pickle.dump(city_base, f)

def make_env(city_base, env_params, seed=0):
    def _init():
        env = EpiSimEnvironment(city_base, **env_params) #coef_infect는 x^2에 대한 가중 효과만 반영. 기본 x가 infection cost에 들어감
        return(env)
    return _init
    
class RandomPolicy():
    def __init__(self, env, p):
        self.p = p
        self.shape = (env.num_envs, *env.action_space.shape)
        
    def predict(self, obs, deterministic):
        return(np.random.choice(2, size=self.shape, 
                                p=[1-self.p,self.p]), None)

class VecDriver():

    def __init__(self, env_params, num_envs):
        self.num_envs = num_envs
        self.env_params = env_params
        self.vec_env = SubprocVecEnv([make_env(city_base, env_params) for i in range(num_envs)])
        self.reset()

    def reset(self):
        self.episode_rewards = [[] for _ in range(self.num_envs)]
        self.cohorts = [[] for _ in range(self.num_envs)]
        self.links = [[] for _ in range(self.num_envs)]
        self.fermeture = [[] for _ in range(self.num_envs)]

    def archive(self, folder, prefix):
        
        path = f"{folder}/{prefix}.pkl"
        with open(path, 'wb') as f:
            pickle.dump({'rewards': self.episode_rewards,
                         'cohort': self.cohorts,
                         'links': self.links,
                         'fermeture': self.fermeture}, f)
        print(f"Result saved to {path}")

        
    def run(self, model):
        episode_counts = np.zeros(self.num_envs, dtype=int)

        obs = self.vec_env.reset()
        dones = [False] * self.num_envs

        for _ in tqdm(range(self.env_params['max_epis_length'])):
            if np.sum(episode_counts>0) == self.num_envs:
                break
            actions, _ = model.predict(obs, deterministic=False)
            # print(actions.shape)
            obs, rewards, dones, infos = self.vec_env.step(actions)
            
            for i in range(self.num_envs):
                if episode_counts[i] < 1: #각자 한번만 해라. 나머지는 기록 안한다.
                    self.episode_rewards[i].append(rewards[i])
                    self.cohorts[i].append(infos[i]['cohort'])
                    self.links[i].append(infos[i]['link'])
                    self.fermeture[i].append(actions[i])
                    if dones[i]:
                        print(f"Env {i} - Episode {episode_counts[i]} reward: {sum(self.episode_rewards[i])}")
                        episode_counts[i] += 1
    def close(self):
        self.vec_env.close()  
    

 

      
# %%
if __name__ == "__main__":
    import os
    with open('city_jecheon_scale20.pkl', 'rb') as f:
        city_base = pickle.load(f)
    
    num_envs = 32
    
    env_params = {
        'max_epis_length': 52*3,
        'ext_rate': 0.001,
        'mean_recover': 5,
        'risk_coexist': 0.04,
        'n_visit_tries': 2,
        'r0': 5,
        'coef_block': 0.05,
        'coef_infect': 0.1,
        'gamma': 0.999
    }
    
    folder = './results/'
    if env_params is not None:
        param_str = ".".join(f"{k}{v}" for k, v in sorted(env_params.items()))
    else:
        param_str = "default"
    folder = './results/'+param_str
    os.makedirs(folder, exist_ok=True)

    driver = VecDriver(env_params=env_params, num_envs=num_envs)
    model = PPO.load("jc20_resnet18policy_ib-2.0_cb0.05_ci0.1_it1500000_2.zip", env=driver.vec_env, device="cuda")

    driver.run(model)
    driver.archive(folder, 'model')
    driver.reset()

    driver.run(RandomPolicy(driver.vec_env, 0.0))
    driver.archive(folder, 'r0.0')
    driver.reset()
    
    driver.run(RandomPolicy(driver.vec_env, 0.5))
    driver.archive(folder, 'r0.5')
    driver.reset()
    
    driver.run(RandomPolicy(driver.vec_env, 1.0))
    driver.archive(folder, 'r1.0')

    driver.close()