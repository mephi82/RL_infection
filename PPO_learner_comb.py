# %%
from episim_comb import *
from featureextractors import *
import pickle
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.utils import get_schedule_fn, LinearSchedule

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from tqdm import tqdm
# %%
# hh0 = init_households(10000, 1000, 3, 1)
# fac0 = init_facilities(20000, pd.read_csv('nodes-simul.csv'), 50, hh0)
# ind0 = init_individuals(30000, hh0)
# city_base = City(2, hh0, fac0, ind0)

# with open('base_city.pkl', 'wb') as f:
#     pickle.dump(city_base, f)

# %%

# with open('city_jecheon_scale20.pkl', 'rb') as f:
#     city_jc = pickle.load(f)

class RewardHistAdvantageLogger(BaseCallback):
    def __init__(self, log_interval=1000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.buffer = []

    def _on_step(self) -> bool:
        # self.locals["rewards"] = 현재 rollout에서 받은 reward 배열
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self.buffer.extend(rewards.flatten().tolist())

        # 일정 step마다 기록
        if self.n_calls % self.log_interval == 0 and len(self.buffer) > 0:
            rewards_arr = np.array(self.buffer)
            self.logger.record("reward/mean", float(rewards_arr.mean()))
            self.logger.record("reward/std", float(rewards_arr.std()))
            # 히스토그램 기록 (TensorBoard에서 확인 가능)
            # self.logger.record("reward/hist", rewards_arr)
            self.buffer.clear()
        return True


class AdvantageLogger(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_rollout_end(self) -> None:
        buf = self.model.rollout_buffer
        self.logger.record("advantages/min", float(buf.advantages.min()))
        self.logger.record("advantages/max", float(buf.advantages.max()))
            # buf.advantages = buf.advantages / (1.0 + abs(buf.advantages)/10)

    def _on_step(self) -> bool:    
        return True
            

class EntCoefScheduler(BaseCallback):
    """
    ent_coef를 선형 스케줄로 줄이는 콜백.
    progress_remaining: 1.0 -> 0.0 (학습이 진행될수록 감소)
    """
    def __init__(self, start=0.05, end=0.0, end_fraction=0.5, verbose=0):
        super().__init__(verbose)
        self.schedule = LinearSchedule(start, end, end_fraction)

    def _on_rollout_start(self) -> None:
        # 현재 학습 진행도 (1 → 0)
        progress = float(self.model._current_progress_remaining)
        new_ent = float(self.schedule(progress))
        self.model.ent_coef = new_ent
        if self.verbose > 0:
            print(f"[EntCoefScheduler] progress={progress:.3f}, ent_coef={new_ent:.5f}")

    def _on_step(self) -> bool:    
        return True
# %%

def make_env(city_base, env_params, seed=1234):
    def _init():
        print(seed)
        env = Monitor(EpiSimEnvironment(city_base, **env_params, 
                                        seed = seed)) #coef_infect는 x^2에 대한 가중 효과만 반영. 기본 x가 infection cost에 들어감
        # env = ActionMasker(env, env.action_masks)
        return(env)
    return _init

if __name__ == "__main__":
    # with open('city_random.pkl', 'rb') as f:
    #     city = pickle.load(f)
    # prefix = 'rdcity'
    #%%
    # city_name = 'city_local_only'
    city_name = 'city_small'
    with open(city_name+'.pkl', 'rb') as f:
        city = pickle.load(f)
    
    # %%
    ## for jc
    # env = EpiSimEnvironment(city_base, max_epis_length=52*7, ext_rate = 0.001, mean_recover=5, 
    #                         risk_coexist = 0.04, n_visit_tries = 2, r0=5,
    #                         coef_block=0.06, 
    #                         coef_infect=0.1, gamma=0.99) #coef_infect는 x^2에 대한 가중 효과만 반영. 기본 x가 infection cost에 들어감

    # for random city
    # env = EpiSimEnvironment(city, max_epis_length=52*7, ext_rate=1/1000/7, mean_recover=5,
    #                         risk_coexist = 0.04, n_visit_tries = 1, r0=10,
    #                         coef_block=0.07, coef_infect=0.1, gamma=0.999)
    
    env_params = {
        'max_epis_length': 180,
        'ext_rate': 1/1000/7,
        # 'ext_rate': 1/1000,
        'mean_recover': 5,
        'risk_coexist': 0.05,
        # 'risk_coexist': 0.025,
        'n_visit_tries': 2,
        'r0': 100,
        'coef_block': 0.06,
        'coef_infect': 0.1,
        'gamma': 1.0,
        'mask_choice': [],
        'ngrid': 50,
        'lag': 0,
    }            
    n_ts = 10_000_000
    n_envs = 32
    base_seed = 123456
    prefix = city_name #+'_'+','.join(np.array(mask_choice).astype(str))
    # env = EpiSimEnvironment(city, **env_params)

    

    # %%
    # %% PPO에 PointNet 붙이기
    pointnet_kwargs = dict(
        features_extractor_class=PointNetExtractor,
        features_extractor_kwargs=dict(features_dim=128)
    )

    transformer_kwargs = dict(
        normalize_images=False,
        features_extractor_class=ImageTransformerExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            patch_size=8,      # 96/8=12, 88/8=11 -> OK
            d_model=64,
            nhead=4,
            num_layers=2,
            pool="mean",       # "cls"로 바꿔도 됨
        ),
        net_arch=dict(pi=[256, 128, 128], vf=[256, 128, 128]),
    )

    # ✅ disable image normalization in the policy
    policy_kwargs = dict(
        normalize_images=False,
        features_extractor_class=ResNet18ExtractorWithTime,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
    )

    
    #%%
    vecenv = SubprocVecEnv([make_env(city, env_params, 
                                     seed = base_seed+i) for 
                                      i in range(n_envs)])
    
    
    
    

    # ##
    # # ##
    normenv = VecNormalize.load(prefix+"_vecnormalize.pkl", vecenv)
    # normenv = VecNormalize(vecenv, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma = env_params['gamma'])
    obs = normenv.reset()
    print('ENV made')
    
    
    # %%
    #원래꺼
    model = MaskablePPO("MultiInputPolicy", normenv, gamma=env_params['gamma'], verbose=0, 
                    n_steps=int(4096/vecenv.num_envs),
                    batch_size=256, 
                    learning_rate=LinearSchedule(4e-4,1e-5,0.5), 
                    # n_epochs=4,
                    tensorboard_log="./ppo_logs/",
                    # ent_coef=0.01,
                    clip_range = 0.2,
                    # target_kl=0.3,
                    device="cuda",
                    policy_kwargs=policy_kwargs,
                )
    
    # %% 여기가 재학습
    model = MaskablePPO.load(
        './logs/best_model/best_model.zip',
    #     # "learned_models/small_resnet18_len180_risk0.025_ex0.100_cb0.03_ci0.10_it10000000_1.zip", 
    #     'learned_models/city_small__ngrid50_len180_risk0.050_ex0.014_cb0.06_ci0.10_it10000000_1.zip',
                     env=normenv, device="cuda")

    # with torch.no_grad():
    #     model.policy.action_net.bias.data[-1] += 10.0
    #     print((model.policy.action_net.bias.data))

    # %% 여기가 재학습
    # model = PPO.load(
    #     #'./logs/best_model/best_model.zip',
    #     # "learned_models/small_resnet18_len180_risk0.025_ex0.100_cb0.03_ci0.10_it10000000_1.zip", 
    #     'learned_models/city_small__ngrid50_len180_risk0.050_ex0.014_cb0.06_ci0.10_it10000000_1.zip',
    #                  env=normenv, device="cuda")
    
    # model_driver = EpiSimDriver(env)
    # model_goal = model_driver.run(model, deterministic=False, verbose=True)
    # print("Model goal initial:", model_goal)

    # %%
    for _ in tqdm(range(env_params['max_epis_length']*10)):  # 2만~5만 스텝 권장
        action, _ = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = normenv.step(action)
    normenv.save("vecnorm_stats.pkl")  # 스냅샷 저장

    print("n_steps:", model.n_steps)
    print("n_envs:", vecenv.num_envs)
    print("→ rollout size:", model.n_steps * vecenv.num_envs)
    #%%
    eval_vec = SubprocVecEnv([make_env(city, env_params, 
                                       seed = base_seed+10000+i) for 
                                      i in range(n_envs)])
    # eval_vec = SubprocVecEnv([
    #     lambda: Monitor(EpiSimEnvironment(
    #         env.city,
    #         max_epis_length=env.epis_length,
    #         ext_rate=env.ext_rate,
    #         mean_recover=env.mean_recover,
    #         risk_coexist=env.risk_coexist,
    #         n_visit_tries=env.n_visit_tries,
    #         r0=env.r0,
    #         coef_block=env.coef_block,
    #         coef_infect=env.coef_infect,
    #         gamma=env.gamma,
    #         seed = base_seed+i
    #     )) for i in range(n_envs)
    # ])
    # %%
    # 학습용 VecNormalize 통계 저장 & 평가 환경에 로드
    eval_env = VecNormalize.load("vecnorm_stats.pkl", eval_vec)

    # 평가에서는 통계 고정 + raw 보상으로
    eval_env.training = False
    eval_env.norm_reward = False
    eval_env.norm_obs = False
    
    # %% PPO 에이전트 학습
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/eval/",
        n_eval_episodes=32,
        eval_freq=int(4096*5/vecenv.num_envs),
        deterministic=True,
        render=False,
    )

    

    progress_callback = ProgressBarCallback()
    rewardhist_callback = RewardHistAdvantageLogger(log_interval=model.n_steps)
    ent_callback = EntCoefScheduler(start=0.005, end=0.0, end_fraction=0.05, verbose=0)
    advantage_callback = AdvantageLogger()

    callback = CallbackList([eval_callback, progress_callback, 
                             rewardhist_callback, 
                             advantage_callback,
                             ent_callback,
                             ])

    log_name = (
        f"{prefix}_"
        f"{','.join(np.array(env_params['mask_choice']).astype(str))}"
        f"_ngrid{env_params['ngrid']}"
        f"_len{env_params['max_epis_length']}"
        f"_risk{env_params['risk_coexist']:.3f}"
        f"_ex{100*env_params['ext_rate']:.3f}"
        f"_cb{env_params['coef_block']:.2f}"
        f"_ci{env_params['coef_infect']:.2f}"
        f"_lag{env_params['lag']}"
        f"_it{n_ts}"
    )

    
    


    model.learn(total_timesteps=n_ts, 
                callback=callback,
                tb_log_name=log_name,
                reset_num_timesteps=True,
                log_interval=1)
                # tb_log_name="ifsq_cb"+str(env.coef_block)+"_it"+str(n_ts))
    # model.save("ppo_transformer_"+"ifsq_cb"+str(env.coef_block)+"_it"+str(n_ts)+".zip")
    model.save(log_name+".zip")
    normenv.save(prefix+"_vecnormalize.pkl")
    # 0.05, 50k -3141
    # # %%
    # driver = EpiSimDriver(env)
    # naive_goal = driver.run(model, verbose=True)
    # naive_goal

    # # %%
    # driver = EpiSimDriver(env)
    # naive_goal = driver.run('blockade', verbose=True)
    # naive_goal

    # %%학습된 에이전트 평가
    # obs, _ = env.reset()
    # done = False
    # total_reward = 0
    # discounting = 1

    # while not done:
    #     st = time()
    #     action, _ = model.predict(obs, deterministic=False)
    #     # action = [0]
    #     # action = np.zeros(env.action_space.shape[0])
    #     obs, reward, terminated, trunc, info = env.step(action)
    #     done = terminated or trunc
    #     total_reward += discounting*reward
    #     discounting *= env.gamma
    #     print("elapsed {}: {:.3f}\t| # of infectees: {}\t| # of blocked: {}".format(
    #                 env.t, time()-st, sum(env.cohort['state']=='I'), sum(env.block_masks[action])))
    # print("Reward:", total_reward)


    # %%
