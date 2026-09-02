# %%
from episim import *
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
from stable_baselines3.common.utils import get_schedule_fn, get_linear_fn
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

class RewardHistLogger(BaseCallback):
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
                self.logger.record("reward/hist", rewards_arr)
                self.buffer.clear()
            return True

class EntCoefScheduler(BaseCallback):
    """
    ent_coef를 선형 스케줄로 줄이는 콜백.
    progress_remaining: 1.0 -> 0.0 (학습이 진행될수록 감소)
    """
    def __init__(self, start=0.05, end=0.0, end_fraction=0.5, verbose=0):
        super().__init__(verbose)
        self.schedule = get_linear_fn(start, end, end_fraction)

    def _on_rollout_start(self) -> None:
        # 현재 학습 진행도 (1 → 0)
        progress = float(self.model._current_progress_remaining)
        new_ent = float(self.schedule(progress))
        self.model.ent_coef = new_ent
        if self.verbose > 0:
            print(f"[EntCoefScheduler] progress={progress:.3f}, ent_coef={new_ent:.5f}")

    def _on_step(self) -> bool:    
        return True

if __name__ == "__main__":
    # with open('city_random.pkl', 'rb') as f:
    #     city = pickle.load(f)
    # prefix = 'rdcity'
    #%%
    with open('city_local.pkl', 'rb') as f:
        city = pickle.load(f)
    prefix = 'local'
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
    env = EpiSimEnvironment(city, max_epis_length=365, ext_rate=1/1000/7, mean_recover=5,
                            risk_coexist = 0.03, n_visit_tries = 2, r0=5,
                            coef_block=0.05, coef_infect=0.1, gamma=0.999)

    n_ts = 5_000_000
    # %%

    check_env(env, warn=True)
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
        features_extractor_class=ResNet18Extractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 128, 128], vf=[256, 128, 128]),
    )

    
    #%%
    vecenv = SubprocVecEnv([lambda: Monitor(EpiSimEnvironment(env.city, 
                                                              max_epis_length=env.epis_length, 
                                                              ext_rate = env.ext_rate, 
                                                              mean_recover=env.mean_recover,
                                                              risk_coexist = env.risk_coexist, 
                                                              n_visit_tries = env.n_visit_tries, 
                                                              r0=env.r0, 
                                                              coef_block=env.coef_block,
                                                              coef_infect=env.coef_infect,
                                                              gamma=env.gamma))                             
                            for _ in range(16)])  # 24개 병렬 환경
    
    # %%
    
    
    init_bias = -4.0

    # ##
    # normenv = VecNormalize(vecenv, norm_obs=False, norm_reward=True, clip_reward=1.0, gamma = env.gamma)
    # --- 통계 워밍업 (학습 없이 스텝만 진행) ---
    # obs = normenv.reset()
    print('ENV made')
    
    # gpt가 알려준 붕괴 방지
    # n_envs = vecenv.num_envs
    # rollout = max(4096 // n_envs, 256)
    # model = PPO(
    #     "CnnPolicy", normenv,
    #     gamma=env.gamma,
    #     n_steps=rollout,
    #     batch_size=128,                 # 64~128 권장
    #     n_epochs=10,                    # 기본값
    #     learning_rate=get_linear_fn(2e-4, 1e-5, 1.0),  # 초반부터 감쇠
    #     clip_range=get_linear_fn(0.2, 0.1, 0.5),
    #     clip_range_vf=0.2,
    #     ent_coef=0.02,        # 중반 이후 서서히 0
    #     max_grad_norm=0.5,
    #     target_kl=0.03,
    #     tensorboard_log="./ppo_logs/",
    #     device="cuda",
    #     policy_kwargs=dict(
    #         **policy_kwargs,
    #         ortho_init=True,
    #     ),
    #     # verbose=0,
    # )

    #원래꺼
    model = PPO("CnnPolicy", vecenv, gamma=env.gamma, verbose=0, 
                    n_steps=int(4096/vecenv.num_envs),
                    batch_size=128, 
                    learning_rate=get_linear_fn(3e-4,1e-5,0.8), 
                    # n_epochs=4,
                    tensorboard_log="./ppo_logs/",
                    ent_coef=0.01,
                    clip_range = 0.2,
                    device="cuda",
                    policy_kwargs=policy_kwargs,
                )
    
    with torch.no_grad():
        model.policy.action_net.bias.fill_(init_bias)
    # ##
    # normenv = VecNormalize.load(prefix+"_vecnormalize.pkl", vecenv)
    # model = PPO.load("/home/ckang/projects/corona2025/local_resnet18policy_len365_risk0.03_ib-4.0_cb0.03_ci0.1_it3000000_1.zip", env=normenv, device="cuda")

    # for _ in tqdm(range(20_000)):  # 2만~5만 스텝 권장
    #     # 벡터 환경이니 env 수만큼 액션 샘플
    #     # print('???')
    #     action = model.predict(obs, deterministic=False)
    #     obs, rew, done, info = normenv.step(action)
    # normenv.save("vecnorm_stats.pkl")  # 스냅샷 저장

    print("n_steps:", model.n_steps)
    print("n_envs:", vecenv.num_envs)
    print("→ rollout size:", model.n_steps * vecenv.num_envs)
    #%%
    eval_vec = SubprocVecEnv([
        lambda: Monitor(EpiSimEnvironment(
            env.city,
            max_epis_length=env.epis_length,
            ext_rate=env.ext_rate,
            mean_recover=env.mean_recover,
            risk_coexist=env.risk_coexist,
            n_visit_tries=env.n_visit_tries,
            r0=env.r0,
            coef_block=env.coef_block,
            coef_infect=env.coef_infect,
            gamma=env.gamma
        )) for _ in range(16)
    ])

    # 학습용 VecNormalize 통계 저장 & 평가 환경에 로드
    # eval_env = VecNormalize.load("vecnorm_stats.pkl", eval_vec)

    # 평가에서는 통계 고정 + raw 보상으로
    # eval_env.training = False
    # eval_env.norm_reward = False
    
    # %% PPO 에이전트 학습
    eval_callback = EvalCallback(
        eval_vec,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/eval/",
        n_eval_episodes=16,
        eval_freq=int(4096*5/vecenv.num_envs),
        deterministic=False,
        render=False,
    )

    

    progress_callback = ProgressBarCallback()
    rewardhist_callback = RewardHistLogger(log_interval=1000)
    ent_callback = EntCoefScheduler(start=0.01, end=0.0, end_fraction=0.2, verbose=1)


    callback = CallbackList([eval_callback, progress_callback, rewardhist_callback, ent_callback])

    log_name = prefix+"_resnet18policy_len"+str(env.epis_length)+"_risk"+str(env.risk_coexist)+ \
        "_ib"+str(init_bias)+ \
        "_cb"+str(env.coef_block)+"_ci"+str(env.coef_infect)+ \
        "_it"+str(n_ts)

    env.reset()
    model_driver = EpiSimDriver(env)
    model_goal = model_driver.run(model, deterministic=False, verbose=True)
    print("Model goal initial:", model_goal)


    model.learn(total_timesteps=n_ts, 
                callback=callback,
                tb_log_name=log_name,
                reset_num_timesteps=True,
                log_interval=1)
                # tb_log_name="ifsq_cb"+str(env.coef_block)+"_it"+str(n_ts))
    # model.save("ppo_transformer_"+"ifsq_cb"+str(env.coef_block)+"_it"+str(n_ts)+".zip")
    model.save(log_name+".zip")
    # normenv.save(prefix+"_vecnormalize.pkl")
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
    obs, _ = env.reset()
    done = False
    total_reward = 0
    discounting = 1

    while not done:
        st = time()
        action, _ = model.predict(obs, deterministic=False)
        # action = [0]
        # action = np.zeros(env.action_space.shape[0])
        obs, reward, terminated, trunc, info = env.step(action)
        done = terminated or trunc
        total_reward += discounting*reward
        discounting *= env.gamma
        print("elapsed {}: {:.3f}\t| # of infectees: {}\t| # of blocked: {}".format(
                    env.t, time()-st, sum(env.cohort['state']=='I'), sum(action)))
    print("Reward:", total_reward)


    # %%
