# %%
from episim import *
from featureextractors import *
import pickle
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.utils import get_schedule_fn, get_linear_fn
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
                            coef_block=0.04, coef_infect=0.1, gamma=0.999,
                            ngrid = 50)

    n_ts = 3_000_000
    # %%

    check_env(env, warn=True)
    # %%
    # %% PPO에 PointNet 붙이기
    pointnet_kwargs = dict(
        features_extractor_class=PointNetExtractor,
        features_extractor_kwargs=dict(features_dim=128)
    )

    transformer_kwargs = dict(
        features_extractor_class=TransformerExtractor,
        features_extractor_kwargs=dict(features_dim=128, nhead=4, num_layers=2)
    )

    # ✅ disable image normalization in the policy
    policy_kwargs = dict(
        normalize_images=False,
        # features_extractor_class=ResNet18Extractor,
        # features_extractor_kwargs=dict(features_dim=512),
        # net_arch=dict(pi=[256, 128, 128], vf=[256, 128, 128]),
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
                                                              gamma=env.gamma,
                                                              ngrid=env.ngrid))                             
                            for _ in range(16)])  # 24개 병렬 환경
    
    # %%
    # 2) 평가 환경(별도 인스턴스)


    print('ENV made')
    init_bias = -4.0

    # ##
    normenv = VecNormalize(vecenv, norm_obs=False, norm_reward=True, clip_reward=100.0)
    model = PPO("CnnPolicy", normenv, gamma=env.gamma, verbose=0, 
                    n_steps=int(1024/vecenv.num_envs),
                    batch_size=256, 
                    learning_rate=get_linear_fn(4e-4,1e-8,0.9), 
                    # n_epochs=4,
                    tensorboard_log="./ppo_logs/",
                    # ent_coef=0.05,
                    clip_range = 0.15,
                    device="cuda",
                    policy_kwargs=policy_kwargs,
                )
    
    with torch.no_grad():
        model.policy.action_net.bias.fill_(init_bias)
    # ##
    # normenv = VecNormalize.load(prefix+"_vecnormalize.pkl", vecenv)
    # model = PPO.load("/home/ckang/projects/corona2025/local_resnet18policy_len730_risk0.03_ib-4.0_cb0.04_ci0.1_it3000000_1.zip", env=normenv, device="cuda")

    
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
            gamma=env.gamma,
            ngrid=env.ngrid
        )) for _ in range(16)
    ])

    # 평가 env도 같은 파라미터로 감싼 다음, 통계 복사
    eval_env = VecNormalize(
        eval_vec,
        norm_obs=False,          # 학습과 동일
        norm_reward=True,
        clip_reward=100.0
    )

    # 통계 복사 + freeze
    eval_env.ret_rms  = normenv.ret_rms
    eval_env.training = False         # ★ 정규화 통계 업데이트 금지

    # 원시 보상으로 평가하려면 (필요 시 토글)
    eval_env.norm_reward = False    # ★ raw 보상 기준

    env.reset()
    model_driver = EpiSimDriver(env)
    model_goal = model_driver.run(model, deterministic=False, verbose=True)
    print("Model goal initial:", model_goal)

    
    # %% PPO 에이전트 학습
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model/",
        log_path="./logs/eval/",
        n_eval_episodes=16,
        eval_freq=512,
        deterministic=True,
        render=False,
    )

    progress_callback = ProgressBarCallback()

    callback = CallbackList([eval_callback, progress_callback])

    log_name = prefix+"_cnnpolicy_len"+str(env.epis_length)+"_risk"+str(env.risk_coexist)+ \
        "_ib"+str(init_bias)+ \
        "_cb"+str(env.coef_block)+"_ci"+str(env.coef_infect)+ \
        "_it"+str(n_ts)
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
