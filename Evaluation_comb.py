# %%
import pickle
from episim_comb import *
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm
import numpy as np
import seaborn as sns
# import seaborn_image as isns
import sys
import os, re
from matplotlib.ticker import StrMethodFormatter
# %%
#     pickle.dump(city_base, f)

def make_env(city_base, env_params, seed=1234):
    # print(seed)
    def _init():
        env = EpiSimEnvironment(city_base, **env_params, 
                                seed = seed) #coef_infect는 x^2에 대한 가중 효과만 반영. 기본 x가 infection cost에 들어감
        return(env)
    return _init

# def abc_to_hue_deg(a, b, c, angles=(0, 120, 240)):
#     a, b, c = map(np.asarray, (a, b, c))
#     s = a + b + c
    
#     pa = np.divide(a, s, out=np.zeros_like(a, dtype=float), where=s!=0)
#     pb = np.divide(b, s, out=np.zeros_like(b, dtype=float), where=s!=0)
#     pc = np.divide(c, s, out=np.zeros_like(c, dtype=float), where=s!=0)

#     # 원하는 각도 지정
#     θ_a, θ_b, θ_c = np.radians(angles)

#     va = (np.cos(θ_a), np.sin(θ_a))
#     vb = (np.cos(θ_b), np.sin(θ_b))
#     vc = (np.cos(θ_c), np.sin(θ_c))

#     x = pa*va[0] + pb*vb[0] + pc*vc[0]
#     y = pa*va[1] + pb*vb[1] + pc*vc[1]

#     theta = np.degrees(np.arctan2(y, x))
#     hue_deg = (theta + 360) % 360
#     hue_deg = np.where(s==0, np.nan, hue_deg)  # 합 0 → NaN
#     return hue_deg, s



# %%
class RandomPolicy():
    def __init__(self, env, city, p, thr= 1/365, seed=None):
        self.p = p
        self.thrinf = city.N * thr
        # self.nact = env.action_space.shape[0]
        self.shape = (env.num_envs, *env.action_space.shape)
        self.rng = np.random.default_rng(seed)
        self.env = env
        block_masks = self.env.get_attr('block_masks')[0]
        city = self.env.get_attr('city')[0]
        self.options = np.where(block_masks.sum(axis=1) == int(len(city.facs)*self.p))[0]
        
    def predict(self, obs, deterministic=False):
        # print(self.options)
        actions = self.rng.choice(self.options, size=self.shape, replace = True)
        infects = obs['image'][:,1,:,:].sum(axis=(1,2))
        # print(actions, infects, self.thrinf)
        actions[infects < (0.0001+math.ceil(self.thrinf))] = 0
        # print(infects[infects < (0.0001+math.ceil(self.thrinf))])
        
        return(actions, None)    
        
            
class VecDriver():

    def __init__(self, city_base, env_params, num_envs, seed = 1234):
        self.num_envs = num_envs
        self.env_params = env_params
        self.vec_env = SubprocVecEnv([make_env(city_base, env_params, seed = seed+i) for 
                                      i in range(num_envs)])
        self.reset()

    def reset(self):
        self.episode_rewards = [[] for _ in range(self.num_envs)]
        self.block_cost = [[] for _ in range(self.num_envs)]
        self.infect_cost = [[] for _ in range(self.num_envs)]
        self.obses = [[] for _ in range(self.num_envs)]
        self.cohorts = [[] for _ in range(self.num_envs)]
        self.links = [[] for _ in range(self.num_envs)]
        self.fermeture = [[] for _ in range(self.num_envs)]

    def archive(self, folder, prefix):
        
        path = f"{folder}/{prefix}.pkl"
        with open(path, 'wb') as f:
            pickle.dump({'rewards': self.episode_rewards,
                         'block': self.block_cost,
                         'infect': self.infect_cost,
                         'obses': self.obses,
                         'cohort': self.cohorts,
                         'links': self.links,
                         'fermeture': self.fermeture}, f)
        print(f"Result saved to {path}")

        
    def run(self, model, deterministic = False, like = False, enforce_zero = True):
        episode_counts = np.zeros(self.num_envs, dtype=int)
        # self.vec_env.seed(seed = seed)
        obs = self.vec_env.reset()
        dones = [False] * self.num_envs
        self.vec_env.env_method('set_like', like)
        block_masks = self.vec_env.get_attr('block_masks')[0]
        
        for _ in tqdm(range(self.env_params['max_epis_length'])):
            if np.sum(episode_counts>0) == self.num_envs:
                break

            actions, _ = model.predict(obs, deterministic=deterministic)

            # 감염자 0이면 무조건 안한다
            infects = obs['image'][:,1,:,:].sum(axis=(1,2))
            if enforce_zero: actions[infects < 0.0001] = 0

            # print(actions.shape)
            b_cost = self.vec_env.get_attr('block_cost')
            i_cost = self.vec_env.get_attr('infection_cost')
            obs, rewards, dones, infos = self.vec_env.step(actions)
            
            for i in range(self.num_envs):
                
                if episode_counts[i] < 1: #각자 한번만 해라. 나머지는 기록 안한다.
                    self.episode_rewards[i].append(rewards[i])
                    self.block_cost[i].append(b_cost[i])
                    self.infect_cost[i].append(i_cost[i])
                    self.obses[i].append(obs['image'][i])
                    self.cohorts[i].append(infos[i]['cohort'])
                    self.links[i].append(infos[i]['link'])
                    fermature = block_masks[actions[i]]
                    self.fermeture[i].append(fermature)
                    if dones[i]:
                        print(f"Env {i} - Episode {episode_counts[i]} reward: {sum(self.episode_rewards[i])}")
                        episode_counts[i] += 1
        print(np.array(self.episode_rewards).sum(axis=1).mean())    
                
    def close(self):
        self.vec_env.close()  
    
class Reporter():
    def __init__(self, path):
        with open(path, 'rb') as f:
            res_dict = pickle.load(f)
        self.load_dict(res_dict)

    def load_dict(self, res_dict):
        self.episode_rewards = res_dict['rewards']
        try:
            self.block_cost = res_dict['block']
            self.infect_cost = res_dict['infect']
        except KeyError:
            print("No block or infect cost storage")
        self.obses = res_dict['obses']
        self.cohorts = res_dict['cohort']
        self.links = res_dict['links']
        self.fermeture = res_dict['fermeture']
    
    def discounted_reward_dist(self, gamma):
        discounted_rewards = np.zeros(len(self.episode_rewards))
        for i, rewards in enumerate(self.episode_rewards):
            for r in rewards[::-1]:
                discounted_rewards[i] = gamma*discounted_rewards[i] + r
        return(discounted_rewards)        

    def ferme_time_series(self, aggregate='sum'):
        num_envs = len(self.cohorts)
        if num_envs == 0:
            return np.array([])

        max_len = max(len(seq) for seq in self.cohorts)
        counts = np.zeros((num_envs, max_len), dtype=int)

        for env_idx, seq in enumerate(self.fermeture):
            for t, arr in enumerate(seq):
                counts[env_idx, t] = arr.sum()

        if aggregate == 'none':
            print(counts.shape)
            df=pd.DataFrame({'count': counts.ravel(),
                          'timestep': np.tile(np.arange(counts.shape[1]), counts.shape[0]),
                          'env': np.repeat(np.arange(counts.shape[0]), counts.shape[1])})
            return df
        elif aggregate == 'mean':
            # number of envs that have data at each timestep
            valid_envs = (counts != 0).sum(axis=0)
            valid_envs[valid_envs == 0] = 1  # avoid div‑by‑zero
            return counts.sum(axis=0) / valid_envs
        else:  # 'sum'
            return counts.sum(axis=0)

    def newinf_time_series(self, aggregate='sum'):
        num_envs = len(self.links)
        if num_envs == 0:
            return np.array([])

        max_len = max(len(seq) for seq in self.links)
        counts = np.zeros((num_envs, max_len), dtype=int)

        for env_idx, seq in enumerate(self.links):
            for t, df in enumerate(seq):
                counts[env_idx, t] = len(df)

        if aggregate == 'none':
            print(counts.shape)
            df=pd.DataFrame({'count': counts.ravel(),
                          'timestep': np.tile(np.arange(counts.shape[1]), counts.shape[0]),
                          'env': np.repeat(np.arange(counts.shape[0]), counts.shape[1])})
            return df
        elif aggregate == 'mean':
            # number of envs that have data at each timestep
            valid_envs = (counts != 0).sum(axis=0)
            valid_envs[valid_envs == 0] = 1  # avoid div‑by‑zero
            return counts.sum(axis=0) / valid_envs
        else:  # 'sum'
            return counts.sum(axis=0)        
            
    def cohort_time_series(self, state='I', aggregate='sum'):
        
        # self.cohorts -> list(env) -> list(timestep) -> DataFrame
        num_envs = len(self.cohorts)
        if num_envs == 0:
            return np.array([])

        max_len = max(len(seq) for seq in self.cohorts)
        counts = np.zeros((num_envs, max_len), dtype=int)

        for env_idx, seq in enumerate(self.cohorts):
            for t, df in enumerate(seq):
                counts[env_idx, t] = (df['state'] == state).sum()

        if aggregate == 'none':
            print(counts.shape)
            df=pd.DataFrame({'count': counts.ravel(),
                          'timestep': np.tile(np.arange(counts.shape[1]), counts.shape[0]),
                          'env': np.repeat(np.arange(counts.shape[0]), counts.shape[1])})
            return df
        elif aggregate == 'mean':
            # number of envs that have data at each timestep
            valid_envs = (counts != 0).sum(axis=0)
            valid_envs[valid_envs == 0] = 1  # avoid div‑by‑zero
            return counts.sum(axis=0) / valid_envs
        else:  # 'sum'
            return counts.sum(axis=0)



class Imager():
    def set_span(self, cohort, padding = 10):
        self.observe_min_x = cohort.xcoor.min() 
        self.observe_min_y = cohort.ycoor.min() 
        self.observe_span_x = cohort.xcoor.max() - cohort.xcoor.min() + 1e-5
        self.observe_span_y = cohort.ycoor.max() - cohort.ycoor.min() + 1e-5
        self.observe_span = max(self.observe_span_x, self.observe_span_y)
        self.nx = np.ceil(self.ngrid * (self.observe_span_x/self.observe_span)).astype(int)
        self.ny = np.ceil(self.ngrid * (self.observe_span_y/self.observe_span)).astype(int)
        
    def fac_view(self, facs, fermeture):
        temp = res_00.cohorts[0][0][['xcoor','ycoor']]
        x = facs['xcoor'] - self.observe_min_x
        y = facs['ycoor'] - self.observe_min_y
        x *= self.nx/self.observe_span_x
        y *= self.ny/self.observe_span_y
        x -= 0.5
        y = self.ny - y - 0.5
        df = pd.DataFrame({'xcoor':x,'ycoor':y,
                      'type':facs['type']})

        return(df.loc[fermeture==0], df.loc[fermeture==1])

    def cohort_view(self, cohort, max_c=25):
        np.random.seed(42)
        img = np.zeros((len(self.state_set), self.nx, self.ny), dtype=np.int32)
        x = cohort['xcoor'] - self.observe_min_x + np.random.normal(scale=self.observe_span_x/self.ngrid,size=len(cohort))/1
        x = np.clip(x, 0, self.observe_span_x - 1e-5) #blurring            
        y = cohort['ycoor'] - self.observe_min_y + np.random.normal(scale=self.observe_span/self.ngrid,size=len(cohort))/1 
        y = np.clip(y, 0, self.observe_span_y - 1e-5) #blurring           
        xi = (x/self.observe_span_x*self.nx).astype(int)
        yi = (y/self.observe_span_y*self.ny).astype(int)
        # df = pd.DataFrame({'xidx': xi, 'yidx': yi})
        # grid_counts = df.groupby(['yidx', 'xidx']).size().unstack(fill_value=0)
        # return(grid_counts)
        
        for idx, state in enumerate(self.state_set):
            mask = cohort['state'] == state
            np.add.at(img[idx], (xi[mask], yi[mask]), 1)

        # max_c = np.max(img)
        print(np.max(img))
        img_scaled = np.clip(img,0,max_c)
        img_scaled = (img_scaled * (255.0 / max_c)).astype(np.uint8) 
        return img_scaled, img
        # return img.astype(np.uint8)
        
    def __init__(self, ngrid):
        self.state_set = ('S', 'I', 'R')
        self.ngrid = ngrid


# with open('city_jecheon_scale20.pkl', 'rb') as f:
#     city = pickle.load(f)



num_envs = 100

env_params = {
        'max_epis_length': 180,
        'ext_rate': 1/1000/7,
        # 'ext_rate': 1/1000,
        'mean_recover': 5,
        'risk_coexist': 0.05,
        # 'risk_coexist': 0.025,
        'n_visit_tries': 2,
        'r0': 100,
        'coef_block': 0.02,
        'coef_infect': 0.1,
        'gamma': 1.0,
        'mask_choice': [],
        'ngrid': 50,
        'lag': 0,
    }            

# city_name = 'city_local_only'
city_name = 'city_small'
review_period = 0

def compute_run_length_dist(df):
    results = []

    for col in df.columns:
        series = df[col]
        
        # [핵심 로직]
        # 값이 이전 행과 다를 때(ne)마다 True가 되고, cumsum으로 그룹 ID를 만듦
        # 예: [0, 0, 1, 1, 1, 0] -> 그룹 ID: [1, 1, 2, 2, 2, 3]
        run_ids = series.ne(series.shift()).cumsum()
        
        # 그룹별 사이즈(길이) 계산
        run_lengths = series.groupby(run_ids).size()
        
        # 값(0인지 1인지)도 같이 저장하고 싶다면 아래 주석 해제
        # run_values = series.groupby(run_ids).first()
        
        # 결과를 시각화하기 좋게 DataFrame으로 정리
        temp_df = pd.DataFrame({
            'col_name': col,
            'duration': run_lengths.values,
            # 'value': run_values.values # 0이 연속된건지 1이 연속된건지 구분이 필요하면 추가
        })
        results.append(temp_df)

    # 전체 결과를 하나로 합침
    dist_df = pd.concat(results).reset_index(drop=True)

    # --- 결과 확인 ---
    return dist_df

use_paper_style(base_fontsize=12, font_family="DejaVu Serif", usetex=False,
                    figure_width=6, figure_height=4.5)       
#%%
if 2==0: #그림을 그리자.
    # %%
    with open(city_name+'.pkl', 'rb') as f:
        city = pickle.load(f)
    env = EpiSimEnvironment(city, **env_params)
    # model = PPO.load("jc20_resnet18policy_ib-2.0_cb0.06_ci0.1_it5000000_2.zip", env=env, device="cuda")
    # model = PPO.load("jc20_resnet18policy_ib-2.0_cb0.15_ci0.1_it5000000_1.zip", env=env, device="cuda")
    # model = PPO.load("jc20_resnet18policy_ib-2.0_cb0.07_ci0.1_it5000000_3.zip", env=env, device="cuda")
    model = PPO.load("./logs/best_model/best_model.zip", env=env, device="cuda")
    
    # %% fig:dist_hhs
    fig, ax = plt.subplots(1,1, figsize=(5, 4))
    df_hh_fac = pd.concat([city.hhs, city.facs])
    df_hh_fac.loc[df_hh_fac.type=='household', 'type'] = 'Households'
    df_hh_fac.loc[df_hh_fac.type!='Households', 'type'] = 'Community centers'

    sns.scatterplot(df_hh_fac, ax=ax,
                    x='xcoor',
                    y='ycoor',
                    hue='type', 
                    style="type",
                    palette={"Households": "white", "Community centers": "yellow"},
                    markers={"Households": "o", "Community centers": '^'}, 
                    size="type",
                    sizes={"Households": 20, "Community centers": 250},
                    edgecolor='black',
                    )    
    # tick과 label 제거
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set(xlabel=None, ylabel=None)
    ax.legend(title=None, fontsize=10)
    # 격자 추가
    ax.grid(True, linewidth=0.8, alpha=0.5)
    for i, row in df_hh_fac.loc[df_hh_fac.type=='Community centers'].drop_duplicates(['xcoor', 'ycoor']).reset_index(drop=True).iterrows():
        ax.text(row.xcoor, row.ycoor-1, i+1, fontsize=10, ha='center', va='center')

    

    plt.savefig("./figs/dist_hhs.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
    # %% fig:visitors
    visit_facs = [20001, 20004]#, 20007, 20010]
    affil_facs = [20002, 20000]
    visitors = city.draw_visit(20000)
    # affiliates = city.draw_visit(20078, mode='affiliated')

    # %% fig:visitors        
    plt.figure(figsize=(5, 4))
    ax = sns.scatterplot(city.hhs, 
                    x='xcoor',
                    y='ycoor',
                    s=30,
                    c='white',
                    edgecolor='black',
                    )    
    # tick과 label 제거
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set(xlabel=None, ylabel=None)
    # ax.legend(title=None)
    # 격자 추가
    ax.grid(True, linewidth=0.8, alpha=0.5)
    map2 = {"Healthcare": "High locality", 
            "School": "yellow", 
            "Workplace": "cyan", 
            "Venue": "Low locality", 
            "Restaurant": "fuchsia",
            "Daily service": "orange"}
    colors = {"Low locality": "red", 
            "School": "yellow", 
            "Workplace": "cyan", 
            "High locality": "mediumblue", 
            "Restaurant": "fuchsia",
            "Daily service": "orange"}
    df_visit = visitors.loc[visitors.fid.isin(visit_facs)].sort_values('fid')
    df_visit = df_visit.merge(city.inds, left_on='iid', right_index=True)
    df_visit = df_visit.merge(city.facs[['type']], left_on='fid', right_index=True)
    df_affil = None
    df_visit['type_z'] = df_visit['type_y'].map(map2)
    # df_affil = affiliates.loc[affiliates.fid.isin(affil_facs)].sort_values('fid')
    # df_affil = df_affil.merge(city.inds, left_on='iid', right_index=True)
    # df_affil = df_affil.merge(city.facs[['type']], left_on='fid', right_index=True)
    
    sns.scatterplot(pd.concat([df_affil,df_visit]), x='xcoor', y='ycoor', 
                    s=20, hue='type_z',
                    palette=colors,
                    edgecolor='black',
                    )
    ax.legend(title=None, fontsize=10)
    
    # sns.scatterplot(city.facs.loc[affil_facs+visit_facs], x='xcoor', y='ycoor', 
    #             s=40, hue='type_z',
    #             palette=colors,
    #             edgecolor='black',
    #             linewidth=1,
    #             marker='^',
    #             legend=False,
    #             )
    plt.savefig("./figs/visitors.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
    # %% fig:heatmaps
    # with open('city_local.pkl', 'rb') as f:
    #     city = pickle.load(f)
    # env = EpiSimEnvironment(city, **env_params)
    driver = EpiSimDriver(env)
    driver.run(model, verbose=True)                        

    # %%
    df_hh_fac = city.facs.loc[driver.fermeture[178]]
    df_hh_fac.loc[df_hh_fac.type=='household', 'type'] = 'Households'
    df_hh_fac.loc[df_hh_fac.type!='Households', 'type'] = 'Community centers'
 
    sns.scatterplot(driver.cohorts[178], 
                    x='xcoor',
                    y='ycoor',
                    hue='state', 
                    # style="type",
                    palette={"S": "white", "I": "red", "R": "blue"},
                    # markers={"Households": "o", "Community centers": '^'}, 
                    # size="type",
                    # sizes={"Households": 15, "Community centers": 40},
                    edgecolor='black',
                    )    
    ax = sns.scatterplot(df_hh_fac, 
                    x='xcoor',
                    y='ycoor',
                    hue='type', 
                    style="type",
                    palette={"Households": "white", "Community centers": "red"},
                    markers={"Households": "o", "Community centers": '^'}, 
                    size="type",
                    sizes={"Households": 15, "Community centers": 40},
                    edgecolor='black',
                    )       
    # path = '/home/ckang/projects/corona2025/results/coef_block0.05.coef_infect0.1.ext_rate0.00014285714285714287.gamma0.999.max_epis_length365.mean_recover5.n_visit_tries2.r05.risk_coexist0.03/'
    # res_00 = Reporter(path+'r0.00'+'.pkl')
    # %%
    imager = ObsImager([obs['image'] for obs in driver.obses[1:]])
    imager.save_gif('./sample.gif')

    # %%
    fig, ax = plt.subplots(1,3, figsize=(12, 4.5))
    timesteps = [1, 30, -1]
    for i, t in enumerate(timesteps):
        rgb = rgb_from_SIR_hsl(driver.obses[t], l_min=0.25, l_max=0.75)
        ax[i].imshow(rgb)#, alpha=np.where((rgb.sum(axis=2))==0, 0.0, 1.0))
        ax[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax[i].set(xlabel=None, ylabel=None)
    ax[0].set_title('Early', fontsize=14)
    ax[1].set_title('Intermediate', fontsize=14)
    ax[2].set_title('Late', fontsize=14)
    plt.savefig("./figs/heatmaps.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)



    # # %%
    # fig, ax = plt.subplots(1,3, figsize=(12, 4.5))
    # tick = 31
    # ax[0].imshow(np.rot90(driver.obses[tick][0]), cmap='Greens')
    # ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # ax[0].set(xlabel=None, ylabel=None)

    # ax[1].imshow(np.rot90(driver.obses[tick][1]), cmap='Reds')
    # ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # ax[1].set(xlabel=None, ylabel=None)

    # ax[2].imshow(np.rot90(driver.obses[tick][2]), cmap='Blues')
    # ax[2].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # ax[2].set(xlabel=None, ylabel=None)


    # %%
    obs, _ = env.reset()
    actions = []
    ninf = []
    for _ in tqdm(range(52*7)):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, _ = env.step(action)
        ninf.append(obs[1].sum())
        actions.append(action)

    
    # %%
    import seaborn as sns
    sns.heatmap(np.stack(actions))
    plt.show()
    sns.lineplot(ninf)
    # %%
    closed = city.facs.loc[actions[1]==1]
    plt.scatter(city.hhs['xcoor'], city.hhs['ycoor'], s=5, c='black')
    plt.scatter(closed['xcoor'], closed['ycoor'], s=5, c='red')

    # %%
    driver = VecDriver(city_base=city, env_params=env_params, num_envs=num_envs)
    driver.run(model, deterministic = True)
    driver.archive('./', 'temp_model_det')



# %%
if 1==0:
# %%

# %%
    
    # path = '/home/ckang/projects/corona2025/results/coef_block0.04.coef_infect0.1.ext_rate0.00014285714285714287.gamma0.999.max_epis_length365.mean_recover5.n_visit_tries2.r010.risk_coexist0.03/'
    # path = '/home/ckang/projects/corona2025/results/coef_block0.05.coef_infect0.1.ext_rate0.00014285714285714287.gamma0.999.max_epis_length180.mean_recover5.n_visit_tries2.r010.risk_coexist0.02/'
    # path = '/home/ckang/projects/corona2025/results/coef_block0.03.coef_infect0.1.ext_rate0.00014285714285714287.gamma0.999.max_epis_length365.mean_recover5.n_visit_tries2.r010.risk_coexist0.03/'
    # path = '/home/ckang/projects/corona2025/results/coef_block0.05.coef_infect0.1.ext_rate0.00014285714285714287.gamma0.999.max_epis_length365.mean_recover5.n_visit_tries2.r010.risk_coexist0.03/'
    # path = './results/coef_block0.06.coef_infect0.10.ext_rate0.00.gamma1.00.max_epis_length180.00.mean_recover5.00.n_visit_tries2.00.r0100.00.risk_coexist0.03/'
    # path = './results/coef_block0.08.coef_infect0.10.ext_rate0.00.gamma1.00.max_epis_length180.00.mean_recover5.00.n_visit_tries2.00.r0100.00.risk_coexist0.03/'
    # path = './results/coef_block0.04.coef_infect0.10.ext_rate0.00.gamma1.00.max_epis_length180.00.mean_recover5.00.n_visit_tries2.00.r0100.00.risk_coexist0.03/'
    path = './results/coef_block0.08.coef_infect0.10.ext_rate0.00.gamma1.00.max_epis_length180.00.mean_recover5.00.n_visit_tries2.00.r0100.00.risk_coexist0.03/'
    with open(city_name+'.pkl', 'rb') as f:
        city = pickle.load(f)
    paths = [
            # './results/coef_block0.01.coef_infect0.10.ext_rate0.00.gamma1.00.max_epis_length180.00.mean_recover5.00.n_visit_tries2.00.r0100.00.risk_coexist0.03/',
            # './results/coef_block0.03.coef_infect0.10.ext_rate0.00.gamma1.00.max_epis_length180.00.mean_recover5.00.n_visit_tries2.00.r0100.00.risk_coexist0.03/',
            # './results/coef_block0.05.coef_infect0.10.ext_rate0.00.gamma1.00.max_epis_length180.00.mean_recover5.00.n_visit_tries2.00.r0100.00.risk_coexist0.03/',
            './results/city_small__ngrid50_len180_risk0.050_ex0.014_cb0.02_ci0.10/',
            './results/city_small__ngrid50_len180_risk0.050_ex0.014_cb0.04_ci0.10/',
            './results/city_small__ngrid50_len180_risk0.050_ex0.014_cb0.06_ci0.10/',
            #  './results/coef_block0.06.coef_infect0.10.ext_rate0.00.gamma1.00.max_epis_length180.00.mean_recover5.00.n_visit_tries2.00.r0100.00.risk_coexist0.03/',
            #  './results/coef_block0.08.coef_infect0.10.ext_rate0.00.gamma1.00.max_epis_length180.00.mean_recover5.00.n_visit_tries2.00.r0100.00.risk_coexist0.03/',
            #  './results/coef_block0.10.coef_infect0.10.ext_rate0.00.gamma1.00.max_epis_length180.00.mean_recover5.00.n_visit_tries2.00.r0100.00.risk_coexist0.03/'
             ]
    # c_bs = [0.04, 0.06, 0.08, 0.10]
    c_bs = [0.02, 0.04, 0.06]
    paths = paths[::-1]
    c_bs = c_bs[::-1]
    # c_bs = [0.06]
    # # res_00 = Reporter(path+'r0.00'+'.pkl')
    # res_25 = Reporter(path+'r0.25'+'.pkl')
    # res_50 = Reporter(path+'r0.50'+'.pkl')
    # res_75 = Reporter(path+'r0.75'+'.pkl')
    # res_1c = Reporter(path+'r1.00'+'.pkl')
    # res_mo = Reporter(path+'model'+'.pkl')
    dfs = []
    res_repo = []
    res_mos = []
    for i, path in enumerate(paths):
        res_11 = Reporter(path+'r0.00'+'.pkl')
        res_1 = Reporter(path+'r1.00'+'.pkl')
        res_0 = Reporter(path+'ld0.000'+'.pkl')
        res_2 = Reporter(path+'ld0.001'+'.pkl')
        res_3 = Reporter(path+'ld0.002'+'.pkl')
        # res_4 = Reporter(path+'ld0.003'+'.pkl')
        # res_5 = Reporter(path+'ld0.004'+'.pkl')
        # res_6 = Reporter(path+'ld0.005'+'.pkl')
        # res_7 = Reporter(path+'ld0.01'+'.pkl')
        # res_8 = Reporter(path+'ld0.05'+'.pkl')
        # res_11 = Reporter(path+'ld0.10'+'.pkl')
        res_9 = Reporter(path+'model'+'.pkl')
        res_10 = Reporter(path+'model.like'+'.pkl')
        
        reses = [res_11, 
                 res_1, 
                 res_0, 
                 res_2, res_3,
                #  , res_4, 
                #  res_5, res_6, 
                #  res_7, res_8,
                res_9, res_10]
        # reses = [res_34, res_4, res_8, res_9]
        res_repo.append(reses)
        names = ['No intervention', 
                'Complete Lockdown', 
                'Adaptive Lockdown (0.0%)', 'Adaptive Lockdown (0.1%)', 'Adaptive Lockdown (0.2%)', 
                # 'Cond. Lockdown (r=0.4%)', 'Cond. Lockdown (r=0.5%)', 
                # 'Cond. Lockdown (r=1.0%)', 'Cond. Lockdown (r=5.0%)', 'Cond. Lockdown (r=10.0%)', 
                'RL-optimized', 'Shuffled RL',]
        # try:
        #     res_mo_det = Reporter(path+'model.deterministic'+'.pkl')
        #     reses = [res_00, res_25, res_50, res_75, res_1c, res_mo, res_mo_det]
        #     names = ['Random 0.0', 'Random 0.25', 'Random 0.5', 'Random 0.75', 'Random 1.0', 'Model', 'Model (Deterministic)']
        # except FileNotFoundError:
        #     print("No deterministic model found, continuing without it.")
        res_mos.append(res_9)
        list_rewards = []
        for res in reses:
            list_rewards.append(-1*res.discounted_reward_dist(gamma = env_params['gamma']))
        df_rewards = pd.DataFrame(list_rewards, index = names).T
        df_melt = df_rewards.melt(var_name="Policy", value_name="Total cost")
        df_melt['c_b'] = c_bs[i]
        dfs.append(df_melt)

    # %%
    fig, ax = plt.subplots(figsize=(9, 5))

    sns.pointplot(
        data=pd.concat(dfs),
        x="Total cost", y="Policy", hue="c_b",
        # dodge=0.5,              # 같은 행에서 좌우로 분리
        errorbar=("ci", 95),    # 신뢰구간
        marker="D",# hue별 마커
        linestyle="none",
        linewidth=1.0,
        markersize=3,
        capsize=0.15,
        palette={0.02:'tab:orange', 0.03: 'tab:blue', 0.05: 'tab:orange',
            0.04:'tab:red', 0.06: 'tab:blue', 0.08: 'tab:orange', 0.10: 'tab:green'},
        #  colors=palette[str(c_bs[i])],
        ax=ax
    )

    # === 세로선 (vertical guideline) 추가 ===
    for x in range(4000, 15000, 1000):
        ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.6, alpha=0.6)

    # === 가로선 (lane separator) 추가 ===
    # ytick 위치 가져오기
    yticks = ax.get_yticks()
    for y in yticks:
        ax.axhline(y=y+0.5, color='lightgray', linestyle='-', linewidth=0.5, alpha=1.0)

    # === 포맷 및 스타일 ===
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.set_xlabel("Total cost")
    ax.set_ylabel("Policy")

    handles, labels = ax.get_legend_handles_labels()
    labels = [r"$C_{e}$="+"{:.2f}".format(float(lbl)) for lbl in labels]  # prefix 붙이기



    ax.legend(handles, labels, loc="lower right", frameon=True)

    plt.grid(False)   # seaborn 기본 grid 제거 (중복 방지)
    plt.tight_layout()
    plt.show()

    # sns.stripplot(
    #     data=pd.concat(dfs), x='Total cost', y='Policy',
    #     hue='c_b', dodge=True, jitter=False,
    #     palette='Set2', marker='D', size=6, linewidth=0.8
    # )
    # plt.legend(title='c_b')
    # plt.xlabel('Total cost')
    # plt.tight_layout()
    # plt.show()
    # %%
    
    fig, ax = plt.subplots(1,3, figsize=(12, 4.5))
    timesteps = [1, 15, 40]
    for i, t in enumerate(timesteps):
        rgb = rgb_from_SIR_hsl(res_11.obses[0][t], l_min=0.25, l_max=0.75)
        ax[i].imshow(rgb)#, alpha=np.where((rgb.sum(axis=2))==0, 0.0, 1.0))
        ax[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax[i].set(xlabel=None, ylabel=None)
    ax[0].set_title('Early', fontsize=14)
    ax[1].set_title('Intermediate', fontsize=14)
    ax[2].set_title('Late', fontsize=14)
    plt.savefig("./figs/heatmaps.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)


    
    # %%
    pd.concat(dfs).groupby(['Policy', 'c_b'])['Total cost'].agg(['mean', 'std']).reset_index(drop=False).pivot(index='Policy',columns='c_b',values='mean').astype(int)
    # %%
    pd.concat(dfs).groupby(['Policy', 'c_b'])['Total cost'].agg(['mean', 'std']).reset_index(drop=False).pivot(index='Policy',columns='c_b',values='std').astype(int)
    # %%
    data = pd.concat(dfs)
    c_b = 0.02
    x1 = data.loc[(data.Policy==names[5]) & (data.c_b==c_b), 'Total cost']
    x2 = data.loc[(data.Policy==names[2]) & (data.c_b==c_b), 'Total cost']
    print(x1.mean(), x2.mean())
    from scipy import stats
    stats.ttest_rel(x1, x2)

    # %%
    for i, path in enumerate(paths):
        
        reses = res_repo[i]
        print(path)
        for res in reses:
            
            print(int(np.array([ic[-1] for ic in res.infect_cost]).mean()))
            print(int(np.array([bc[-1] for bc in res.block_cost]).mean()))

        

    #%% closure rate anova
    res_mo = res_mos[2]
    j = 8
    facility = city.facs.iloc[j]
    fac_copy = city.facs.copy()
    code, val = pd.factorize(fac_copy.xcoor)
    fac_copy['cluster'] = code
    for j,fac in enumerate(city.facs.index):
        ferm_vec = np.concatenate([np.array(e)[:,j] for e in res_mo.fermeture]).astype(int)
        fac_copy.loc[fac, 'ferme_rate'] = ferm_vec.mean()
    # %%
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    def one_way_anova(df, factor):
        model = smf.ols(f"ferme_rate ~ C({factor})", data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)   # typ=2 or typ=1 둘 다 가능
        return anova_table

    anova_locality = one_way_anova(fac_copy, "locality")
    anova_risk     = one_way_anova(fac_copy, "risk")
    anova_cluster  = one_way_anova(fac_copy, "cluster")

    print("=== ANOVA: locality ===")
    print(anova_locality)

    print("\n=== ANOVA: risk ===")
    print(anova_risk)

    print("\n=== ANOVA: cluster ===")
    print(anova_cluster)
    # %%
    
    model = smf.ols("ferme_rate ~ C(locality) * C(risk)", data=fac_copy).fit()
    anova_2way = sm.stats.anova_lm(model, typ=2)   # typ=2 많이 씀

    print(anova_2way)

    # %%
    ferm_mat = np.concatenate([np.array(e) for e in res_mo.fermeture]).astype(int)
    #%%
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    for i, res_mo in enumerate(res_mos[::-1]):
    
        for j,fac in enumerate(city.facs.index):
            ferm_vec = np.concatenate([np.array(e)[:,j] for e in res_mo.fermeture]).astype(int)
            fac_copy.loc[fac, 'ferme_rate'] = ferm_vec.mean()

        s= fac_copy.groupby(['risk', 'locality'])['ferme_rate'].mean()

        heatmap_data = s.unstack()
        df = pd.DataFrame(s).reset_index()
        df["locality"] = df["locality"].replace({
                            0.5: "Low",
                            5.0: "High",
                        })
        # im = ax.imshow(heatmap_data.values, aspect="auto", cmap='hot_r', vmin=0, vmax=0.8)
        im = sns.lineplot(df, x='risk', y='ferme_rate', 
                          hue='locality', style='locality', 
                          markersize=10, markeredgecolor='white', 
                          markeredgewidth=0,
                          markers={"Low": "o", "High": "s"},
                          palette={"Low":'tab:red', "High": 'tab:blue'},
                          ax=ax[i])

        # 축 tick 설정
        ax[i].set_xticks([0.25, 0.5])
        ax[i].set_yticks(np.arange(0.2, 0.8, 0.1))
        ax[i].set_xlim(0.15, 0.6)
        ax[i].set_xticklabels(['Low', 'High'])
        # ax[i].set_yticklabels(heatmap_data.index)
        ax[i].grid(axis='y', linewidth=0.8, alpha=0.5)
        ax[i].set_xlabel("Risk")
        ax[i].set_ylabel("Closure rate")
        ax[i].legend(title="Locality", loc="upper left")
        items = ['a','b','c']
        ax[i].set_title(f"$({items[i]})~c_e$ = {c_bs[::-1][i]:.2f}")

        # # 값 텍스트 표시
        # for i in range(heatmap_data.shape[0]):
        #     for j in range(heatmap_data.shape[1]):
        #         ax.text(j, i, f"{heatmap_data.values[i, j]:.3f}",
        #                 ha="center", va="center", color="white")

        # plt.colorbar(im, ax=ax)
        # plt.title("2x2 Heatmap")
        plt.tight_layout()
    plt.savefig("./figs/closure_rate.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
    # %%
    visits = where_to_go(city.block_visit(city.facs.index[res_mo.fermeture[0][16]]))
    temp = city.inds.copy()
    temp = temp.merge(visits, left_index=True, right_index=True)
    sns.scatterplot(temp.loc[temp.fid==20005], x='xcoor', y='ycoor', hue='fid')
    plt.xlim(0,100)
    plt.ylim(0,100)

    unique, counts = np.unique((np.concatenate([res_repo[0][-2].cohorts[i][-2].finfected for i in range(100)])), return_counts=True)


    # %%
    # %% for rl shuffle
    np.concatenate([np.array(res_repo[0][2].fermeture[j]) for j in range(100)]).mean(axis=0)
    unique, counts = np.unique((np.concatenate([res_repo[0][3].cohorts[i][-2].finfected for i in range(100)])), return_counts=True)
    #%% Fermeture와 obervation의 corrleation analysis
    def conditional_prob(ferm_vec, state_vec, ferm_value=1, state_value=1):
        # P(ferm_value | state_value ) = P(state_value and ferm_value) / P(state_value)
        joint_occurrence = np.sum((ferm_vec == ferm_value) & (state_vec == state_value))
        state_occurrence = np.sum(state_vec == state_value)
        
        if state_occurrence == 0:
            return 0.0  # P(state_value)가 0인 경우 처리
        
        cond_prob = joint_occurrence / state_occurrence
        return cond_prob


    fig = plt.figure(figsize=(6, 18))
    # fig.supxlabel("Common X label")
    # fig.supylabel("Common Y label")
    # fig.suptitle("Common Title")
    # 바깥 grid (1행 2열)
    gs = fig.add_gridspec(2, 2)#, width_ratios=[1, 1], height_ratios=[1, 1])
    rowi = -1
    coli = -1
    # ax = fig.add_axes([0, 0, 0.8, 0.8])  # 전체판 axis
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)

    # # 가운데 선(가로/세로 반)
    # ax.axvline(0.5, linewidth=2)
    # ax.axhline(0.5, linewidth=2)

    # # ✅ tick 위치 2개만 남기고 label을 low/high로 변경
    # ax.set_xticks([0.25, 0.75])
    # ax.set_xticklabels(["Low", "High"])

    # ax.set_yticks([0.25, 0.75])
    # ax.set_yticklabels(["Low", "High"])

    # # 보기 좋게 tick 위치(아래/왼쪽만)
    # ax.tick_params(bottom=True, left=True, top=False, right=False)

    # gs_ll_lr = gs[1, 0].subgridspec(3, 1)
    # gs_hl_lr = gs[1, 1].subgridspec(3, 1)
    # gs_ll_hr = gs[0, 0].subgridspec(3, 1)
    # gs_hl_hr = gs[0, 1].subgridspec(3, 1)

    env = EpiSimEnvironment(city, **env_params)

    for j in range(len(city.facs)):
        facility = city.facs.iloc[j]

        x = facility['xcoor']-env.observe_min_x#.clip(0, 1 - 1e-8)
        y = facility['ycoor']-env.observe_min_y#.clip(0, 1 - 1e-8)
        xi = (x/env.observe_span_x*env.nx)#.astype(int)
        yi = (y/env.observe_span_y*env.ny)#.astype(int)

        ferm_vec = np.concatenate([np.array(e)[1:,j] for e in res_mo.fermeture]).astype(int)
        inf_mats = np.concatenate([np.array(e)[:-1,1,:,:] for e in res_mo.obses]).clip(0,1).astype(int)
        rec_mats = np.concatenate([np.array(e)[:-1,2,:,:] for e in res_mo.obses]).clip(0,1).astype(int)
        # sus_mats = np.concatenate([np.array(e)[:,0,:,:] for e in res_mo.obses]).astype(int) 


        # corr_mat = np.nan_to_num(np.apply_along_axis(lambda x: np.corrcoef(ferm_vec, x)[0, 1], 0, inf_mats))
        corr_mat = np.nan_to_num(np.apply_along_axis(lambda x: conditional_prob(ferm_vec, x, 1, 1), 0, inf_mats))
        # corr_mat = np.nan_to_num(np.apply_along_axis(lambda x: conditional_prob(x, ferm_vec, 1, 0), 0, inf_mats))

        if j>=0 and j<3:
            rowi = 0
            coli = 1
            jprime = j
        if j>=3 and j<6:
            rowi = 1
            coli = 0
            jprime = j-3
        if j>=6 and j<9:
            rowi = 0
            coli = 0
            jprime = j-6
        if j>=9 and j<12:
            rowi = 1
            coli = 1
            jprime = j-9

        if j % 3 == 0:
            gs_sub = gs[rowi, coli].subgridspec(4, 1, height_ratios=[1,1,1,0.05], hspace=0.1)
            ax = fig.add_subplot(gs_sub[3,0])
            norm = mpl.colors.Normalize(vmin=0, vmax=1.25)
            cb = mpl.colorbar.ColorbarBase(
                ax,
                cmap=plt.cm.hot_r,
                norm=norm,
                orientation="horizontal"
            )
            cb.set_ticks(np.arange(0, 1.25, 0.5))
            pos = ax.get_position()  # 원래 위치
            scalex = 0.9
            ax.set_position([pos.x0+pos.width*((1-scalex)/2), pos.y0, 
                             pos.width * scalex, pos.height])
            ax.set_xlim(0,1)
            ax.tick_params(axis="both", labelsize=18)
            # if rowi == 1:
            #     if coli == 0:
            #         ax.set_xlabel('Low')
            #     else :
            #         ax.set_xlabel('High')
                

        
        ax = fig.add_subplot(gs_sub[jprime,0])
        
        im = ax.imshow(np.rot90(corr_mat), cmap='hot_r', vmin=0, vmax=1.25)
        ax.scatter(xi, env.ny-yi, marker="*", s=150, c="deepskyblue", edgecolors="k", linewidths=1.0)
    # tick과 label 제거
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set(xlabel=None, ylabel=None)
        # ✅ colorbar는 vmax=1.0까지만 표시 (legend 자체가 1.0에서 끝남)

    import matplotlib as mpl
    mpl.rcParams["svg.fonttype"] = "none"   # 텍스트를 path로 바꾸지 않고 글자로 유지(편집/검색 쉬움)

    plt.savefig("condi_prob.svg", bbox_inches="tight")    

    # ax[axi,axj].legend(title=None, fontsize=10)
    # plt.text(xi, env.ny-yi, facility.type, color='black')
    # %% 로지스틱 리그레션
    from sklearn.linear_model import LogisticRegression

    # 예시 입력
    # ferm_vec: (n,)
    # inf_mat: (n, k, l)
    # ferm_vec, inf_mat = ...

    n, k, l = inf_mats.shape

    coef_mat = np.zeros((k, l))

    # 각 (k, l) 위치마다 로지스틱 회귀 수행
    for i in range(k):
        for j in range(l):
            X = inf_mats[:, i, j].reshape(-1, 1)  # predictor (n, 1)
            y = ferm_vec                         # binary response (n,)

            # 로지스틱 회귀 (규제 제거: C를 크게)
            model = LogisticRegression(C=1e6, solver='lbfgs')
            model.fit(X, y)

            coef_mat[i, j] = model.coef_[0, 0]   # 단일 feature의 coefficient

    print(coef_mat)
    # %%
    plt.imshow(np.rot90(coef_mat), cmap='seismic', vmin=-5, vmax=5)
    plt.plot(xi, env.ny-yi, 'k*', ms=7)
    # plt.imshow(coef_mat, cmap='seismic', vmin=-5, vmax=5)
    # plt.plot(facility.ycoor, facility.xcoor, 'yx')
    # plt.text(10,10, facility.type, color='black')
    # %%
    x1 = list_rewards[2]
    x2 = list_rewards[5]
    print(x1.mean(), x2.mean())
    from scipy import stats
    stats.ttest_rel(x1, x2)
    # %% 현재 감염자 수
    reses = res_repo[0]
    fig, ax = plt.subplots(1,2,figsize=(10, 3.5))

    styles = ["--", "-.", ":", "-"]
    infects = []
    for i, res in enumerate(reses[2:6]):
        infected = res.cohort_time_series(state='I', aggregate='none')
        infected['count'] /= city.N/100
        sns.lineplot(infected, x='timestep', y='count', linestyle=styles[i], errorbar=None, ax=ax[0])
    
    ax[0].set_xticks(range(0,181,30))
    ax[0].set_ylabel('Infected individuals (%)')
    ax[0].set_xlabel('Time steps (days)')

    yticks = ax[0].get_yticks()
    for y in yticks:
        ax[0].axhline(y=y, color='lightgray', linestyle='-', linewidth=0.5, alpha=1.0)
    # plt.legend(np.repeat(names[2:6],2))
    plt.legend(names[2:6])

    #  폐쇄된 시설 수
    infects = []
    for i, res in enumerate(reses[2:6]):
        fermeture = res.ferme_time_series(aggregate='none')
        fermeture['prop'] = fermeture['count'] / city.facs.shape[0] * 100
        
        sns.lineplot(fermeture, x='timestep', y='prop', linestyle=styles[i], errorbar=None,ax=ax[1])

    ax[1].set_xticks(range(0,181,30))
    ax[1].set_ylabel('Closed places (%)')
    ax[1].set_ylim(30,100)
    ax[1].set_xlabel('Time steps (days)')

    yticks = ax[1].get_yticks()
    for y in yticks:
        ax[1].axhline(y=y, color='lightgray', linestyle='-', linewidth=0.5, alpha=1.0)
    # plt.legend(np.repeat(names[2:6],2))
    plt.legend(names[2:6])
    plt.savefig("./figs/plot_ninf_nferme.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.show()

    

    # %% 신규 감염자 수
    for res in reses[0:1]:
        infected = res.newinf_time_series(aggregate='none')
        sns.lineplot(infected, x='timestep', y='count', errorbar='sd')
    plt.legend(np.repeat(names,2))
    plt.show()

    # fermetures = []
    # for i, res in enumerate(reses):
    #     if i<2 or i>5:
    #         continue
    #     fermeture = res.ferme_time_series(aggregate='none')
    #     # fermeture['count'] = fermeture['count'].rolling(window=5).mean()
    #     fermeture['policy'] = names[i]
    #     fermetures.append(fermeture) 
    # ax2=sns.lineplot(pd.concat(fermetures), x='timestep', y='count', hue='policy', errorbar=None)

    # # plt.legend(np.repeat(names,2))    
    # plt.show()
    # %%
    stat = 0
    npersons = 0
    gengaps = []
    for envi in reses[0].cohorts:
        for cohort in envi:
            newbie = cohort.loc[(cohort.tinfected==0) & (cohort.spreader>0)]
            speaders = newbie['spreader'].values
            stat += cohort.loc[speaders].tinfected.sum()
            gengaps.append(cohort.loc[speaders].tinfected)
            npersons += len(newbie)
    print(stat, npersons)

    # %% 시설별 정책 변환 횟수
    res_counts = []
    run_lengths = []
    for res in reses[2:6]:
        counts = []
        run_dists = []
        for f in res.fermeture:
            count = 0
            last_closure = f[0]
            for closure in f:
                count += (closure != last_closure).sum()
                last_closure = closure
            counts.append(count)
            run_dist = compute_run_length_dist(pd.DataFrame(np.array(f)))
            run_dists.append(run_dist)
        res_counts.append(counts)
        run_lengths.append(pd.concat(run_dists))


    # %%

    # %%
    ferm = res_9.fermeture
    ferm_mo = pd.DataFrame({'env_'+str(i):
                            np.array(x[:250]).sum(axis=1) for i,x in enumerate(ferm[5:6])})
    infected = res_9.cohort_time_series(state='I', aggregate='none')
    sns.lineplot(infected.loc[infected.env==5], x='timestep', y='count', hue='env')
    sns.lineplot(ferm_mo)
    plt.show()
    # %%
    sns.lineplot(infected.loc[infected.env<7], x = 'timestep', y = 'count', hue = 'env')
    plt.show()
    # %%
    ts = 18
    ferms = []
    for e in ferm:
        ferms.append(e[ts])
    ferms = np.array(ferms)
    ferms.mean(axis=0)
    # %% GIF
    
    for i, name in enumerate(names):
        imager = ObsImager(reses[i].obses[31])
        imager.save_gif(path+name)


    # %%
    fig, ax = plt.subplots(1,3)   
    imager = Imager(196)
    imager.set_span(pd.concat([city.inds[['xcoor','ycoor']], 
                               city.facs[['xcoor','ycoor']]]))
    
    env=25
    timestep=10
    img_cohort, img_raw = imager.cohort_view(res_mo.cohorts[env][timestep], max_c=30)
    
    ax[0].imshow(np.rot90(img_cohort[0],1) , cmap='hot')    
    ax[1].imshow(np.rot90(img_cohort[1],1) , cmap='hot')    
    ax[2].imshow(np.rot90(img_cohort[2],1) , cmap='hot')
    ouvr_facs, ferm_facs = imager.fac_view(city.facs, res_mo.fermeture[env][timestep])
    
    # city.inds.plot.scatter(x='xcoor',y='ycoor', c='red')
    
    ouvr_facs.plot.scatter(x='xcoor',y='ycoor', ax=ax[0], 
                            s=35, 
                            c='none',
                            edgecolors='gray'   # 테두리 색상 지정
                         )
    ferm_facs.plot.scatter(x='xcoor',y='ycoor', ax=ax[0], 
                            s=35, 
                            c='none',
                            edgecolors='yellow'   # 테두리 색상 지정
                         )
    ferm_facs.plot.scatter(x='xcoor',y='ycoor', ax=ax[0], marker='x',
                            s=15, 
                            c='yellow',
                            # edgecolors='yellow'   # 테두리 색상 지정
    
                          )
    for axi in ax:
        axi.set_xlim(45*imager.ngrid/128,95*imager.ngrid/128)
        axi.set_ylim(60*imager.ngrid/128,10*imager.ngrid/128)
    plt.show()
    # %%
    fig, ax = plt.subplots(3,3) 
    for env in range(9):
        ax[env//3, env % 3].imshow(np.array(res_mo_det.fermeture[10+env]))
    plt.show()


    # %%
    plt.imshow(np.array(res_mo.fermeture).mean(axis=0), cmap='hot')
    plt.show()
    # %%

if __name__ == "__main__":

    #%%

    with open(city_name+'.pkl', 'rb') as f:
        city = pickle.load(f)    
    prefix = city_name
    folder = './results/'    
    # model = PPO.load("/home/ckang/projects/corona2025/logs/best_model/best_model.zip", env=driver.vec_env, device="cuda")

    if len(sys.argv) > 2 :
        seeds = [123456, 234567, 345678, 456789, 567890, 5858, 3846, 154248, 95464, 121212]
        
        res_dict = {}
        for model_string in sys.argv[1:]:
            means = []
            for seed in seeds:
                match = re.search(r"cb([0-9.]+)", model_string)
                env_params['coef_block'] = float(match.group(1))

                driver = VecDriver(city_base=city, env_params=env_params, num_envs=num_envs, seed=seed)
                model = PPO.load(model_string+".zip", env=driver.vec_env, device="cuda")
                driver.run(model, deterministic = True)

                means.append(np.array(driver.episode_rewards).sum(axis=1).mean())
                driver.close()
            res_dict[model_string] = means
        res_df = pd.DataFrame(res_dict)
        res_df['seed'] = seeds
        res_df.to_csv('model_comparison.csv', index=False)
        print(res_df.set_index('seed'))

    # %%
    else:
        # seed = 456789
        # seed = 154248
        # seed = 234567 # for risk 0.03
        seed = 5858 # for risk 0.05
        model_string = sys.argv[1]
        match = re.search(r"cb([0-9.]+)", model_string)
        env_params['coef_block'] = float(match.group(1))


        if env_params is not None:
            param_str = (
                            f"{prefix}_"
                            f"{','.join(np.array(env_params['mask_choice']).astype(str))}"
                            f"_rp{review_period}"
                            f"_ngrid{env_params['ngrid']}"
                            f"_len{env_params['max_epis_length']}"
                            f"_risk{env_params['risk_coexist']:.3f}"
                            f"_ex{100*env_params['ext_rate']:.3f}"
                            f"_cb{env_params['coef_block']:.2f}"
                            f"_ci{env_params['coef_infect']:.2f}"
                            f"_lag{env_params['lag']}"
                        )
        else:
            param_str = "default"
        folder += param_str

        os.makedirs(folder, exist_ok=True)

        driver = VecDriver(city_base=city, env_params=env_params, num_envs=num_envs, seed=seed)
        model = PPO.load(model_string+".zip", env=driver.vec_env, device="cuda")

        driver.run(RandomPolicy(driver.vec_env,  city, p=1.0, thr= -0.01, seed = seed,), enforce_zero=False)
        driver.archive(folder, 'r1.00')
        driver.reset()
        
        driver.run(model, deterministic = True, enforce_zero=False)
        driver.archive(folder, 'model')
        driver.reset()

        driver.run(model, deterministic = True, like = True, enforce_zero=False)
        driver.archive(folder, 'model.like')
        driver.reset()

        driver.run(RandomPolicy(driver.vec_env,  city, p=0.0, thr= -0.01, seed = seed,))
        driver.archive(folder, 'r0.00')
        driver.reset()

        driver.run(RandomPolicy(driver.vec_env, city, p=1.0, thr = 0.000, seed = seed,))
        driver.archive(folder, 'ld0.000')
        driver.reset()

        driver.run(RandomPolicy(driver.vec_env, city, p=1.0, thr = 0.001, seed = seed,))
        driver.archive(folder, 'ld0.001')
        driver.reset()

        driver.run(RandomPolicy(driver.vec_env, city, p=1.0, thr = 0.002, seed = seed,))
        driver.archive(folder, 'ld0.002')
        driver.reset()
        
        driver.run(RandomPolicy(driver.vec_env,  city, p=1.0, thr= 0.003, seed = seed,))
        driver.archive(folder, 'ld0.003')
        driver.reset()

        driver.run(RandomPolicy(driver.vec_env, city, p=1.0, thr = 0.004, seed = seed,))
        driver.archive(folder, 'ld0.004')
        driver.reset()

        driver.run(RandomPolicy(driver.vec_env, city, p=1.0, thr = 0.005, seed = seed,))
        driver.archive(folder, 'ld0.005')
        driver.reset()

        # driver.run(RandomPolicy(driver.vec_env, city, p=1.0, thr = 0.01, seed = seed,))
        # driver.archive(folder, 'ld0.01')
        # driver.reset()

        # driver.run(RandomPolicy(driver.vec_env, city, p=1.0, thr = 0.05, seed = seed,))
        # driver.archive(folder, 'ld0.05')
        # driver.reset()

        # driver.run(RandomPolicy(driver.vec_env, city, p=1.0, thr = 0.10, seed = seed,))
        # driver.archive(folder, 'ld0.10')
        # driver.reset()
        
        

    driver.close()


