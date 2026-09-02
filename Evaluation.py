# %%
import pickle
from episim import *
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm
import numpy as np
import seaborn as sns
# import seaborn_image as isns
import sys
import os

# %%
#     pickle.dump(city_base, f)

def make_env(city_base, env_params, seed=0):
    def _init():
        env = EpiSimEnvironment(city_base, **env_params) #coef_infect는 x^2에 대한 가중 효과만 반영. 기본 x가 infection cost에 들어감
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
        self.nact = env.action_space.shape[0]
        self.shape = (env.num_envs, *env.action_space.shape)
        self.rng = np.random.default_rng(seed)
        
    def predict(self, obs, deterministic=False):
        actions = np.zeros(self.shape)
        for i,o in enumerate(obs):
            if o[1].sum() < self.thrinf:
                continue
            ferme = self.rng.choice(self.nact, int(self.nact*self.p), replace=False)
            actions[i,ferme] = 1
        return(actions, None)
        # return(self.rng.choice(2, size=self.shape, 
        #                         p=[1-self.p,self.p]), None)

class VecDriver():

    def __init__(self, city_base, env_params, num_envs, seedi = 42):
        self.num_envs = num_envs
        self.env_params = env_params
        self.vec_env = SubprocVecEnv([make_env(city_base, env_params) for 
                                      i in range(num_envs)])
        self.reset()

    def reset(self):
        self.episode_rewards = [[] for _ in range(self.num_envs)]
        self.obses = [[] for _ in range(self.num_envs)]
        self.cohorts = [[] for _ in range(self.num_envs)]
        self.links = [[] for _ in range(self.num_envs)]
        self.fermeture = [[] for _ in range(self.num_envs)]

    def archive(self, folder, prefix):
        
        path = f"{folder}/{prefix}.pkl"
        with open(path, 'wb') as f:
            pickle.dump({'rewards': self.episode_rewards,
                         'obses': self.obses,
                         'cohort': self.cohorts,
                         'links': self.links,
                         'fermeture': self.fermeture}, f)
        print(f"Result saved to {path}")

        
    def run(self, model, seed = None, deterministic = False):
        episode_counts = np.zeros(self.num_envs, dtype=int)
        self.vec_env.seed(seed = seed)
        obs = self.vec_env.reset()
        dones = [False] * self.num_envs

        for _ in tqdm(range(self.env_params['max_epis_length'])):
            if np.sum(episode_counts>0) == self.num_envs:
                break
            actions, _ = model.predict(obs, deterministic=deterministic)
            # print(actions.shape)
            obs, rewards, dones, infos = self.vec_env.step(actions)
            
            for i in range(self.num_envs):
                
                if episode_counts[i] < 1: #각자 한번만 해라. 나머지는 기록 안한다.
                    self.episode_rewards[i].append(rewards[i])
                    self.obses[i].append(obs[i])
                    self.cohorts[i].append(infos[i]['cohort'])
                    self.links[i].append(infos[i]['link'])
                    self.fermeture[i].append(actions[i])
                    if dones[i]:
                        print(f"Env {i} - Episode {episode_counts[i]} reward: {sum(self.episode_rewards[i])}")
                        episode_counts[i] += 1
                
                
    def close(self):
        self.vec_env.close()  
    
class Reporter():
    def __init__(self, path):
        with open(path, 'rb') as f:
            res_dict = pickle.load(f)
        self.load_dict(res_dict)

    def load_dict(self, res_dict):
        self.episode_rewards = res_dict['rewards']
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
        """
        Return the time‑series of counts for the specified ``state`` across
        all environments in the batch.

        Parameters
        ----------
        state : str
            Value in the ``state`` column to count (default 'I').
        aggregate : {'sum', 'mean', 'none'}
            - 'sum'  : sum counts over environments (default)
            - 'mean' : mean count across environments (ignores missing steps)
            - 'none' : return raw 2‑D array (envs × timesteps)

        Returns
        -------
        np.ndarray
            If aggregate in {'sum', 'mean'} → 1‑D array of length = max timesteps.
            If aggregate == 'none'          → 2‑D array with shape (n_envs, max timesteps).
        """
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



num_envs = 32

env_params = {
    'max_epis_length': 180,
    'ext_rate': 1/1000/7,
    'mean_recover': 5,
    'risk_coexist': 0.03,
    'n_visit_tries': 2,
    'r0': 100,
    'coef_block': 0.06,
    'coef_infect': 0.1,
    'gamma': 1.0
}        
#%%
if 2==0: #그림을 그리자.
    # %%
    with open('city_small.pkl', 'rb') as f:
        city = pickle.load(f)
    env = EpiSimEnvironment(city, **env_params)
    # model = PPO.load("jc20_resnet18policy_ib-2.0_cb0.06_ci0.1_it5000000_2.zip", env=env, device="cuda")
    # model = PPO.load("jc20_resnet18policy_ib-2.0_cb0.15_ci0.1_it5000000_1.zip", env=env, device="cuda")
    # model = PPO.load("jc20_resnet18policy_ib-2.0_cb0.07_ci0.1_it5000000_3.zip", env=env, device="cuda")
    model = PPO.load("./logs/best_model/best_model.zip", env=env, device="cuda")
    
    # %% fig:dist_hhs
    use_paper_style(base_fontsize=10, font_family="DejaVu Serif", usetex=False,
                    figure_width=6, figure_height=4.5)       
    df_hh_fac = pd.concat([city.hhs, city.facs])
    df_hh_fac.loc[df_hh_fac.type=='household', 'type'] = 'Households'
    df_hh_fac.loc[df_hh_fac.type!='Households', 'type'] = 'Community centers'
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
    # tick과 label 제거
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set(xlabel=None, ylabel=None)
    ax.legend(title=None, fontsize=10)
    # 격자 추가
    ax.grid(True, linewidth=0.8, alpha=0.5)

    plt.savefig("./figs/dist_hhs.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
    # %% fig:visitors
    visit_facs = [20000, 20018, 20021, 20033]
    affil_facs = [20055, 20078]
    visitors = city.draw_visit(20018)
    affiliates = city.draw_visit(20078, mode='affiliated')

    # %% fig:visitors        
    ax = sns.scatterplot(city.hhs, 
                    x='xcoor',
                    y='ycoor',
                    s=15,
                    c='white',
                    edgecolor='black',
                    )    
    # tick과 label 제거
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set(xlabel=None, ylabel=None)
    # ax.legend(title=None)
    # 격자 추가
    ax.grid(True, linewidth=0.8, alpha=0.5)
    colors = {"Healthcare": "purple", 
            "School": "yellow", 
            "Workplace": "cyan", 
            "Venue": "mediumblue", 
            "Restaurant": "fuchsia",
            "Daily service": "orange"}
    df_visit = visitors.loc[visitors.fid.isin(visit_facs)].sort_values('fid')
    df_visit = df_visit.merge(city.inds, left_on='iid', right_index=True)
    df_visit = df_visit.merge(city.facs[['type']], left_on='fid', right_index=True)
    df_affil = affiliates.loc[affiliates.fid.isin(affil_facs)].sort_values('fid')
    df_affil = df_affil.merge(city.inds, left_on='iid', right_index=True)
    df_affil = df_affil.merge(city.facs[['type']], left_on='fid', right_index=True)
    
    sns.scatterplot(pd.concat([df_affil,df_visit]), x='xcoor', y='ycoor', 
                    s=20, hue='type_y',
                    palette=colors,
                    edgecolor='black',
                    )
    ax.legend(title=None, fontsize=10)
    
    sns.scatterplot(city.facs.loc[affil_facs+visit_facs], x='xcoor', y='ycoor', 
                s=40, hue='type',
                palette=colors,
                edgecolor='black',
                linewidth=1,
                marker='^',
                legend=False,
                )
    plt.savefig("./figs/visitors.pdf", format="pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
    # %% fig:heatmaps
    with open('city_local.pkl', 'rb') as f:
        city = pickle.load(f)
    env = EpiSimEnvironment(city, **env_params)
    driver = EpiSimDriver(env)
    driver.run('random00', verbose=True)                        

    # path = '/home/ckang/projects/corona2025/results/coef_block0.05.coef_infect0.1.ext_rate0.00014285714285714287.gamma0.999.max_epis_length365.mean_recover5.n_visit_tries2.r05.risk_coexist0.03/'
    # res_00 = Reporter(path+'r0.00'+'.pkl')
    # %%
    imager = ObsImager(driver.obses)
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
    path = '/home/ckang/projects/corona2025/results/coef_block0.04.coef_infect0.1.ext_rate0.00014285714285714287.gamma0.999.max_epis_length365.mean_recover5.n_visit_tries2.r010.risk_coexist0.03/'
    # path = '/home/ckang/projects/corona2025/results/coef_block0.05.coef_infect0.1.ext_rate0.00014285714285714287.gamma0.999.max_epis_length180.mean_recover5.n_visit_tries2.r010.risk_coexist0.02/'
    # path = '/home/ckang/projects/corona2025/results/coef_block0.03.coef_infect0.1.ext_rate0.00014285714285714287.gamma0.999.max_epis_length365.mean_recover5.n_visit_tries2.r010.risk_coexist0.03/'
    # path = '/home/ckang/projects/corona2025/results/coef_block0.05.coef_infect0.1.ext_rate0.00014285714285714287.gamma0.999.max_epis_length365.mean_recover5.n_visit_tries2.r010.risk_coexist0.03/'
    path = './results/coef_block0.05.coef_infect0.1.ext_rate0.00014285714285714287.gamma0.999.max_epis_length365.mean_recover5.n_visit_tries2.r010.risk_coexist0.03/'
    res_00 = Reporter(path+'r0.00'+'.pkl')
    res_25 = Reporter(path+'r0.25'+'.pkl')
    res_50 = Reporter(path+'r0.50'+'.pkl')
    res_75 = Reporter(path+'r0.75'+'.pkl')
    res_1c = Reporter(path+'r1.00'+'.pkl')
    res_mo = Reporter(path+'model'+'.pkl')
    reses = [res_00, res_25, res_50, res_75, res_1c, res_mo]
    names = ['Random 0.0', 'Random 0.25', 'Random 0.5', 'Random 0.75', 'Random 1.0', 'Model']
    try:
        res_mo_det = Reporter(path+'model.deterministic'+'.pkl')
        reses = [res_00, res_25, res_50, res_75, res_1c, res_mo, res_mo_det]
        names = ['Random 0.0', 'Random 0.25', 'Random 0.5', 'Random 0.75', 'Random 1.0', 'Model', 'Model (Deterministic)']
    except FileNotFoundError:
        print("No deterministic model found, continuing without it.")
    # %%
    list_rewards = []
    for res in reses:
        list_rewards.append(res.discounted_reward_dist(gamma = env_params['gamma']))
    df_rewards = pd.DataFrame(list_rewards, index = names).T
    df_rewards.boxplot()
    plt.show()
    # %%

    # %% 현재 감염자 수
    for res in reses:
        infected = res.cohort_time_series(state='I', aggregate='none')
        sns.lineplot(infected, x='timestep', y='count', errorbar='sd')
    plt.legend(np.repeat(names,2))
    plt.show()

    # %% 신규 감염자 수
    for res in reses:
        infected = res.newinf_time_series(aggregate='none')
        sns.lineplot(infected, x='timestep', y='count', errorbar='sd')
    plt.legend(np.repeat(names,2))
    plt.show()
    # %% 폐쇄된 시설 수
    fermetures = []
    for i, res in enumerate(reses):
        fermeture = res.ferme_time_series(aggregate='none')
        fermeture['policy'] = names[i]
        fermetures.append(fermeture) 
    sns.lineplot(pd.concat(fermetures), x='timestep', y='count', hue='policy', errorbar=None)
    # plt.legend(np.repeat(names,2))    
    plt.show()
    # %%
    ferm = res_mo_det.fermeture
    ferm_mo = pd.DataFrame({'env_'+str(i):
                            np.array(x[:250]).sum(axis=1) for i,x in enumerate(ferm[0:7])})

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
    with open('city_local.pkl', 'rb') as f:
        city = pickle.load(f)
    # %% GIF
    
    for i, name in enumerate(names):
        imager = ObsImager(reses[i].obses[31])
        imager.save_gif(path+name)


    # %%
    fig, ax = plt.subplots(1,3)   
    imager = Imager(196)
    with open('city_local.pkl', 'rb') as f:
        city = pickle.load(f)
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
    with open('city_local.pkl', 'rb') as f:
        city = pickle.load(f)
    
    folder = './results/'
    if env_params is not None:
        param_str = ".".join(f"{k}{v}" for k, v in sorted(env_params.items()))
    else:
        param_str = "default"
    folder = './results/'+param_str

    os.makedirs(folder, exist_ok=True)

    driver = VecDriver(city_base=city, env_params=env_params, num_envs=num_envs)
    model = PPO.load(sys.argv[1]+".zip", env=driver.vec_env, device="cuda")
    

    # %%
    seed = 42

    driver.run(RandomPolicy(driver.vec_env, city, 0.0), seed = seed)
    driver.archive(folder, 'r0.00')
    driver.reset()
    
    driver.run(RandomPolicy(driver.vec_env, city, 0.25), seed = seed)
    driver.archive(folder, 'r0.25')
    driver.reset()

    driver.run(RandomPolicy(driver.vec_env,  city, 0.5), seed = seed)
    driver.archive(folder, 'r0.50')
    driver.reset()

    driver.run(RandomPolicy(driver.vec_env,  city, 0.75), seed = seed)
    driver.archive(folder, 'r0.75')
    driver.reset()
    
    driver.run(RandomPolicy(driver.vec_env,  city, 1.0), seed = seed)
    driver.archive(folder, 'r1.00')
    driver.reset()

    driver.run(model, seed = seed, deterministic = False)
    driver.archive(folder, 'model')
    driver.reset()

    driver.run(model, seed = seed, deterministic = True)
    driver.archive(folder, 'model.deterministic')
    driver.reset()

    driver.close()


