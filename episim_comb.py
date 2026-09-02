# %% episim
from city_init import *

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from common_funcs import *
from time import time
from stable_baselines3 import PPO

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from collections import deque

import math
import colorsys

STATES = ('S', 'I', 'R')

def all_combinations(a: np.ndarray):
    """
    모든 부분집합(공집합 포함)을 리스트로 반환.
    a: 1차원 ndarray
    """
    n = a.size
    # 0..(2^n-1)까지의 비트마스크 만들기
    masks = (np.arange(1 << n)[:, None] & (1 << np.arange(n))) > 0
    return masks

class EpiSimEnvironment(gym.Env):
    def __init__(self, city, 
                 max_epis_length, ext_rate, mean_recover, risk_coexist, n_visit_tries, 
                 coef_block=0.01, coef_infect=1.0, 
                 gamma=0.99, r0=3, ngrid=96, seed = 42, mask_choice=[], oracle=False, lag=0):
        super().__init__()
        self.city = city
        self.epis_length = max_epis_length
        self.ext_rate = ext_rate
        self.mean_recover = mean_recover
        self.risk_coexist = risk_coexist
        self.n_visit_tries = n_visit_tries
        self.coef_block = coef_block
        self.coef_infect = coef_infect
        self.gamma = gamma
        self.r0 = r0
        self.like = False

        self.state_set = STATES
        self.t = 0
        self.lag = lag
        N = len(city.inds)
        
        # self.city.boundingbox(0,0,1,1)

        self.ngrid = ngrid  # grid size for image observation
        self._set_img_span(self.city.inds)        
        
        # self.observation_space = spaces.Box(low=0, high=num_attr_classes, shape=(N, 3), dtype=np.float32)
        # self.observation_space = spaces.Box(
        #     low=0, high=1, shape=(len(city.inds), len(self.state_set)), dtype=np.int32
        # )
        num_attr_classes = len(self.state_set)
        
        
        self.block_masks = all_combinations(self.city.facs.index)
        self.block_masks = self.block_masks[mask_choice] if len(mask_choice)>0 else self.block_masks
        N_ACTIONS = len(self.block_masks)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Dict({
            'image':spaces.Box(
                low=0,
                high=255,
                shape=(num_attr_classes, self.nx, self.ny),  # 채널, 높이, 너비
                dtype=np.uint8),
            'time': spaces.Box(low=0, high=self.epis_length, shape=(1,), dtype=np.int64)
        })
        # 마스크를 미리 만들어 재사용 (매 step마다 4096 배열 새로 만들지 않음)
        self._mask_all = np.ones(N_ACTIONS, dtype=bool)
        self._mask_only_zero = np.zeros(N_ACTIONS, dtype=bool)
        self._mask_only_zero[0] = True

        # self.observation_space = spaces.Dict({
        #     "image": spaces.Box(
        #         low=0,
        #         high=255,
        #         shape=(num_attr_classes, self.nx, self.ny),  # 채널, 높이, 너비
        #         dtype=np.uint8),
        #     "time": spaces.Box(low=0, high=self.epis_length, shape=(1,), dtype=np.int32)
        # })
        self.np_random = None
        self.seed = seed
        self.oracle = oracle
        # self._init_simulation()
        self._last_n_infected = 0
        
        # # 좌표 범위 지정
        
        

        # # xcoor, ycoor, state (as category code)        
        # x_low, x_high = city.hhs.xcoor.min(), city.hhs.xcoor.max()
        # y_low, y_high = city.hhs.ycoor.min(), city.hhs.ycoor.max()

        # # attr은 정수지만 float로 입력될 것이므로, 0~num_attr_classes-1 범위로 설정
        # low = np.array([x_low, y_low, 0] * N, dtype=np.float32)
        # high = np.array([x_high, y_high, num_attr_classes - 1] * N, dtype=np.float32)

        # self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        ##

        

        # self.observation_space = spaces.Box(
        #     low=0, high=1, shape=(len(city.inds), len(self.state_set)), dtype=np.int32
        # )

    def set_like(self, like):
        self.like = like        

    def _init_simulation(self):
        self.cohort = self.city.inds.copy()
        # self.cohort['xcoor'] = self.cohort['xcoor'].clip(0, self.city.ngrid - 1e-5)
        # self.cohort['ycoor'] = self.cohort['ycoor'].clip(0, self.city.ngrid - 1e-5)
        self.cohort = self.cohort.astype({col: np.float32 for col in self.cohort.select_dtypes(include=['float64']).columns})
        self.cohort = self.cohort.astype({col: np.int32 for col in self.cohort.select_dtypes(include=['int64']).columns})
        self.last_links = None
        # self._external_infection(self.cohort)
        externals = self.cohort.sample(int(0.005 * len(self.cohort)), 
                                       random_state=self.np_random,
                                       replace=False).index
        # print("externals", externals)
        
        self.cohort.loc[externals, 'state'] = 'I'
        self.cohort.loc[externals, 'tinfected'] = 0
        self.cohort.loc[externals, 'finfected'] = -1
        self.cohort.loc[externals, 'spreader'] = -1

        self.infection_cost = 0
        self.block_cost = 0
        self._last_n_infected = sum(self.cohort['state']=='I')

    def action_masks(self):
        # obs를 내부에 저장해두는 방식(아래 wrapper에서 저장해줄 예정)
        cond = (self._last_n_infected == 0)  # obs의 2번째 매트릭스 합이 0
        if cond:
            return self._mask_only_zero
        return self._mask_all


    def _set_img_span(self, cohort):
        self.observe_min_x = cohort.xcoor.min()
        self.observe_min_y = cohort.ycoor.min()
        self.observe_span_x = cohort.xcoor.max() - cohort.xcoor.min() + 1e-5
        self.observe_span_y = cohort.ycoor.max() - cohort.ycoor.min() + 1e-5
        self.observe_span = max(self.observe_span_x, self.observe_span_y)
        self.nx = np.ceil(self.ngrid * (self.observe_span_x/self.observe_span)).astype(int)
        self.ny = np.ceil(self.ngrid * (self.observe_span_y/self.observe_span)).astype(int)

        x = cohort['xcoor']-self.observe_min_x#.clip(0, 1 - 1e-8)
        y = cohort['ycoor']-self.observe_min_y#.clip(0, 1 - 1e-8)
        self.xi = (x/self.observe_span_x*self.nx).astype(int)
        self.yi = (y/self.observe_span_y*self.ny).astype(int)

    def reset(self, *, seed = None, options=None):
        super().reset(seed=self.seed)
        self.t = 0
        self._init_simulation()
        self.cohort_lag = deque([self.cohort.copy()], maxlen=self.lag+1)
        # img 범위 조정
        
        # return {'image':self._img_observe(), "time":self.t}, {'cohort': self.cohort.copy()}
        return {"image":self._img_observe(), "time":np.array([self.t])}, {'cohort': self.cohort.copy()}

    def check_terminated(self):
        # return bool((self.cohort.state=='R').mean() > ((self.r0-1)/self.r0))
        return bool(self.t > self.epis_length - 1)

    def step(self, action):
        self.t += 1
        block_mask = self.block_masks[action].copy()
        if self.like:
            self.np_random.shuffle(block_mask)
        if self.oracle: #일단 블락 없이 돌리고
            block_mask.fill(False)
        blocked = self.city.facs.index[block_mask]
        # print(blocked)
        link = self._advance_cohort(blocked)
        self.last_links = link

        if self.oracle: 
            #걸린 애들은 안걸리게 만들고
            self.cohort.loc[link['iid'], 'state'] = 'S'
            self.cohort.loc[link['iid'], 'tinfected'] = -1
            #걸린 곳만 블락했던 걸로
            blocked = link['fid'].unique()

        nblck_affil, nblck_visit = self.city.facs.loc[blocked, ['affiliated', 'visit']].sum()
        blocking_cost = nblck_affil + self.n_visit_tries * nblck_visit
        
        # n_internal_inf = len(link)
        # state_counts = self.cohort.state.value_counts()
        currently_infected = sum(self.cohort.state=='I')#sum(self.cohort=='I')
        # print(blocking_cost, currently_infected, self.t, self.epis_length)
        this_block = blocking_cost * self.coef_block
        this_infect = self.coef_infect * currently_infected ** 1.5 + currently_infected
        self.block_cost += this_block
        self.infection_cost += this_infect
        reward = -(this_block + this_infect)
        
        terminated = self.check_terminated()
        # terminated = bool((action.sum() == 0) and (currently_infected==0))
        
        trunc = False
        # print('step:', terminated, trunc)
        # obs, reward, terminated, trunc, info
        self.cohort_lag.append(self.cohort.copy())
        
        # return self._onehot_observe(), reward, terminated, trunc, {}#trunc, {}
        # return self._xy_observe(), reward, terminated, trunc, {}#trunc, {}
        return ({"image":self._img_observe(), "time":np.array([self.t])}, \
                reward, terminated, trunc, \
                {'cohort':self.cohort.copy(), 'link':self.last_links.copy()})#trunc, {}

        # return ({'image':self._img_observe(), "time":self.t}, \
        #         reward, terminated, trunc, \
        #         {'cohort':self.cohort.copy(), 'link':self.last_links.copy()})#trunc, {}

    def _advance_cohort(self, blocked):
        self._update_ts(self.cohort)
        self._recover(self.cohort)
        link = self._internal_infection(self.cohort, blocked)
        self._external_infection(self.cohort)
        return link

    def observe_detail(self):
        return self.cohort.copy(), self.last_links.copy()
        
    def _onehot_observe(self):
        return np.column_stack([
            (self.cohort['state'] == state).astype(int) for state in self.state_set
        ]).astype(np.int32)

    def _xy_observe(self):
        coords = self.cohort[['xcoor','ycoor']]
        states = self.cohort.state.astype('category').cat.codes
        return np.column_stack([coords, states])
    
    def _img_observe(self):
        # Create a grid image with shape (num_states, n, n)
        img = np.zeros((len(self.state_set), self.nx, self.ny), dtype=np.int32)
        obs_cohort = self.cohort_lag[0]
        self._last_n_infected = sum(obs_cohort['state']=='I')
        
        # df = pd.DataFrame({'xidx': xi, 'yidx': yi})
        # grid_counts = df.groupby(['yidx', 'xidx']).size().unstack(fill_value=0)
        # return(grid_counts)
        
        for idx, state in enumerate(self.state_set):
            mask = obs_cohort['state'] == state
            np.add.at(img[idx], (self.xi[mask], self.yi[mask]), 1)
        # return np.log2(img+1).astype(np.uint8)
        return img.astype(np.uint8)

    def _external_infection(self, cohort):
        externals = (self.np_random.random(self.city.N) < self.ext_rate) & (cohort['state'] == 'S')
        cohort.loc[externals, 'state'] = 'I'
        cohort.loc[externals, 'tinfected'] = 0
        cohort.loc[externals, 'finfected'] = -1
        cohort.loc[externals, 'spreader'] = -1

    def _update_ts(self, cohort):
        cohort.loc[cohort['state'] == 'I', 'tinfected'] += 1
        cohort.loc[cohort['state'] == 'R', 'trecovered'] += 1

    def _recover(self, cohort):
        infectees = cohort.index[cohort['state'] == 'I']
        tinfects = cohort.loc[infectees, 'tinfected'].values
        rprob = (stats.expon.cdf(tinfects, scale=self.mean_recover) - stats.expon.cdf(tinfects - 1, scale=self.mean_recover)) / (1 - stats.expon.cdf(tinfects - 1, scale=self.mean_recover))
        recovered = infectees[self.np_random.random(len(rprob)) < rprob]
        cohort.loc[recovered, 'state'] = 'R'
        cohort.loc[recovered, 'trecovered'] = 0

    def _get_infectees(self, cohort, realized_links, risks):
        realized_links = realized_links[realized_links != -1].reset_index()
        realized_links = realized_links.merge(risks * self.risk_coexist, left_on='fid', right_index=True)
        infectives = realized_links.merge(cohort[['state']], left_on='iid', right_index=True)

        infection_hubs = infectives[infectives['state'] == 'I'][['fid', 'iid']]
        infection_hubs.columns = ['fid', 'spreader']

        infectees = infection_hubs.merge(infectives, on='fid')
        infectees['expose'] = self.np_random.random(len(infectees))
        infectees = infectees[(infectees['expose'] < infectees['risk']) & (infectees['state'] == 'S')]
        return infectees

    def _internal_infection(self, cohort, blocked):
        link_house = self.city.inds['hid'].rename('fid')
        link_affil = self.city.block_affil(blocked).rename('fid')

        affil_infectees = self._get_infectees(cohort, pd.concat([link_house, link_affil]),
                                              pd.concat([self.city.hhs['risk'], self.city.facs['risk']]))

        infectees = [affil_infectees] * (self.n_visit_tries + 1)
        for i in range(self.n_visit_tries):
            visit_links = where_to_go(self.city.block_visit(blocked), rng=self.np_random)
            infectees[i + 1] = self._get_infectees(cohort, visit_links, self.city.facs['risk'])

        infectees = pd.concat(infectees)
        cohort.loc[infectees['iid'], 'state'] = 'I'
        cohort.loc[infectees['iid'], 'tinfected'] = 0
        cohort.loc[infectees['iid'], 'finfected'] = infectees['fid'].values.astype(np.int32)
        cohort.loc[infectees['iid'], 'spreader'] = infectees['spreader'].values.astype(np.int32)

        return infectees

class EpiSimEnvironmentSansExt(EpiSimEnvironment):
    def __init__(self, city, max_epis_length, ext_rate, mean_recover, risk_coexist, n_visit_tries, coef_block=0.01, coef_infect=1.0, 
                 gamma=0.99, r0=3):
        super().__init__(city, max_epis_length, ext_rate, mean_recover, risk_coexist, n_visit_tries, coef_block, coef_infect, gamma, r0)
    
    def check_terminated(self):
        return bool((self.cohort.state=='I').sum() == 0)

    def _advance_cohort(self, blocked):
        self._update_ts(self.cohort)
        self._recover(self.cohort)
        link = self._internal_infection(self.cohort, blocked)
        # self._external_infection(self.cohort)
        return link
        

class EpiSimSimpleEnvironment(EpiSimEnvironment):
    def __init__(self, city, max_epis_length, ext_rate, mean_recover, risk_coexist, n_visit_tries, coef_block=0.01, coef_infect=1.0, 
                 gamma=0.99, r0=3):
        super().__init__(city, max_epis_length, ext_rate, mean_recover, risk_coexist, n_visit_tries, coef_block, coef_infect, gamma, r0)
        self.action_space = spaces.MultiBinary(1)

    def step(self, action):
        # action shape: (1,), need to expand to (len(city.facs),)
        expanded_action = np.full(len(self.city.facs), action, dtype=np.int32)
        return super().step(expanded_action)




# %%
class EpiSimDriver():
    def __init__(self, env, nrepeat = 100, seed = 42):
        self.env = env
        self.nrepeat = nrepeat
        self.obses = [None]
        self.cohorts = [None]
        self.links = [None]
        self.fermeture = [None]
        self.rng = np.random.default_rng(seed)
        

    def run(self, model, deterministic=True, verbose= False, like=False, review_period = 0):
        env = self.env
        obs, _ = env.reset()
        terminated = False
        trunc = False
        self.total_reward = 0
        discounting = 1
        self.env.set_like(like)
        last_mask = None
        print(model)
        
        cont = 0
        while not (terminated or trunc): #numpy.bool_이라 is False에 안걸림. 아놔 -_-
            st = time()
            if cont == 0 or cont > review_period:
                # print(env.t, 'reviewing action...')
                cont = 1
                if isinstance(model, str):
                    # if (obs['image'][1].sum() < self.env.city.N*-0.01) and  \
                    #    (obs['image'][2].sum() > self.env.city.N*-0.01):
                    #     action = 0
                    # else:
                    action = self.policy(obs, model)
                    action_mask = env.block_masks[action]                
                elif isinstance(model, np.ndarray):
                    action = model    
                    action_mask = env.block_masks[action]
                else:
                    if (obs['image'][1].sum() <= np.ceil(self.env.city.N*0.000)):
                        action = 0
                    else:
                        action = model.predict(obs, deterministic=deterministic)[0]
                    action_mask = env.block_masks[action]
                    # if like:
                    #     action = self.policy(obs, 
                    #                          'random'+str((action_mask.sum()/len(action_mask)*100).astype(float)))
                    #     action_mask = env.block_masks[action]
            # print(action)
            obs, reward, terminated, trunc, info = env.step(action)

            change_cost = 0 if last_mask is None else (last_mask!=action_mask).sum()
            done = terminated or trunc
            self.total_reward += discounting*reward

            last_mask = env.block_masks[action]
            discounting *= self.env.gamma
            cohort = info['cohort']
            link = info['link']
            self.fermeture.append(action_mask)
            self.cohorts.append(cohort)
            self.links.append(link)
            self.obses.append(obs)
            if verbose: print("elapsed {}: {:.3f}\t| # of infectees: {}\t| # of blocked: {}".format(
                env.t, time()-st, sum(cohort['state']=='I'), sum(action_mask)))
            cont+=1
        return self.total_reward
    
    def policy(self, obs, policy_tag):
        # print(policy_tag)
        if policy_tag == 'naive':
            return np.zeros(len(self.env.city.facs))

        if policy_tag == 'blockade':
            return np.ones(len(self.env.city.facs))    

        if policy_tag.startswith('cond'):

            p1 = float(policy_tag.split('cond')[1])/100
            if (obs['image'][1].sum() <= np.ceil(self.env.city.N*p1)):
                return 0
            else: 
                return -1

        if policy_tag.startswith('random'):
            action = 0
            if (obs['image'][1].sum() < np.ceil(self.env.city.N*0.001)):
                return action
            
            p1 = float(policy_tag.split('random')[1])/100
            options = np.where(env.block_masks.sum(axis=1) == int(len(self.env.city.facs)*p1))[0]
            action = self.rng.choice(options, 1, replace = False)[0]
            
            # ferme = self.rng.choice(len(self.env.city.facs), int(len(self.env.city.facs)*p1),
            #                         replace = False)
            # action[ferme]=1
            return action

    def count_infected(self):
        return np.array([sum(cohort['state']=='I') for cohort in self.cohorts[1:]])

    def action_prob_on_ts(self, model, ts):
        obs = self.obses[ts]
        tensor_obs = model.policy.obs_to_tensor(obs)[0]
        tensor_prob = model.policy.get_distribution(tensor_obs).distribution.probs[0]
        return tensor_prob.cpu().detach().numpy()

    def plot_probs(self, tss: list, model):
        for ts in tss:
            plt.plot(self.action_prob_on_ts(model, ts), label=f'Timestep {ts}')
            plt.xlabel('Facility index')
            plt.legend()

    def graph_at(self, t, scale = 1, width= 20, height = 20, cohort_only = False):
        facilities = self.env.city.facs
        cohort = self.cohorts[t]
        cohort = cohort.sort_values(by=['state'], ascending=False)

        links = self.links[t]
        links = links.loc[links['fid'].isin(facilities.index)]

        fermeture = self.fermeture[t]
        closed_facs = facilities.loc[fermeture == 1]

        G = nx.DiGraph()

        G.add_nodes_from(cohort.index)
        pos_dict = {index: (x.xcoor, x.ycoor) for index, x in cohort.iterrows()}
        node_color = [COL_DICT[value] for index, value in cohort['state'].items()]
        node_size = [100*scale]*len(cohort)
        if (cohort_only==False) :
            G.add_nodes_from(facilities.index)
            pos_dict.update({index: (x.xcoor, x.ycoor) for index, x in facilities.iterrows()})
            node_color += [COL_DICT[value] for index, value in facilities['type'].items()]
            node_size += [200*scale]*len(facilities)
            node_labels = {index: x.type for index, x in facilities.iterrows()}
            
            link_infectee = links[['fid','iid']].drop_duplicates()
            G.add_edges_from(link_infectee.values)

            link_spreader = links[['spreader', 'fid']].drop_duplicates()
            G.add_edges_from(link_spreader.values)

        print('# of infected:',str(sum(cohort['state']=='I')))

        fig, ax = plt.subplots(figsize=(width,height))
        if (cohort_only==False) :
            nx.draw(G, pos = pos_dict, ax = ax, node_color = node_color, node_size =node_size, edgelist = list(G.edges), 
                labels= node_labels, verticalalignment= 'top', font_size=20*scale)            
            # return(G, pos_dict, node_color, node_size, node_labels)
            # Draw white circles over closed facilities
            for index, x in closed_facs.iterrows():
                ax.scatter(x.xcoor, x.ycoor, s=200*scale, c='white', marker='X', edgecolors='black', zorder=10)
        else :
            nx.draw(G, pos = pos_dict, ax = ax, node_color = node_color, node_size =node_size, edgelist = list(G.edges), 
                verticalalignment= 'top', font_size=20*scale)            

        plt.show()
# %%
def get_networkx(city, linktype):
    B = nx.Graph()

    # 사람 노드 추가
    person_nodes = [i for i in city.inds.index]
    B.add_nodes_from(person_nodes, bipartite='person')

    # 시설 노드 추가
    facility_nodes = [j for j in city.facs.index]
    B.add_nodes_from(facility_nodes, bipartite='facility')

    P = city.linkp[linktype]
    # 간선 추가: 방문 확률을 edge weight로 사용
    for i, row in P.iterrows():
        for j in P.columns:
            if P.loc[i, j] > 0.0:
                B.add_edge(i, j, weight=P.loc[i, j])
    return(B)       

import numpy as np
from scipy.sparse import csr_matrix
from sknetwork.visualization import svg_graph
from sknetwork.utils import *
from sknetwork.ranking import *

def analyze_network(P):
    biadjacency = csr_matrix(P)
    adjacency = bipartite2directed(biadjacency)
    
    return adjacency

# %%
if 1==0:
    # %% for check 1/R0
    # with open('city_local.pkl', 'rb') as f:
    #     city = pickle.load(f)        
    # env = EpiSimEnvironment(city, max_epis_length=365, ext_rate=1/1000, mean_recover=5,
    #                         risk_coexist = 0.03, n_visit_tries = 2, r0=100,
    #                         coef_block=0.04, coef_infect=0.1, gamma=0.999)
    # %%
    env.reset()
    # env.cohort.loc[env.cohort.sample(frac=0.6).index, 'state'] = 'R'
    naive_driver = EpiSimDriver(env)
    naive_goal = naive_driver.run('naive', verbose=True)    
    # %%
    x1=np.array([sum(c.state=='I') for c in naive_driver.cohorts[1:]])
    horizon = 180
    plt.plot(x1[:horizon])
# %%
if __name__ == "__main__":
    # %%
    import pickle
    # with open('base_city.pkl', 'rb') as f:
    # with open('city_jecheon_scale20.pkl', 'rb') as f:
    #     city = pickle.load(f)
    # with open('city_random.pkl', 'rb') as f:
    #     city = pickle.load(f)    
    with open('city_small.pkl', 'rb') as f:
        city = pickle.load(f)    
    env = EpiSimEnvironment(city, max_epis_length=180, ext_rate=1/1000, 
                            mean_recover=5,
                            risk_coexist = 0.03, 
                            n_visit_tries = 2, r0=10,
                            coef_block=0.02, coef_infect=0.1, gamma=1.0,
                            ngrid=32,
                            seed = 1550)        
    # env = EpiSimEnvironment(city, max_epis_length=180, ext_rate=1/1000/7, mean_recover=5,
    #                         risk_coexist = 0.02, n_visit_tries = 2, r0=10,
    #                         coef_block=0.02, coef_infect=0.1, gamma=0.999,
    #                         # ngrid = 36
    #                         )
    # %%
    env.reset()  
    ret = 0
    for _ in range(env.epis_length):
        # obs, rew, ter, trun, info = env.step(env.action_space.sample())
        obs, rew, ter, trun, info = env.step(-1)
        ret += rew
        print(sum(env.cohort.state=='I'))
    print(f"Episode return:{ret}, Block Cost:{env.block_cost}, Infect Cost:{env.infection_cost}")
    #0:11699, sample:13000, -1:13119
    
    # %%
    #52주 3년 차단               50 25 naive
    #coef_block=0.075 : -92299.50232550885 -91715.79185695387 -83044.71864613418
    #coef_block=0.07 : -91469.93907427888 -90963.21894880386 -90380.12595844168
    #                  -90538.88508213636 -88302.79804978849 -87885.7219527444
    #coef_block=0.06 : -85215.66127109587 -89656.20897175779 -88516.8170174391
    #coef_block=0.05 : random100 < -71957.2037523374  -78705.795351355   -83460.56146553713
    #coef_block=0.04 : -69449.35840556923 -77391.76341905104 -82476.22694321761
    #coef_block=0.025 : -56701.44858788229 -69886.91050018118 -86795.2181344513


    # %%
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    env.reset()
    
    #%%    
    img_raw = env._img_observe()
    img = img_raw.transpose(1, 2, 0)
    img = (img * (255.0 / img.max())).astype(np.uint8) if img.max() > 0 else img.astype(np.uint8)
    
    plt.imshow(np.rot90(img))
    plt.axis("off")
    plt.title("CnnPolicy Observation")
    plt.show()
    
    # %%
    
    # model = PPO.load("/home/ckang/projects/corona2025/rdcity_resnet18policy_ib0.0_cb0.05_ci0.1_it3000000_1.zip", env=env, device="cuda")
    
    # model = PPO.load("/home/ckang/projects/corona2025/learned_models/trash/small_resnet18_len180_risk0.03_cb0.06_ci0.1_it8000000_1.zip", env=env, device="cuda")
    
    # model = PPO.load("/home/ckang/projects/corona2025/learned_models/small_resnet18_len180_risk3.00_ex0.000_cb0.05_ci0.10_it8000000_1.zip", env=env, device="cuda")
    # model = PPO.load("/home/ckang/projects/corona2025/learned_models/small_resnet18_len180_risk0.03_cb0.06_ci0.1_it8000000_1.zip", env=env, device="cuda")
    # model = PPO.load("/home/ckang/projects/corona2025/local_resnet18policy_len365_risk0.03_ib-4.0_cb0.03_ci0.1_it8000000_1.zip", env=env, device="cuda")
    # model = PPO.load("/home/ckang/projects/corona2025/local_resnet18policy_len365_risk0.03_ib-4.0_cb0.04_ci0.1_it8000000_2.zip", env=env, device="cuda")
    
    
    
    # model = PPO.load("cnnpolicy1m.zip", env=env, device="cuda")

    # rand25_driver = EpiSimDriver(env)
    # rand_goal25 = rand25_driver.run('random25', verbose=True)
    
    

    
    # %%
    env = EpiSimEnvironment(city, max_epis_length=180, ext_rate=1/1000/7, 
                            mean_recover=5,
                            risk_coexist = 0.05, 
                            n_visit_tries = 2, r0=100,
                            coef_block=0.06, coef_infect=0.1, gamma=1.0,
                            seed = 5865, mask_choice=[], ngrid=50,oracle=False, lag=0)        
    from sb3_contrib import MaskablePPO
    # model = PPO.load("./logs/best_model/best_model.zip", env=env, device="cuda")
    ms_model = MaskablePPO.load("./learned_models/city_small__ngrid50_len180_risk0.050_ex0.014_cb0.06_ci0.10_lag0_it10000000_1.zip", env=env, device="cuda")

    model = PPO.load("./learned_models/city_small__ngrid50_len180_risk0.050_ex0.014_cb0.06_ci0.10_it10000000_1.zip", 
                     env=env, device="cuda")
    policies = [
        # 'cond0.0',
                #  'cond-1.1',
                # 'cond0.1',
                # 'cond0.2',
                # 'cond1.5',
        #         # 'cond2.5',
        #         # 'cond3.0',
                # 'cond5.0',
                # 'cond10.0',
                ms_model,                
                model,
                # 'cond100',
                # model,
                # 'random00', 
                # 'random10',
                # 'random20',
                # # 'random30',
                # # 'random40',
                # 'random50', 
                # # 'random60', 
                # 'random70', 
                # 'random80', 
                ]
    drivers = [EpiSimDriver(env, seed=1234) for _ in policies]

    for i, pol in enumerate(policies):
        env.reset()
        print(drivers[i].run(pol, verbose=False, like=False, review_period=0))
        print(env.block_cost, env.infection_cost)

    print([driver.total_reward for driver in drivers])
    # 365, r0.03, cb0.03 : [-23094.865730624966, -38957.75995199776, -25391.722266907203, -24396.38067215054, -29884.032989591065, -37807.15840623095]
    #                      [-22804.622000689214, -34660.98281631096, -31155.350022058155, -25301.585326463977, -30319.627025532354, -37954.70027059465]
    # 
    # %%
    # %%
    # sns.barplot(model_driver.action_prob_on_ts(model,3)-0.5)
    sns.barplot(model_driver.action_prob_on_ts(model,30)-0.5)
    imager = ObsImager(model_driver.obses[1:])
    imager.save_gif('model')

    # %%
    # imager = ObsImager(rand50_driver.obses[1:])
    # imager.save_gif('rand50.gif')

    # print(static_goal, model_goal, naive_driver.total_reward, rand50_driver.total_reward, rand100_driver.total_reward)


    xs = []
    for driver in drivers:
        xs.append(np.array([sum(c.state=='I') for c in driver.cohorts[1:]]))

    
    # print(naive_driver.total_reward, rand25_driver.total_reward, rand50_driver.total_reward, rand100_driver.total_reward)

    # x0=np.array([sum(c.state=='I') for c in static_driver.cohorts[1:]])
    # x1=np.array([sum(c.state=='I') for c in naive_driver.cohorts[1:]])
    # x5=np.array([sum(c.state=='I') for c in rand25_driver.cohorts[1:]])
    # x2=np.array([sum(c.state=='I') for c in rand50_driver.cohorts[1:]])
    # x6=np.array([sum(c.state=='I') for c in rand75_driver.cohorts[1:]])
    # x3=np.array([sum(c.state=='I') for c in rand100_driver.cohorts[1:]])
    # x4=np.array([sum(c.state=='I') for c in model_driver.cohorts[1:]])

    # # %% 누적
    # x1=np.array([sum(c.state!='S') for c in naive_driver.cohorts[1:]])
    # x5=np.array([sum(c.state!='S') for c in rand25_driver.cohorts[1:]])
    # x2=np.array([sum(c.state!='S') for c in rand50_driver.cohorts[1:]])
    # x3=np.array([sum(c.state!='S') for c in rand100_driver.cohorts[1:]])
    # x4=np.array([sum(c.state!='S') for c in model_driver.cohorts[1:]])
    # %%
    horizon = 180
    for i, x in enumerate(xs):
        plt.plot(xs[i][:horizon])
    plt.legend(policies)
    plt.show()
    # %%
    for i, x in enumerate(drivers[:1]):
        plt.plot(np.array([f.sum() for f in x.fermeture[1:]]))
    plt.legend(policies)
    plt.show()

    # %%    
    # plt.plot(np.array([f.sum() for f in naive_driver.fermeture[1:]]))
    # plt.plot(np.array([f.sum() for f in rand25_driver.fermeture[1:]]))
    # plt.plot(np.array([f.sum() for f in rand50_driver.fermeture[1:]]))
    # plt.plot(np.array([f.sum() for f in rand100_driver.fermeture[1:]]))
    # plt.plot(np.array([f.sum() for f in model_driver.fermeture[1:]]))
    
    # city.facs.loc[model_driver.fermeture[-1]==1]
    # city.facs.loc[model_driver.fermeture[1]==1]
    # %%
    from stable_baselines3 import PPO
    model = PPO.load("/home/ckang/projects/corona2025/logs/best_model/best_model.zip", env=env, device="cuda")
    model_driver = EpiSimDriver(env)
    model_goal = model_driver.run(model, verbose=True)
    # %%
    print(rand100_driver.total_reward,  
          rand50_driver.total_reward, 
          rand25_driver.total_reward, 
          model_driver.total_reward,
          naive_driver.total_reward)
    

    # %%
    
    plt.plot(rand100_driver.count_infected())
    plt.plot(rand50_driver.count_infected())
    plt.plot(rand25_driver.count_infected())
    plt.plot(model_driver.count_infected())
    plt.plot(naive_driver.count_infected())
    plt.legend(['rand100','rand50','rand25','model','naive'])
    # %%
    plt.plot(np.array([f.sum() for f in rand50_driver.fermeture[1:]]))
    plt.plot(np.array([f.sum() for f in rand25_driver.fermeture[1:]]))
    plt.plot(np.array([f.sum() for f in model_driver.fermeture[1:]]))
    plt.plot(np.array([f.sum() for f in naive_driver.fermeture[1:]]))
    # %%
    model_driver.graph_at(10, scale=0.25, width=10, height=10)

# %%
    plt.imshow(model_driver.fermeture[1:])
# %%
    env.city.facs.iloc[np.argsort(np.array(model_driver.fermeture[1:]).sum(axis=0))[::-1]]
    



# %%
    bigraph = get_networkx(city, 'visit')

# %%
    # pos = nx.spring_layout(bigraph, seed=42)
    # nx.draw(bigraph, pos)
    # ctr = nx.degree_centrality(bigraph)
    eig = nx.eigenvector_centrality(bigraph, weight='weight')

# %%
    adj = analyze_network(city.linkp['visit'].values)
    # pagerank.argsort()[::-1]
# %%

# %%

