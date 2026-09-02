# %% episim
from city_init import *

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from common_funcs import *
from time import time

class EpiSimSimpleEnvironment(gym.Env):
    def __init__(self, city, max_epis_length, ext_rate, mean_recover, risk_coexist, n_visit_tries, coef_block=0.01, coef_infect=1.0, 
                 gamma=0.99, r0=3):
        
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

        self.state_set = ('S', 'I', 'R')
        self.t = 0

        self.action_space = spaces.MultiBinary(1)
        
        # self.observation_space = spaces.Box(
        #     low=0, high=1, shape=(len(city.inds), len(self.state_set)), dtype=np.int32
        # )
        
        N = len(city.inds)
        num_attr_classes = len(self.state_set)
        self.city.boundingbox(0,0,1,1)
        # # 좌표 범위 지정
        
        

        # # xcoor, ycoor, state (as category code)        
        # x_low, x_high = city.hhs.xcoor.min(), city.hhs.xcoor.max()
        # y_low, y_high = city.hhs.ycoor.min(), city.hhs.ycoor.max()

        # # attr은 정수지만 float로 입력될 것이므로, 0~num_attr_classes-1 범위로 설정
        # low = np.array([x_low, y_low, 0] * N, dtype=np.float32)
        # high = np.array([x_high, y_high, num_attr_classes - 1] * N, dtype=np.float32)

        # self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        ##

        self.observation_space = spaces.Box(low=0, high=num_attr_classes, shape=(N, 3), dtype=np.float32)

        # self.observation_space = spaces.Box(
        #     low=0, high=1, shape=(len(city.inds), len(self.state_set)), dtype=np.int32
        # )

        self._init_simulation()

    def _init_simulation(self):
        self.cohort = self.city.inds.copy()
        self.cohort = self.cohort.astype({col: np.float32 for col in self.cohort.select_dtypes(include=['float64']).columns})
        self.cohort = self.cohort.astype({col: np.int32 for col in self.cohort.select_dtypes(include=['int64']).columns})
        self.last_links = None
        self._external_infection(self.cohort)
        self.infection_cost = 0
        self.block_cost = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self._init_simulation()
        return self._xy_observe(), {}

    def step(self, action):
        self.t += 1
        action = np.full(len(self.city.facs), action, dtype=np.int32)
        blocked = self.city.facs.index[action == 1]
        link = self._advance_cohort(blocked)
        self.last_links = link

        nblck_affil, nblck_visit = self.city.facs.loc[blocked, ['affiliated', 'visit']].sum()
        blocking_cost = nblck_affil + self.n_visit_tries * nblck_visit
        # n_internal_inf = len(link)
        # state_counts = self.cohort.state.value_counts()
        currently_infected = sum(self.cohort.state=='I')#sum(self.cohort=='I')
        # print(blocking_cost, currently_infected, self.t, self.epis_length)

        self.block_cost += blocking_cost
        self.infection_cost += currently_infected
        reward = -(self.coef_block * blocking_cost + self.coef_infect * currently_infected ** 2)
        
        terminated = bool((self.cohort.state=='R').mean() > ((self.r0-1)/self.r0))
        
        trunc = bool(self.t > self.epis_length - 1)
        # print('step:', terminated, trunc)
        # obs, reward, terminated, trunc, info
        
        # return self._onehot_observe(), reward, terminated, trunc, {}#trunc, {}
        return self._xy_observe(), reward, terminated, trunc, {}#trunc, {}
        # return (self._onehot_observe(), reward, False, False, {})#trunc, {}

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

    def _external_infection(self, cohort):
        externals = (np.random.rand(self.city.N) < self.ext_rate) & (cohort['state'] == 'S')
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
        recovered = infectees[np.random.rand(len(rprob)) < rprob]
        cohort.loc[recovered, 'state'] = 'R'
        cohort.loc[recovered, 'trecovered'] = 0

    def _get_infectees(self, cohort, realized_links, risks):
        realized_links = realized_links[realized_links != -1].reset_index()
        realized_links = realized_links.merge(risks * self.risk_coexist, left_on='fid', right_index=True)
        infectives = realized_links.merge(cohort[['state']], left_on='iid', right_index=True)

        infection_hubs = infectives[infectives['state'] == 'I'][['fid', 'iid']]
        infection_hubs.columns = ['fid', 'spreader']

        infectees = infection_hubs.merge(infectives, on='fid')
        infectees['expose'] = np.random.rand(len(infectees))
        infectees = infectees[(infectees['expose'] < infectees['risk']) & (infectees['state'] == 'S')]
        return infectees

    def _internal_infection(self, cohort, blocked):
        link_house = self.city.inds['hid'].rename('fid')
        link_affil = self.city.block_affil(blocked).rename('fid')

        affil_infectees = self._get_infectees(cohort, pd.concat([link_house, link_affil]),
                                              pd.concat([self.city.hhs['risk'], self.city.facs['risk']]))

        infectees = [affil_infectees] * (self.n_visit_tries + 1)
        for i in range(self.n_visit_tries):
            visit_links = where_to_go(self.city.block_visit(blocked))
            infectees[i + 1] = self._get_infectees(cohort, visit_links, self.city.facs['risk'])

        infectees = pd.concat(infectees)
        cohort.loc[infectees['iid'], 'state'] = 'I'
        cohort.loc[infectees['iid'], 'tinfected'] = 0
        cohort.loc[infectees['iid'], 'finfected'] = infectees['fid'].values.astype(np.int32)
        cohort.loc[infectees['iid'], 'spreader'] = infectees['spreader'].values.astype(np.int32)

        return infectees


# %%
class EpiSimDriver():
    def __init__(self, env, nrepeat = 100):
        self.env = env
        self.nrepeat = nrepeat
        self.cohorts = [None]
        self.links = [None]

    def run(self, model, verbose= False):
        env = self.env
        obs, info = env.reset()
        terminated = False
        trunc = False
        total_reward = 0
        discounting = 1
        
        
        while not (terminated or trunc): #numpy.bool_이라 is False에 안걸림. 아놔 -_-
            st = time()
            if isinstance(model, str):
                action = self.policy(obs, model)
            else:
                action = model.predict(obs, deterministic=False)[0]
            
            obs, reward, terminated, trunc, info = env.step(action)
            done = terminated or trunc
            total_reward += discounting*reward
            discounting *= self.env.gamma
            cohort, link = env.observe_detail()
            self.cohorts.append(cohort)
            self.links.append(link)
            print("elapsed {}: {:.3f}\t| # of infectees: {}\t| # of blocked: {}".format(
                env.t, time()-st, sum(cohort['state']=='I'), sum(action)))
        return total_reward
    
    def policy(self, obs, policy_tag):
        if policy_tag == 'naive':
            return np.zeros(len(self.env.city.facs))

        if policy_tag == 'blockade':
            return np.ones(len(self.env.city.facs))    

        if policy_tag == 'random':
            return np.random.choice(2,size=len(self.env.city.facs))
        

    def graph_at(self, t, scale = 1, width= 20, height = 20):
        facilities = self.env.city.facs
        cohort = self.cohorts[t]
        cohort = cohort.sort_values(by=['state'], ascending=False)

        links = self.links[t]
        links = links.loc[links['fid'].isin(facilities.index)]

        G = nx.DiGraph()

        G.add_nodes_from(cohort.index)
        pos_dict = {index: (x.xcoor, x.ycoor) for index, x in cohort.iterrows()}
        node_color = [COL_DICT[value] for index, value in cohort['state'].items()]
        node_size = [100*scale]*len(cohort)

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
        nx.draw(G, pos = pos_dict, ax = ax, node_color = node_color, node_size =node_size, edgelist = list(G.edges), 
            labels= node_labels, verticalalignment= 'top', font_size=20*scale)            
        # return(G, pos_dict, node_color, node_size, node_labels)
        plt.show()
    