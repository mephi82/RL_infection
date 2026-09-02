import gymnasium as gym
from gymnasium import spaces
import numpy as np
from common_funcs import *

class EpiSim():
    def __init__(self, city, nrepeat, ext_rate, mean_recover, risk_coexist, n_visit_tries):
        self.city = city
        self.nrepeat = nrepeat
        self.cohorts = [None]*nrepeat
        self.links = [None]*nrepeat
        self.ext_rate = ext_rate
        self.mean_recover = mean_recover
        self.risk_coexist = risk_coexist
        self.state_set = ('S','I','R')
        
        
        self.n_visit_tries = n_visit_tries
        # self.linkps = [None]*nrepeat
        self.cohort = self.city.inds.copy()
        self.cohorts[0] = self.cohort.copy()
        self.external_infection(self.cohort)
    
    def copy(self):
        return copy.copy(self)
    
    def onehot_observe(self):
        return np.column_stack([(self.cohort['state']==state).values.astype(int) for state in self.state_set])
        
    def graph_at(self, t, scale = 1, width= 20, height = 20):
        facilities = self.city.facs
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


    def external_infection(self, cohort):
        externals = (np.random.rand(self.city.N)<self.ext_rate) & (cohort['state']=='S')
        cohort.loc[externals, 'state'] = 'I'
        cohort.loc[externals, 'tinfected'] = 0
        cohort.loc[externals, 'finfected'] = -1
        cohort.loc[externals, 'spreader'] = -1

    def update_ts(self, cohort):
        cohort.loc[cohort['state']=='I', 'tinfected'] += 1
        cohort.loc[cohort['state']=='R', 'trecovered'] += 1

    
    def recover(self, cohort):
        infectees = cohort.index[cohort['state']=='I']
        tinfects = cohort.loc[infectees, 'tinfected'].values
        rprob = (stats.expon.cdf(tinfects, scale = self.mean_recover) - stats.expon.cdf(tinfects-1, scale = self.mean_recover))/(1-stats.expon.cdf(tinfects-1, scale = self.mean_recover))
        recovered = infectees[np.random.rand(len(rprob))<rprob]
        cohort.loc[recovered, 'state'] = 'R'
        cohort.loc[recovered, 'trecovered'] = 0
        
    def get_infectees2(self, cohort, went_to, risk_factor):
        st = time()
        went_to = went_to.drop(-1,axis=1)
        A = csr_matrix(went_to.values)
        
        R = diags((risk_factor*city_base.facs.loc[went_to.columns,'risk']).values)
        realized_links = A.dot(R).dot(A.transpose())
        realized_links = realized_links[np.where((sim.cohorts[0]['state']=='I'))[0]]
        infected = realized_links.data > np.random.rand(realized_links.nnz)
        infectors, infectees = realized_links.nonzero()
        
        spreader = went_to.iloc[infectors[infected]].index
        infected = went_to.iloc[infectees[infected]].index
        fid = went_to.idxmax(axis=1).loc[spreader].reset_index()
        fid['infected']=infected.values
        print(time()-st)
        return(fid)

    def get_infectees(self, cohort, realized_links, risks):
        # st = time()

        realized_links = realized_links[realized_links!=-1].reset_index()
        realized_links = realized_links.merge(risks*self.risk_coexist, left_on='fid', right_index= True)
        infectives = realized_links.merge(cohort[['state']], left_on='iid', right_index=True)#.reset_index(drop=True)
        
        infection_hubs = infectives[infectives['state']=='I'][['fid', 'iid']]#.drop_duplicates(subset=['fid'])
        #확진자 여러명이 방문하면 감염 기회도 여러번
        
        infection_hubs.columns = ['fid','spreader']

        infectees = infection_hubs.merge(infectives, on = 'fid')#.drop_duplicates(subset=['iid'])
        infectees['expose'] = np.random.rand(len(infectees))
        infectees = infectees.loc[infectees['expose']<infectees['risk']]
        infectees = infectees.loc[infectees['state']=='S']
        
        # print(time()-st)
        return(infectees)

#      

    def internal_infection(self, cohort, blocked):# link_affil, link_visit, link_house):

        # realized_links = pd.concat([link_house,link_affil,link_visit])
        # infectees = self.get_infectees(cohort, realized_links, 
        #                                pd.concat([self.city.hhs['risk'],self.city.facs['risk']]))
        
        # ngroups_affil = (self.city.facs['affiliated']/self.max_coexisst).apply(np.ceil)
        # ngroups_visit = (self.city.facs['visit']/self.max_coexisst).apply(np.ceil)
        # st = time()
        link_house = self.city.inds['hid'].rename('fid')
        link_affil = self.city.block_affil(blocked).rename('fid')
        
        affil_infectees = self.get_infectees(cohort, pd.concat([link_house,link_affil]), 
                                       pd.concat([self.city.hhs['risk'],self.city.facs['risk']]))
        # print(time()-st)
        infectees = [affil_infectees]*(self.n_visit_tries+1)
        # 링크를 합쳐서 하면 서로 다른 시간대에 같은 장소에 가더다로 같은 시간대에 간 것 처럼 취급되어(시간대 정보가 사라지므로) 감염이 늘어남.
        for i in range(self.n_visit_tries):
            infectees[i+1] = self.get_infectees(cohort, where_to_go(self.city.block_visit(blocked)), self.city.facs['risk'])
            # print(time()-st)
        infectees = pd.concat(infectees)
        

        cohort.loc[infectees['iid'], 'state'] = 'I'
        cohort.loc[infectees['iid'], 'tinfected'] = 0
        cohort.loc[infectees['iid'], 'finfected'] = infectees['fid'].values
        cohort.loc[infectees['iid'], 'spreader'] = infectees['spreader'].values
        # print(time()-st)
        return(infectees)        
        
    def advance_cohort(self, blocked):
        
        self.update_ts(self.cohort)
        self.recover(self.cohort)
        # blocked = self.get_blocked(cohort)
        
        infection_links = self.internal_infection(self.cohort, blocked)
                                       # self.city.block_affil(blocked).rename('fid'),
                                       # where_to_go(self.city.block_visit(blocked)),
                                       # self.city.inds['hid'].rename('fid'))
        # print(infection_links)
        self.external_infection(self.cohort)
        
        return(infection_links)
    
    def get_blocked(self):
        return(None)
    
    def run(self, verbose = True):
        for t in range(1, self.nrepeat):
            st = time()
            link = self.advance_cohort(self.get_blocked())
            self.cohorts[t] = self.cohort.copy()
            self.links[t] = link
            if verbose: print("elapsed ", t, ": ", time()-st, "\t| # of infectees: ", sum(self.cohort['state']=='I'))


class EpiSimEnvironment(gym.Env):
    def __init__(self, episim, coef_block=0.01, coef_infect=1.0, gamma=0.99):
        self._episim = episim
        self._coef_block = coef_block
        self._coef_infect = coef_infect
        self._gamma = gamma
        self.t = 0
        self.infection_cost = 0
        self.block_cost = 0

        self.action_space = spaces.MultiBinary(len(episim.city.facs))
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(episim.city.inds), len(episim.state_set)),
            dtype=np.int32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._episim = EpiSim(
            self._episim.city,
            self._episim.nrepeat,
            self._episim.ext_rate,
            self._episim.mean_recover,
            self._episim.risk_coexist,
            self._episim.n_visit_tries
        )
        self.t = 0
        self.infection_cost = 0
        self.block_cost = 0

        obs = self._episim.onehot_observe().astype(np.int32)
        return obs, {}

    def step(self, action):
        self.t += 1
        blocked = self._episim.city.facs.index[action == 1]
        nblck_affil, nblck_visit = self._episim.city.facs.loc[blocked, ['affiliated', 'visit']].sum()
        blocking_cost = nblck_affil + self._episim.n_visit_tries * nblck_visit

        link = self._episim.advance_cohort(blocked)
        n_internal_inf = len(link)

        self._episim.cohorts[self.t] = self._episim.cohort.copy()
        self._episim.links[self.t] = link

        self.block_cost += blocking_cost
        self.infection_cost += n_internal_inf
        reward = -(self._coef_block * blocking_cost + self._coef_infect * n_internal_inf)

        done = self.t > self._episim.nrepeat - 2
        obs = self._episim.onehot_observe().astype(np.int32)

        return obs, reward, done, False, {}


class EpiSimEnvironmentAAO(gym.Env):
    def __init__(self, episim, coef_block=0.01, coef_infect=1.0, gamma=0.99):
        self._episim = episim
        self._coef_block = coef_block
        self._coef_infect = coef_infect
        self._gamma = gamma
        self.t = 0
        self.infection_cost = 0
        self.block_cost = 0

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(episim.city.inds), len(episim.state_set)),
            dtype=np.int32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._episim = EpiSim(
            self._episim.city,
            self._episim.nrepeat,
            self._episim.ext_rate,
            self._episim.mean_recover,
            self._episim.risk_coexist,
            self._episim.n_visit_tries
        )
        self.t = 0
        self.infection_cost = 0
        self.block_cost = 0

        obs = self._episim.onehot_observe().astype(np.int32)
        return obs, {}

    def step(self, action):
        self.t += 1
        blocked = self._episim.city.facs.index[np.random.sample(len(self._episim.city.facs)) < action]

        nblck_affil, nblck_visit = self._episim.city.facs.loc[blocked, ['affiliated', 'visit']].sum()
        blocking_cost = nblck_affil + self._episim.n_visit_tries * nblck_visit

        link = self._episim.advance_cohort(blocked)
        n_internal_inf = len(link)

        self._episim.cohorts[self.t] = self._episim.cohort.copy()
        self._episim.links[self.t] = link

        self.block_cost += blocking_cost
        self.infection_cost += n_internal_inf
        reward = -(self._coef_block * blocking_cost + self._coef_infect * n_internal_inf)

        done = self.t > self._episim.nrepeat - 2
        obs = self._episim.onehot_observe().astype(np.int32)

        return obs, reward, done, False, {}
