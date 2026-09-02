# %% episim
from city_init import *

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from common_funcs import *
from time import time

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import math
import colorsys


class ObsImager():
    def __init__(self, obses):
        self.obses = obses

    def rgb_image(self, obs):
        green = obs[0] #S
        red = obs[1] #I
        blue = obs[2] #R
        saturation = obs.sum(axis=0)
        saturation = saturation/saturation.max() * 255
        cell_max =  obs.max(axis=0)
        cell_max[cell_max==0]=255
        red = red / cell_max * saturation
        green = green / cell_max * saturation
        blue = blue / cell_max * saturation

        red[cell_max==255] = 255
        blue[:,:] = 255
        green[cell_max==255] = 255
        img = np.stack([red, green, blue], axis=-1)
        
        
        return np.rot90(img.astype(np.uint8))


    def save_gif(self, filename):
        # PIL Image 객체로 변환
        pil_frames = []
        for t, obs in enumerate(self.obses):
            img = Image.fromarray(rgb_from_SIR_hsl(obs))
            # Draw 객체 생성
            draw = ImageDraw.Draw(img)

            # 폰트 설정 (폰트 파일 필요, 예: Arial.ttf)
            font = ImageFont.load_default()

            # 텍스트 위치, 내용, 색상 지정
            draw.text((10, 70), "ts="+str(t), font=font, fill=(255, 255, 255))

            # 결과 저장
            pil_frames.append(img)

        # GIF 저장 (loop=0이면 무한 반복)
        pil_frames[0].save(
            filename+".gif",
            save_all=True,
            append_images=pil_frames[1:],
            duration=100,   # 프레임당 200ms
            loop=0
        )
    
    def action_prob_on_ts(self, model, env, ts):
        obs = self.obses[env][ts]
        tensor_obs = model.policy.obs_to_tensor(obs)[0]
        tensor_prob = model.policy.get_distribution(tensor_obs).distribution.probs[0]
        return tensor_prob.cpu().detach().numpy()

    def plot_probs(self, tss: list, env, model):
        for ts in tss:
            plt.plot(self.action_prob_on_ts(model, env, ts), label=f'Timestep {ts}')
            plt.xlabel('Facility index')


def use_paper_style(
    base_fontsize=10,           # 논문용 기본 폰트(10~11pt 권장)
    font_family="DejaVu Serif", # 라텍스와 톤을 맞추려면 serif 권장(Computer Modern 계열 사용 시 usetex=True)
    usetex=False,               # LaTeX 렌더링을 쓸지 여부(True면 TeX 설치 필요)
    figure_width=3.5,           # 단일 칼럼 폭(inch). 2-column이면 7.2 정도
    figure_height=2.6,          # 종횡비에 맞춰 조정(예: 3.5x2.6, 7.2x4.6 등)
):
    # 폰트/텍스트
    mpl.rcParams.update({
        "font.size": base_fontsize,
        "font.family": font_family,
        "mathtext.fontset": "dejavuserif",  # usetex=False일 때 수식 폰트
        "text.usetex": usetex,
        
        # PDF/PS 폰트 타입: 42(Truetype) -> 일러스트/인크스케이프 호환성↑
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # 선/마커/격자
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.grid": False,
        "axes.linewidth": 0.8,

        # 축/틱/라벨
        "axes.titlesize": base_fontsize + 1,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,

        # 범례
        "legend.fontsize": base_fontsize - 1,
        "legend.frameon": False,

        # 저장 기본
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        # raster 요소(heatmap 등) 있을 때 해상도. 벡터 요소엔 영향 없음
        "savefig.dpi": 300,
    })

    # 기본 그림 크기 설정
    plt.rcParams["figure.figsize"] = (figure_width, figure_height)

# (a,b,c) → 각도(0~360°), 합 0은 NaN
def abc_to_hue_deg(a, b, c):
    a, b, c = map(np.asarray, (a, b, c))
    s = a + b + c

    # 비율 (합=0은 0으로)
    pa = np.divide(a, s, out=np.zeros_like(a, dtype=float), where=s!=0)
    pb = np.divide(b, s, out=np.zeros_like(b, dtype=float), where=s!=0)
    pc = np.divide(c, s, out=np.zeros_like(c, dtype=float), where=s!=0)

    # 꼭짓점: a(0°) b(120°) c(240°)
    xa, ya =  1.0, 0.0
    xb, yb = -0.5,  np.sqrt(3)/2
    xc, yc = -0.5, -np.sqrt(3)/2

    x = pa*xa + pb*xb + pc*xc
    y = pa*ya + pb*yb + pc*yc

    theta = np.degrees(np.arctan2(y, x))
    hue_deg = (theta + 360) % 360
    hue_deg = np.where(s==0, np.nan, hue_deg)  # 합 0 → NaN
    return hue_deg, s

# RGB 배열만 얻고 싶을 때
def rgb_from_SIR_hsl(mat3d, sat=1.0, l_min=0.3, l_max=0.6):
    S, I, R = mat3d
    hue_deg, mag = abc_to_hue_deg(I, S, R)

    # Lightness: 합(mag)을 전역 정규화해 l_min~l_max로
    m0, m1 = np.nanmin(mag), np.nanmax(mag)
    if m1 > m0:
        L = (mag - m0) / (m1 - m0)
    else:
        L = np.zeros_like(mag, dtype=float)
    L = l_min + L * (l_max - l_min)

    H = (np.mod(hue_deg, 360) / 360.0).astype(float)  # [0,1]
    S = np.full_like(L, float(sat), dtype=float)

    # HLS → RGB (colorsys는 (H,L,S) 순서!)
    hls = np.stack([H, L, S], axis=-1)
    flat = hls.reshape(-1, 3)
    rgb_flat = [colorsys.hls_to_rgb(*t) if not np.isnan(t[0]) else (0,0,0) for t in flat]
    rgb = np.array(rgb_flat, dtype=float).reshape(hls.shape)
    return np.rot90(rgb*255).astype(np.uint8)


class EpiSimEnvironment(gym.Env):
    def __init__(self, city, max_epis_length, ext_rate, mean_recover, risk_coexist, n_visit_tries, 
                 coef_block=0.01, coef_infect=1.0, 
                 gamma=0.99, r0=3, ngrid=96):
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

        self.state_set = ('S', 'I', 'R')
        self.t = 0
        N = len(city.inds)
        
        # self.city.boundingbox(0,0,1,1)

        self.ngrid = ngrid  # grid size for image observation
        self._set_img_span(self.city.inds)        
        
        # self.observation_space = spaces.Box(low=0, high=num_attr_classes, shape=(N, 3), dtype=np.float32)
        # self.observation_space = spaces.Box(
        #     low=0, high=1, shape=(len(city.inds), len(self.state_set)), dtype=np.int32
        # )
        num_attr_classes = len(self.state_set)
        self.action_space = spaces.MultiBinary(len(city.facs))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(num_attr_classes, self.nx, self.ny),  # 채널, 높이, 너비
            dtype=np.uint8
        )

        self._init_simulation()
        
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

        

    def _init_simulation(self):
        self.cohort = self.city.inds.copy()
        # self.cohort['xcoor'] = self.cohort['xcoor'].clip(0, self.city.ngrid - 1e-5)
        # self.cohort['ycoor'] = self.cohort['ycoor'].clip(0, self.city.ngrid - 1e-5)
        self.cohort = self.cohort.astype({col: np.float32 for col in self.cohort.select_dtypes(include=['float64']).columns})
        self.cohort = self.cohort.astype({col: np.int32 for col in self.cohort.select_dtypes(include=['int64']).columns})
        self.last_links = None
        # self._external_infection(self.cohort)
        externals = self.cohort.sample(int(self.ext_rate * 7 * len(self.cohort)), replace=False).index
        
        self.cohort.loc[externals, 'state'] = 'I'
        self.cohort.loc[externals, 'tinfected'] = 0
        self.cohort.loc[externals, 'finfected'] = -1
        self.cohort.loc[externals, 'spreader'] = -1

        self.infection_cost = 0
        self.block_cost = 0

    def _set_img_span(self, cohort):
        self.observe_min_x = cohort.xcoor.min()
        self.observe_min_y = cohort.ycoor.min()
        self.observe_span_x = cohort.xcoor.max() - cohort.xcoor.min() + 1e-5
        self.observe_span_y = cohort.ycoor.max() - cohort.ycoor.min() + 1e-5
        self.observe_span = max(self.observe_span_x, self.observe_span_y)
        self.nx = np.ceil(self.ngrid * (self.observe_span_x/self.observe_span)).astype(int)
        self.ny = np.ceil(self.ngrid * (self.observe_span_y/self.observe_span)).astype(int)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self._init_simulation()
        # img 범위 조정
        
        # return self._xy_observe(), {}
        
        


        return self._img_observe(), {}

    def step(self, action):
        self.t += 1
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
        reward = -(self.coef_block * blocking_cost + self.coef_infect * currently_infected ** 1.5 + currently_infected)
        
        terminated = bool((self.cohort.state=='R').mean() > ((self.r0-1)/self.r0))
        # terminated = bool((action.sum() == 0) and (currently_infected==0))
        
        trunc = bool(self.t > self.epis_length - 1)
        # print('step:', terminated, trunc)
        # obs, reward, terminated, trunc, info
        
        # return self._onehot_observe(), reward, terminated, trunc, {}#trunc, {}
        # return self._xy_observe(), reward, terminated, trunc, {}#trunc, {}
        return (self._img_observe(), \
                reward, terminated, trunc, \
                {'cohort':self.cohort.copy(), 'link':self.last_links.copy()})#trunc, {}

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
        x = self.cohort['xcoor']-self.observe_min_x#.clip(0, 1 - 1e-8)
        y = self.cohort['ycoor']-self.observe_min_y#.clip(0, 1 - 1e-8)
        xi = (x/self.observe_span_x*self.nx).astype(int)
        yi = (y/self.observe_span_y*self.ny).astype(int)
        # df = pd.DataFrame({'xidx': xi, 'yidx': yi})
        # grid_counts = df.groupby(['yidx', 'xidx']).size().unstack(fill_value=0)
        # return(grid_counts)
        
        for idx, state in enumerate(self.state_set):
            mask = self.cohort['state'] == state
            np.add.at(img[idx], (xi[mask], yi[mask]), 1)
        # return np.log2(img+1).astype(np.uint8)
        return img.astype(np.uint8)

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
    def __init__(self, env, nrepeat = 100):
        self.env = env
        self.nrepeat = nrepeat
        self.obses = [None]
        self.cohorts = [None]
        self.links = [None]
        self.fermeture = [None]

    def run(self, model, deterministic=True, verbose= False):
        env = self.env
        obs =  env._img_observe()
        terminated = False
        trunc = False
        self.total_reward = 0
        discounting = 1
        
        
        while not (terminated or trunc): #numpy.bool_이라 is False에 안걸림. 아놔 -_-
            st = time()
            if isinstance(model, str):
                action = self.policy(obs, model)
            elif isinstance(model, np.ndarray):
                action = model    
            else:
                action = model.predict(obs, deterministic=deterministic)[0]
            
            obs, reward, terminated, trunc, info = env.step(action)
            done = terminated or trunc
            self.total_reward += discounting*reward
            discounting *= self.env.gamma
            cohort = info['cohort']
            link = info['link']
            self.fermeture.append(action)
            self.cohorts.append(cohort)
            self.links.append(link)
            self.obses.append(obs)
            if verbose: print("elapsed {}: {:.3f}\t| # of infectees: {}\t| # of blocked: {}".format(
                env.t, time()-st, sum(cohort['state']=='I'), sum(action)))
        return self.total_reward
    
    def policy(self, obs, policy_tag):
        if policy_tag == 'naive':
            return np.zeros(len(self.env.city.facs))

        if policy_tag == 'blockade':
            return np.ones(len(self.env.city.facs))    

        if policy_tag.startswith('random'):
            if (obs[1].sum() < self.env.city.N*0.01) and (obs[2].sum() > self.env.city.N*0.1):
                return np.zeros(len(self.env.city.facs))
            
            p1 = int(policy_tag.split('random')[1])/100
            return np.random.choice(2,size=len(self.env.city.facs), p=[1-p1,p1])

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
    with open('city_local.pkl', 'rb') as f:
        city = pickle.load(f)        
    env = EpiSimEnvironment(city, max_epis_length=180, ext_rate=1/1000/7, mean_recover=5,
                            risk_coexist = 0.02, n_visit_tries = 2, r0=5,
                            coef_block=0.05, coef_infect=0.1, gamma=0.999,
                            # ngrid = 36
                            )
    
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
    static = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 1., 0., 0., 0., 0., 0., 0., 0.])
    # static = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
    #    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #    0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    #    0., 1., 0., 0., 0., 0., 0., 0., 0.])
    env.reset()
    static_driver = EpiSimDriver(env)
    static_goal = static_driver.run(static, verbose=True)

    # %%
    from stable_baselines3 import PPO
    # model = PPO.load("/home/ckang/projects/corona2025/rdcity_resnet18policy_ib0.0_cb0.05_ci0.1_it3000000_1.zip", env=env, device="cuda")
    model = PPO.load("/home/ckang/projects/corona2025/logs/best_model/best_model.zip", env=env, device="cuda")
    # model = PPO.load("/home/ckang/projects/corona2025/local_resnet18policy_len365_risk0.03_ib-4.0_cb0.05_ci0.1_it8000000_2.zip", env=env, device="cuda")
    # model = PPO.load("/home/ckang/projects/corona2025/local_resnet18policy_len365_risk0.03_ib-4.0_cb0.03_ci0.1_it3000000_1.zip", env=env, device="cuda")
    
    # model = PPO.load("cnnpolicy1m.zip", env=env, device="cuda")
    # %%
    env.reset()
    model_driver = EpiSimDriver(env)
    model_goal = model_driver.run(model, deterministic=True, verbose=True)
    model_goal
    # rand25_driver = EpiSimDriver(env)
    # rand_goal25 = rand25_driver.run('random25', verbose=True)
    # %%
    # sns.barplot(model_driver.action_prob_on_ts(model,3)-0.5)
    sns.barplot(model_driver.action_prob_on_ts(model,30)-0.5)
    imager = ObsImager(model_driver.obses[1:])
    imager.save_gif('model.gif')

    # %%
    probs = np.column_stack([model_driver.action_prob_on_ts(model,3),
                             model_driver.action_prob_on_ts(model,80)])

    plt.imshow(probs,aspect='auto')
    # %%
    actions = np.column_stack([model_driver.fermeture[3],
                               model_driver.fermeture[80],
                               model_driver.fermeture[120],
                               model_driver.fermeture[150],
                               model_driver.fermeture[200],])
    plt.imshow(actions,aspect='auto')
    # %%
    plt.scatter(city.hhs.xcoor, city.hhs.ycoor)
    selected = city.facs.iloc[[0,2,4,5,7,9]]
    plt.scatter(selected.xcoor, selected.ycoor, color='red')
    unselected = city.facs.iloc[[1,3,6,8]]
    plt.scatter(unselected.xcoor, unselected.ycoor, color='black')
    plt.show()

    #%%
    env.reset()
    rand50_driver = EpiSimDriver(env)
    rand_goal50 = rand50_driver.run('random50', verbose=True)
    #%%
    for _ in range(100):
        env.step(np.zeros(env.action_space.shape))
        print(sum(env.cohort.state=='I'))

    #%%
    env.reset()
    naive_driver = EpiSimDriver(env)
    naive_goal = naive_driver.run('naive', verbose=True)
    env.reset()
    rand25_driver = EpiSimDriver(env)
    rand_goal25 = rand25_driver.run('random25', verbose=True)
    env.reset()
    rand100_driver = EpiSimDriver(env)
    rand_goal100 = rand100_driver.run('random100', verbose=True)
    # %%
    # imager = ObsImager(rand50_driver.obses[1:])
    # imager.save_gif('rand50.gif')

    # print(static_goal, model_goal, naive_driver.total_reward, rand50_driver.total_reward, rand100_driver.total_reward)
    print(model_goal,naive_driver.total_reward, rand25_driver.total_reward, rand50_driver.total_reward, rand100_driver.total_reward)
    # print(naive_driver.total_reward, rand25_driver.total_reward, rand50_driver.total_reward, rand100_driver.total_reward)

    # x0=np.array([sum(c.state=='I') for c in static_driver.cohorts[1:]])
    x1=np.array([sum(c.state=='I') for c in naive_driver.cohorts[1:]])
    x5=np.array([sum(c.state=='I') for c in rand25_driver.cohorts[1:]])
    x2=np.array([sum(c.state=='I') for c in rand50_driver.cohorts[1:]])
    x3=np.array([sum(c.state=='I') for c in rand100_driver.cohorts[1:]])
    x4=np.array([sum(c.state=='I') for c in model_driver.cohorts[1:]])

    # %% 누적
    x1=np.array([sum(c.state!='S') for c in naive_driver.cohorts[1:]])
    x5=np.array([sum(c.state!='S') for c in rand25_driver.cohorts[1:]])
    x2=np.array([sum(c.state!='S') for c in rand50_driver.cohorts[1:]])
    x3=np.array([sum(c.state!='S') for c in rand100_driver.cohorts[1:]])
    x4=np.array([sum(c.state!='S') for c in model_driver.cohorts[1:]])
    # %%
    horizon = 365
    plt.plot(x1[:horizon])
    plt.plot(x5[:horizon])
    plt.plot(x2[:horizon])
    plt.plot(x3[:horizon])
    plt.plot(x4[:horizon])
    plt.legend(['naive','rand25','rand50','rand100','model'])
    plt.show()

    # %%    
    plt.plot(np.array([f.sum() for f in naive_driver.fermeture[1:]]))
    plt.plot(np.array([f.sum() for f in rand25_driver.fermeture[1:]]))
    plt.plot(np.array([f.sum() for f in rand50_driver.fermeture[1:]]))
    plt.plot(np.array([f.sum() for f in rand100_driver.fermeture[1:]]))
    plt.plot(np.array([f.sum() for f in model_driver.fermeture[1:]]))
    
    city.facs.loc[model_driver.fermeture[-1]==1]
    city.facs.loc[model_driver.fermeture[1]==1]
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

