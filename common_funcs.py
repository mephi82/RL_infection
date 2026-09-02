# common_functions
import numpy as np
import pandas as pd
import matplotlib as mpl
from PIL import Image, ImageDraw, ImageFont
import colorsys
import matplotlib.pyplot as plt
import multiprocessing
from time import time
import math

from scipy.optimize import fsolve, minimize, least_squares
from scipy import stats
import warnings

import networkx as nx

COL_DICT = {'household': "gray", 'S': "gray", 'I': "red", 'R': "green",
              'school': "black", 'office': "black", 'neighbor': "black", 
              'restaurant': "magenta", 'gym': "magenta", 'singing': "magenta",
              'swimming': "cyan", 'religion': "cyan", 'meeting': "cyan", 'bath': "cyan", 'beauty': "cyan", 
              'shopping': "yellow", 'studying': "yellow", 'gaming': "yellow", 'concert': "yellow",
              'fermeture': "white", 'medical': "red"}

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


def conditional_prob_matrix(X, eps=1e-12):
    """
    X: (N, d) binary matrix (0/1)
    return: C (d, d), where C[i,j] = P(X_j=1 | X_i=1)
    """
    X = (X > 0).astype(float)  # ensure 0/1
    cooc = X.T @ X             # (d,d): count of (Xi=1 and Xj=1)
    count_i = X.sum(axis=0)    # (d,): count of (Xi=1)
    
    C = cooc / (count_i[:, None] + eps)
    return C

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


def where_to_go(linkprob, rng: np.random.Generator = None):
    # return(linkprob.apply(lambda row: np.random.choice(linkprob.columns, p=row), axis=1).rename('fid'))
    # return(linkprob.apply(lambda row: linkprob.columns[bisect(row.values, np.random.rand())], axis=1).rename('fid'))
    if rng is None:
        rng = np.random.default_rng()
    cumlinkprob = linkprob.cumsum(axis=1)
    # return((cumlinkprob.sub(np.random.rand(len(cumlinkprob)), axis=0)>0).idxmax(axis=1).rename('fid'))

    marker = (cumlinkprob.sub(rng.random(len(cumlinkprob)), axis=0)>0).astype(int)
    res = np.where(marker - marker.shift(periods=1, axis=1, fill_value=0))
    return(pd.Series(linkprob.columns[res[1]], index=linkprob.index[res[0]], name='fid'))

def where_to_go_sparse(linkprob):
    cumlinkprob = linkprob.cumsum(axis=1)

    marker = (cumlinkprob.sub(np.random.rand(len(cumlinkprob)), axis=0)>0).astype(int)
    return (marker - marker.shift(periods=1, axis=1, fill_value=0)).drop(columns = -1).values
    
def count_numbers(aresult):
    indis = aresult.cohorts[1:]
    link = aresult.links[1:]
    
    Icount = np.fromiter(map(lambda x: (x['state']=='I').sum(), indis), dtype=int)
    Ncount = np.fromiter(map(lambda x: (x['tinfected']==0).sum(), indis), dtype=int)
    Rcount = np.fromiter(map(lambda x: (x['trecovered']==0).sum(), indis), dtype=int)
    Ccount = np.fromiter(map(lambda x: (x['state']!='S').sum(), indis), dtype=int)
    # Vcount = np.fromiter(map(lambda x: len(x), visit[1:]), dtype=int)
    # return({'I':Icount, 'N':Ncount, 'R':Rcount, 'C':Ccount})
    return((Icount, Ncount, Rcount, Ccount))
# Icount = [(x['state']=='I').sum() for x in indis]

def analyze_result(res, parallel = False):

    st = time()
    if parallel:
        with multiprocessing.Pool(processes=len(res)) as pool:
            counts = pool.map(count_numbers, res)
    else:
        counts=list(map(count_numbers, res))

    Icounts, Ncounts, Rcounts, Ccounts = np.swapaxes(np.array(counts), 0, 1)
    print(time()-st)
    return(Icounts, Ncounts, Rcounts, Ccounts)