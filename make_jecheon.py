# %%
import folium
folium.__version__
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common_funcs import *
from city_init import *

# %%
import pandas as pd
import requests
import time
from tqdm import tqdm

def get_lat_lon_from_address_kakao(address: str, api_key: str):
    url = 'https://dapi.kakao.com/v2/local/search/address.json'
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"query": address}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None, None

    result = response.json()

    try:
        first_match = result['documents'][0]
        # print(first_match)
        return first_match['address']['address_name'], first_match['address']['region_3depth_name'], \
                float(first_match['y']), float(first_match['x'])  # (latitude, longitude)
    except (IndexError, KeyError):
        # print("Not Found:", address)
        return address, None, None, None

def geocode_series(address_series: pd.Series, api_key: str, delay: float = 0.3):
    addresses, dongs, latitudes, longitudes = [], [], [], []
    for addr in tqdm(address_series):
        aname, dong, lat, lon = get_lat_lon_from_address_kakao(addr, api_key)

        addresses.append(aname if aname else addr)  # Use the original address if geocoding fails
        dongs.append(dong)
        latitudes.append(lat)
        longitudes.append(lon)
        # time.sleep(delay)  # Too many rapid requests may result in being throttled

    return pd.DataFrame({
        "address": addresses,
        "dong": dongs,
        "latitude": latitudes,
        "longitude": longitudes
    })

# 예시 사용
api_key = '9499f0e50a0ad0a8962ffdfe620765a3'
# address_series = pd.Series([
#     '서울특별시 강남구 역삼동 832-7',
#     '부산광역시 해운대구 우동 123-45'
# ])

# result_df = geocode_series(address_series, api_key)
# print(result_df)

# # %%
# a = pd.read_csv('jc_employee_1.csv')
# b = pd.read_csv('jc_employee_2.csv')
# df = pd.concat([a,b])
# df = df.drop(['성립일자','사업종류'],axis=1)
# df = df.drop_duplicates(subset=['사업장명','주소'])

# df.loc[df['상시인원']>4].to_csv('jc_employee.csv', index = False, encoding='utf-8-sig')

## %%
# dfh_raw = pd.read_csv("hh_jecheon.csv").dropna()
# dfh = dfh_raw.drop(['xcoor','ycoor'], axis=1)
# dfh = dfh.groupby(['adress','type']).sum('nhh').reset_index()
# result_df = geocode_series(dfh.adress, api_key)
# # dfh = dfh.loc[(dfh['xcoor']<130) & (dfh['xcoor']>127)]
# # %%
# dfh['adress'] = result_df['address'].values
# dfh['ycoor'] = result_df['latitude'].values
# dfh['xcoor'] = result_df['longitude'].values
# dfh['dong'] = result_df['dong'].values
# dfh.dropna().to_csv("hh_jecheon_newgeocoding.csv", index=False)

# # %%
# dff_raw = pd.read_csv("fac_jecheon.csv").dropna()
# dff = dff_raw.copy().reset_index(drop=True)

# result_df = geocode_series(dff.address, api_key)
# #%%
# result_df = result_df.dropna()
# dff = dff.loc[result_df.index]

# dff['ycoor'] = result_df['latitude'].values
# dff['xcoor'] = result_df['longitude'].values
# dff['dong'] = result_df['dong'].values
# dff.to_csv("fac_jecheon_newgeocoding.csv", index=False)
ngrid = 96
scale=20

dfh = pd.read_csv("hh_jecheon_newgeocoding.csv").dropna()
dfh = dfh.loc[(dfh['xcoor']<128.5) & (dfh['xcoor']>127.5)]
dfh = dfh.loc[(dfh['ycoor']<37.5) & (dfh['xcoor']>36.5)]
dongunit = dfh['adress'].str.split(expand=True)[2]
dfh = dfh[~((dongunit.str.endswith('읍') | dongunit.str.endswith('면')))]

# dfh = grid_aggregate(dfh, ngrid,1/scale)
households = dfh.loc[dfh.index.repeat(dfh['nhh']), ['xcoor','ycoor','type']].reset_index(drop=True)
probs = np.array([26350, 17272, 9536, 7323, 2142, 506, 119, 18, 13, 7]) #세대원수 구성
probs = probs / probs.sum()
households['affiliated'] = np.random.choice(np.arange(1, 11), size=len(households), p=probs)

households['visit'] = 0
households['permanent'] = 1
households['sojourn'] = 8
households['locality'] = 1.0
households['risk'] = 0.5

dff = pd.read_csv("fac_jecheon_newgeocoding.csv").dropna()
df_factory = dff.loc[dff.type=='factory']
idxdong = dff.loc[~dff.dong.str.endswith('리')].index
dff = pd.concat([df_factory,dff.loc[idxdong]]).drop(['dong'], axis=1)
dff = dff.drop_duplicates()

dff.loc[dff.type=='factory','type'] = 'office' # 공장 -> 사무실

dff = dff.loc[(dff['xcoor']<128.5) & (dff['xcoor']>127.5)]
dff = dff.loc[(dff['ycoor']<37.5) & (dff['xcoor']>36.5)]
dff.drop_duplicates(subset=['xcoor','ycoor','name','type'], inplace=True)
# dff = grid_aggregate(dff, ngrid, 1/7)


meta = pd.read_csv("meta_jecheon.csv")
dff = dff.loc[dff['type'].isin(meta['type'])].reset_index(drop=True)

facs = dff.merge(meta.drop(['count','frequency'],axis=1), left_on='type', right_on='type')

households = down_scailing(households, scale=1/scale,seed=5).reset_index(drop=True)
facs = down_scailing(facs, scale=1/scale,seed=5).reset_index(drop=True)


facs['visit'] = np.clip(np.random.normal(loc=facs['contact_avg'], scale=facs['contact_std'], size=len(facs)),0,10000)
facs['affiliated'] = np.random.choice(range(1,10),len(facs)).astype(int)
facs.loc[facs['type']=='office','affiliated'] = 5*np.exp(facs[facs['type']=='office']['visit']).astype(int)
facs.loc[facs['type']=='office','visit'] = 0
facs.loc[facs['type']=='school','affiliated'] = facs[facs['type']=='school']['visit'].astype(int)
facs.loc[facs['type']=='school','visit'] = 0
facs['visit'] = facs['visit'].astype(int)
facs = facs.drop(['contact_avg','contact_std'], axis=1)


households['hid'] = households.index + 100000
households.set_index('hid', inplace=True)

facs['fid'] = facs.index + 200000
facs.set_index('fid', inplace=True)

plt.scatter(households['xcoor'], households['ycoor'],s=0.5)
plt.scatter(facs['xcoor'], facs['ycoor'],s=0.5)    
print(households.affiliated.sum(), facs.affiliated.sum()/households.affiliated.sum(), facs.visit.sum()/households.affiliated.sum())

# htemp = households.sample(1000, random_state=42)
# ftemp = facs.sample(70, random_state=42)
# individuals = init_individuals(300000, htemp)

# city = City(2,htemp, ftemp, individuals)

# from episim import *
# env = EpiSimEnvironment(city, max_epis_length=365*3, ext_rate=0.001, mean_recover=5,
#                             risk_coexist=0.05, n_visit_tries=2, r0=5,
#                             coef_block=0.025, coef_infect=0.1, gamma=0.999,
#                             ngrid=128)
# naive_driver = EpiSimDriver(env)
# naive_goal = naive_driver.run('naive', verbose=True)
# %%

#%%
# facs['type'].value_counts()
individuals = init_individuals(300000, households)
city_jc = City(2,households, facs, individuals)
# %%
import pickle
with open('city_jecheon_scale'+str(scale)+'.pkl', 'wb') as f:
    pickle.dump(city_jc, f) 

# %%
# linkp = city_jc.linkp['visit']
# for g in city_jc.facs.type.unique():
#     fmembers = city_jc.facs.index[city_jc.facs.type==g]
#     print("{}:{}", g, linkp.loc[:,fmembers].sum(axis=1).iloc[0])
# df

# %%
# center = [(df['xcoor'].min()+df['xcoor'].max())/2, (df['ycoor'].min()+df['ycoor'].max())/2]
df = pd.concat([dff[['xcoor','ycoor','type']], dfh[['xcoor','ycoor','type']]], axis=0,
               ignore_index=True)
def draw_with_map(df):
    center = [df['ycoor'].mean(), df['xcoor'].mean()]
    m = folium.Map(location=center, zoom_start=13)

    for i in df.index:
        loc = df.loc[i, ['ycoor','xcoor']].tolist()
        try:
            folium.Circle(
                location = loc,
                tooltip = i,
                radius = 1,
                color = COL_DICT[df.loc[i,'type']]
            ).add_to(m)
        except KeyError:
            pass
       

    return(m)
draw_with_map(df)

# %%
