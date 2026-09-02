# %% city_init
from common_funcs import *
import quadprog
import seaborn as sns



def init_households(istart, nh, mean_members, sd_members, real=None, nperh = None, nquart=7, centroids=None):
    if real is not None:
        households = real.loc[real.index.repeat(real['nhh']), ['xcoor','ycoor','type']].reset_index(drop=True)
        households['affiliated'] = np.random.choice(a = range(1,len(nperh)+1), size = len(households), p = nperh/sum(nperh))
        np.maximum(1,np.random.normal(mean_members,sd_members,len(households))).astype(int) #가족구성원 수 : 평균 3, 표편 1
    else:
        #거점을 두고
        quartiers = []
        for i in range(nquart):
            if centroids is None:
                centroid = np.random.rand(1,2).flatten()*nquart
            else:
                centroid = centroids[i]
            print(centroid)
            cov = np.array([[0.25, 0.01], [0.01, 0.25]])
            quartiers.append(pd.DataFrame(
                np.random.multivariate_normal(centroid, cov/2, size=int(nh/nquart/2)),
                # centroid-np.array([0.5,0.5])+np.random.rand(int(nh/nquart),2), 
                                          columns=['xcoor', 'ycoor']))
            quartiers.append(pd.DataFrame(
                np.random.multivariate_normal(centroid, cov*2, size=int(nh/nquart/2)),
                # centroid-np.array([0.5,0.5])+np.random.rand(int(nh/nquart),2), 
                                          columns=['xcoor', 'ycoor']))
        households = pd.concat(quartiers)
        # households = pd.DataFrame(np.random.rand(nh,2), columns=['xcoor', 'ycoor'])
        households['type'] = 'household'
        households['affiliated'] = np.maximum(1,np.random.normal(mean_members,sd_members,len(households))).astype(int) #가족구성원 수 : 평균 3, 표편 1

    households['hid'] = np.arange(istart, istart+len(households))
    households['visit'] = 0
    households['permanent'] = 1
    households['sojourn'] = 8
    households['locality'] = 1.0
    households['risk'] = 1.0
    households=households.set_index('hid')
    return(households)


def simulate_fac(row, class_size, border):
    facs = []
    for i in range(row['Count']):
        xcoor, ycoor = np.random.rand(1,2)[0]
        xcoor = xcoor * border[0]
        ycoor = ycoor * border[1]
        affiliated = np.random.randint(1,4)
        contacts = max(1,int(np.random.normal(row['Contact_avg'], row['Contact_std'],1)[0]))
        nclasses =1
        # 1/nclasses로 risk adjust
        if row['Residential']==1:
            affiliated = contacts
            nclasses = math.ceil(affiliated/class_size)
            contacts = int(affiliated*0.1)#/nclasses)
        
        # for _ in range(nclasses):
        facs.append([xcoor, ycoor, row['Type'], affiliated, contacts, row['Residential'], row['Sojourn'], row['Locality'], 
                     row['Risk']*1/nclasses])
            
    return(pd.DataFrame(facs, columns = ['xcoor', 'ycoor', 'type', 'affiliated', 'visit', 'permanent', 'sojourn', 'locality','risk' ]))
# simulate_fac(meta.loc[0])

def init_facilities(istart, meta, class_size, households, real=None):
    if real is None:
        facilities = pd.concat(meta.apply(simulate_fac, axis=1, class_size = class_size, 
                                          border = (households.xcoor.max(), households.ycoor.max()) 
                                         ).tolist(), ignore_index=True)
    else:
        facilities = real
    facilities['fid'] = np.arange(istart, istart+len(facilities))
    # pd.concat(meta.apply(simulate_fac, axis=1).values().tolist())
    return(facilities.set_index('fid'))

def init_individuals(istart, households):
    individuals = pd.DataFrame([hid for hid in households.index for _ in range(households['affiliated'][hid])], columns=['hid']) #hid를 가족구성원 수만큼 반복
    individuals = individuals.merge(households[['xcoor','ycoor']], on='hid')
    individuals['type'] = 'individual'
    individuals['state'] = 'S'
    individuals['affiliation'] = individuals['hid']
    individuals['tinfected'] = -1 #when infected
    individuals['trecovered'] = -1 #when recovered
    individuals['spreader'] = 0 #from whom infected
    individuals['finfected'] = 0 #where infected
    individuals['iid'] = np.arange(istart, istart+len(individuals)) 
    individuals = individuals.set_index('iid')
    return(individuals)


def create_mega(ngrid):
    mega_city = City(2, None, None, None)

    ngrid = 3
    for xleft in range(ngrid):
        for ybottom in range(ngrid):
            hh = init_households(10000+100000*(1+xleft+ybottom*ngrid), 1000, 3, 1)
            fac = init_facilities(20000+100000*(1+xleft+ybottom*ngrid), pd.read_csv('nodes-simul.csv'), bir = 0.5)
            ind = init_individuals(30000+100000*(1+xleft+ybottom*ngrid), hh)
            mega_city.add(hh, fac, ind, xleft, ybottom)
    mega_city.update_travel()
    return(mega_city)

def down_scailing(df, scale, seed=42):
    # # 영역 범위 계산
    # x_min, x_max = df['xcoor'].min(), df['xcoor'].max()
    # y_min, y_max = df['ycoor'].min(), df['ycoor'].max()

    # # 그리드 크기 설정
    # x_grid_size = (x_max - x_min) / ngrid
    # y_grid_size = (y_max - y_min) / ngrid

    # # 각 지점이 속한 그리드 셀 인덱스 계산
    # x_idx = ((df['xcoor'] - x_min) / x_grid_size).astype(int)
    # y_idx = ((df['ycoor'] - y_min) / y_grid_size).astype(int)

    # # index가 ngrid 이상으로 나가는 것 방지 (max 경계점에 걸릴 때)
    # x_idx = x_idx.clip(upper=ngrid - 1)
    # y_idx = y_idx.clip(upper=ngrid - 1)

    # df['x_bin'] = x_idx
    # df['y_bin'] = y_idx
    dfs = []
    df['type'] = df['type'].str.strip()
    for grp in df['type'].unique():
        df_part = df.loc[df['type']==grp]
        # if False : #grp=='household':
        #     # 각 그리드 셀에서 nhh 합산
        #     grouped = df_part.groupby(['x_bin', 'y_bin'])['nhh'].sum().reset_index()
        #     grouped.nhh = np.ceil(grouped.nhh*scale).astype(int)

        #     # 각 셀의 중심 좌표 계산
        #     grouped['xcoor'] = x_min + (grouped['x_bin'] + 0.5) * x_grid_size
        #     grouped['ycoor'] = y_min + (grouped['y_bin'] + 0.5) * y_grid_size

        #     # 필요한 열만 정리
        #     result_df = grouped[['xcoor', 'ycoor', 'nhh']].assign(type=grp)
        #     # print(result_df)
                         
        # else:
        result_df = df_part.sample(frac=scale, random_state=seed)
        dfs.append(result_df)
    
    return pd.concat(dfs)


class City():

    # a city stores a snapshot of any time
    def __init__(self, lf, hhs, facs, inds, ngrid=96):
        #locality factor
        self.N = 0
        self.lf = lf
        self.hhs = hhs.copy()
        self.inds = inds.copy()
        self.facs = facs.copy()
        self.ngrid = ngrid
        self.boundingbox(0, 0, ngrid, ngrid)  # set bounding box to [0,0] to [ngrid, ngrid]
        

        ## for debugging
        if self.inds is not None:
            self.N = len(self.inds)
            self.update_travel()

    # def compute_dists(self):
    #     ds = [None]*len(self.facs.index)
    #     for i, c in enumerate(self.facs.index):
    #         ds[i] = np.sqrt((self.inds['xcoor'] - self.facs.loc[c]['xcoor']).pow(2) + (self.inds['ycoor'] - self.facs.loc[c]['ycoor']).pow(2)).rename(c)
    #     return(pd.concat(ds, axis =1))  

    def scaling(self, scale):
        if scale==1.0:
            return
        return



    def boundingbox(self, xleft, ybottom, xright, ytop):
        # Calculate global min and max for both x and y across hhs, facs, inds
        x_min = min(self.hhs['xcoor'].min(), self.facs['xcoor'].min(), self.inds['xcoor'].min())
        x_max = max(self.hhs['xcoor'].max(), self.facs['xcoor'].max(), self.inds['xcoor'].max())
        y_min = min(self.hhs['ycoor'].min(), self.facs['ycoor'].min(), self.inds['ycoor'].min())
        y_max = max(self.hhs['ycoor'].max(), self.facs['ycoor'].max(), self.inds['ycoor'].max())
        # Adjust x and y ranges to be equal by padding the narrower dimension
        x_range = x_max - x_min
        y_range = y_max - y_min
        if x_range > y_range:
            pad = (x_range - y_range) / 2
            y_min -= pad
            y_max += pad
        elif y_range > x_range:
            pad = (y_range - x_range) / 2
            x_min -= pad
            x_max += pad

        # Scale all coordinates to the bounding box
        if x_max > x_min:
            self.hhs['xcoor'] = (((self.hhs['xcoor'] - x_min) / (x_max - x_min)) * (xright - xleft) + xleft).astype(np.float32)
            self.facs['xcoor'] = (((self.facs['xcoor'] - x_min) / (x_max - x_min)) * (xright - xleft) + xleft).astype(np.float32)
            self.inds['xcoor'] = (((self.inds['xcoor'] - x_min) / (x_max - x_min)) * (xright - xleft) + xleft).astype(np.float32)
        else:
            self.hhs['xcoor'] = (xleft).astype(np.float32)
            self.facs['xcoor'] = (xleft).astype(np.float32)
            self.inds['xcoor'] = (xleft).astype(np.float32)
        if y_max > y_min:
            self.hhs['ycoor'] = (((self.hhs['ycoor'] - y_min) / (y_max - y_min)) * (ytop - ybottom) + ybottom).astype(np.float32)
            self.facs['ycoor'] = (((self.facs['ycoor'] - y_min) / (y_max - y_min)) * (ytop - ybottom) + ybottom).astype(np.float32)
            self.inds['ycoor'] = (((self.inds['ycoor'] - y_min) / (y_max - y_min)) * (ytop - ybottom) + ybottom).astype(np.float32)
        else:
            self.hhs['ycoor'] = (ybottom).astype(np.float32)
            self.facs['ycoor'] = (ybottom).astype(np.float32)
            self.inds['ycoor'] = (ybottom).astype(np.float32)

    def add(self, hhs, facs, inds, xleft, ybottom):
        hhs[['xcoor','ycoor']] += [xleft, ybottom]
        facs[['xcoor','ycoor']] += [xleft, ybottom]
        inds[['xcoor','ycoor']] += [xleft, ybottom]
        
        self.hhs = pd.concat([hhs,self.hhs])
        self.facs = pd.concat([facs,self.facs])
        self.inds = pd.concat([inds,self.inds])
        self.N = len(self.inds)
        # self.update_travel()
        
        
    def compute_fdists(self):
        ds = [None]*len(self.facs.index)
        for i, c in enumerate(self.facs.index):
            exponent = self.lf*self.facs.loc[c]['locality']/2
            # print(exponent)
            dist = (self.inds['xcoor'] - self.facs.loc[c]['xcoor']).pow(2) + (self.inds['ycoor'] - self.facs.loc[c]['ycoor']).pow(2)
            dist = np.clip(dist,1,np.inf) # + np.random.random(len(dist)) * 1e-08 - 5e-09  # avoid division by zero
            ds[i] = 1/(np.power(dist,
                            exponent).rename(c))
        return(pd.concat(ds, axis =1))  
    
    def objAttr(self, a, d, n):
        error = (1/d.dot(a)).dot(d*a) - n / (sum(n)/self.N)
        # error = (1/v.dot(p)).dot(v).multiply(p)-n_c/prob_nest
        
        # 솔브 방식
        return error
        # 최적화 방식
        # return(error.dot(error)) 

    def residual(self, x, D, y):
        Dx = D @ x
        inv_Dx = 1.0 / Dx
        pred = inv_Dx @ (D @ np.diag(x))
        return pred - y  # residual vector
    
    def compute_attractiveness(self, fgroup, mode):
        fmembers = self.facs.loc[self.facs['type']==fgroup].index.values
        d = self.fdists[fmembers].values
        n = self.facs.loc[fmembers, mode].values
        if sum(n) == 0:
            return(pd.Series(data = np.zeros(len(fmembers)), index = fmembers))
        # print(d)
        guess = n*0.1
        # print(fgroup, mode,#d,n,guess, 
        #       self.residual(guess, d, n / (sum(n)/self.N)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 최적화 방식
            # sol = minimize(self.objAttr, guess, bounds=[(0, None)]*len(guess), args=(d, n))
            # sol = sol.x
            # 솔브 방식
            sol = fsolve(self.objAttr, guess, args=(d, n))

        
        # least squares
        # sol = least_squares(self.residual, guess, args=(d, n / (sum(n)/self.N)), method='trf')          
        return(pd.Series(data =sol, index = fmembers) )
        
        
    
    def compute_linkprob(self, fgroup, attr, ns):
        fmembers = self.facs.loc[self.facs['type']==fgroup].index.values
        d = self.fdists[fmembers]
        
        # a = self.attr[mode][fmembers]
        # n = self.facs.loc[fmembers, mode]
        n = ns[fmembers]
        a = attr[fmembers]
        
        if sum(n) == 0:
            return pd.DataFrame(0.0, index=d.index, columns=d.columns)

        g = d*a
        
        return g * 1/(d @ a).values.reshape(-1,1)*(sum(n)/self.N)
        # return((g.T*(1/g.sum(axis=1))).T*(sum(n)/self.N))
        
    
    def update_travel(self):
        self.fdists = self.compute_fdists()
        fgroups = self.facs['type'].unique()
        
        self.attr = {}
        for mode in ['affiliated', 'visit']:
            st = time()
            args = [(group, mode) for group in fgroups]
            with multiprocessing.Pool(processes=len(fgroups)) as pool:
            # with multiprocessing.Pool(processes=1) as pool:    
                self.attr[mode] = pd.concat(pool.starmap(self.compute_attractiveness, args))
            print('Computed', mode, 'attractiveness, Elapsed:', time()-st)    

        self.linkp = {}
        for mode in ['affiliated', 'visit']:
            self.linkp[mode] = pd.concat([self.compute_linkprob(fgroup, self.attr[mode], self.facs[mode]) for fgroup in fgroups], axis = 1)
            self.linkp[mode][-1] = 1- self.linkp[mode].sum(axis=1)
            # self.cumlinkp[mode] = self.linkp[mode].cumsum(axis=1)
            # self.cumlinkp[mode][-1] = 1
            # Count NaN values in city_temp.linkp['affiliated']
            # Example: print(city_temp.linkp['affiliated'].isna().sum().sum())
        # debuging : erase
        # 왜 linkp에 nan이 나오는가? compute_linkprob에서 뭐가 문제인가?
        self.inds['affiliation'] = where_to_go(self.linkp['affiliated'])
        
        
        #self.link_affil#.idxmax(axis=1) 
        # self.link_affil = self.link_affil.loc[self.link_affil[-1]<1].drop(-1, axis =1)
        
    def draw_visit(self, fid, mode = 'visit'):
        plt.Figure()
        fac = self.facs.loc[fid]
        # print(fac)
        links = where_to_go(self.linkp[mode]).reset_index()
        visitors = links.loc[links['fid']==fid].merge(self.inds, left_on='iid', right_index= True)
        plt.title('VISIT')
        plt.scatter(self.hhs['xcoor'], self.hhs['ycoor'], s=5)
        plt.scatter(visitors['xcoor'], visitors['ycoor'], s=5)
        plt.scatter(fac['xcoor'], fac['ycoor'], s=10, c='red')
        plt.show()
        return(links)    
    
    def draw_facility(self, type):
        plt.Figure()
        fac = self.facs.loc[self.facs.type==type]
        # print(fac)
        
        plt.title('Facs')
        plt.scatter(self.hhs['xcoor'], self.hhs['ycoor'], s=5)
        plt.scatter(fac['xcoor'], fac['ycoor'], s=10, c='red')
        plt.show()
        
    

    def block_visit(self, blocked, baloon = False, mode = 'visit'):
        linkp = self.linkp[mode].copy()
        if blocked is not None:
            if baloon:
                blocked_attr = self.attr[mode]
                blocked_attr[blocked] = 0
                fgroups = self.facs['type'].unique()
                linkp = pd.concat([self.compute_linkprob(fgroup, blocked_attr, self.facs[mode]) for fgroup in fgroups], axis = 1)
            else:
                linkp[-1] += self.linkp[mode][blocked].sum(axis=1)
                linkp[blocked] = 0
        return(linkp)
    
    def block_affil(self, blocked):
        link_affil = self.inds['affiliation'].copy()
        if blocked is not None: 
            link_affil[link_affil.isin(blocked)] = -1
        return(link_affil)

# %%
def make_small_affil_city():
    # %%
    hsize = 1000
    nquart = 3
    centroids = np.random.rand(nquart,2)*1.5*nquart
    households = init_households(10000, hsize, 1, 0.0, nquart=nquart, centroids=centroids)
    plt.scatter(households['xcoor'], households['ycoor'],s=5)
    individuals = init_individuals(30000, households)
    len(individuals)
    # %%
    volumes = (100, 50)
    # volume = 25
    nfacs = (0,1,2,3,4)
    risks = (0.25, 0.5)
    locality = (0.5, 5.0)
    # type: (risk, locality)
    meta_facs = {
                 'School': (volumes[0], nfacs[1], locality[1], risks[0]), 
                 'Workplace': (volumes[0], nfacs[1], locality[0], risks[0]), 
                 'Local center': (volumes[1], nfacs[1], locality[1], risks[0]), 
                 'Global center': (volumes[1], nfacs[1], locality[0], risks[0]),
                }
    
    facs = []
    for fac_type, (volume, nfac, locality, risk) in meta_facs.items():
        # sampled_locs = centroids
        xcoors = np.repeat(centroids[:,0],nfac)#np.random.uniform(xbound[0], xbound[1], neach_fac)
        ycoors = np.repeat(centroids[:,1],nfac)#np.random.uniform(ybound[0], ybound[1], neach_fac)

        # if fac_type == 'Workplace':
        #     volume = np.array([volumes[1]]*int(neach_fac))
        # elif fac_type == 'School':
        #     volume = np.array([volumes[1]]*int(neach_fac))
        # else: 
        #     volume = np.array([volumes[0]]*int(neach_fac/2) + [volumes[1]]*int(neach_fac/2))

        facs.append(pd.DataFrame({'xcoor': xcoors, 'ycoor': ycoors, 'type': fac_type, 
                                  'affiliated': volume*int(fac_type in ['Workplace', 'School']), 
                                  'visit': volume*int(fac_type not in ['Workplace', 'School']), 
                                  'permanent': int(fac_type in ['Workplace', 'School']), 
                                  'sojourn': 1+7*int(fac_type in ['Workplace', 'School']), 
                                  'locality': locality, 'risk': risk}))
    facs = pd.concat(facs, ignore_index=True)
    # %%
    plt.scatter(households['xcoor'], households['ycoor'],s=5)
    for fac_type, (volume, locality, locality, risk) in meta_facs.items():
        plt.scatter(facs.loc[facs.type==fac_type]['xcoor'], 
                    facs.loc[facs.type==fac_type]['ycoor'],s=5)    
    # %%
    facs['fid'] = np.arange(20000, 20000+len(facs))
    facilities = facs.set_index('fid')
    
    city_local = City(2, households, facilities, individuals)
    # %%
    city_local.draw_visit(20004, mode='affiliated')
    # city_local.draw_visit(20000, mode='visit')
    # city_local.draw_visit(20009, mode='visit')

    import pickle
    with open('city_small_affil.pkl', 'wb') as f:
        pickle.dump(city_local, f)


# %%
def make_small_city():
    # %%
    hsize = 1000
    nquart = 3
    centroids = np.random.rand(nquart,2)*1.5*nquart
    households = init_households(10000, hsize, 1, 0, nquart=nquart, centroids=centroids)
    plt.scatter(households['xcoor'], households['ycoor'],s=5)
    len(households)
    # %%
    volumes = (25, 50)
    # volume = 25
    nfacs = (0,1,2,3,4)
    risks = (0.25, 0.5)
    locality = (0.5, 5.0)
    # type: (risk, locality)
    meta_facs = {
                 'Healthcare': (volumes[1], nfacs[1], locality[1], risks[1]), 
                 'Venue': (volumes[1], nfacs[1], locality[0], risks[0]), 
                 'Restaurant': (volumes[1], nfacs[1], locality[0], risks[1]), 
                 'Daily service': (volumes[1], nfacs[1], locality[1], risks[0]),
                }
    
    facs = []
    for fac_type, (volume, nfac, locality, risk) in meta_facs.items():
        # sampled_locs = centroids
        xcoors = np.repeat(centroids[:,0],nfac)#np.random.uniform(xbound[0], xbound[1], neach_fac)
        ycoors = np.repeat(centroids[:,1],nfac)#np.random.uniform(ybound[0], ybound[1], neach_fac)

        # if fac_type == 'Workplace':
        #     volume = np.array([volumes[1]]*int(neach_fac))
        # elif fac_type == 'School':
        #     volume = np.array([volumes[1]]*int(neach_fac))
        # else: 
        #     volume = np.array([volumes[0]]*int(neach_fac/2) + [volumes[1]]*int(neach_fac/2))

        facs.append(pd.DataFrame({'xcoor': xcoors, 'ycoor': ycoors, 'type': fac_type, 
                                  'affiliated': volume*int(fac_type in ['Workplace', 'School']), 
                                  'visit': volume*int(fac_type not in ['Workplace', 'School']), 
                                  'permanent': int(fac_type in ['Workplace', 'School']), 
                                  'sojourn': 1+7*int(fac_type in ['Workplace', 'School']), 
                                  'locality': locality, 'risk': risk}))
    facs = pd.concat(facs, ignore_index=True)
    # %%
    plt.scatter(households['xcoor'], households['ycoor'],s=5)
    for fac_type, (volume, locality, locality, risk) in meta_facs.items():
        plt.scatter(facs.loc[facs.type==fac_type]['xcoor'], 
                    facs.loc[facs.type==fac_type]['ycoor'],s=5)    
    # %%
    facs['fid'] = np.arange(20000, 20000+len(facs))
    facilities = facs.set_index('fid')
    individuals = init_individuals(30000, households)
    city_local = City(2, households, facilities, individuals)
    # %%
    city_local.draw_visit(20010, mode='affiliated')
    city_local.draw_visit(20000, mode='visit')
    city_local.draw_visit(20009, mode='visit')

    import pickle
    with open('city_small.pkl', 'wb') as f:
        pickle.dump(city_local, f)
# %%
def make_localized_city():
    
    hsize = 1000
    nquart = 10
    centroids = np.random.rand(nquart,2)*nquart/2
    households = init_households(10000, hsize, 3, 0, nquart=nquart, centroids=centroids)
    plt.scatter(households['xcoor'], households['ycoor'],s=5)
    len(households)
    # %%
    volumes = (25, 50)
    # volume = 25
    nfacs = (0,1,2,3,4)
    risks = (0.25, 0.5)
    locality = (0.5, 5.0)
    # type: (risk, locality)
    meta_facs = {'Healthcare': (volumes[0], nfacs[1], locality[1], risks[1]), 
                 'Venue': (volumes[0], nfacs[1], locality[0], risks[0]), 
                 'Restaurant': (volumes[0], nfacs[1], locality[0], risks[1]), 
                 'Daily service': (volumes[0], nfacs[1], locality[1], risks[0]),
                 'Workplace': (volumes[1], nfacs[2], locality[0], risks[0]),
                 'School': (volumes[1], nfacs[2], locality[1],risks[0]),
                 }
    
    facs = []
    for fac_type, (volume, nfac, locality, risk) in meta_facs.items():
        # sampled_locs = centroids
        xcoors = np.repeat(centroids[:,0],nfac)#np.random.uniform(xbound[0], xbound[1], neach_fac)
        ycoors = np.repeat(centroids[:,1],nfac)#np.random.uniform(ybound[0], ybound[1], neach_fac)

        # if fac_type == 'Workplace':
        #     volume = np.array([volumes[1]]*int(neach_fac))
        # elif fac_type == 'School':
        #     volume = np.array([volumes[1]]*int(neach_fac))
        # else: 
        #     volume = np.array([volumes[0]]*int(neach_fac/2) + [volumes[1]]*int(neach_fac/2))

        facs.append(pd.DataFrame({'xcoor': xcoors, 'ycoor': ycoors, 'type': fac_type, 
                                  'affiliated': volume*int(fac_type in ['Workplace', 'School']), 
                                  'visit': volume*int(fac_type not in ['Workplace', 'School']), 
                                  'permanent': int(fac_type in ['Workplace', 'School']), 
                                  'sojourn': 1+7*int(fac_type in ['Workplace', 'School']), 
                                  'locality': locality, 'risk': risk}))
    facs = pd.concat(facs, ignore_index=True)
    # %%
    plt.scatter(households['xcoor'], households['ycoor'],s=5)
    for fac_type, (volume, locality, locality, risk) in meta_facs.items():
        plt.scatter(facs.loc[facs.type==fac_type]['xcoor'], 
                    facs.loc[facs.type==fac_type]['ycoor'],s=5)    
    # %%
    facs['fid'] = np.arange(20000, 20000+len(facs))
    facilities = facs.set_index('fid')
    individuals = init_individuals(30000, households)
    city_local = City(2, households, facilities, individuals)
    # %%
    city_local.draw_visit(20050, mode='affiliated')
    city_local.draw_visit(20000, mode='visit')
    city_local.draw_visit(20020, mode='visit')
    # %%
    import pickle
    with open('city_local.pkl', 'wb') as f:
        pickle.dump(city_local, f)

# %%
def make_random_city(hsize=1000):
    # %%
    households = init_households(10000, hsize, 3, 0, nquart=5)
    plt.scatter(households['xcoor'], households['ycoor'],s=5)
    len(households)
    
    # %%
    xbound = (min(households.xcoor), max(households.xcoor))
    ybound = (min(households.ycoor), max(households.ycoor))

    volumes = (25, 50)
    risks = (0.25, 0.5)
    locality = 1.0
    # type: (risk, locality)
    meta_facs = {'Healthcare': (volumes[0], risks[1]), 
                 'Venue': (volumes[1], risks[0]), 
                 'Restaurant': (volumes[1], risks[1]), 
                 'Daily service': (volumes[0], risks[0]),
                 'Workplace': (volumes[0], risks[0]),
                 'School': (volumes[1], risks[0]),
                 }
    neach_fac = 10
    facs = []
    for fac_type, (volume, risk) in meta_facs.items():
        sampled_locs = households.sample(neach_fac)
        xcoors = sampled_locs.xcoor#np.random.uniform(xbound[0], xbound[1], neach_fac)
        ycoors = sampled_locs.ycoor#np.random.uniform(ybound[0], ybound[1], neach_fac)

        # if fac_type == 'Workplace':
        #     volume = np.array([volumes[1]]*int(neach_fac))
        # elif fac_type == 'School':
        #     volume = np.array([volumes[1]]*int(neach_fac))
        # else: 
        #     volume = np.array([volumes[0]]*int(neach_fac/2) + [volumes[1]]*int(neach_fac/2))

        facs.append(pd.DataFrame({'xcoor': xcoors, 'ycoor': ycoors, 'type': fac_type, 
                                  'affiliated': volume*int(fac_type in ['Workplace', 'School']), 
                                  'visit': volume*int(fac_type not in ['Workplace', 'School']), 
                                  'permanent': int(fac_type in ['Workplace', 'School']), 
                                  'sojourn': 1+7*int(fac_type in ['Workplace', 'School']), 
                                  'locality': locality, 'risk': risk}))
    facs = pd.concat(facs, ignore_index=True)
    # %%
    plt.scatter(households['xcoor'], households['ycoor'],s=5)
    for fac_type, (locality, risk) in meta_facs.items():
        plt.scatter(facs.loc[facs.type==fac_type]['xcoor'], 
                    facs.loc[facs.type==fac_type]['ycoor'],s=5)    
    # %%
    facs['fid'] = np.arange(20000, 20000+len(facs))
    # pd.concat(meta.apply(simulate_fac, axis=1).values().tolist())
    facilities = facs.set_index('fid')
    # %%
    individuals = init_individuals(30000, households)
    city_random = City(2, households, facilities, individuals)

    # %%
    import pickle
    with open('city_random.pkl', 'wb') as f:
        pickle.dump(city_random, f)

# %%
def make_random_jc(size=1500):
    
    # %%
    xbound = (128.23, 128.19)
    ybound = (37.16, 37.12)
    dfh = pd.read_csv("hh_jecheon_newgeocoding.csv").dropna()
    dfh = dfh.loc[(dfh['xcoor']<xbound[0]) & (dfh['xcoor']>xbound[1])]
    dfh = dfh.loc[(dfh['ycoor']<ybound[0]) & (dfh['ycoor']>ybound[1])]
    dongunit = dfh['adress'].str.split(expand=True)[2]
    dfh = dfh[~((dongunit.str.endswith('읍') | dongunit.str.endswith('면')))]        
    # %%
    size=1500
    households = dfh.loc[dfh.index.repeat(dfh['nhh']), ['xcoor','ycoor','type']].reset_index(drop=True)
    pertubation = np.random.multivariate_normal([0,0], 
                                                10**(-2.5)*np.array([[households.xcoor.var(),0],[0,households.ycoor.var()]]), 
                                                len(households))
    # sns.scatterplot(x=pertubation[:,0], y=pertubation[:,1], s=0.5)
    households.xcoor = households.xcoor + pertubation[:,0]
    households.ycoor = households.ycoor + pertubation[:,1]
    households=households.sample(size)
    
    # %%
    probs = np.array([26350, 17272, 9536, 7323, 2142, 506, 119, 18, 13, 7]) #세대원수 구성
    probs = probs / probs.sum()
    households['affiliated'] = np.random.choice(np.arange(1, 11), size=len(households), p=probs)
    households['visit'] = 0
    households['permanent'] = 1
    households['sojourn'] = 8
    households['locality'] = 1.0
    households['risk'] = 0.5
    # %%
    dff = pd.read_csv("fac_jecheon_newgeocoding.csv").dropna()
    df_factory = dff.loc[dff.type=='factory']
    idxdong = dff.loc[~dff.dong.str.endswith('리')].index
    dff = pd.concat([df_factory,dff.loc[idxdong]]).drop(['dong'], axis=1)
    dff = dff.drop_duplicates()
    dff.loc[dff.type=='factory','type'] = 'office' # 공장 -> 사무실
    dff = dff.loc[(dff['xcoor']<xbound[0]) & (dff['xcoor']>xbound[1])]
    dff = dff.loc[(dff['ycoor']<ybound[0]) & (dff['ycoor']>ybound[1])]
    facs = dff#.sample(int(households.affiliated.sum()/10))
    # facs = pd.concat([facs, dff.loc[dff.type=='shopping']])
    facs.drop_duplicates(subset=['xcoor','ycoor'], inplace=True)
    # plt.scatter(x=households.xcoor, y=households.ycoor, s=0.5)
    # plt.scatter(x=facs.xcoor, y=facs.ycoor, s=0.5)
    # %%
    # Create the dictionary
    data = {
        "origin": [
            "gym", "medical", "park", "restaurant", "meeting", "office",
            "hotel", "bath", "beauty", "factory", "religion", "school", "shopping"
        ],
        "type": [
            "Medical", "Medical", "Venue", "Restaurant", "Venue", "Workplace",
            "Venue", "Local services", "Local services", "Workplace",
            "Venue", "School", "Local services"
        ]
    }

    # Create the DataFrame
    facmap = pd.DataFrame(data)
    facs = facs.merge(facmap, left_on='type', right_on='origin', how='left', suffixes=('', '_agg'))
    #%%
    factype = facs#.loc[facs.type=='hotel'].copy()

    plt.scatter(x=households.xcoor, y=households.ycoor, s=0.5)
    plt.scatter(x=factype.xcoor, y=factype.ycoor, s=1)


# %%
def make_city(scale=12):
    # %%
    dfh = pd.read_csv("hh_jecheon_newgeocoding.csv").dropna()
    dfh = dfh.loc[(dfh['xcoor']<128.5) & (dfh['xcoor']>127.5)]
    dfh = dfh.loc[(dfh['ycoor']<37.5) & (dfh['xcoor']>36.5)]
    dongunit = dfh['adress'].str.split(expand=True)[2]
    dfh = dfh[~((dongunit.str.endswith('읍') | dongunit.str.endswith('면')))]

    # ngrid = 96
    # scale=12
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
    facs['visit'] = np.clip(np.random.normal(loc=facs['contact_avg'], scale=facs['contact_std'], size=len(dff)),0,10000)

    facs['affiliated'] = np.random.choice(range(1,10),len(facs)).astype(int)
    facs.loc[facs['type']=='office','affiliated'] = 5*np.exp(facs[facs['type']=='office']['visit']).astype(int)
    facs.loc[facs['type']=='office','visit'] = 0
    facs.loc[facs['type']=='school','affiliated'] = facs[facs['type']=='school']['visit'].astype(int)
    facs.loc[facs['type']=='school','visit'] = 0
    facs['visit'] = facs['visit'].astype(int)
    facs = facs.drop(['contact_avg','contact_std'], axis=1)

    households = households.sample(frac=1/scale, random_state=124).reset_index(drop=True)
    facs = facs.sample(frac=1/scale, random_state=124).reset_index(drop=True)
    
    households['hid'] = households.index + 100000
    households.set_index('hid', inplace=True)

    facs['fid'] = facs.index + 200000
    facs.set_index('fid', inplace=True)

    plt.scatter(households['xcoor'], households['ycoor'],s=0.5)
    plt.scatter(facs['xcoor'], facs['ycoor'],s=0.5)    
    print(households.affiliated.sum(), facs.affiliated.sum(), facs.visit.sum())


# %%
    htemp = households.sample(1000, random_state=42)
    ftemp = facs.sample(100, random_state=42)
    individuals = init_individuals(300000, htemp)

    city = City(2,htemp, ftemp, individuals)
# %%
    from time import time 
    
    
    # city.N = len(city.inds)
    # city.fdists = city.compute_fdists()
    # st=time()
    for fgroup in city.facs['type'].unique()[5:6]:
        # city.compute_attractiveness(fgroup, 'affiliated')
        st=time()
        
        city.compute_attractiveness(fgroup, 'affiliated')
        print(time()-st)
    # for fgroup in city.facs['type'].unique():
    #     city.compute_linkprob(fgroup, city.attr['affiliated'], city.facs['affiliated'])
    #     city.compute_linkprob(fgroup, city.attr['visit'], city.facs['visit'])

# %%
    # Dx = d @ guess
    # inv_Dx = 1.0 / Dx
    # pred = inv_Dx @ (d @ np.diag(guess))
    d = city.fdists[fmembers]
        
    # a = self.attr[mode][fmembers]
    # n = self.facs.loc[fmembers, mode]
    n = city.facs.visit[fmembers]
    a = city.attr['visit'][fmembers]
    
    g = d*a
    probs=g * 1/(d @ a).values.reshape(-1,1)*(sum(n)/city.N)
    # (g.T*(1/g.sum(axis=1))).T*(sum(n)/city.N)
# %%
    import pickle
    with open('city_jecheon_opt.pkl', 'rb') as f:
        city_opt = pickle.load(f) 
# %%
# %%
if __name__ == "__main__":
    pass