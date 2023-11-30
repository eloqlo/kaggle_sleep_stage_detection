import pandas as pd
import numpy as np
import gc
import os, glob
from tqdm import tqdm 

class SubmissionFHDataset(Dataset):

    def __init__(self, series_path):
        self.series_list=self._preprocess_test_series(series_path)
        self.ds_type='ea'
        self.use_feat=False
        self.use_hour=False
        self.make_hour_emb=False
        self.rolling_ranges = [17, 33, 65]
        self.train_len = (6+24+6)*60*12
        self.X=[]
        self.ids=[]
        for cur_id, x in tqdm(self.series_list):
            # gen feat
            x = self._gen_feat(x)
            # change columns order - hour 을 맨 뒤에 놓는다.
            cols=x.columns.tolist()
            cols=cols[1:2]+cols[3:9]+cols[0:1]+cols[9:]+cols[2:3]
            x=x[cols]
            # crop series into fixed size
            it=len(x)//self.train_len
            for i in range(it+1):
                # 마지막 iteration zeropad로 길이 맞춰주기
                if i==it:
                    num_pad = self.train_len-len(x)%self.train_len
                    st= i*self.train_len
                    h=x['hour']
                    x_padded = np.pad(x.drop(columns='hour').to_numpy()[st:].copy(), ((0,num_pad),(0,0)),'constant',constant_values=0)
                    h_padded = np.pad(h.to_numpy()[st:].copy(), (0,num_pad),'constant',constant_values=24)[:,np.newaxis]
                    # concat x-pad and h-pad
                    x_padded = np.concatenate((x_padded,h_padded),axis=1)
                    X = pd.DataFrame(x_padded, columns=x.columns)
                    self.X.append(X)
                    self.ids.append((cur_id,i))
                else:
                    st = i*self.train_len
                    ed = (i+1)*self.train_len
                    self.X.append(x[st:ed])
                    self.ids.append((cur_id,i))

                    
    # before evaluate, change state according to 내 모델.
    def change_mode(self,ds_type, use_feat, use_hour, make_hour_emb):
        self.ds_type=ds_type
        self.use_feat=use_feat
        self.use_hour=use_hour
        self.make_hour_emb=make_hour_emb
        
            
    def __len__(self):
        return len(self.data_paths)
    
    
    def _gen_feat(self,xy):
        # enmo
        for r in self.rolling_ranges:
            tmp_feat = xy['enmo'].rolling(r, center=True)
            xy[f'enmo_mean_{r}'] = tmp_feat.mean()
            xy[f'enmo_std_{r}'] = tmp_feat.std()
        # anglez
        for r in self.rolling_ranges:
            tmp_feat = xy['anglez'].rolling(r, center=True)
            xy[f'anglez_mean_{r}'] = tmp_feat.mean()
            xy[f'anglez_std_{r}'] = tmp_feat.std()
        return xy.fillna(0)
    
    
    
    def _preprocess_test_series(self,data_path):
        series = pd.read_parquet(data_path)
        # make hour feature
        series['timestamp'] = pd.to_datetime(series.timestamp,format = '%Y-%m-%dT%H:%M:%S%z').astype("datetime64[ns, UTC-04:00]")
        series['hour'] = series.timestamp.dt.hour
        series = series.drop(columns=['timestamp','step'])
        # normalize enmo and anglez
        mean_enmo = series['enmo'].mean()
        std_enmo = series['enmo'].std()
        series.enmo = (series['enmo'] - mean_enmo)/std_enmo
        mean_anglez = series['anglez'].mean()
        std_anglez = series['anglez'].std()
        series.anglez = (series['anglez'] - mean_anglez)/std_anglez
        # seperate into ids
        ids = pd.DataFrame({'series_id':series.series_id.unique()})
        series_list=[]
        for cur_id in tqdm(ids.series_id, total=len(ids)):
            cur_series = series.loc[series.series_id==cur_id].copy().reset_index(drop=True).drop(columns='series_id')
            series_list.append((cur_id,cur_series))
        del series
        gc.collect()
        return series_list
    
    
    def __getitem__(self,i):
        
        x = self.X[i].copy()
        h = False
        id_ = self.ids[i]
        
        if not self.use_feat:
            x = x[['enmo','anglez','hour']]
        if not self.use_hour:
            x.drop(columns=['hour'], inplace=True)
        if self.use_hour and self.make_hour_emb:
            h = x[['hour']]
            x.drop(columns=['hour'], inplace=True)
        if self.ds_type=='e':
            x.drop(list(x.filter(regex='anglez')), axis=1, inplace=True)
        elif self.ds_type=='a':
            x.drop(list(x.filter(regex='enmo')), axis=1, inplace=True)
        x = x.to_numpy()
        if self.use_hour and self.make_hour_emb:
            h = h.to_numpy()
        return x, h, y, id_
