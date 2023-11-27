from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import glob
import numpy as np

# final version, sanity check completed.
class TrainFHDataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths=data_paths
        self.exclude_time=0
        self.ds_type='ea'
        self.use_feat=True
        self.use_hour=True
        self.make_hour_emb=False
        self.rolling_ranges = [17, 33, 65]
        self.train_len = (6+24+6)*60*12
        self.X=[]
        self.Y=[]
        for path in tqdm(self.data_paths):
            xy = pd.read_csv(path)
            y = xy[['onset','wakeup']]
            # gen feat
            x = self._gen_feat(xy)
            # change columns order - hour 을 맨 뒤에 놓는다.
            cols=x.columns.tolist()
            cols=cols[1:2]+cols[3:9]+cols[0:1]+cols[9:]+cols[2:3]
            x=x[cols]
            
            self.X.append(x)
            self.Y.append(y)
    
    def change_mode(self,ds_type, use_feat, use_hour, make_hour_emb, exclude_time):
        self.ds_type = ds_type
        self.use_feat = use_feat
        self.use_hour = use_hour
        self.make_hour_emb = make_hour_emb
        self.exclude_time = exclude_time
    
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
        return xy.drop(columns=['onset','wakeup']).fillna(0)
    
    def __getitem__(self,i):
        x = self.X[i].copy()
        h = False
        y = self.Y[i].copy()
        
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
        
        if not self.exclude_time==0:
            # 36 - exclude_time 
            st = self.exclude_time*60*12
            ed = -self.exclude_time*60*12
            x = x[st:ed]
            y = y[st:ed]
            if self.use_hour and self.make_hour_emb:
                h = h[st:ed]
        x = x.to_numpy()
        y = y.to_numpy()
        if self.use_hour and self.make_hour_emb:
            h = h.to_numpy()
        
        return x, h, y
