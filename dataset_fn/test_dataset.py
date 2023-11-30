from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import glob
import numpy as np

class InferenceFHDataset(Dataset):

    def __init__(self, data_paths):
        self.id_map = pd.read_parquet('/kaggle/input/id-series-hour-inference/archive/id_map.parquet')
        self.data_paths=data_paths
        self.ds_type='ea'
        self.use_feat=True
        self.use_hour=True
        self.make_hour_emb=False
        self.rolling_ranges = [17, 33, 65]
        self.train_len = (6+24+6)*60*12
        self.X=[]
        self.Y=[]
        self.ids=[]
        for path in tqdm(self.data_paths):
            cur_id = self.id_map.iloc[int(path.split('/')[-1].split('.')[0][2:])].series_id
            xy = pd.read_csv(path)
            y = xy[['onset','wakeup']]
            # gen feat
            x = self._gen_feat(xy)
            # change columns order - hour 을 맨 뒤에 놓는다.
            cols=x.columns.tolist()
            cols=cols[1:2]+cols[3:9]+cols[0:1]+cols[9:]+cols[2:3]
            x=x[cols]
            # crop series into fixed size
            it=len(x)//self.train_len
            num_pad = self.train_len-len(x)%self.train_len
            for i in range(it+1):
                # 마지막 iteration zeropad로 길이 맞춰주기
                if i==it:
                    st= i*self.train_len
                    x_padded = np.pad(x.to_numpy()[st:].copy(), ((0,num_pad),(0,0)),'constant',constant_values=0)
                    y_padded = np.pad(y.to_numpy()[st:].copy(), ((0,num_pad),(0,0)),'constant',constant_values=0)
                    X = pd.DataFrame(x_padded, columns=x.columns)
                    Y = pd.DataFrame(y_padded, columns=y.columns)
                    self.X.append(X)
                    self.Y.append(Y)
                    self.ids.append((cur_id,i))
                else:
                    st = i*self.train_len
                    ed = (i+1)*self.train_len
                    self.X.append(x[st:ed])
                    self.Y.append(y[st:ed])
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
        return xy.drop(columns=['onset','wakeup']).fillna(0)
    
    def __getitem__(self,i):
        
        x = self.X[i].copy()
        h = False
        y = self.Y[i].copy()
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
        y = y.to_numpy()
        if self.use_hour and self.make_hour_emb:
            h = h.to_numpy()
        return x, h, y, id_
