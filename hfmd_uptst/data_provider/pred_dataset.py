import os
import pandas as pd
import os
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import StandardScaler
import warnings
import numpy as np
warnings.filterwarnings('ignore')
        
class hand_foot(Dataset):
    def __init__(self, root_path='/home/Paradise/UPTST/dataset',data_type='train', size=None,split_ratio=[0.7,0.2,0.1],
                    data_path='cq_hfmd_2015_21.csv',tasks='MS_1',
                    scale=True):
        # size [seq_len, label_len, pred_len]

        self.seq_len = size[0]
        self.pred_len = size[1]
        self.split_ratio=split_ratio
        assert data_type in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[data_type]
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.train_len,self.test_len,self.val_len=self.__read_data__()
        self.tasks=tasks
        
        
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        window_size=self.seq_len+self.pred_len
        train_val_data_len=1854-window_size
        test_len=2558-2001-window_size
        train_len=int(0.875*train_val_data_len)
        val_len=int(0.125*train_val_data_len)
        
        if (train_len+val_len)!=train_val_data_len:
            val_len+=1
            
        border1s = [0, train_len , 2001]
        border2s = [train_len+window_size, 1854, 2558]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            # 获取标准差和均值
            std = self.scaler.scale_
            mean = self.scaler.mean_
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        return train_len,test_len,val_len

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        if self.tasks=='S':
            seq_x=seq_x[:,-1:]
            seq_y = self.data_y[r_begin:r_end,-1:]
        elif self.tasks=='MS_0':
           seq_y = self.data_y[r_begin:r_end,-1:]
           seq_x=seq_x[:,-2:]
        elif self.tasks=='MS_1':
           seq_y = self.data_y[r_begin:r_end,-1:]
           seq_x=np.concatenate((seq_x[:,:-2], seq_x[:,-1:]), axis=-1)
        elif self.tasks=='MS_2':
            seq_y = self.data_y[r_begin:r_end,-1:]
        elif self.tasks=='M2':
            seq_y = self.data_y[r_begin:r_end,-2:]
        
        return torch.tensor(seq_x, dtype=torch.float32),torch.tensor(seq_y, dtype=torch.float32)
    
    def __len__(self):
        
        if self.set_type==0:
            return self.train_len
        elif self.set_type==1:
            return self.val_len
        elif self.set_type==2:
            return self.test_len
        