import pandas as pd
import numpy as np

from itertools import product
import os

import sys
import torchvision
import torch as t
from torch.utils.data import DataLoader
import torchvision.transforms
import random

from tqdm import tqdm
from network import LatentModel

file_dir=os.getcwd()

df=pd.read_csv('./df_prac.csv')

def normalized_df(df,normalize_col=['WFX','WFY','RFX','RFY']):
    for ele in normalize_col:
        M_ele, m_ele = max(df[ele]), min(df[ele])
        df[ele] = df[ele] - m_ele / (M_ele - m_ele)

    return df


df_norm=normalized_df(df)

def batch_sample(df_norm,batch,X_col=['WFX','WFY','RFX','RFY'],Y_col=['OVX','OVY']):

    id_list = list(pd.Series.unique(df_norm.id))
    sel_id=random.choices(id_list,k=batch)

    max_num_context=max([len(df_norm[df_norm.id==sel]) for sel in sel_id])
    n_context=np.random.randint(50,max_num_context)
    n_target=np.random.randint(0,max_num_context-n_context)
    n_tot=n_context+n_target

    c_x,c_y,t_x,t_y=list(),list(),list(),list()

    for sel in sel_id:
        df_sel=df_norm[df_norm.id==sel]
        total_index=np.random.choice(list(df_sel.index),n_tot,replace=False)
        c_idx=total_index[:n_context]
        t_idx=total_index[n_context:]

        context_x=df_sel[df_sel.index.isin(c_idx)][X_col].values.tolist()
        context_y=df_sel[df_sel.index.isin(c_idx)][Y_col].values.tolist()

        target_x=df_sel[df_sel.index.isin(t_idx)][['WFX','WFY','RFX','RFY']].values.tolist()
        target_y = df_sel[df_sel.index.isin(t_idx)][['OVX', 'OVY']].values.tolist()

        context_x, context_y, target_x, target_y = list(map(lambda x: t.FloatTensor(x), (context_x, context_y, target_x, target_y)))

        c_x.append(context_x)
        c_y.append(context_y)
        t_x.append(target_x)
        t_y.append(target_y)

    c_x_torch = t.stack(c_x, dim=0)
    c_y_torch= t.stack(c_y, dim=0)
    t_x_torch = t.stack(t_x, dim=0)
    t_y_torch = t.stack(t_y, dim=0)

    return c_x_torch,c_y_torch,t_x_torch,t_y_torch

cx,cy,tx,ty=batch_sample(df_norm,8)




model = LatentModel(128)
y_pred, kl, loss = model(cx,cy,tx,ty)





