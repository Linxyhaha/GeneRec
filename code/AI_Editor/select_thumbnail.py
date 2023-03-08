# -*- encoding: utf-8 -*-
'''
Discription: Thumbnail Selection. calculate similarity of user preference and frame, for each user-video pair.
'''

import torch
import clip
import numpy as np
from PIL import Image

import pdb 
# load feature
cover_feat = torch.load("cover_feature.pt")
video_f_feat = torch.load('video_feature.pt')

# load data
path = '../../data/'
tr = np.load(path+"training_dict.npy", allow_pickle=True).item()
tst= np.load(path+"testing_dict.npy", allow_pickle=True).item()

with torch.no_grad():

    # calculate user rep via historically like item's cover feature
    user_rep = torch.zeros(len(tr),512)
    for u_id in tr:
        rep = torch.zeros(512)
        for i_id in tr[u_id]:
            rep += (cover_feat[i_id]/len(tr[u_id]))
        user_rep[u_id] = rep

    # normalization
    user_rep /= user_rep.norm(dim=-1,keepdim=True)
    video_f_feat /= video_f_feat.norm(dim=-1,keepdim=True)
    user_rep = user_rep.cuda()
    video_f_feat = video_f_feat.cuda()


    # for each item, calculate the similarity of user rep and each frame, then select the frame with the highest similarity score

    select_covers = []
    for i_id in range(len(video_f_feat)):
        sim_score = torch.mm(user_rep,(video_f_feat[i_id]).T.to(torch.float32)) #(user, frame)
        _, indices = torch.topk(sim_score, 1)
        select_covers.append(indices.cpu())
    res = torch.stack(select_covers)

# save results
torch.save(res, 'results/select_thumbnail.pt') # (item,) each entry is the select cover idx for all users
