# https://github.com/edenton/svg/blob/master/data/kth.py
import numpy as np
import os
import pickle
import torch
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .h5 import HDF5Dataset

import torch.nn.functional as F
import pdb


class Huawei_edit_Dataset(Dataset):

    def __init__(self, interaction_path, data_path, frame_p, frames_, frames_per_sample=5, image_size=64, train=True, valid=False, test=False, random_time=True, random_horizontal_flip=True,
                 total_videos=-1, skip_videos=0, with_target=True, ng=False, cond_u=True, user_emb_path='./user_emb/user_rep.npy',dictname=None):

        self.data_path = data_path                    # '/path/to/Datasets/UCF101_64_h5' (with .hdf5 file in it), or to the hdf5 file itself
        self.train = train
        self.valid = valid
        self.test = test

        self.frame_p = frame_p
        self.frames_ = frames_

        self.frames_per_sample = frames_per_sample
        self.image_size = image_size
        self.random_time = random_time
        self.random_horizontal_flip = random_horizontal_flip

        self.total_videos = total_videos            # If we wish to restrict total number of videos (e.g. for val)
        self.with_target = with_target
        self.ng = ng

        self.dictname = dictname
        # Read h5 files as dataset
        self.videos_ds = HDF5Dataset(self.data_path)

        # Read interactions
        self.ui_dict = np.load('/storage/xylin/recommendation/generative_rec/data/huawei/data_1220/training_dict.npy',allow_pickle=True).item()
        self.edit_pairs = self.get_uipairs(interaction_path)
        self.get_group_index(interaction_path)
        self.samples = self.edit_pairs
        
        # classifier = torch.load(user_emb_path)
        # self.user_emb = classifier.id_embedding[:classifier.num_user]
        self.cond_u = cond_u
        self.user_emb = torch.FloatTensor(np.load(user_emb_path,allow_pickle=True))

        print(f"Dataset length: {self.__len__()}")

    def get_uipairs(self,path):
        self.item2id = np.load(path+'item_map.npy',allow_pickle=True).item()    
        edit_dict = np.load(path+f'{self.dictname}.npy',allow_pickle=True).item()
        edit_list = []
        for u_id in edit_dict:
            for i_id in edit_dict[u_id]:
                if i_id ==3569:
                    continue
                edit_list.append([u_id,i_id])
        return edit_list

    def get_group_index(self,path):
        self.user_group = np.load(path+'user_group/group_dict.npy',allow_pickle=True).item()
        self.u2g = {}
        for g_idx, g_uids in self.user_group.items():
            for u_id in g_uids:
                self.u2g[u_id] = g_idx

    def len_of_vid(self, index):
        video_index = index % self.__len__()
        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)
        with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
            video_len = f['len'][str(idx_in_shard)][()]
        return video_len

    def __len__(self):
        # return self.total_videos if self.total_videos > 0 else self.num_train_vids if self.train else self.num_test_vids
        return self.total_videos if self.total_videos>0 else len(self.samples) # 暂时没考虑test和validation

    def max_index(self):
        # return self.num_train_vids if self.train else self.num_test_vids
        return len(self.samples)

    def __getitem__(self, index, time_idx=0):

        # Use `index` to select the video, and then
        # randomly choose a `frames_per_sample` window of frames in the video

        u_id, video_index = self.samples[index]
        user_rep = self.user_emb[u_id].repeat(3).reshape(1,3,32,1)

        user_c = F.interpolate(user_rep,(self.image_size,self.image_size)) #(1,3,64,64)
        user_c = user_c.squeeze(0)

        if self.cond_u==False:
            if len(self.ui_dict[u_id]):
                his_i = random.choice(self.ui_dict[u_id])
                while (his_i == video_index or his_i==3569) and len(self.ui_dict[u_id])>1:
                    his_i = random.choice(self.ui_dict[u_id])
                while his_i == 3569:
                    his_i = random.choice(range(4570))
            else:
                his_i = random.choice(range(4570))
                while his_i ==3569:
                    his_i = random.choice(range(4570))
            shard_idx_c, idx_in_shard_c = self.videos_ds.get_indices(his_i)

        shard_idx, idx_in_shard = self.videos_ds.get_indices(video_index)

        # read data
        prefinals = []
        with self.videos_ds.opener(self.videos_ds.shard_paths[shard_idx]) as f:
            if self.cond_u:
                for i in range(self.frame_p):
                    prefinals.append(user_c)
            else:
                # slice data
                video_len = f['len'][str(idx_in_shard_c)][()]

                # 1: history item (conditional)
                if self.random_time and video_len > self.frame_p:
                    time_idx = np.random.choice(video_len - self.frame_p)

                h,w,_ = f[str(idx_in_shard_c)][str(0)][()].shape
                # random crop
                crop_c = np.random.randint(int(self.image_size/h*w) - self.image_size) if self.train else int((self.image_size/h*w - self.image_size)/2)
                # random horizontal flip
                flip_p = np.random.randint(2) == 0 if self.random_horizontal_flip else 0
                for i in range(time_idx, min(time_idx + self.frame_p, video_len)):
                    img = f[str(idx_in_shard_c)][str(i)][()]
                    arr = transforms.RandomHorizontalFlip(flip_p)(transforms.ToTensor()(img[:, crop_c:crop_c + self.image_size]))
                    prefinals.append(arr)

            # 2: current item
            video_len = f['len'][str(idx_in_shard)][()]
            if self.random_time and video_len > self.frames_:
                time_idx = np.random.choice(video_len - self.frames_)

            h,w,_ = f[str(idx_in_shard)][str(0)][()].shape
            # random crop
            crop_c = np.random.randint(int(self.image_size/h*w) - self.image_size) if self.train else int((self.image_size/h*w - self.image_size)/2)
            # random horizontal flip
            flip_p = np.random.randint(2) == 0 if self.random_horizontal_flip else 0
            for i in range(time_idx, min(time_idx + self.frames_, video_len)):
                img = f[str(idx_in_shard)][str(i)][()]
                arr = transforms.RandomHorizontalFlip(flip_p)(transforms.ToTensor()(img[:, crop_c:crop_c + self.image_size]))
                assert arr.shape==(3,64,64)
                prefinals.append(arr)
    
        video = torch.stack(prefinals)
            
        return video, torch.tensor(u_id)
