import os
import glob
import pickle
import torch
import math
import torch.utils.data
import torch.nn as nn
import numpy as np
import random

class ScoreDatasetREGRAD(torch.utils.data.Dataset):
    def __init__(self, all_points_num, path, tag, data_seed, data_width):
        self.all_points_num = all_points_num
        self.base_path = path
        self.tag = tag
        self.width = np.array(data_width, dtype=np.float32)

        np.random.seed(data_seed)
        
        if self.tag == "test":
            self.base_path = '/data0/cxg19/regrad/eval_data'
            p_path = []
            for scene in os.listdir(self.base_path):
                p_file =glob.glob('/data0/cxg19/regrad/eval_data/{}/*.p'.format(scene))
                p_path.extend(p_file)
            p_path.sort()
            p_path = np.array(p_path)[:1000]
            self.data_name = p_path

    def _noise_color(self, color, label):
        table_color_time = np.random.rand(3)
        obj_color_time = 1-np.random.rand(3) / 5
        for i in range(3):
            color[label==0, i] *= table_color_time[i]
            color[label!=0, i] *= obj_color_time[i]
        return color

    def __getitem__(self, index):
        # print(self.data_name[index])
        data_path = os.path.join(self.base_path, self.data_name[index])
        data = np.load(data_path, allow_pickle=True)
        view = data['view_cloud'].astype(np.float32)
        print(view.mean(axis=0))
        view[:,2] += 0.25
        view_color = data['view_cloud_color'].astype(np.float32)
        view_score = data['view_cloud_score'].astype(np.float32)
        view_label = data['view_cloud_label'].astype(np.float32)

        select_point_index = None
        if len(view) >= self.all_points_num:
            select_point_index = np.random.choice(len(view), self.all_points_num, replace=False)
        elif len(view) < self.all_points_num:
            select_point_index = np.random.choice(len(view), self.all_points_num, replace=True)

        view, view_color, view_label, view_score = view[select_point_index], view_color[select_point_index], \
                                                view_label[select_point_index], view_score[select_point_index]
        view_color = self._noise_color(view_color, view_label)
        view_mean = np.mean(view, axis=0)
        view = np.c_[view, view_color]

        view_score = np.tanh(view_score)
        return view, view_score, view_label, data_path, self.width

    def __len__(self):
        return len(self.data_name)
        # return int(len(self.data_name) / 1200)

