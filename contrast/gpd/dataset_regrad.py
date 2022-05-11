import os
import glob
import pickle
import torch
import math
import torch.utils.data
import torch.nn as nn
import numpy as np
import random
import open3d
# from .eval_score.configs import config
# from open3d.open3d.geometry import voxel_down_sample, estimate_normals, orient_normals_towards_camera_location


class PointNetGPDDatasetREGRAD(torch.utils.data.Dataset):
    def __init__(self, all_points_num, path, tag, data_seed, thresh_score, frame_num):
        self.all_points_num   = all_points_num
        self.base_path = path
        self.tag = tag
        self.thresh = thresh_score
        self.frame_num = frame_num

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

    def __getitem__(self, index):
        data_path = os.path.join(self.base_path, self.data_name[index])
        data = np.load(data_path, allow_pickle=True)
        view = data['view_cloud'].astype(np.float32)
        view_color = data['view_cloud_color'].astype(np.float32)
        view = np.concatenate((view, view_color), axis=1)

        if 'frame' in data.keys():
            grasp              = data['frame']
            grasp_score        = data['antipodal_score']

        else:
            grasp              = data['select_frame']
            grasp_score        = np.array(data['select_antipodal_score'])

        frame_idx   = np.random.choice(len(grasp), self.frame_num, replace=False)
        grasp       = grasp[frame_idx]
        grasp_score = grasp_score[frame_idx]

        select_point_index = None
        if len(view) >= self.all_points_num:
            select_point_index = np.random.choice(len(view), self.all_points_num, replace=False)
        elif len(view) < self.all_points_num:
            select_point_index = np.random.choice(len(view), self.all_points_num, replace=True)
        view = view[select_point_index]

        label = np.zeros((len(frame_idx)), dtype=np.int64)
        label[grasp_score<self.thresh]  = 0
        label[grasp_score>=self.thresh] = 1

        return view, grasp, label, data_path

    def __len__(self):
        return len(self.data_name) 
