import os
import os.path as osp
import pickle, glob
import random
import logging
import numpy as np
# from path import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from tdgpd.utils.md5 import get_md5


# mp.set_start_method('forkserver')


class ScoreFunc(nn.Module):
    def forward(self, search_score, antipodal_score, object_num):
        object_num = object_num + (object_num < 0.5).astype(np.float) * 10000
        scored_grasp = np.minimum(np.log(search_score + 1) / 4, np.ones([1, 1, 1])) * antipodal_score \
                       / np.sqrt(1 - np.power(antipodal_score, 2)) / np.power(object_num, 2)
        LENGTH_SEARCH, GRASP_PER_LENGTH = search_score.shape[1:]
        scored_grasp = np.reshape(scored_grasp, (-1, LENGTH_SEARCH * GRASP_PER_LENGTH))
        score = np.max(scored_grasp, axis=1)
        local_seach_index = np.argmax(scored_grasp, axis=1)

        return scored_grasp, score, local_seach_index

    def extra_repr(self):
        extra_str = "Grasp: min(log(search_score)/4, 1)/tan(theta)/obj**2; Point: max of grasps"
        return extra_str


class RobustScoreFunc(nn.Module):
    def forward(self, search_score, antipodal_score):
        scored_grasp = np.minimum(np.log(search_score + 1) / 4, np.ones([1, 1, 1])) * antipodal_score \
                       / np.sqrt(1 - np.power(antipodal_score, 2))
        scored_grasp_smooth = scored_grasp.copy()
        LENGTH_SEARCH, GRASP_PER_LENGTH = search_score.shape[1:]
        padding = 1
        weight = np.array([[[0.2, 0.6, 0.2]]])
        scored_grasp_padding = np.zeros((scored_grasp.shape[0], LENGTH_SEARCH, GRASP_PER_LENGTH + 2 * padding))
        scored_grasp_padding[:, :, padding:-padding] = scored_grasp.copy()
        for p in range(padding):
            scored_grasp_padding[:, :, padding - 1 - p] = scored_grasp[:, :, -1 - p].copy()
            scored_grasp_padding[:, :, -1 - p] = scored_grasp[:, :, p].copy()
        for i in range(GRASP_PER_LENGTH):
            scored_grasp_smooth[:, :, i] = np.sum(scored_grasp_padding[:, :, i:i + 2 * padding + 1] * weight, axis=-1)
            # scored_grasp_smooth[:, :, i] = np.min(scored_grasp_padding[:, :, i:i + 2 * padding + 1], axis=-1)
        scored_grasp_smooth = np.reshape(scored_grasp_smooth, (-1, LENGTH_SEARCH * GRASP_PER_LENGTH))
        scored_grasp = np.reshape(scored_grasp, (-1, LENGTH_SEARCH * GRASP_PER_LENGTH))
        score = np.max(scored_grasp_smooth, axis=1)
        local_seach_index = np.argmax(scored_grasp_smooth, axis=1)

        return scored_grasp, score, local_seach_index

    def extra_repr(self):
        extra_str = "Grasp: min(log(search_score)/4, 1)/tan(theta)/obj**2, min of 3 neighbors; Point: max of grasps"
        return extra_str


class GraspScoreFunc(nn.Module):
    def forward(self, antipodal_score):
        return antipodal_score / np.sqrt(1 - np.power(antipodal_score, 2))

    def extra_repr(self):
        extra_str = "Baseline 1/gamma, where gamma is the friction coefficient"
        return extra_str


class YCBScenesAll(Dataset):
    score_grad_dict = {2: np.array([1 / 1.5]),
                       3: np.array([0.4, 1.0]),
                       4: np.array([0.25, 1 / 1.5, 1.1])
                       }

    def __init__(self,
                 root_dir,
                 mode="train",
                 score_classes=2,
                 num_points=102400,
                 score_func=RobustScoreFunc,
                 distance_threshold=0.05,
                 std_R=0.0,
                 std_t=0.0, ):
        super(YCBScenesAll, self).__init__()
        self.root_dir = os.path.abspath(root_dir)
        self.mode = mode
        self.score_classes = score_classes
        self.score_grad = self.score_grad_dict[score_classes]
        self.num_points = num_points
        self.num_frame_points = num_points // 8
        self.score_func = score_func()
        self.distance_threshold = distance_threshold

        self.std_R = std_R
        self.std_t = std_t
        
        # self.path_list = np.array(sorted(os.listdir(self.root_dir)))
        #np.random.seed(1)

        if 'eval_data' in self.root_dir:
            p_path = np.array(sorted(os.listdir(self.root_dir)))
            index = np.random.choice(len(p_path), int(len(p_path)*0.8), replace=False)
            if self.tag != "train":
                ori = np.arange(len(p_path)) 
                index = np.array(list( set(list(ori)) - set(list(index)) ))
        else:
            if mode == "test":
                base_dir_name = "training_data_test"
            else:
                base_dir_name = "training_data"
            self.root_dir = os.path.join(self.root_dir, base_dir_name) 
            p_path = np.array(sorted(os.listdir(self.root_dir)))

            if mode == "test":
                index = np.arange(len(p_path)) 
            else:
                index = np.random.choice(len(p_path), int(len(p_path)*0.8), replace=False)
                if mode != "train":
                    ori = np.arange(len(p_path)) 
                    index = np.array(list( set(list(ori)) - set(list(index)) ))
        self.path_list = p_path[index]
        

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.path_list[index])
        data = np.load(path, allow_pickle=True)
        point_cloud = data["view_cloud"].T

        chosen_inds = None
        if point_cloud.shape[1] >= self.num_points:
            chosen_inds = np.random.choice(point_cloud.shape[1], self.num_points, replace=False)
        elif point_cloud.shape[1] < self.num_points:
            chosen_inds = np.random.choice(point_cloud.shape[1], self.num_points, replace=True)
        scene_points = point_cloud[:, chosen_inds]

        return {
            "scene_points": torch.tensor(scene_points).float(),
            "data_path":  path,
        }

    def __len__(self):
        return len(self.path_list)

    def get_stat(self):
        num_point_in_data_list = []
        num_local_search_list = []
        num_frame_list = []

        scored_scene_points_list = []

        for path in tqdm(self.path_list):
            data = np.load(path, allow_pickle=True)
            search_score = data["search_score"]
            antipodal_score = data["antipodal_score"]
            object_num = data["objects_num"]
            point_cloud = data["point_cloud"]
            frame_index = data["frame_index"]

            num_frame = frame_index.shape[0]
            num_frame_list.append(num_frame)

            num_point_in_data, num_local_search = search_score.shape
            num_point_in_data_list.append(num_point_in_data)
            num_local_search_list.append(num_local_search)

            _, scored_scene_points = self.score_func(search_score, antipodal_score, object_num)

            scored_scene_points_list.append(scored_scene_points)

        scored_scene_points_list = np.concatenate(scored_scene_points_list, axis=0)

        print("Number of points:\n mean: {:.2f}, std: {:.2f}, min: {:d}, max: {:d}".format(
            np.mean(num_point_in_data_list), np.std(num_point_in_data_list, ddof=1), np.min(num_point_in_data_list),
            np.max(num_point_in_data_list)))

        print("Number of local search:\n mean: {:.2f}, std: {:.2f}, min: {:d}, max: {:d}".format(
            np.mean(num_local_search_list), np.std(num_local_search_list, ddof=1), np.min(num_local_search_list),
            np.max(num_local_search_list)))

        print("Number of frames: \n mean: {:.2f}, std: {:.2f}, min: {:d}, max: {:d}".format(
            np.mean(num_frame_list), np.std(num_frame_list, ddof=1), np.min(num_frame_list),
            np.max(num_frame_list)))

        print("Scored scene points: \n mean: {:.2f}, std: {:.2f}, min: {:.2f}, max: {:.2f}".format(
            np.mean(scored_scene_points_list), np.std(scored_scene_points_list, ddof=1),
            np.min(scored_scene_points_list),
            np.max(scored_scene_points_list)))

        norm_scored_scene_points = (scored_scene_points_list - np.min(scored_scene_points_list)) / (
                np.max(scored_scene_points_list) - np.min(scored_scene_points_list) + 1e-4)

        lins = np.linspace(0, 1, 51)
        num_grasp = []
        num_points = []

        for l in lins:
            num_points.append(np.sum(np.logical_and(norm_scored_scene_points >= l, norm_scored_scene_points < l + 1)))

        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(lins * (np.max(scored_scene_points_list) - np.min(scored_scene_points_list)) + np.min(
            scored_scene_points_list),
                 np.asarray(num_points) / norm_scored_scene_points.size, 'b', linewidth=2)
        plt.show()
        print("Scored scene point distribution: \n", num_points)

class YCBScenesREGRAD(Dataset):
    score_grad_dict = {2: np.array([1 / 1.5]),
                       3: np.array([0.4, 1.0]),
                       4: np.array([0.25, 1 / 1.5, 1.1])
                       }

    def __init__(self,
                 root_dir,
                 mode="train",
                 score_classes=2,
                 num_points=102400,
                 score_func=RobustScoreFunc,
                 distance_threshold=0.05,
                 std_R=0.0,
                 std_t=0.0, ):
        super(YCBScenesREGRAD, self).__init__()
        #self.root_dir = os.path.abspath(root_dir)
        self.mode = mode
        self.score_classes = score_classes
        self.score_grad = self.score_grad_dict[score_classes]
        self.num_points = num_points
        self.num_frame_points = num_points // 8
        self.score_func = score_func()
        self.distance_threshold = distance_threshold

        self.std_R = std_R
        self.std_t = std_t
        
        self.root_dir = '/data0/cxg19/regrad/eval_data'
        p_path = []
        for scene in os.listdir(self.root_dir):
            p_file =glob.glob(self.root_dir+'/{}/*.p'.format(scene))
            p_path.extend(p_file)
        p_path.sort()
        p_path = np.array(p_path)[:1000]
        self.path_list = p_path
        

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.path_list[index])
        data = np.load(path, allow_pickle=True)
        point_cloud = data["view_cloud"].T

        chosen_inds = None
        if point_cloud.shape[1] >= self.num_points:
            chosen_inds = np.random.choice(point_cloud.shape[1], self.num_points, replace=False)
        elif point_cloud.shape[1] < self.num_points:
            chosen_inds = np.random.choice(point_cloud.shape[1], self.num_points, replace=True)
        scene_points = point_cloud[:, chosen_inds]

        return {
            "scene_points": torch.tensor(scene_points).float(),
            "data_path":  path,
        }

    def __len__(self):
        return len(self.path_list)

class YCBScenes(Dataset):
    score_grad_dict = {2: np.array([1 / 1.5]),
                       3: np.array([0.4, 1.0]),
                       4: np.array([0.25, 1 / 1.5, 1.1])
                       }

    def __init__(self,
                 root_dir,
                 mode="train",
                 score_classes=2,
                 num_points=102400,
                 score_func=RobustScoreFunc,
                 distance_threshold=0.05,
                 std_R=0.0,
                 std_t=0.0, ):
        super(YCBScenes, self).__init__()
        self.root_dir = os.path.abspath(root_dir)
        self.mode = mode
        self.score_classes = score_classes
        self.score_grad = self.score_grad_dict[score_classes]
        self.num_points = num_points
        self.num_frame_points = num_points // 8
        self.score_func = score_func()
        self.distance_threshold = distance_threshold

        self.std_R = std_R
        self.std_t = std_t
        
        # self.path_list = np.array(sorted(os.listdir(self.root_dir)))
        np.random.seed(1)

        if 'eval_data' in self.root_dir:
            p_path = np.array(sorted(os.listdir(self.root_dir)))
            index = np.random.choice(len(p_path), int(len(p_path)*0.8), replace=False)
            if self.tag != "train":
                ori = np.arange(len(p_path)) 
                index = np.array(list( set(list(ori)) - set(list(index)) ))
        else:
            if mode == "test":
                base_dir_name = "merged_data_test"
            else:
                base_dir_name = "merged_data"
            self.root_dir = os.path.join(self.root_dir, base_dir_name) 
            p_path = np.array(sorted(os.listdir(self.root_dir)))

            if mode == "test":
                index = np.arange(len(p_path)) 
            else:
                index = np.random.choice(len(p_path), int(len(p_path)*0.8), replace=False)
                if mode != "train":
                    ori = np.arange(len(p_path)) 
                    index = np.array(list( set(list(ori)) - set(list(index)) ))
        self.path_list = p_path[index]
        
        logger = logging.getLogger('tdgpd.dataset')
        logger.info("Num of classes: {}, grad: {}".format(self.score_classes, self.score_grad))
        logger.info("Data length: {}, score function: {}".format(len(self.path_list), self.score_func))

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.path_list[index])
        data = np.load(path, allow_pickle=True)
        search_score = data["search_score"]
        antipodal_score = data["antipodal_score"]
        # point_cloud = data["point_cloud"]
        point_cloud = data["view_cloud"].T

        frame_index = data["valid_index"]
        valid_frame = data["valid_frame"]
        direction = data["direction"]
        LENGTH_SEARCH, GRASP_PER_LENGTH = search_score.shape[1:]
        
        # direction = np.zeros(180)
        movable = (direction > self.distance_threshold).astype(np.float)
        objects_label = np.min(data["objects_label"], axis=(1, 2))  # (N, )
        scene_point_movable = movable[objects_label]

        num_frame = frame_index.shape[0]
        if num_frame == 0:
            print("No valid frame in ", path)
            os.system("rm {}".format(path))
            valid_frame = np.zeros((0, 48, 4, 4))
        else:
            valid_frame = np.reshape(valid_frame, (num_frame, -1, 4, 4))

        LOCAL_SEARCH = valid_frame.shape[1]

        # augmentation
        H = np.eye(4)
        if self.std_R > 0:
            rx, ry, rz = np.clip(np.random.randn(3) * self.std_R, -2 * self.std_R, 2 * self.std_R)
            rx = np.array([[1.0, 0.0, 0.0],
                           [0.0, np.cos(rx), -np.sin(rx)],
                           [0.0, np.sin(rx), np.cos(rx)]])
            ry = np.array([[np.cos(ry), 0.0, np.sin(ry)],
                           [0.0, 1.0, 0.0],
                           [-np.sin(ry), 0.0, np.cos(ry)]])
            rz = np.array([[np.cos(rz), -np.sin(rz), 0.0],
                           [np.sin(rz), np.cos(rz), 0.0],
                           [0.0, 0.0, 1.0]])
            R = rx @ ry @ rz
            H[:3, :3] = R
        if self.std_t > 0:
            t = np.clip(np.random.randn(3) * self.std_t, -2 * self.std_t, 2 * self.std_t)
            H[:3, 3] = t

        H4 = np.tile(H[np.newaxis, np.newaxis, ...], (valid_frame.shape[0], valid_frame.shape[1], 1, 1))
        valid_frame = np.matmul(H4, valid_frame)

        num_point_in_data = point_cloud.shape[1]
        scored_grasp, scored_scene_points, local_search_index = self.score_func(search_score, antipodal_score)
        length_search_index = local_search_index // GRASP_PER_LENGTH
        # print(length_search_index)
        # print(local_search_index)
        # print(GRASP_PER_LENGTH)

        # advanced indexing
        ind0 = np.arange(0, num_frame).reshape((num_frame, 1, 1))
        ind0 = np.tile(ind0, (1, 4, 4))

        ind1 = np.reshape(local_search_index, (num_frame, 1, 1))
        ind1 = np.tile(ind1, (1, 4, 4))

        ind2 = np.arange(4).reshape((1, 4, 1))
        ind2 = np.tile(ind2, (num_frame, 1, 4))

        ind3 = np.arange(4).reshape((1, 1, 4))
        ind3 = np.tile(ind3, (num_frame, 4, 1))

        best_frame = valid_frame[ind0, ind1, ind2, ind3]

        scored_scene_point_labels = np.sum(scored_scene_points[:, np.newaxis] > self.score_grad[np.newaxis, :], axis=1)
        scored_grasp_labels = np.sum(scored_grasp[..., np.newaxis] > self.score_grad[np.newaxis, np.newaxis, ...],
                                     axis=-1)  # (N, LOCAL_SEARCH)

        if num_frame >= self.num_frame_points:
            frame_point_inds = np.random.choice(np.arange(num_frame), self.num_frame_points, replace=False)
        else:
            frame_point_inds = np.random.choice(np.arange(num_frame), self.num_frame_points, replace=True)

        remain_point_inds = np.setdiff1d(np.arange(num_point_in_data), frame_index)
        if remain_point_inds.shape[0] > (self.num_points - self.num_frame_points):
            remain_point_inds = np.random.choice(remain_point_inds, self.num_points - self.num_frame_points,
                                                 replace=False)
        else:
            remain_point_inds = np.random.choice(remain_point_inds, self.num_points - self.num_frame_points,
                                                 replace=True)

        best_frame = best_frame[frame_point_inds, ...]
        local_search_frame = valid_frame[frame_point_inds, ...]  # (N, LOCAL_SEARCH, 4, 4)
        R = local_search_frame[:, :, :3, :3].reshape(local_search_frame.shape[0], local_search_frame.shape[1], 9)
        t = local_search_frame[:, :, :3, 3].reshape(local_search_frame.shape[0], local_search_frame.shape[1], 3)
        local_search_frame = np.concatenate([R, t], axis=-1)
        frame_index = frame_index[frame_point_inds]
        scored_scene_point_labels = np.concatenate([scored_scene_point_labels[frame_point_inds],
                                                    np.zeros_like(remain_point_inds)])

        length_search_index = length_search_index[frame_point_inds]
        # print(length_search_index.max())

        scored_grasp_labels = scored_grasp_labels[frame_point_inds]  # (N, LOCAL_SEARCH)

        scene_point_movable = np.concatenate([scene_point_movable[frame_point_inds],
                                              np.zeros((remain_point_inds.shape[0], scene_point_movable.shape[1]))])
        scored_scene_points = np.concatenate([scored_scene_points[frame_point_inds], np.zeros_like(remain_point_inds)])
        chosen_inds = np.concatenate([frame_index, remain_point_inds])
        scene_points = point_cloud[:, chosen_inds]

        scene_points = np.dot(H[:3, :3], scene_points) + H[:3, 3:4]

        return {
            "scene_points": torch.tensor(scene_points).float(),
            "scene_movable_labels": torch.tensor(scene_point_movable).transpose(0, 1).float(),
            "scene_score_labels": torch.tensor(scored_scene_point_labels).long(),
            "scene_score": torch.tensor(scored_scene_points).float(),
            "frame_index": torch.tensor(np.arange(self.num_frame_points)).long(),
            "best_frame_R": torch.tensor(best_frame[:, :3, :3]).float().view(-1, 9).transpose(0, 1),
            # "best_frame_t": torch.tensor(best_frame[:, :3, 3]).float().transpose(0, 1),
            "best_frame_t": torch.tensor(length_search_index).long(),
            # "local_search_frame": torch.tensor(local_search_frame).float().permute(2, 0, 1),  # (12, N, LOCAL_SEARCH)
            # "scored_grasp_labels": torch.tensor(scored_grasp_labels).long(),  # (N, LOCAL_SEARCH)
            "data_path":  path,
        }

    def __len__(self):
        return len(self.path_list)#//100

    def get_stat(self):
        num_point_in_data_list = []
        num_local_search_list = []
        num_frame_list = []

        scored_scene_points_list = []

        for path in tqdm(self.path_list):
            data = np.load(path, allow_pickle=True)
            search_score = data["search_score"]
            antipodal_score = data["antipodal_score"]
            object_num = data["objects_num"]
            point_cloud = data["point_cloud"]
            frame_index = data["frame_index"]

            num_frame = frame_index.shape[0]
            num_frame_list.append(num_frame)

            num_point_in_data, num_local_search = search_score.shape
            num_point_in_data_list.append(num_point_in_data)
            num_local_search_list.append(num_local_search)

            _, scored_scene_points = self.score_func(search_score, antipodal_score, object_num)

            scored_scene_points_list.append(scored_scene_points)

        scored_scene_points_list = np.concatenate(scored_scene_points_list, axis=0)

        print("Number of points:\n mean: {:.2f}, std: {:.2f}, min: {:d}, max: {:d}".format(
            np.mean(num_point_in_data_list), np.std(num_point_in_data_list, ddof=1), np.min(num_point_in_data_list),
            np.max(num_point_in_data_list)))

        print("Number of local search:\n mean: {:.2f}, std: {:.2f}, min: {:d}, max: {:d}".format(
            np.mean(num_local_search_list), np.std(num_local_search_list, ddof=1), np.min(num_local_search_list),
            np.max(num_local_search_list)))

        print("Number of frames: \n mean: {:.2f}, std: {:.2f}, min: {:d}, max: {:d}".format(
            np.mean(num_frame_list), np.std(num_frame_list, ddof=1), np.min(num_frame_list),
            np.max(num_frame_list)))

        print("Scored scene points: \n mean: {:.2f}, std: {:.2f}, min: {:.2f}, max: {:.2f}".format(
            np.mean(scored_scene_points_list), np.std(scored_scene_points_list, ddof=1),
            np.min(scored_scene_points_list),
            np.max(scored_scene_points_list)))

        norm_scored_scene_points = (scored_scene_points_list - np.min(scored_scene_points_list)) / (
                np.max(scored_scene_points_list) - np.min(scored_scene_points_list) + 1e-4)

        lins = np.linspace(0, 1, 51)
        num_grasp = []
        num_points = []

        for l in lins:
            num_points.append(np.sum(np.logical_and(norm_scored_scene_points >= l, norm_scored_scene_points < l + 1)))

        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(lins * (np.max(scored_scene_points_list) - np.min(scored_scene_points_list)) + np.min(
            scored_scene_points_list),
                 np.asarray(num_points) / norm_scored_scene_points.size, 'b', linewidth=2)
        plt.show()
        print("Scored scene point distribution: \n", num_points)


class YCBScenesContact(Dataset):
    score_grad_dict = {2: np.array([1 / 1.5]),
                       3: np.array([3, 6.0]),
                       4: np.array([0.25, 1 / 1.5, 1.1])
                       }

    def __init__(self,
                 root_dir,
                 mode="train",
                 score_classes=2,
                 num_points=102400,
                 score_func=RobustScoreFunc,
                 distance_threshold=0.05,
                 std_R=0.0,
                 std_t=0.0, ):
        super(YCBScenesContact, self).__init__()
        self.root_dir = os.path.abspath(root_dir)
        self.mode = mode
        self.score_classes = score_classes
        self.score_grad = self.score_grad_dict[score_classes]
        self.num_points = num_points
        self.num_frame_points = num_points // 8
        self.score_func = score_func()
        self.distance_threshold = distance_threshold

        self.std_R = std_R
        self.std_t = std_t

        self.path_list = sorted(self.root_dir.files("*.p"))
        logger = logging.getLogger('tdgpd.dataset')
        logger.info("Num of classes: {}, grad: {}".format(self.score_classes, self.score_grad))
        logger.info("Data length: {}, score function: {}".format(len(self.path_list), self.score_func))

    def __getitem__(self, index):
        path = self.path_list[index]
        data = np.load(path, allow_pickle=True)
        search_score = data["search_score"]
        antipodal_score = data["antipodal_score"]
        point_cloud = data["point_cloud"]
        frame_index = data["valid_index"]
        valid_frame = data["valid_frame"]
        direction = data["direction"]
        movable = (direction > self.distance_threshold).astype(np.float)
        objects_label = data["objects_label"]  # (N, )
        scene_point_movable = movable[objects_label]
        num_frame = search_score.shape[0]
        if num_frame == 0:
            print("No valid frame in ", path)
            os.system("rm {}".format(path))
        # augmentation
        H = np.eye(4)
        if self.std_R > 0:
            rx, ry, rz = np.clip(np.random.randn(3) * self.std_R, -2 * self.std_R, 2 * self.std_R)
            rx = np.array([[1.0, 0.0, 0.0],
                           [0.0, np.cos(rx), -np.sin(rx)],
                           [0.0, np.sin(rx), np.cos(rx)]])
            ry = np.array([[np.cos(ry), 0.0, np.sin(ry)],
                           [0.0, 1.0, 0.0],
                           [-np.sin(ry), 0.0, np.cos(ry)]])
            rz = np.array([[np.cos(rz), -np.sin(rz), 0.0],
                           [np.sin(rz), np.cos(rz), 0.0],
                           [0.0, 0.0, 1.0]])
            R = rx @ ry @ rz
            H[:3, :3] = R
        if self.std_t > 0:
            t = np.clip(np.random.randn(3) * self.std_t, -2 * self.std_t, 2 * self.std_t)
            H[:3, 3] = t

        H4 = np.tile(H[np.newaxis, ...], (valid_frame.shape[0], 1, 1))
        valid_frame = np.matmul(H4, valid_frame)

        num_point_in_data = point_cloud.shape[1]

        scored_scene_points = np.minimum(np.log(search_score + 1) / 4, np.ones([1])) * antipodal_score \
                              / np.sqrt(1 - np.power(antipodal_score, 2))

        scored_scene_point_labels = np.sum(scored_scene_points[:, np.newaxis] > self.score_grad[np.newaxis, :], axis=1)

        if num_frame >= self.num_frame_points:
            frame_point_inds = np.random.choice(np.arange(num_frame), self.num_frame_points, replace=False)
        else:
            frame_point_inds = np.random.choice(np.arange(num_frame), self.num_frame_points, replace=True)

        remain_point_inds = np.setdiff1d(np.arange(num_point_in_data), frame_index)
        if remain_point_inds.shape[0] > (self.num_points - self.num_frame_points):
            remain_point_inds = np.random.choice(remain_point_inds, self.num_points - self.num_frame_points,
                                                 replace=False)
        else:
            remain_point_inds = np.random.choice(remain_point_inds, self.num_points - self.num_frame_points,
                                                 replace=True)

        best_frame = valid_frame[frame_point_inds, ...]
        frame_index = frame_index[frame_point_inds]
        scored_scene_point_labels = np.concatenate([scored_scene_point_labels[frame_point_inds],
                                                    np.zeros_like(remain_point_inds)])

        scene_point_movable = np.concatenate([scene_point_movable[frame_point_inds],
                                              np.zeros((remain_point_inds.shape[0], scene_point_movable.shape[1]))])
        scored_scene_points = np.concatenate([scored_scene_points[frame_point_inds], np.zeros_like(remain_point_inds)])
        chosen_inds = np.concatenate([frame_index, remain_point_inds])
        scene_points = point_cloud[:, chosen_inds]

        scene_points = np.dot(H[:3, :3], scene_points) + H[:3, 3:4]

        return {
            "scene_points": torch.tensor(scene_points).float(),
            "scene_movable_labels": torch.tensor(scene_point_movable).transpose(0, 1).float(),
            "scene_score_labels": torch.tensor(scored_scene_point_labels).long(),
            "scene_score": torch.tensor(scored_scene_points).float(),
            "frame_index": torch.tensor(np.arange(self.num_frame_points)).long(),
            "best_frame_R": torch.tensor(best_frame[:, :3, :3]).float().view(-1, 9).transpose(0, 1),
            "best_frame_t": torch.tensor(best_frame[:, :3, 3]).float().transpose(0, 1),
        }

    def __len__(self):
        return len(self.path_list)


class YCBScenesBaseLine(Dataset):
    score_grad_dict = {2: np.array([1 / 1.5]),
                       3: np.array([0.5, 1.0]),
                       }

    def __init__(self, root_dir, mode="train", score_classes=2,
                 num_close_region_points=128, gpd_in_channels=3,
                 score_func=GraspScoreFunc):
        super(YCBScenesBaseLine, self).__init__()
        self.root_dir = os.path.abspath(root_dir)
        self.mode = mode
        self.score_classes = score_classes
        self.score_grad = self.score_grad_dict[score_classes]
        self.num_close_region_points = num_close_region_points
        assert gpd_in_channels in [3, 12], "Only supports 3 or 12 channels."
        self.gpd_in_channels = gpd_in_channels
        self.score_func = score_func()
        if osp.exists(self.root_dir / "{}_list.pkl".format(mode)):
            with open(self.root_dir / "{}_list.pkl".format(mode), "rb") as f:
                self.grasp_list = pickle.load(f)
        else:
            self.grasp_list = self._generate_balance_list()
            with open(self.root_dir / "{}_list.pkl".format(mode), "wb") as f:
                pickle.dump(self.grasp_list, f)
        logger = logging.getLogger('tdgpd.dataset_baseline')
        logger.info("Data length: {}, score function: {}, MD5: {}".format(len(self.grasp_list), self.score_func,
                                                                          get_md5(self.root_dir / "{}_list.pkl".format(
                                                                              mode))))

    def _generate_balance_list(self):
        grasp_list = []
        path_list = sorted(self.root_dir.files("*_?.p"))
        grad_list = np.array([1 / 2.5, 1 / 2.0, 1 / 1.5, 1.1])
        grasp_list_per_grad = {}
        for p in tqdm(path_list):
            with open(p, "rb") as f:
                try:
                    data = pickle.load(f)
                except:
                    print(p)
                    os.system("rm -f {}".format(p))
                    continue
            antipodal_score = data["antipodal_score"]
            grasp_score = self.score_func(antipodal_score)
            for i in range(grasp_score.shape[0]):
                score = grasp_score[i]
                grad = np.sum(grad_list < score).astype(int)
                grasp_path = p[:-2] + "_{:08d}.pkl".format(i)
                if not os.path.exists(grasp_path):
                    with open(grasp_path, "wb") as fw:
                        d = {"antipodal_score": antipodal_score[i],
                             "close_region_points": data["close_region_points_set"][i],
                             "close_region_projection_map": data["close_region_projection_map_set"][i],
                             }
                        pickle.dump(d, fw)
                if grad in grasp_list_per_grad.keys():
                    grasp_list_per_grad[grad].append(grasp_path)
                else:
                    grasp_list_per_grad[grad] = [grasp_path]

        num_grasp_per_grad = [len(grasp_list_per_grad[i]) for i in grasp_list_per_grad.keys()]
        print("Number of grasp per grad: ")
        for k in grasp_list_per_grad.keys():
            print(k, ": ", len(grasp_list_per_grad[k]))
        min_num = min(num_grasp_per_grad)
        for k in grasp_list_per_grad.keys():
            grasp_list.extend(random.sample(grasp_list_per_grad[k], min_num))

        return grasp_list

    def convert_to_numpy(self):
        for p in tqdm(self.path_list):
            data = np.load(p, allow_pickle=True)
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                    data[k] = v
            with open(p, "wb") as f:
                pickle.dump(data, f)

    def __getitem__(self, index):
        path = self.grasp_list[index]
        with open(path, "rb") as f:
            data = pickle.load(f)
        antipodal_score = data["antipodal_score"]
        close_region_point = data["close_region_points"]
        close_region_projection_map = data["close_region_projection_map"]

        scored_grasp = self.score_func(antipodal_score)
        scored_label = np.sum(self.score_grad < scored_grasp)
        num_close_region_point = close_region_point.shape[1]
        assert num_close_region_point > 0, "There should at least 1 point in close region. {}.".format(path)
        if num_close_region_point > self.num_close_region_points:
            close_region_inds = np.random.choice(np.arange(num_close_region_point),
                                                 self.num_close_region_points, False)
        else:
            close_region_inds = np.random.choice(np.arange(num_close_region_point),
                                                 self.num_close_region_points, True)

        close_region_point = close_region_point[:, close_region_inds]
        if self.gpd_in_channels == 3:
            close_region_projection_map = close_region_projection_map[1:4]

        return {
            "close_region_points": torch.tensor(close_region_point).float(),
            "close_region_projection_maps": torch.tensor(close_region_projection_map).float(),
            "grasp_score_labels": torch.tensor(scored_label).long(),
            "grasp_score": torch.tensor(scored_grasp).float(),
        }

    def __len__(self):
        return len(self.grasp_list)


class YCBScenesEval(Dataset):
    def __init__(self, root_dir, mode="test",
                 num_scene_points=25600,
                 score_classes=2,
                 num_close_region_points=1024, gpd_in_channels=3):
        super(YCBScenesEval, self).__init__()
        self.root_dir = os.path.abspath(root_dir)
        self.mode = mode
        self.num_scene_points = num_scene_points
        self.score_classes = score_classes
        self.num_close_region_points = num_close_region_points
        assert gpd_in_channels in [3, 12], "Only supports 3 or 12 channels."
        self.gpd_in_channels = gpd_in_channels
        self.scene_list = self._generate_scene_list()

        logger = logging.getLogger('tdgpd.dataset_eval')
        logger.info("Data length: {}.".format(len(self.scene_list)))

    def _generate_scene_list(self):
        path_list = sorted(self.root_dir.files("*_*.p"))
        return path_list

    def __getitem__(self, index):
        path = self.scene_list[index]
        with open(path, "rb") as f:
            data = pickle.load(f)

        view_frame = np.array([[0.79885968, 0., 0.60151742, 0.8],
                               [0., 1., 0., 0.],
                               [-0.60151742, 0., 0.79885968, 1.7],
                               [0., 0., 0., 1.]])
        view_frame = np.linalg.inv(view_frame)
        view_R = view_frame[:3, :3]
        view_t = view_frame[:3, 3]
        frame = data["frame"]
        frame = np.stack([np.matmul(view_frame, f) for f in frame])
        antipodal_score = data["antipodal_score"]
        non_collision_bool = data["non_collision_bool"]
        single_label_bool = data["single_label_bool"]
        valid_baseline_grasp_num = antipodal_score.shape[0]
        view_cloud = data["view_cloud"]
        view_cloud = (np.matmul(view_R, view_cloud.T) + view_t[..., np.newaxis]).T

        if view_cloud.shape[0] > self.num_scene_points:
            rand_inds = np.random.choice(np.arange(view_cloud.shape[0]), self.num_scene_points, replace=False)
        else:
            rand_inds = np.random.choice(np.arange(view_cloud.shape[0]), self.num_scene_points, replace=True)
        view_cloud = view_cloud[rand_inds, :]

        scene_cloud = data["scene_cloud"]
        scene_label = data["scene_label"]
        scene_normal = data["scene_normal"]

        scene_cloud = (np.matmul(view_R, scene_cloud.T) + view_t[..., np.newaxis]).T
        scene_normal = scene_normal @ view_R.T
        close_region_points_set = []
        close_region_projection_map_set = []

        for i in range(valid_baseline_grasp_num):
            if data["close_region_points_set"][i].shape[1] >= self.num_close_region_points:
                rand_inds = np.random.choice(np.arange(data["close_region_points_set"][i].shape[1]),
                                             self.num_close_region_points, replace=False)
            else:
                rand_inds = np.random.choice(np.arange(data["close_region_points_set"][i].shape[1]),
                                             self.num_close_region_points, replace=True)
            close_region_points_set.append(data["close_region_points_set"][i][:, rand_inds])
            if self.gpd_in_channels == 3:
                close_region_projection_map_set.append(data["close_region_projection_map_set"][i][1:4])
            else:
                close_region_projection_map_set.append(data["close_region_projection_map_set"][i])

        try:
            close_region_points_set = np.stack(close_region_points_set, axis=0)
        except:
            print(path)
        close_region_projection_map_set = np.stack(close_region_projection_map_set, axis=0)

        return {
            "frame": torch.tensor(frame).float(),
            "antipodal_score": torch.tensor(antipodal_score).float(),
            "non_collision_bool": torch.tensor(non_collision_bool).long(),
            "single_label_bool": torch.tensor(single_label_bool).long(),
            "scene_points": torch.tensor(view_cloud.T).float(),
            "complete_cloud": torch.tensor(scene_cloud.T).float(),
            "close_region_points": torch.tensor(close_region_points_set).float(),
            "close_region_projection_maps": torch.tensor(close_region_projection_map_set).float(),
            "scene_label": torch.tensor(scene_label).long(),
            "scene_normal": torch.tensor(scene_normal).float()
        }

    def __len__(self):
        return len(self.scene_list)


def build_data_loader(cfg, mode="train"):
    print(cfg.DATA.NUM_POINTS)
    if cfg.DATA.TYPE == "Scene":
        if mode == "train":
            dataset = YCBScenes(
                root_dir=cfg.DATA.TRAIN.ROOT_DIR,
                mode="train",
                score_classes=cfg.DATA.SCORE_CLASSES,
                num_points=cfg.DATA.NUM_POINTS,
                std_R=cfg.DATA.STD_R,
                std_t=cfg.DATA.STD_T,
            )
        elif mode == "val":
            dataset = YCBScenes(
                root_dir=cfg.DATA.VAL.ROOT_DIR,
                mode="val",
                score_classes=cfg.DATA.SCORE_CLASSES,
                num_points=cfg.DATA.NUM_POINTS,
            )
        elif mode == "test":
            dataset = YCBScenes(
                root_dir=cfg.DATA.TEST.ROOT_DIR,
                mode="test",
                score_classes=cfg.DATA.SCORE_CLASSES,
                num_points=cfg.DATA.NUM_POINTS,
            )
        elif mode == "test all":
            dataset = YCBScenesAll(
                root_dir=cfg.DATA.TEST.ROOT_DIR,
                mode="test",
                score_classes=cfg.DATA.SCORE_CLASSES,
                num_points=cfg.DATA.NUM_POINTS,
            )
        elif mode == "test regrad":
            dataset = YCBScenesREGRAD(
                root_dir=cfg.DATA.TEST.ROOT_DIR,
                mode="test",
                score_classes=cfg.DATA.SCORE_CLASSES,
                num_points=cfg.DATA.NUM_POINTS,
            )
            # dataset = YCBScenesEval(
            #     root_dir=cfg.DATA.TEST.ROOT_DIR,
            #     mode="test",
            #     score_classes=cfg.DATA.SCORE_CLASSES,
            #     num_scene_points=cfg.DATA.NUM_POINTS,
            # )
        else:
            raise ValueError("Unknown mode: {}.".format(mode))
    elif cfg.DATA.TYPE == "Grasp":
        if mode == "train":
            dataset = YCBScenesBaseLine(
                root_dir=cfg.DATA.TRAIN.ROOT_DIR,
                mode="train",
                score_classes=cfg.DATA.SCORE_CLASSES,
                num_close_region_points=cfg.DATA.NUM_CLOSE_REGION_POINTS,
                gpd_in_channels=cfg.DATA.GPD_IN_CHANNELS
            )
        elif mode == "val":
            dataset = YCBScenesBaseLine(
                root_dir=cfg.DATA.VAL.ROOT_DIR,
                mode="val",
                score_classes=cfg.DATA.SCORE_CLASSES,
                num_close_region_points=cfg.DATA.NUM_CLOSE_REGION_POINTS,
                gpd_in_channels=cfg.DATA.GPD_IN_CHANNELS
            )
        elif mode == "test":
            dataset = YCBScenesEval(
                root_dir=cfg.DATA.TEST.ROOT_DIR,
                mode="test",
                score_classes=cfg.DATA.SCORE_CLASSES,
                num_close_region_points=cfg.DATA.NUM_CLOSE_REGION_POINTS,
                gpd_in_channels=cfg.DATA.GPD_IN_CHANNELS,
            )
        else:
            raise ValueError("Unknown mode: {}.".format(mode))
    elif cfg.DATA.TYPE == "CONTACT":
        if mode == "train":
            dataset = YCBScenesContact(
                root_dir=cfg.DATA.TRAIN.ROOT_DIR,
                mode="train",
                score_classes=cfg.DATA.SCORE_CLASSES,
                num_points=cfg.DATA.NUM_POINTS,
                std_R=cfg.DATA.STD_R,
                std_t=cfg.DATA.STD_T,
            )
        elif mode == "val":
            dataset = YCBScenesContact(
                root_dir=cfg.DATA.VAL.ROOT_DIR,
                mode="val",
                score_classes=cfg.DATA.SCORE_CLASSES,
                num_points=cfg.DATA.NUM_POINTS,
            )
        elif mode == "test":
            dataset = YCBScenesEval(
                root_dir=cfg.DATA.TEST.ROOT_DIR,
                mode="test",
                score_classes=cfg.DATA.SCORE_CLASSES,
                num_scene_points=cfg.DATA.NUM_POINTS,
            )
    else:
        raise ValueError("Unknown data type: {}.".format(cfg.DATA.TYPE))

    if mode == "train":
        batch_size = cfg.TRAIN.BATCH_SIZE
    else:
        batch_size = cfg.TEST.BATCH_SIZE

    data_loader = DataLoader(
        dataset,
        batch_size,
        shuffle=(mode == "train"),
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=True,
    )

    return data_loader


if __name__ == "__main__":
    # dataset = YCBScenesBaseLine(
    #     root_dir="/home/rayc/Projects/3DGPD/data/training_data_baseline",
    #     gpd_in_channels=12,
    # )
    # print("YCB BaseLine Scenes: ", len(dataset))
    # print("Score function: ", dataset.score_func)
    # for k, v in dataset[0].items():
    #     print(k, v.shape)

    # dataset = YCBScenes(
    #     root_dir="/home/rayc/Projects/3DGPD/data/ycb_data/training_data",
    # )
    # print("YCB Scenes: ", len(dataset))
    # print("Score function: ", dataset.score_func)
    # for k, v in dataset[0].items():
    #     print(k, v.shape)
    # dataset.get_stat()
    pass
