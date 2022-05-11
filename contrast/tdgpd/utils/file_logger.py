import os
import os.path as osp
from os import mkdir
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d
import matplotlib.pyplot as plt
#from tdgpd.utils.io_utils import mkdir
import time

def _transform_grasp(grasp_ori, antipodal_score_ori, gpu_id):
    '''
      Input:
        grasp_ori: [B, center_num, 3, 4] 
                   [[x1, y1, z1, c1],
                    [x2, y2, z2, c2],
                    [x3, y3, z3, c3]]
        antipodal_score_ori: [B, center_num]
      Output:
        grasp_trans:[B, center_num, 9] (center[3], axis_y[3], grasp_angle[1], antipodal_score[1], center_score[1])
    '''
    B, CN = antipodal_score_ori.shape
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    grasp_trans = torch.full((B, CN, 9), -1.0).to(device)

    axis_x = grasp_ori[:,:,:3,0].view(B*CN, 3)
    axis_y = grasp_ori[:,:,:3,1].view(B*CN, 3)
    axis_z = grasp_ori[:,:,:3,2].view(B*CN, 3)
    grasp_angle = torch.atan2(axis_x[:,2], axis_z[:,2])  ## torch.atan(torch.div(axis_x[:,2], axis_z[:,2])) is not OK!!!

    grasp_angle[axis_y[:,0] < 0]         = np.pi-grasp_angle[axis_y[:,0] < 0]
    axis_y[axis_y[:,0] < 0]              = -axis_y[axis_y[:,0] < 0]
    grasp_angle[grasp_angle >= 2*np.pi]  = grasp_angle[grasp_angle >= 2*np.pi] - 2*np.pi
    grasp_angle[grasp_angle <= -2*np.pi] = grasp_angle[grasp_angle <= -2*np.pi] + 2*np.pi
    grasp_angle[grasp_angle > np.pi]     = grasp_angle[grasp_angle > np.pi] - 2*np.pi
    grasp_angle[grasp_angle <= -np.pi]   = grasp_angle[grasp_angle <= -np.pi] + 2*np.pi
    
    no_grasp_mask = ((axis_x[:,0]==-1) & (axis_x[:,1]==-1) & (axis_x[:,2]==-1))
    grasp_angle[no_grasp_mask] = -1
    axis_y[no_grasp_mask] = -1

    grasp_trans[:,:,:3]  = grasp_ori[:,:,:3,3].view(B, CN, 3)
    grasp_trans[:,:,3:6] = axis_y.view(B, CN, 3)
    grasp_trans[:,:,6]   = grasp_angle.view(B, CN)
    grasp_trans[:,:,7]   = antipodal_score_ori
    grasp_trans[:,:,8]   = 0.
    return grasp_trans

def file_logger_noselect(data_batch, preds, step, output_dir, prefix="", with_label=True, gpu_id=0):
    if "grasp_logits" in preds.keys():
        tic = time.time()
        grasp_logits = preds["grasp_logits"]
        grasp_logits = F.softmax(grasp_logits, dim=1).detach().cpu().numpy()
        num_classes = grasp_logits.shape[1]
        score = np.linspace(0, 1, num_classes + 1)[:-1][np.newaxis, :]
        scene_pred = np.sum(score * grasp_logits, axis=1)
        top_ind = np.argsort(-scene_pred)[:100]
        frame = data_batch["frame"][0].cpu().numpy()
        return frame[top_ind], scene_pred[top_ind]

    # total_R, total_t = [], []
    B, _, N = data_batch["scene_points"].shape
    grasps = np.full((B,N,3,4), -1.0)
    scores = np.full((B,N), -1.0)
    if "scene_score_logits" in preds.keys():
        T_STRIDE = 0.002

        for i in range(B):
            scene_points = data_batch["scene_points"][i].cpu().numpy().T
            if with_label:
                scene_score = data_batch["scene_score"][i].cpu().numpy()
                scene_score_labels = data_batch["scene_score_labels"][i].cpu().numpy()
            scene_score_logits = preds["scene_score_logits"][i]
            scene_score_logits = F.softmax(scene_score_logits, dim=0).detach().cpu().numpy().T

            pred_frame_R = preds["frame_R"]
            pred_frame_R = pred_frame_R[i].transpose(0, 1).detach().cpu().numpy()
            pred_frame_R = np.reshape(pred_frame_R, (-1, 3, 3))
            pred_frame_t = preds["frame_t"]
            pred_frame_t = F.softmax(pred_frame_t[i], dim=0).transpose(0, 1).detach().cpu().numpy()
            t_classes = pred_frame_t.shape[1]
            t_score = np.linspace(1, 0, t_classes + 1)[1:][np.newaxis, :]
            pred_frame_t = - (pred_frame_t * t_score).sum(1, keepdims=True) * T_STRIDE * pred_frame_R[:, :, 0] + scene_points

            # pred_frame_t = pred_frame_t[i].transpose(0, 1).detach().cpu().numpy()

            if with_label:
                gt_frame_R = data_batch["best_frame_R"]
                batch_size, _, num_frame_points = gt_frame_R.shape
                gt_frame_R = gt_frame_R[i].transpose(0, 1).detach().cpu().numpy()
                gt_frame_R = np.reshape(gt_frame_R, (num_frame_points, 3, 3))
                gt_frame_t = data_batch["best_frame_t"][i].view(1,-1).transpose(0, 1).detach().cpu().numpy().astype(np.float)

            score_classes = scene_score_logits.shape[1]
            score = np.linspace(0, 1, score_classes + 1)[:-1][np.newaxis, :]
            scene_pred = np.sum(score * scene_score_logits, axis=1)

            score_idx = scene_pred > 0.1
            len_idx = score_idx.sum()
            R = pred_frame_R[score_idx]
            t = pred_frame_t[score_idx]

            grasps[i,:len_idx,:3,:3] = pred_frame_R[score_idx]
            grasps[i,:len_idx,:3,3]  = pred_frame_t[score_idx]
            scores[i,:len_idx] = scene_pred[score_idx]

    grasps, scores = torch.Tensor(grasps), torch.Tensor(scores)  
    grasps = _transform_grasp(grasps, scores, gpu_id)
    return grasps
