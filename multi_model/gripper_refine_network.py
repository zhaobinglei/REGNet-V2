import numpy as np
import os, sys
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_jit as focal_loss

from multi_model.proposal_network import GraspProposalNetwork
from multi_model.proposal_network_less import GraspLessProposalNetwork
from multi_model.proposal_network_direct import GraspDirectNetwork
from multi_model.utils.pointnet2 import PointNet2TwoStage, PointNet2Refine

sys.path.append('..')
from dataset_utils.graspdataset import compute_distance

class GripperRefineNetwork(nn.Module):
    def __init__(self, gripper_num, radius_refine, radius, grasp_channel, method, rgb_flag=True, \
                    multi_flag=False, sample_layer=1, conf_times=0.0025, use_region=True, use_fps=False, regrad=False):
        super(GripperRefineNetwork, self).__init__()
        self.gripper_number = gripper_num
        self.radius = radius
        self.multi_flag = multi_flag
        # 2: grasp classification
        # grasp_channel: grasp params(x,y,z,rx,ry,rz,theta)+antipodal_score+center_score(n) )
        self.grasp_channel = grasp_channel
        self.conf_times = conf_times
        self.rgb_flag = rgb_flag
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.criterion_cos = nn.CosineEmbeddingLoss(reduction='mean')
        self.smoothl1_loss = nn.SmoothL1Loss(reduction='mean')
        self.criterion_cls = nn.CrossEntropyLoss(reduction='mean')

        if method == 'class_01':
            self.region_model = GraspProposalNetwork(radius, grasp_channel, rgb_flag, multi_flag, \
                                            sample_layer=sample_layer, conf_times=self.conf_times, \
                                            use_region=use_region, use_fps=use_fps, regrad=regrad)
        elif method == 'class_anchornum':
            self.region_model = GraspLessProposalNetwork(radius, grasp_channel, rgb_flag, multi_flag, \
                                            sample_layer=sample_layer, conf_times=self.conf_times)
        elif method == 'noanchor':
            self.region_model = GraspDirectNetwork(radius, grasp_channel, rgb_flag, multi_flag, \
                                            sample_layer=sample_layer, conf_times=self.conf_times)

        self.refine_model = PointNet2Refine(radius_refine, gripper_num, grasp_channel, rgb_flag, sample_layer=0) # sample_layer not use

    def _cls_loss(self, pre, g_stage1, gt, mask):
        '''
          Input:
          # pre     : [B, 2] predicted classification
          # g_stage1: [B, 9]
          # mask    : [B]
          # gt      : [B, 9]
        '''
        _, order = torch.sort(pre, dim=-1, descending=True)

        # center_diff = g_stage1[:,:3] - gt[:,:3]
        # center_mask = torch.norm(center_diff, dim=1) < 0.02
        # necenter_mask = torch.norm(center_diff, dim=1) > 0.04
        # ori_diff    = com_sim_ori(g_stage1[:,3:6], gt[:,3:6]).view(-1)  # 1 - cos(alpha)
        # ori_mask    = ori_diff < 0.3 # < 45 dgree
        # neori_mask  = ori_diff > 0.5 # > 60 dgree
        # theta_diff  = torch.abs(g_stage1[:,6] - gt[:,6]) / math.pi * 180
        # theta_mask  = theta_diff < 45 # < 60 dgree
        # netheta_mask  = theta_diff > 65 # < 60 dgree

        center_diff = g_stage1[:,:3] - gt[:,:3]
        center_mask = torch.norm(center_diff, dim=1) < 0.015
        necenter_mask = torch.norm(center_diff, dim=1) > 0.02
        ori_diff    = com_sim_ori(g_stage1[:,3:6], gt[:,3:6]).view(-1)  # 1 - cos(alpha)
        ori_mask    = ori_diff < 0.3 # < 45 dgree
        neori_mask  = ori_diff >= 0.5 # > 60 dgree
        theta_diff  = torch.abs(g_stage1[:,6] - gt[:,6]) / math.pi * 180
        theta_mask  = theta_diff < 45 # < 60 dgree
        netheta_mask  = theta_diff >= 60 # < 60 dgree

        # class1_mask, class0_mask, mask: [B]
        class1_mask  = center_mask & ori_mask & theta_mask & mask
        # class0_mask  = ~(center_mask & ori_mask & theta_mask) & mask
        class0_mask  = necenter_mask & neori_mask & netheta_mask #& mask
        print("Refine true length", class1_mask.sum(), "false length", class0_mask.sum())

        gt_cls1, gt_cls0 = torch.nonzero(class1_mask == 1).view(-1), torch.nonzero(class0_mask == 1).view(-1)
        len_cls0, len_cls1 = len(gt_cls0), len(gt_cls1)
        select_len = min(len_cls0, len_cls1)
        gt_cls0 = gt_cls0[np.random.choice(len_cls0, select_len, replace=False)]
        gt_cls1 = gt_cls1[np.random.choice(len_cls1, select_len, replace=False)]
        select_index = torch.cat((gt_cls0, gt_cls1))
        target = torch.full([len(gt)], 2).long().to(self.device)
        target[class1_mask], target[class0_mask] = 1, 0
        loss = self.criterion_cls(pre[select_index], target[select_index])

        # mask = class0_mask | class1_mask
        # loss = focal_loss(
        #      pre[mask],
        #      F.one_hot(target[mask], num_classes=2).to(pre[mask]),
        #      alpha = 0.6,
        #      gamma = 2,
        #      reduction="mean",
        # )

        tp = ((order[:,0] == 1) & (target == 1) ).sum()
        tn = ((order[:,0] == 0) & (target == 0) ).sum()
        fp = ((order[:,0] == 1) & (target == 0) ).sum()
        fn = ((order[:,0] == 0) & (target == 1) ).sum()
        acc = (tp+tn) / (fp+fn+tp+tn)
        # acc = tp / (tp+fp)
        recall = tp / (tp+fn)
        print("cls loss:", loss.data, "acc:", acc, "recall:", recall)
        return class1_mask, loss, [acc, recall]

    def _grasp_loss(self, x_reg, g_stage2, g_stage1, gt, pre_select, gt_select):
        '''
          Input:
          # x_reg     : [B, 9] 
          # g_stage2  : [B, 9]
          # g_stage1  : [B, 9]
          # gt        : [B, 9]
          # pre_select: [B]
          # gt_select : [B]
        '''
        loss_gt1 = self.smoothl1_loss(x_reg[gt_select,:3],  (gt[gt_select,:3] -g_stage1[gt_select,:3])  / self.radius)
        loss_gt2 = self.smoothl1_loss(x_reg[gt_select,3:6], (gt[gt_select,3:6]-g_stage1[gt_select,3:6]) )
        loss_gt3 = self.smoothl1_loss(x_reg[gt_select,6:7], (gt[gt_select,6:7]-g_stage1[gt_select,6:7]) / np.pi)
        loss_gt4 = self.smoothl1_loss(x_reg[gt_select,7:],  (gt[gt_select,7:] -g_stage1[gt_select,7:]) )
        print("regress loss:", loss_gt1.data, loss_gt2.data, loss_gt3.data, loss_gt4.data)
        loss = loss_gt1 + loss_gt2 + loss_gt3 + loss_gt4 

        y_gt = torch.ones(gt_select.sum(), 1).to(self.device)
        loss_gt_center = self.smoothl1_loss(g_stage2[gt_select,:3] , gt[gt_select,:3])
        loss_gt_ori    = self.criterion_cos(g_stage2[gt_select,3:6], gt[gt_select,3:6], y_gt)
        loss_gt_theta  = self.smoothl1_loss(g_stage2[gt_select,6:7], gt[gt_select,6:7])
        loss_gt_score  = self.smoothl1_loss(g_stage2[gt_select,7:] , gt[gt_select,7:])
        print("under gt class loss", loss_gt_center, loss_gt_ori, loss_gt_theta, loss_gt_score)
    
        y_pre = torch.ones(pre_select.sum(), 1).to(self.device)
        loss_pre_center = self.smoothl1_loss(g_stage2[pre_select,:3] , gt[pre_select,:3] )
        loss_pre_ori    = self.criterion_cos(g_stage2[pre_select,3:6], gt[pre_select,3:6], y_pre)
        loss_pre_theta  = self.smoothl1_loss(g_stage2[pre_select,6:7], gt[pre_select,6:7])
        loss_pre_score  = self.smoothl1_loss(g_stage2[pre_select,7:] , gt[pre_select,7:] )
        print("under pre class loss", loss_pre_center, loss_pre_ori, loss_pre_theta, loss_pre_score)
        loss_tuple = [loss_gt1.data, loss_gt2.data, loss_gt3.data, loss_gt4.data, \
                        loss_pre_center, loss_pre_ori, loss_pre_theta, loss_pre_score]

        return loss, loss_tuple

    def compute_loss(self, g_stage1, x_reg, x_cls, mask, gt=None):
        '''
          Input:
            # g_stage1 : [B, N*anchor_number,9] regressed grasp from the stage1
            # x_reg    : [B*N*anchor_number,9] delta grasp from the stage2 (loss)
            # x_cls    : [B*N*anchor_number,2]
            # mask     : [B*N*anchor_number] grasp indexes which contains more than 0 points
            # gt       : [B*N*anchor_number,9] 
          Return:
            final_grasp_select       : [len(class_select), 8] 
            select_grasp_class_stage2: [len(class_select), 8]
            class_select             : [len(class_select)]
            loss_stage2              : tuple
            correct_stage2_tuple     : tuple
        '''
        B, all_anchor_num, _ = g_stage1.shape
        print('====')
        print("Average Anchor Number:\t", all_anchor_num)
        print("Average Refine Init Number:\t", mask.sum().item() // B)
        
        pre_cls = torch.max(x_cls, dim=-1)[1]
        pre_select  = (pre_cls==1) & mask
        print("Average After Refine Number:\t", pre_select.sum().item()  // B)
        print(pre_select.sum().item() )

        g_stage1   = g_stage1.view(-1, self.grasp_channel)
        pre_center = x_reg[:, :3] * self.radius + g_stage1[:,:3]

        pre_ori      = x_reg[:,3:6] + g_stage1[:,3:6]
        pre_sum_ori_ = torch.sqrt(torch.sum(torch.mul(pre_ori, pre_ori), dim=1).add_(1e-12) ).view(-1,1)
        pre_ori      = torch.div(pre_ori, pre_sum_ori_)

        pre_theta  = x_reg[:,6:7] * np.pi + g_stage1[:,6:7]
        pre_score  = x_reg[:,7:]  + g_stage1[:,7:]
        g_stage2   = torch.cat((pre_center, pre_ori, pre_theta, pre_score), dim=-1)

        loss = torch.zeros(1).to(self.device)
        loss_tuple = [torch.zeros(1).to(self.device) for i in range(12)] 
        if gt is not None:
            gt_select, cls_loss, acc_tuple = self._cls_loss(x_cls, g_stage1, gt, mask)
            # gt_select, pre_select = mask, mask
            reg_loss, loss_tuple = self._grasp_loss(x_reg, g_stage2.data, g_stage1, gt, pre_select, gt_select)
            loss = cls_loss / 10 + reg_loss#reg_loss#
            loss_tuple.extend(acc_tuple)
            loss_tuple.append(cls_loss.data)
            loss_tuple.append(reg_loss.data)

        g_stage2[~pre_select] = -1.0
        return g_stage2.view(B,-1,self.grasp_channel), loss, loss_tuple

    def _compute_distance(self, points1, points2):
        '''
        points1:  [B, len(points1), 3]
        points2:  [B, len(points2), 3]
        distance: [B, len(points2), len(points1)]
        '''
        B, len1, len2 = points1.shape[0], points1.shape[1], points2.shape[1]
        # for i in range(B):
        #     distance = compute_distance(points1[i], points2[i])
        #     print(distance)
        points1 = points1.repeat(1, len2, 1)
        points2 = points2.view(B,len2,1,3).repeat(1, 1, len1, 1).view(B,-1,3)
        distance = torch.norm(points1.view(-1,3)-points2.view(-1,3), dim=1)
        distance = distance.view(B, len2, len1)
        return distance
        
    def forward(self, pc, gripper_params, pc_conf=None, pc_label=None, data_path=None, cur_idx=None, data_width=None, epoch=-1):
        '''
         Input:
          pc              :[B,A,6]
          pc_conf         :[B,A]
          pc_label        :[B,A]
        '''
        if data_path is not None:
            data_path = data_path[cur_idx.cpu()]
        # all_feature: [B, 256, A],  pre_grasp: [B, N*anchor_number, 9], gt: [B*N*anchor_number, 9]
        all_feature, group_feature, pre_grasp, loss1, loss_tuple1, gt = self.region_model(pc, pc_conf, pc_label, data_path, data_width=data_width)
        # x_cls: [B*N*anchor_number, 2], x_reg: [B*N*anchor_number, self.reg], mask: [B*N*anchor_number] N-> select_points_number
        x_cls, x_reg, mask             = self.refine_model(pre_grasp, pc, all_feature, group_feature, gripper_params)
        pre2_grasp, loss2, loss_tuple2 = self.compute_loss(pre_grasp.data, x_reg, x_cls, mask, gt)
        # loss = loss1
        # if epoch != 0:
        #     loss += 4*loss2
        loss = loss1 + loss2
        return pre2_grasp, pre_grasp, loss, loss_tuple2, loss_tuple1

def com_sim_ori(a, b):
    '''
      input:
         a :[N, 3]
         b :[N, 3]
      output:
         sim :[N, 1] the samller the sim, the closer a and b
    '''
    a_b = torch.sum(torch.mul(a, b), dim=1)
    epsilon = 1e-12
    a2 = torch.add(torch.sum(torch.mul(a, a), dim=1), (epsilon))
    b2 = torch.add(torch.sum(torch.mul(b, b), dim=1), (epsilon))
    div_ab = torch.sqrt(torch.mul(a2, b2))
    sim = torch.div(a_b, div_ab).mul_(-1).add_(1).view(-1,1)
    return sim
    
if __name__ == '__main__':
    pass
