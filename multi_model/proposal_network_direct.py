import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import sys 
sys.path.append("..")
from dataset_utils.graspdataset import get_grasp_gt
from multi_model.utils.pointnet2 import PointNet2Direct
from fvcore.nn import sigmoid_focal_loss_jit as focal_loss

class GraspDirectNetwork(nn.Module):
    def __init__(self, radius=0.06, grasp_channel=9, rgb_flag=True, multi_flag=False, k_obj=2, sample_layer=1, conf_times=0.0025):
        super(GraspDirectNetwork, self).__init__()
        self.k_obj = k_obj
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.multi_flag = multi_flag
        self.conf_times = conf_times
        
        self.anchor_number = 1
        # (1+n: grasp classification, grasp params(x,y,z,rx,ry,rz,theta), antipodal_score, center_score)
        self.reg_channel = grasp_channel
        self.radius = radius
        self.sample_layer = sample_layer
        
        self.network = PointNet2Direct(sample_layer=self.sample_layer, k_reg=self.reg_channel, \
                                k_obj=self.k_obj, use_rgb=rgb_flag, use_multi=multi_flag)

        # self.criterion_cls  = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor([4]))
        self.criterion_cls = nn.CrossEntropyLoss(reduction='mean')
        self.criterion_reg = nn.MSELoss(reduction='mean')
        self.smoothl1_loss = nn.SmoothL1Loss(reduction='mean')
        self.criterion_cos = nn.CosineEmbeddingLoss(reduction='mean')
        
    def compute_conf_loss(self, pre_conf, select_index, select_thre_index, gt_conf):
        '''
          Input:  
            pre_conf     : [B, select_points_before_conf_thre_num] torch.float32
            select_index : [B, select_points_before_conf_thre_num] torch.int64
            gt_conf      : [B, N]                                  torch.float32
        '''
        loss = torch.zeros(1).to(self.device)
        if gt_conf is not None:
            loss = self.criterion_reg(pre_conf, gt_conf)
            # loss = self.criterion_reg(pre_conf, gt_conf.gather(1,select_index))
        return loss

    def compute_reg_loss(self, predict_cls, predict_reg, select_xyz, data_paths):
        '''
          Input:  
            predict_cls: [B, 2*self.anchor_number, select_points_number] 
            predict_reg: [B, (self.reg_channel-2)*self.anchor_number, select_points_number] 
            select_xyz : [B, select_points_number, 3]
            data_paths : [B]
        '''
        B, N, _ = select_xyz.shape
        # gt: [B, select_points_number, 9]
        gt_cls, gt = None, None
        if data_paths is not None:
            gt = get_grasp_gt(select_xyz, data_paths, score_thre=None)
            # gt_cls: [B*select_points_number]
            gt_cls = self._generate_cls_gt(gt[:,:,3:7])

        pre_anchor_cls = predict_cls.view(B, self.anchor_number, -1, N).permute(0,3,1,2)
        pre_grasp      = predict_reg.view(B, self.anchor_number, -1, N).permute(0,3,1,2)
        
        pre_index, gt_index, cls_loss, acc_tuple = self._cls_loss(pre_anchor_cls.contiguous().view(-1,2), gt_cls)
        pre_label, reg_loss, loss_tuple          = self._grasp_label_loss(select_xyz, pre_grasp, pre_index, gt_index, gt)
        loss = reg_loss #cls_loss/10+

        loss_tuple.extend(acc_tuple)
        loss_tuple.append(cls_loss.data)
        loss_tuple.append(reg_loss.data)
        return pre_label, loss, loss_tuple

    def forward(self, pc, pc_conf=None, pc_label=None, data_path=None, cur_idx=None, data_width=None):
        '''
         Input:
          pc              :[B,A,6]
          pc_conf         :[B,A]
          pc_label        :[B,A]
         Output:
          pre_grasp       :[B,A,Feature(128)])
          loss
        '''
        B, N, _ = pc.shape
        # x_conf [B, layer_points_number], x_reg[B, self.reg_channel*self.anchor_number, select_points_number]
        # select_xyz[B, 3, select_points_number], select_index [B, layer_points_number]
        x_conf, x_cls, x_reg, select_xyz, select_index, select_thre_index = \
                            self.network(pc[:,:,:6].permute(0,2,1), conf_times=self.conf_times, add_widths=data_width) 
        # x_cls, x_reg, select_xyz = self.network(pc[:,:,:6].permute(0,2,1), conf_times=0.15, add_widths=data_width) 

        if data_path is not None and cur_idx is not None:
            data_path = data_path[cur_idx.cpu()]
        pre_grasp, loss_grasp, loss_tuple = self.compute_reg_loss(x_cls, x_reg, select_xyz.transpose(2,1), data_path)
        ## don't use confidence 
        # loss_tuple.append(torch.tensor([0]).float().to(self.device))
        # loss = loss_grasp

        ## use confidence 
        loss_conf = self.compute_conf_loss(x_conf, select_index, select_thre_index, pc_conf)
        loss = loss_grasp + loss_conf
        loss_tuple.append(loss_conf)
        return None, None, pre_grasp, loss, loss_tuple, None

    def _cls_loss(self, pre, target):
        '''
            pre    : [B*select_points_number*self.anchor_number] torch.float32
            target : [B*select_points_number*self.anchor_number] torch.float32
        '''
        _, order = torch.sort(pre, dim=-1, descending=True)
        pre_index = torch.nonzero(order[:,0] <= 1).view(-1)
        print("predict cls", len(pre_index))

        loss = torch.zeros(1).to(self.device)
        acc, recall = torch.zeros(1).to(self.device), torch.zeros(1).to(self.device)
        gt_index = None
        if target is not None:
            gt_cls1, gt_cls0 = torch.nonzero(target == 1).view(-1), torch.nonzero(target == 0).view(-1)
            print("true length", len(gt_cls1), "false length", len(gt_cls0))
            gt_index = gt_cls1.clone()
            # len_cls0, len_cls1 = len(gt_cls0), len(gt_cls1)
            # select_len = min(len_cls0, len_cls1)
            # gt_cls0 = gt_cls0[np.random.choice(len_cls0, select_len, replace=False)]
            # gt_cls1 = gt_cls1[np.random.choice(len_cls1, select_len, replace=False)]
            select_index = torch.cat((gt_cls0, gt_cls1))
            loss = self.criterion_cls(pre[select_index], target[select_index])
            # loss = focal_loss(
            #     pre[select_index],
            #     F.one_hot(target[select_index], num_classes=2).to(pre[select_index]),
            #     alpha = 0.75,
            #     gamma = 0,
            #     reduction="mean",
            # )

            tp = ((order[:,0] == 1) & (target == 1) ).sum()
            tn = ((order[:,0] == 0) & (target == 0) ).sum()
            fp = ((order[:,0] == 1) & (target == 0) ).sum()
            fn = ((order[:,0] == 0) & (target == 1) ).sum()
            acc = (tp+tn) / (fp+fn+tp+tn)
            # acc = tp / (tp+fp)
            recall = tp / (tp+fn)
            print("cls loss:", loss.data, "acc:", acc, "recall:", recall)
            
        return pre_index, gt_index, loss, [acc, recall]

    def _grasp_label_loss(self, select_xyz, pre, pre_index, gt_index, target):
        '''
            select_xyz: [B, select_points_number, 3] torch.float32
            pre       : [B, select_points_number, self.anchor_number, 9] torch.float32
            pre_index : [len(pre_index)]
            gt_index  : [len(gt_index)]
            target    : [B, select_points_number, 9] torch.float32
        '''
        B, N, _, channel = pre.shape
        pre    = pre.contiguous().view(-1,channel)
        center = select_xyz.view(B,N,1,-1).repeat(1,1,self.anchor_number,1).view(-1,3)
        
        pre_ori      = pre[:,3:6]
        pre_sum_ori_ = torch.sqrt(torch.sum(torch.mul(pre_ori, pre_ori), dim=1).add_(1e-12) ).view(-1,1)
        pre_ori      = torch.div(pre_ori, pre_sum_ori_)
        pre_center   = pre[:, :3] * self.radius + center
        pre_theta    = pre[:,6:7] * np.pi
        pre_score    = pre[:,7:]
        pre_label = torch.cat((pre_center, pre_ori, pre_theta, pre_score), dim=-1)
    
        loss = torch.zeros(1).to(self.device)
        loss_tuple = [torch.zeros(1).to(self.device)] * 8
        if gt_index is not None and target is not None:
            target = target.view(B,N,1,-1).repeat(1,1,self.anchor_number,1).view(-1,channel)
            delta_ori = torch.mul(pre[:,3:6], pre_sum_ori_)
            loss_gt1  = self.smoothl1_loss(pre[:,:3][gt_index] , (target[:,:3]-center)[gt_index] / self.radius)
            loss_gt2  = self.smoothl1_loss(pre[:,3:6][gt_index] , (target[:,3:6])[gt_index])
            loss_gt3  = self.smoothl1_loss(pre[:,6:7][gt_index], (target[:,6:7])[gt_index] / np.pi)
            loss_gt4  = self.smoothl1_loss(pre[:,7:][gt_index] , target[:,7:][gt_index])
            print("regress loss:", loss_gt1.data, loss_gt2.data, loss_gt3.data, loss_gt4.data)
            loss = loss_gt1 + loss_gt2 + loss_gt3 + loss_gt4

            y_gt = torch.ones(len(gt_index), 1).to(self.device)
            loss_gt_center = self.smoothl1_loss(pre_center[gt_index], target[:,:3][gt_index]).data
            loss_gt_ori    = self.criterion_cos(pre_ori[gt_index]   , target[:,3:6][gt_index], y_gt).data
            loss_gt_theta  = self.smoothl1_loss(pre_theta[gt_index] , target[:,6:7][gt_index]).data
            loss_gt_score  = self.smoothl1_loss(pre_score[gt_index] , target[:,7:][gt_index]).data
            print("under gt class loss", loss_gt_center, loss_gt_ori, loss_gt_theta, loss_gt_score)
        
            y_pre = torch.ones(len(pre_index), 1).to(self.device)
            loss_pre_center = self.smoothl1_loss(pre_center[pre_index], target[:,:3][pre_index]).data
            loss_pre_ori    = self.criterion_cos(pre_ori[pre_index]   , target[:,3:6][pre_index], y_pre).data
            loss_pre_theta  = self.smoothl1_loss(pre_theta[pre_index] , target[:,6:7][pre_index]).data
            loss_pre_score  = self.smoothl1_loss(pre_score[pre_index] , target[:,7:][pre_index]).data
            print("under pre class loss", loss_pre_center, loss_pre_ori, loss_pre_theta, loss_pre_score)
            loss_tuple = [loss_gt1.data, loss_gt2.data, loss_gt3.data, loss_gt4.data, \
                            loss_pre_center, loss_pre_ori, loss_pre_theta, loss_pre_score]

        return_grasp = torch.full((B*N*self.anchor_number, channel), -1.0).to(self.device)
        return_grasp[pre_index] = pre_label[pre_index]
        return return_grasp.view(B,-1,channel), loss, loss_tuple

    def _generate_cls_gt(self, gt):
        # gt: [B, select_points_number, 4]
        # self.templates [self.anchor_number, 4]
        gt = gt.view(-1, 4)
        grasp_mask = (torch.sum(gt, dim=1) != -4)

        # [B*select_points_number]
        cls_gt = torch.zeros((len(gt))).long()
        cls_gt[grasp_mask] = 1
        return cls_gt.to(self.device)


if __name__ == '__main__':
    pass
