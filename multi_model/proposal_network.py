import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import sys 
sys.path.append("..")
from dataset_utils.graspdataset import get_grasp_gt
from multi_model.utils.pointnet2 import PointNet2Region
from fvcore.nn import sigmoid_focal_loss_jit as focal_loss

class GraspProposalNetwork(nn.Module):
    def __init__(self, radius=0.06, grasp_channel=9, rgb_flag=True, multi_flag=False, k_obj=2, \
                    sample_layer=1, conf_times=0.0025, use_region=True, use_fps=False, regrad=False):
        super(GraspProposalNetwork, self).__init__()
        self.k_obj = k_obj
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.multi_flag = multi_flag
        self.conf_times = conf_times
        
        self.templates = _enumerate_templates()#.to(self.device)
        self.anchor_number = self.templates.shape[0]
        # 1: point confidence
        # 2: grasp classification
        # grasp_channel: grasp params(x,y,z,rx,ry,rz,theta)+antipodal_score+center_score(n) )
        self.reg_channel = grasp_channel
        self.radius = radius
        self.regrad = regrad
        self.network = PointNet2Region(sample_layer=sample_layer, k_reg=self.reg_channel, \
                        k_anchor=self.anchor_number, k_obj=self.k_obj, use_rgb=rgb_flag, \
                        use_multi=multi_flag, use_region=use_region, use_fps=use_fps)

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
            predict_reg: [B, self.reg_channel*self.anchor_number, select_points_number] 
            select_xyz : [B, select_points_number, 3]
            data_paths : [B]
        '''
        B, N, _ = select_xyz.shape
        # gt: [B, select_points_number, 9]
        gt_anchor_cls, gt = None, None
        if data_paths is not None:
            gt = get_grasp_gt(select_xyz, data_paths, score_thre=None, regrad=self.regrad)
            # gt_anchor_cls: [B*select_points_number, self.anchor_number]
            gt_anchor_cls = self._generate_anchor_gt(gt[:,:,3:7]).view(-1)

        pre_anchor_cls = predict_cls.view(B, self.anchor_number, -1, N).permute(0,3,1,2)
        pre_grasp      = predict_reg.view(B, self.anchor_number, -1, N).permute(0,3,1,2)
        
        pre_index, gt_index, cls_loss, acc_tuple = self._anchor_cls_loss(pre_anchor_cls.contiguous().view(-1,2), gt_anchor_cls)
        pre_label, reg_loss, loss_tuple, gt      = self._grasp_label_loss(select_xyz, pre_grasp, pre_index, gt_index, gt)
        loss = cls_loss / 10 + reg_loss

        loss_tuple.extend(acc_tuple)
        loss_tuple.append(cls_loss.data)
        loss_tuple.append(reg_loss.data)
        return pre_label, loss, loss_tuple, gt

    def forward(self, pc, pc_conf=None, pc_label=None, data_path=None, cur_idx=None, data_width=None):
        '''
         Input:
          pc              :[B,A,6]
          pc_conf         :[B,A]
          pc_label        :[B,A]
        '''
        B, N, _ = pc.shape
        
        # all_feature [B, 256, A],  x_conf [B, layer_points_number]
        # x_cls  [B, 2*self.anchor_number, select_points_number]
        # x_reg  [B, self.reg_channel*self.anchor_number, select_points_number]
        # select_xyz   [B, 3, select_points_number]   
        # select_index [B, layer_points_number], select_thre_index [B, select_points_number]
        # select_points_number = conf_times * layer_points_number
        all_feature, group_feature, x_conf, x_cls, x_reg, select_xyz, select_index, select_thre_index = \
                    self.network(pc[:,:,:6].permute(0,2,1), conf_times=self.conf_times, add_widths=data_width) #0.2  0.05
        
        group_feature = group_feature.transpose(2,1)
        group_feature = group_feature.view(B,group_feature.shape[1],1,-1).repeat(1,1,self.anchor_number,1).view(-1,group_feature.shape[2])
        loss_conf = self.compute_conf_loss(x_conf, select_index, select_thre_index, pc_conf)
        if data_path is not None and cur_idx is not None:
            data_path = data_path[cur_idx.cpu()]
        pre_grasp, loss_grasp, loss_tuple, gt = self.compute_reg_loss(x_cls, x_reg, select_xyz.transpose(2,1), data_path)

        loss = loss_conf + loss_grasp
        loss_tuple.append(loss_conf)
        return all_feature, group_feature, pre_grasp, loss, loss_tuple, gt

    def _anchor_cls_loss(self, pre, target):
        '''
            pre    : [B*select_points_number*self.anchor_number] torch.float32
            target : [B*select_points_number*self.anchor_number] torch.float32
        '''
        _, order = torch.sort(pre, dim=-1, descending=True)
        pre_index = torch.nonzero(order[:,0] == 1).view(-1)
        print(len(pre_index))

        loss = torch.zeros(1).to(self.device)
        acc, recall = torch.zeros(1).to(self.device), torch.zeros(1).to(self.device)
        gt_index = None
        if target is not None:
            gt_cls1, gt_cls0 = torch.nonzero(target == 1).view(-1), torch.nonzero(target == 0).view(-1)
            gt_index = gt_cls1.clone()
            print("Region true length", len(gt_cls1), "false length", len(gt_cls0))
            # len_cls0, len_cls1 = len(gt_cls0), len(gt_cls1)
            # select_len = min(len_cls0, len_cls1)
            # gt_cls0 = gt_cls0[np.random.choice(len_cls0, int(select_len), replace=False)]
            # gt_cls1 = gt_cls1[np.random.choice(len_cls1, select_len, replace=False)]
            select_index = torch.cat((gt_cls0, gt_cls1))
            # loss = self.criterion_cls(pre[select_index], target[select_index])
            loss = focal_loss(
                pre[select_index],
                F.one_hot(target[select_index], num_classes=2).to(pre[select_index]),
                alpha = 0.25,
                gamma = 2,
                reduction="mean",
            )

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
        anchor = self.templates.repeat(B*N, 1).to(self.device)
        
        pre_ori      = pre[:,3:6] + anchor[:,:3]
        pre_sum_ori_ = torch.sqrt(torch.sum(torch.mul(pre_ori, pre_ori), dim=1).add_(1e-12) ).view(-1,1)
        pre_ori      = torch.div(pre_ori, pre_sum_ori_)
        pre_center   = pre[:, :3] * self.radius + center
        pre_theta    = pre[:,6:7] * np.pi + anchor[:,3:4]
        pre_score    = pre[:,7:]
        pre_label = torch.cat((pre_center, pre_ori, pre_theta, pre_score), dim=-1)
    
        loss = torch.zeros(1).to(self.device)
        loss_tuple = [torch.zeros(1).to(self.device) for i in range(8)] 
        if gt_index is not None and target is not None:
            target = target.view(B,N,1,-1).repeat(1,1,self.anchor_number,1).view(-1,channel)  #repeat ground truth
            # delta_ori = torch.mul(pre[:,3:6], pre_sum_ori_)
            loss_gt1  = self.smoothl1_loss(pre[:,:3][gt_index] , (target[:,:3]-center)[gt_index] / self.radius)
            loss_gt2  = self.smoothl1_loss(pre[:,3:6][gt_index], (target[:,3:6]-anchor[:,:3])[gt_index])
            loss_gt3  = self.smoothl1_loss(pre[:,6:7][gt_index], (target[:,6:7]-anchor[:,3:4])[gt_index] / np.pi)
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
        return return_grasp.view(B,-1,channel), loss, loss_tuple, target

    def _generate_anchor_gt(self, gt):
        # gt: [B, select_points_number, 4]
        # self.templates [self.anchor_number, 4]
        gt = gt.view(-1, 4)
        no_grasp_mask = (torch.sum(gt, dim=1) == -4)

        # [B*select_points_number, self.anchor_number]
        sim_ori   = com_sim_ori(gt[:,:3], self.templates[:,:3].to(self.device))
        # sim_theta = com_sim_theta(gt[:,3:], self.templates[:,3:].to(self.device))
        sim = sim_ori# (sim_ori + sim_theta)/2
        # sort_sim, sort_index = torch.sort(sim, dim=1, descending=False)

        anchor_gt = torch.zeros_like(sim).long()
        anchor_gt[(sim >= 0.5) & (sim < 0.75)] = 2
        anchor_gt[sim < 0.5] = 1
        anchor_gt[no_grasp_mask] = 0
        # anchor_gt: [B*select_points_number, self.anchor_number]
        return anchor_gt


def _enumerate_templates():
    '''
      Enumerate all grasp anchors:
      For one score center, we generate 120 anchors.

      grasp configuration:(p, r, theta)
      r -> (1,0,0),                   (sqrt(2)/2, 0, sqrt(2)/2),           (sqrt(2)/2, 0, -sqrt(2)/2),           \
           (sqrt(2)/2,sqrt(2)/2,0),   (sqrt(3)/3, sqrt(3)/3, sqrt(3)/3),   (sqrt(3)/3, sqrt(3)/3, -sqrt(3)/3),   \
           (0,1,0),                   (0, sqrt(2)/2, sqrt(2)/2),           (0, sqrt(2)/2, -sqrt(2)/2),           \
           (-sqrt(2)/2,sqrt(2)/2,0),  (-sqrt(3)/3, sqrt(3)/3, sqrt(3)/3),  (-sqrt(3)/3, sqrt(3)/3, -sqrt(3)/3),  \
           (-1,0,0),                  (-sqrt(2)/2, 0, sqrt(2)/2),          (-sqrt(2)/2, 0, -sqrt(2)/2),          \
           (-sqrt(2)/2,-sqrt(2)/2,0), (-sqrt(3)/3, -sqrt(3)/3, sqrt(3)/3), (-sqrt(3)/3, -sqrt(3)/3, -sqrt(3)/3), \
           (0,-1,0),                  (0, -sqrt(2)/2, sqrt(2)/2),          (0, -sqrt(2)/2, -sqrt(2)/2),          \
           (sqrt(2)/2,-sqrt(2)/2,0),  (sqrt(3)/3, -sqrt(3)/3, sqrt(3)/3),  (sqrt(3)/3, -sqrt(3)/3, -sqrt(3)/3)
      theta -> {-pi/2, -pi/4, 0, pi/4, pi/2}
    '''
    sqrt2 = math.sqrt(2)/2
    sqrt3 = math.sqrt(3)/3
    t_r = torch.FloatTensor([
                        [sqrt3, sqrt3, sqrt3], [sqrt3, sqrt3, -sqrt3], \
                        [sqrt3, -sqrt3, -sqrt3], [sqrt3, -sqrt3, sqrt3]\
                        ]).view(4,1,3).repeat(1,1,1)#repeat(1,3,1)
    #t_r = torch.FloatTensor([
    #                    [sqrt3, sqrt3, sqrt3], [sqrt3, sqrt3, -sqrt3], \
    #                    [-sqrt3, sqrt3, -sqrt3], [-sqrt3, sqrt3, sqrt3], \
    #                    [-sqrt3, -sqrt3, sqrt3], [-sqrt3,-sqrt3, -sqrt3], \
    #                    [sqrt3, -sqrt3, -sqrt3], [sqrt3,-sqrt3, sqrt3]\
    #                    ]).view(1,8,1,3).repeat(1,1,1,1)#repeat(1,1,5,1)

    #t_r = torch.FloatTensor([#[1.0,0,0], [-1.0,0,0], [0,1.0,0], [0,-1.0,0],
    #                    [sqrt2, sqrt2, 0], [sqrt2, -sqrt2, 0]
    #                    ]).view(1,2,1,3).repeat(1,1,1,1)#repeat(1,1,5,1)
    # t_theta = torch.FloatTensor([-math.pi/2, 0, math.pi/2]).view(1,3,1).repeat(4,1,1)
    t_theta = torch.FloatTensor([0]).view(1,1,1).repeat(4,1,1)
    tem = torch.cat([t_r, t_theta], dim=2).view(-1,4)#.half()
    return tem

def com_sim_ori(a, b):
    '''
      input:
         a (gt) :[M, 3];  b (anchor) :[N, 3] 
      output:
         sim :[M, N] the samller the sim, the closer a and b
    '''
    M, N = len(a), len(b)
    a, b = a.repeat(1, N).view(M*N, -1), b.repeat(M, 1)

    a_b = torch.sum(torch.mul(a, b), dim=1)
    epsilon = 1e-12
    a2 = torch.add(torch.sum(torch.mul(a, a), dim=1), (epsilon))
    b2 = torch.add(torch.sum(torch.mul(b, b), dim=1), (epsilon))
    div_ab = torch.sqrt(torch.mul(a2, b2))
    sim = torch.div(a_b, div_ab).mul_(-1).add_(1).view(-1,1).view(M, N)
    return sim

def com_sim_theta(a, b):
    '''
      input:
         a (gt) :[M, 1];  b (anchor) :[N, 1] 
      output:
         sim :[M, N]
    '''
    M, N = len(a), len(b)
    a, b = a.repeat(1, N).view(M*N, -1), b.repeat(M, 1)
    sim = torch.abs(a - b) / math.pi
    return sim.view(M, N)

if __name__ == '__main__':
    pass
