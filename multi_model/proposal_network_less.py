import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import sys 
sys.path.append("..")
from dataset_utils.graspdataset import get_grasp_gt
from multi_model.utils.pointnet2 import PointNet2Less
from fvcore.nn import sigmoid_focal_loss_jit as focal_loss

class GraspLessProposalNetwork(nn.Module):
    def __init__(self, radius=0.06, grasp_channel=9, multi_flag=False, k_obj=2, conf_times=0.0025):
        super(GraspLessProposalNetwork, self).__init__()
        self.k_obj = k_obj
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.multi_flag = multi_flag
        self.conf_times = conf_times
        
        self.templates = _enumerate_templates()#.to(self.device)
        self.anchor_number = self.templates.shape[0]
        # (2+n: grasp classification(2), grasp params(x,y,z,rx,ry,rz,theta)+antipodal_score+center_score(n) )
        self.reg_channel = 2+grasp_channel
        self.radius = radius
        
        self.network = PointNet2Less(input_chann=6, sample_layer=0, k_reg=self.reg_channel, \
                                            k_anchor=self.anchor_number, k_obj=self.k_obj)

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
            predict_cls: [B, self.anchor_number, select_points_number] 
            predict_reg: [B, (self.reg_channel-2)*self.anchor_number, select_points_number] 
            select_xyz : [B, select_points_number, 3]
            data_paths : [B]
        '''
        B, N, _ = select_xyz.shape
        # gt: [B, select_points_number, 9]
        gt_anchor_cls, gt = None, None
        if data_paths is not None:
            gt = get_grasp_gt(select_xyz, data_paths, score_thre=None)
            # gt_anchor_cls: [B*select_points_number]
            gt_anchor_cls = self._generate_anchor_gt(gt[:,:,3:7]).view(-1)

        pre_anchor_cls = predict_cls.view(B, self.anchor_number, N).permute(0,2,1).contiguous()
        pre_grasp      = predict_reg.view(B, self.anchor_number, -1, N).permute(0,3,1,2).contiguous()
        
        pre_index, gt_index, cls_loss, acc_tuple = self._anchor_cls_loss(pre_anchor_cls.view(-1,self.anchor_number), gt_anchor_cls)
        pre_label, reg_loss, loss_tuple          = self._grasp_label_loss(select_xyz, pre_grasp, pre_index, gt_index, gt)
        loss = cls_loss  + reg_loss

        loss_tuple.extend(acc_tuple)
        loss_tuple.append(cls_loss.data)
        loss_tuple.append(reg_loss.data)
        return pre_label, loss, loss_tuple

    def forward(self, pc, pc_conf=None, pc_label=None, data_path=None, cur_idx=None):
        '''
         Input:
          pc              :[B,A,6]
          pc_conf         :[B,A]
          pc_label        :[B,A]
         Output:
          pre_grasp       :[B,A,Feature(128)])
          loss
        '''
        print(cur_idx)
        B, N, _ = pc.shape
        # x_conf [B, layer_points_number], x_reg[B, self.reg_channel*self.anchor_number, select_points_number]
        # select_xyz[B, 3, select_points_number], select_index [B, layer_points_number]
        x_conf, x_cls, x_reg, select_xyz, select_index, select_thre_index = self.network(pc[:,:,:6].permute(0,2,1), conf_times=self.conf_times) 

        loss_conf = self.compute_conf_loss(x_conf, select_index, select_thre_index, pc_conf)
        # return loss_conf
        pre_grasp, loss_grasp, loss_tuple = self.compute_reg_loss(x_cls, x_reg, select_xyz.transpose(2,1), data_path[cur_idx.cpu()])

        loss = loss_conf + loss_grasp
        loss_tuple.append(loss_conf)
        return pre_grasp, loss, loss_tuple

    def _anchor_cls_loss(self, pre, target):
        '''
            pre    : [B*select_points_number, self.anchor_number] torch.float32
            target : [B*select_points_number] torch.int64
        '''
        _, order = torch.sort(pre, dim=-1, descending=True)
        # pre_index [B*select_points_number] The value is from -1 to self.anchor_number-1
        pre_index = order[:,0].view(-1)

        loss = torch.zeros(1).to(self.device)
        gt_index = None
        if target is not None:
            gt_index = target
            select_index = torch.nonzero(target != -1).view(-1)
            loss = self.criterion_cls(pre[select_index], target[select_index])

            t = (pre_index == target).sum()
            f = (pre_index != target).sum()
            acc = t/(t+f)
            recall = t/(t+f)
            print("cls loss:", loss.data, "acc:", acc, "recall:", recall)
            
        return pre_index, gt_index, loss, [acc, recall]

    def _grasp_label_loss(self, select_xyz, pre, pre_index, gt_index, target):
        '''
            select_xyz: [B, select_points_number, 3] torch.float32
            pre       : [B, select_points_number, self.anchor_number, 9] torch.float32
            pre_index : [B*select_points_number] torch.int64
            gt_index  : [B*select_points_number] torch.int64
            target    : [B, select_points_number, 9] torch.float32
        '''
        B, N, _, channel = pre.shape
        pre    = pre.contiguous().view(-1, self.anchor_number, channel)
        center = select_xyz.view(B,N,1,-1).repeat(1,1,self.anchor_number,1).view(-1, self.anchor_number, 3)
        anchor = self.templates.view(1,self.anchor_number,-1).repeat(B*N, 1, 1).to(self.device)
        
        pre_ori      = pre[:,:,3:6] + anchor[:,:,:3]
        pre_sum_ori_ = torch.sqrt(torch.sum(torch.mul(pre_ori.view(-1,3), pre_ori.view(-1,3)), dim=1).add_(1e-12) ).view(-1,1)
        pre_ori      = torch.div(pre_ori.view(-1,3), pre_sum_ori_).view(-1,self.anchor_number,3)
        pre_center   = pre[:,:,:3] * self.radius + center
        pre_theta    = pre[:,:,6:7] * np.pi + anchor[:,:,3:4]
        pre_score    = pre[:,:,7:]
        pre_label = torch.cat((pre_center, pre_ori, pre_theta, pre_score), dim=-1)

        pre_select_score = pre_score[:,:,0].gather(1,pre_index.view(-1,1)).view(-1)
        pre_grasp_mask = (pre_select_score > 0.4)
        pre_nograsp_mask = (pre_select_score <= 0.4)
    
        loss = torch.zeros(1).to(self.device)
        loss_tuple = [torch.zeros(1).to(self.device)] * 8
        if gt_index is not None and target is not None:
            target = target.view(B,N,1,-1).repeat(1,1,self.anchor_number,1).view(-1,self.anchor_number,channel)
            delta_ori = torch.mul(pre[:,3:6], pre_sum_ori_)
            gt_grasp_mask = (gt_index != -1)
            gt_nograsp_mask = (gt_index == -1)
            gt_index[gt_nograsp_mask] = 0
            score_gt = target[:,:,7:].gather(1,gt_index.view(-1,1,1).repeat(1,1,channel-7)).view(-1,channel-7)
            score_gt[gt_nograsp_mask] = torch.zeros((gt_nograsp_mask.sum(), channel-7)).to(score_gt)
            
            loss_gt1  = self.smoothl1_loss(pre[:,:,:3].gather(1,gt_index.view(-1,1,1).repeat(1,1,3)).view(-1,3)[gt_grasp_mask] , 
                                    (target[:,:,:3]-center).gather(1,gt_index.view(-1,1,1).repeat(1,1,3)).view(-1,3)[gt_grasp_mask] / self.radius)
            loss_gt2  = self.smoothl1_loss(pre[:,:,3:6].gather(1,gt_index.view(-1,1,1).repeat(1,1,3)).view(-1,3)[gt_grasp_mask], 
                                    (target[:,:,3:6]-anchor[:,:,:3]).gather(1,gt_index.view(-1,1,1).repeat(1,1,3)).view(-1,3)[gt_grasp_mask])
            loss_gt3  = self.smoothl1_loss(pre[:,:,6:7].gather(1,gt_index.view(-1,1,1).repeat(1,1,1)).view(-1,1)[gt_grasp_mask], 
                                    (target[:,:,6:7]-anchor[:,:,3:4]).gather(1,gt_index.view(-1,1,1).repeat(1,1,1)).view(-1,1)[gt_grasp_mask] / np.pi)
            loss_gt4  = self.smoothl1_loss(pre[:,:,7:].gather(1,gt_index.view(-1,1,1).repeat(1,1,channel-7)).view(-1,channel-7), score_gt)
            print("regress loss:", loss_gt1.data, loss_gt2.data, loss_gt3.data, loss_gt4.data)
            loss = loss_gt1 + loss_gt2 + loss_gt3 + loss_gt4

            y_gt = torch.ones(gt_grasp_mask.sum(), 1).to(self.device)
            loss_gt_center = self.smoothl1_loss(pre_center.gather(1,gt_index.view(-1,1,1).repeat(1,1,3)).view(-1,3)[gt_grasp_mask], 
                                        target[:,:,:3].gather(1,gt_index.view(-1,1,1).repeat(1,1,3)).view(-1,3)[gt_grasp_mask]).data
            loss_gt_ori    = self.criterion_cos(pre_ori.gather(1,gt_index.view(-1,1,1).repeat(1,1,3)).view(-1,3)[gt_grasp_mask]   , 
                                        target[:,:,3:6].gather(1,gt_index.view(-1,1,1).repeat(1,1,3)).view(-1,3)[gt_grasp_mask], y_gt).data
            loss_gt_theta  = self.smoothl1_loss(pre_theta.gather(1,gt_index.view(-1,1,1).repeat(1,1,1)).view(-1,1)[gt_grasp_mask] , 
                                        target[:,:,6:7].gather(1,gt_index.view(-1,1,1).repeat(1,1,1)).view(-1,1)[gt_grasp_mask]).data
            loss_gt_score  = loss_gt4.data
            print("under gt class loss", loss_gt_center, loss_gt_ori, loss_gt_theta, loss_gt_score)
        
            y_pre = torch.ones(pre_grasp_mask.sum(), 1).to(self.device)
            loss_pre_center = self.smoothl1_loss(pre_center.gather(1,pre_index.view(-1,1,1).repeat(1,1,3)).view(-1,3)[pre_grasp_mask], 
                                        target[:,:,:3].gather(1,pre_index.view(-1,1,1).repeat(1,1,3)).view(-1,3)[pre_grasp_mask]).data
            loss_pre_ori    = self.criterion_cos(pre_ori.gather(1,pre_index.view(-1,1,1).repeat(1,1,3)).view(-1,3)[pre_grasp_mask]   , 
                                        target[:,:,3:6].gather(1,pre_index.view(-1,1,1).repeat(1,1,3)).view(-1,3)[pre_grasp_mask], y_pre).data
            loss_pre_theta  = self.smoothl1_loss(pre_theta.gather(1,pre_index.view(-1,1,1).repeat(1,1,1)).view(-1,1)[pre_grasp_mask] , 
                                        target[:,:,6:7].gather(1,pre_index.view(-1,1,1).repeat(1,1,1)).view(-1,1)[pre_grasp_mask]).data
            loss_pre_score  = self.smoothl1_loss(pre_score.gather(1,pre_index.view(-1,1,1).repeat(1,1,channel-7)).view(-1,channel-7)[pre_grasp_mask] , 
                                        target[:,:,7:].gather(1,pre_index.view(-1,1,1).repeat(1,1,channel-7)).view(-1,channel-7)[pre_grasp_mask]).data
            print("under pre class loss", loss_pre_center, loss_pre_ori, loss_pre_theta, loss_pre_score)
            loss_tuple = [loss_gt1.data, loss_gt2.data, loss_gt3.data, loss_gt4.data, \
                            loss_pre_center, loss_pre_ori, loss_pre_theta, loss_pre_score]

        return_grasp = torch.full((B*N, channel), -1.0).to(self.device)
        return_grasp[pre_grasp_mask] = pre_label.gather(1,pre_index.view(-1,1,1).repeat(1,1,channel)).view(-1,channel)[pre_grasp_mask]
        return return_grasp.view(B,-1,channel), loss, loss_tuple

    def _generate_anchor_gt(self, gt):
        # gt: [B, select_points_number, 4]
        # self.templates [self.anchor_number, 4]
        gt = gt.view(-1, 4)
        no_grasp_mask = (torch.sum(gt, dim=1) == -4)

        # [B*select_points_number, self.anchor_number]
        sim_ori   = com_sim_ori(gt[:,:3], self.templates[:,:3].to(self.device))
        # sim_theta = com_sim_theta(gt[:,3:], self.templates[:,3:].to(self.device))
        sim = sim_ori# (sim_ori + sim_theta)/2
        sort_sim, sort_index = torch.sort(sim, dim=1, descending=False)

        anchor_gt = sort_index[:,0].view(-1)
        anchor_gt[no_grasp_mask] = -1
        # anchor_gt: [B*select_points_number]
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
