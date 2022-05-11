import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from time import time
import numpy as np
import math
#from .pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
from multi_model.utils.pn2_utils.modules import PointNetSAModule, PointnetFPModule
import multi_model.utils.pn2_utils.function as _F
from multi_model.utils.pn2_utils.nn import SharedMLP

class PointNet2Region(nn.Module):
    """PointNet++ part segmentation with single-scale grouping

    PointNetSA: PointNet Set Abstraction Layer
    PointNetFP: PointNet Feature Propagation Layer

    Args:
        score_classes (int): the number of grasp score classes
        num_centroids (tuple of int): the numbers of centroids to sample in each set abstraction module
        radius (tuple of float): a tuple of radius to query neighbours in each set abstraction module
        num_neighbours (tuple of int): the numbers of neighbours to query for each centroid
        sa_channels (tuple of tuple of int): the numbers of channels within each set abstraction module
        fp_channels (tuple of tuple of int): the numbers of channels for feature propagation (FP) module
        num_fp_neighbours (tuple of int): the numbers of nearest neighbor used in FP
        seg_channels (tuple of int): the numbers of channels in segmentation mlp
        dropout_prob (float): the probability to dropout input features

    References:
        https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_part_seg.py

    """
    _SA_MODULE = PointNetSAModule
    _FP_MODULE = PointnetFPModule

    def __init__(self, sample_layer=0, k_reg=9, k_anchor=4, k_obj=2, use_rgb=True, 
                use_multi=False, use_region=True, use_fps=False, dropout_prob=0.5, prior_prob=0.01):
        super(PointNet2Region, self).__init__()
        self.k_reg = k_reg
        self.k_anchor = k_anchor
        self.sample_layer = sample_layer
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_rgb   = use_rgb
        self.use_multi = use_multi

        self.use_region = use_region
        self.use_fps = use_fps

        num_centroids     = (5120, 1024, 256)
        radius            = (0.02, 0.08, 0.32)
        num_neighbours    = (64, 64, 64)
        sa_channels       = ((128, 128, 256), (256, 256, 512), (512, 512, 1024))
        fp_channels       = ((1024, 1024), (512, 512), (256, 256, 256))
        num_fp_neighbours = (3, 3, 3)
        seg_channels      = (512, 256, 256, 128)

        num_sa_layers = len(num_centroids)
        num_fp_layers = len(fp_channels)

        # Set Abstraction Layers
        input_chann = 3
        if use_rgb:
            input_chann += 3
        if use_multi:
            input_chann += 3

        feature_channels = input_chann - 3 
        self.sa_modules = nn.ModuleList()
        for ind in range(num_sa_layers):
            sa_module = self._SA_MODULE(in_channels=feature_channels,
                                        mlp_channels=sa_channels[ind],
                                        num_centroids=num_centroids[ind],
                                        radius=radius[ind],
                                        num_neighbours=num_neighbours[ind],
                                        use_xyz=True)
            self.sa_modules.append(sa_module)
            feature_channels = sa_channels[ind][-1]

        inter_channels = [input_chann - 3]
        inter_channels.extend([x[-1] for x in sa_channels])
        # Feature Propagation Layers
        self.fp_modules = nn.ModuleList()
        feature_channels = inter_channels[-1]
        for ind in range(num_fp_layers):
            in_channels = feature_channels + inter_channels[-2 - ind] 
            if inter_channels[-2 - ind] == 3 or inter_channels[-2 - ind] == 6: # use_rgb=True, use_multi=True
                in_channels += 3
            fp_module = self._FP_MODULE(in_channels=in_channels,
                                        mlp_channels=fp_channels[ind],
                                        num_neighbors=num_fp_neighbours[ind])
            self.fp_modules.append(fp_module)
            feature_channels = fp_channels[ind][-1]

        # MLP
        feature_channels = 256#fp_channels[-1-self.sample_layer][-1]
        self.mlp = SharedMLP(feature_channels, seg_channels, ndim=1, dropout_prob=dropout_prob)
        #self.select_cls_mlp = SharedMLP(feature_channels, seg_channels, ndim=1, dropout_prob=dropout_prob)
        #self.select_reg_mlp = SharedMLP(feature_channels, seg_channels, ndim=1, dropout_prob=dropout_prob)
        self.select_cls_mlp = SharedMLP(256, seg_channels, ndim=1, dropout_prob=dropout_prob)
        self.select_reg_mlp = SharedMLP(256, seg_channels, ndim=1, dropout_prob=dropout_prob)
            
        self.conv_conf = nn.Conv1d(seg_channels[-1], 1, 1)
        self.bn_conf = nn.BatchNorm1d(1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-2)

        self.conv_cls = nn.Conv1d(seg_channels[-1], 2*self.k_anchor, 1)
        self.bn_cls = nn.BatchNorm1d(2*self.k_anchor)
        self.conv_reg = nn.Conv1d(seg_channels[-1], self.k_reg*self.k_anchor, 1)
        self.bn_reg = nn.BatchNorm1d(self.k_reg*self.k_anchor)

        self.region_radius = 0.03
        self.group_num = 256
        self.mp = nn.MaxPool1d(self.group_num)
        #self.ap = torch.nn.AvgPool1d(self.group_num)
        #self.dc = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=self.group_num, groups=256)
        #self.dc = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=self.group_num, groups=512)

        for modules in [self.sa_modules, self.fp_modules, self.mlp, self.select_reg_mlp, 
                            self.select_cls_mlp, self.conv_conf, self.conv_cls, self.conv_reg]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv1d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)  
        
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.conv_cls.bias, bias_value)      

    def select_using_conf(self, x_conf, select_feature, select_xyz, conf_times=0.5):
        '''
          Input:
            x_conf          [B, N]
            select_feature  [B, C, N]
            select_xyz      [B, 3, N]
        '''
        sample_num = int(x_conf.shape[-1] * conf_times)
        B, feature_channel, xyz_channel = len(select_feature), select_feature.shape[1], select_xyz.shape[1]
        # thre_index: [B, len_select]
        thre_index = torch.sort(x_conf, dim=-1, descending=True)[1][:,:sample_num]
        
        # thre_index_feature: [B, C, len_select], thre_index_xyz: [B, 3, len_select]
        thre_index_feature, thre_index_xyz = thre_index.unsqueeze(1).expand(B, feature_channel, sample_num), \
                                thre_index.unsqueeze(1).expand(B, xyz_channel, sample_num)
        select_feature = select_feature.gather(2, thre_index_feature)
        select_xyz     = select_xyz.gather(2, thre_index_xyz)
        #select_index   = select_index.gather(1, thre_index)
        
        return select_feature, select_xyz, thre_index

    def select_using_conf_layer1(self, x_conf, select_feature, select_xyz, conf_times=0.5, select_index=None):
        '''
          Input:
            x_conf          [B, N]
            select_feature  [B, C, N1]
            select_xyz      [B, 3, N]
        '''
        sample_num = int(x_conf.shape[-1] * conf_times)
        B, feature_channel, xyz_channel = len(select_feature), select_feature.shape[1], select_xyz.shape[1]
        # thre_index: [B, len_select]
        # thre_index = torch.sort(x_conf, dim=-1, descending=True)[1][:,:sample_num]
        thre_index = torch.topk(x_conf, dim=-1, k=sample_num)[1]
        select_thre_index   = select_index.gather(1, thre_index)
        # print(thre_index.shape, select_thre_index.shape)
        
        # thre_index_feature: [B, C, len_select], thre_index_xyz: [B, 3, len_select]
        thre_index_feature, thre_index_xyz = select_thre_index.unsqueeze(1).expand(B, feature_channel, sample_num), \
                                thre_index.unsqueeze(1).expand(B, xyz_channel, sample_num)
        select_feature = select_feature.gather(2, thre_index_feature)
        select_xyz     = select_xyz.gather(2, thre_index_xyz)
        return select_feature, select_xyz, thre_index

    def select_fps_layer1(self, x_conf, select_feature, select_xyz, conf_times=0.5, select_index=None):
        '''
          Input:
            x_conf          [B, N]
            select_feature  [B, C, N1]
            select_xyz      [B, 3, N1]
        '''
        sample_num = int(x_conf.shape[-1] * conf_times)
        B, feature_channel, xyz_channel = len(select_feature), select_feature.shape[1], select_xyz.shape[1]
        # thre_index: [B, len_select]
        
        thre_indexs = []
        for i in range(len(x_conf)):
            all_thre_index = torch.nonzero(x_conf[i] >= 0.7).view(-1)
            no_thre_index = torch.nonzero(x_conf[i] < 0.7).view(-1)
            if len(all_thre_index) >= sample_num:
                thre_index     = _F.farthest_point_sample(select_xyz[i][:3].view(1,3,-1), sample_num)
            else:
                print("{} less than {} points".format(len(all_thre_index), sample_num))
                thre_index     = _F.farthest_point_sample(select_xyz[i][:3,no_thre_index].view(1,3,-1), sample_num-len(all_thre_index))
                thre_index     = torch.cat((thre_index.view(-1), all_thre_index), dim=0)
            thre_indexs.append(thre_index.view(-1))
        thre_index = torch.stack(thre_indexs)
        select_thre_index   = select_index.gather(1, thre_index)
        
        # thre_index_feature: [B, C, len_select], thre_index_xyz: [B, 3, len_select]
        thre_index_feature, thre_index_xyz = select_thre_index.unsqueeze(1).expand(B, feature_channel, sample_num), \
                                thre_index.unsqueeze(1).expand(B, xyz_channel, sample_num)
        select_feature = select_feature.gather(2, thre_index_feature)
        select_xyz     = select_xyz.gather(2, thre_index_xyz)
        return select_feature, select_xyz, thre_index

    def forward(self, points, conf_times=0.5, add_widths=None):
        B,C,N = points.size()
        xyz = points[:,:3,:]
        feature = points[:,3:6,:] if self.use_rgb else None
        add_feature = add_widths.to(xyz).view(B,1,1).repeat(1,3,N) if self.use_multi else None
        
        if feature is not None:
            if add_feature is not None:
                feature = torch.cat((feature, add_feature), dim=1)
        else:
            if add_feature is not None:
                feature = add_feature

        # save intermediate results
        inter_xyz = [xyz]
        inter_index = [torch.arange(N).view(1,-1).repeat(B,1).to(self.device)]
        inter_feature = [feature]
        # Set Abstraction Layers
        for sa_module in self.sa_modules:
            xyz, feature, index = sa_module(xyz, feature)
            inter_xyz.append(xyz)
            inter_feature.append(feature)
            inter_index.append(inter_index[-1].gather(1, index))

        # Feature Propagation Layers
        sparse_xyz = xyz
        sparse_feature = feature
        for fp_ind, fp_module in enumerate(self.fp_modules):
            dense_xyz = inter_xyz[-2 - fp_ind]
            dense_feature = inter_feature[-2 - fp_ind]
            fp_feature = fp_module(dense_xyz, sparse_xyz, dense_feature, sparse_feature)
            sparse_xyz = dense_xyz
            sparse_feature = fp_feature

            if 2-fp_ind == self.sample_layer:
                select_feature = sparse_feature
                select_xyz     = sparse_xyz
                select_index   = inter_index[-2 - fp_ind]
                
        # # (batch_size, channels, point_number)
        # x = self.mlp(select_feature)
        # # (batch_size, channels, point_number*conf_times)
        # x_conf = self.bn_conf(self.conv_conf(x))
        # x_conf = self.sigmoid(x_conf)
        # x_conf = x_conf.transpose(2,1).contiguous().view(B, -1)
      
        x = self.mlp(sparse_feature)
        # (batch_size, channels, point_number*conf_times)
        x_conf = self.bn_conf(self.conv_conf(x))
        x_conf = self.sigmoid(x_conf)
        x_conf = x_conf.transpose(2,1).contiguous().view(B, -1)

        if not self.use_fps:
            if self.use_region:
                # if use region
                xyz_formal = sparse_xyz.clone()
                feature_formal = sparse_feature.clone()
                #xyz_formal = select_xyz.clone()
                #feature_formal = select_feature.clone()

                # select_feature, select_xyz, thre_index = self.select_using_conf(x_conf, select_feature, select_xyz, conf_times)
                select_feature, select_xyz, thre_index = self.select_using_conf_layer1(x_conf.gather(1, select_index), sparse_feature, select_xyz, conf_times, select_index)
                # index: [B, N1, num_neigh],  unique_count: [B, N1]
                index, unique_count = _F.ball_query(xyz_formal, select_xyz, self.region_radius, self.group_num)
                # select_feature: [B, C1, N1, num_neigh]
                select_feature = _F.group_points(feature_formal, index)
                select_feature = select_feature.transpose(2,1).contiguous().view(-1,select_feature.shape[1],self.group_num)
                select_feature = self.mp(select_feature)
                select_feature = select_feature.view(B,-1,select_feature.shape[1]).transpose(2,1).contiguous()
            else:
                # don't use region
                select_feature, select_xyz, thre_index = self.select_using_conf_layer1(x_conf.gather(1, select_index), sparse_feature, select_xyz, conf_times, select_index)
        else:
            if self.use_region:
                # regnet!! -> 64 grasp centers -> conf_times=0.0025
                # if use region
                xyz_formal = sparse_xyz.clone()
                feature_formal = sparse_feature.clone()
                #xyz_formal = select_xyz.clone()
                #feature_formal = select_feature.clone()

                select_feature, select_xyz, thre_index = self.select_fps_layer1(x_conf.gather(1, select_index), sparse_feature, sparse_xyz, 0.005, select_index)
                # index: [B, N1, num_neigh],  unique_count: [B, N1]
                index, unique_count = _F.ball_query(xyz_formal, select_xyz, self.region_radius, self.group_num)
                # select_feature: [B, C1, N1, num_neigh]
                select_feature = _F.group_points(feature_formal, index)
                select_feature = select_feature.transpose(2,1).contiguous().view(-1,select_feature.shape[1],self.group_num)
                select_feature = self.mp(select_feature)
                select_feature = select_feature.view(B,-1,select_feature.shape[1]).transpose(2,1).contiguous()
            else:
                # don't use region
                select_feature, select_xyz, thre_index = self.select_fps_layer1(x_conf.gather(1, select_index), sparse_feature, sparse_xyz, 0.005, select_index)

        # (batch_size, channels, point_number*conf_times)
        select_x_cls = self.select_cls_mlp(select_feature)
        # (batch_size, reg_channels, point_number*conf_times)
        # x_cls = self.bn_cls(self.conv_cls(select_x))
        x_cls = self.conv_cls(select_x_cls)
        # (2: grasp classification)
        # x_cls = self.softmax(x_cls.view(B, self.k_anchor, 2, -1))
        # x_cls = x_cls.view(B, self.k_anchor*2, -1)

        # x_reg = self.bn_reg(self.conv_reg(select_x))
        select_x_reg = self.select_reg_mlp(select_feature)
        x_reg = self.conv_reg(select_x_reg)
        # (n: grasp params(x,y,z,rx,ry,rz,theta), grasp scores)
        x_reg = x_reg.view(B, self.k_anchor, self.k_reg, -1)
        x_reg[:,:,7:,:] = self.sigmoid(x_reg[:,:,7:,:])
        x_reg = x_reg.view(B, self.k_anchor * self.k_reg, -1)

        return sparse_feature, select_feature, x_conf, x_cls, x_reg, select_xyz, select_index, select_index.gather(1,thre_index)

class PointNet2Refine(nn.Module):
    def __init__(self, radius, gripper_num, grasp_channel, use_rgb=False, dropout_prob=0.5, prior_prob=0.01, sample_layer=0):
        super(PointNet2Refine, self).__init__()
        self.radius = radius 
        self.gripper_num = gripper_num
        self.num_neighbours = gripper_num * 3
        self.use_rgb = use_rgb
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.k_reg = grasp_channel 

        feature_chan_map = {0: 256, 1: 512, 2:1024}
        # if use only 1 point, sample layer keep same with the PointNet2Region layer
        # if else, features are extracted from the first layer feature: [B, 256, N] -> sample_layer=0
        # last_feature_channel = feature_chan_map[sample_layer]
        last_feature_channel = 6 if self.use_rgb else 3
        seg_feature_channels  = (512, 256, 256, 128)
        self.mix_mlp   = SharedMLP(last_feature_channel, seg_feature_channels, ndim=1, dropout_prob=dropout_prob)
        self.conv_reg  = nn.Conv1d(128+256, self.k_reg, 1) 
        self.conv_cls  = nn.Conv1d(128+256, 2, 1)
        self.mp = nn.MaxPool1d(gripper_num)
        self.ap = torch.nn.AvgPool1d(gripper_num)
        self.dc_reg = nn.Conv1d(in_channels=self.k_reg, out_channels=self.k_reg, kernel_size=gripper_num, groups=self.k_reg)
        self.dc_cls = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=gripper_num, groups=2)

        for modules in [self.mix_mlp, self.conv_reg, self.conv_cls]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv1d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)  
        
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.conv_cls.bias, bias_value) 

    def get_gripper_transform(self, group_points, group_feature, unique_count, grasp, gripper_params):
        '''
          Return the transformed local points in the closing area of gripper.
          Input: 
            # group_points   : [B, 6, N*anchor_number, num_neighbours]
            # group_feature  : [B, 256, N*anchor_number, num_neighbours]
            # unique_count   : [B, N*anchor_number]
            # grasp:         : [B, N*anchor_number, self.k_reg]
            # gripper_params : List [torch.tensor(),float,float] widths, height, depth

          Return:    
            # gripper_pc      : [B, 6, self.gripper_num]
            # gripper_feature : [B, C_feature, self.gripper_num] 
            # mask            : [B] True or False
        '''
        batch_size, _, num, _ = group_points.shape
        group_points  = group_points.permute(0,2,3,1).contiguous().view(-1,self.num_neighbours,6)
        group_feature = group_feature.permute(0,2,3,1).contiguous().view(-1,self.num_neighbours,group_feature.shape[1])
        grasp         = grasp.view(-1, self.k_reg)
        grasp_mask    = torch.sum(grasp, dim=1) != -1*(self.k_reg)
        unique_count  = unique_count.view(-1)#[grasp_mask]

        # group_feature1 = group_feature.clone()
        # group_points = group_points[grasp_mask]
        # group_feature = group_feature[grasp_mask]
        # grasp = grasp[grasp_mask]
        
        widths, height, depths = gripper_params
        B, _ = grasp.shape
        center, axis_y, angle = grasp[:, 0:3], grasp[:, 3:6], grasp[:, 6]

        cos_t, sin_t = torch.cos(angle), torch.sin(angle)
        one, zero = torch.ones((B, 1), dtype=torch.float32).to(self.device), torch.zeros((B, 1), dtype=torch.float32).to(self.device)
        R1 = torch.cat( (cos_t.view(B,1), zero, -sin_t.view(B,1), zero, one, zero, 
                            sin_t.view(B,1), zero, cos_t.view(B,1)), dim=1).view(B,3,3)

        norm_y = torch.norm(axis_y, dim=1).add_(1e-12)
        axis_y = torch.div(axis_y, norm_y.view(-1,1))
        axis_y[torch.nonzero(torch.eq(norm_y, 0))] = torch.tensor(([0,1,0]), dtype=torch.float).to(self.device)
            
        axis_x = torch.cat((axis_y[:, 1].view(-1,1), -axis_y[:, 0].view(-1,1), zero), 1)
        norm_x = torch.norm(axis_x, dim=1).add_(1e-12)
        axis_x = torch.div(axis_x, norm_x.view(-1,1))
        axis_x[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float).to(self.device)

        axis_z = torch.cross(axis_x, axis_y, dim=1)
        norm_z = torch.norm(axis_z, dim=1)
        axis_z = torch.div(axis_z, norm_z.view(-1,1))
        axis_z[torch.nonzero(torch.eq(norm_z, 0))] = torch.tensor(([0,0,1]), dtype=torch.float).to(self.device)

        matrix = torch.cat((axis_x.view(-1,3,1), axis_y.view(-1,3,1), axis_z.view(-1,3,1)), dim=2)
        matrix = torch.bmm(matrix, R1)
        approach = matrix[:,:,0]
        norm_x = torch.norm(approach, dim=1).add_(1e-12)
        approach = torch.div(approach, norm_x.view(-1,1))
        approach[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float).to(self.device)

        minor_normal = torch.cross(approach, axis_y, dim=1)
        matrix = torch.cat((approach.view(-1,3,1), axis_y.view(-1,3,1), minor_normal.view(-1,3,1)), dim=2).permute(0,2,1)
        ## pcs_t: [B,G,3]
        pcs_t = torch.bmm(matrix, (group_points[:,:,:3].float() - \
                            center.view(-1,1,3).repeat(1, group_points.shape[1], 1).float()).permute(0,2,1)).permute(0,2,1)

        x_limit = depths   #float
        z_limit = height/2 # float
        # torch.tensor [batch_size]  or  float
        y_limit = (widths/2).float().view(batch_size,1,1).repeat(1,num,self.num_neighbours).\
                        view(-1,self.num_neighbours) if type(widths) is torch.Tensor else widths / 2

        
        x1 = pcs_t[:,:,0] > 0
        x2 = pcs_t[:,:,0] < x_limit
        y1 = pcs_t[:,:,1] > -y_limit
        y2 = pcs_t[:,:,1] < y_limit
        z1 = pcs_t[:,:,2] > -z_limit
        z2 = pcs_t[:,:,2] < z_limit
        
        a = torch.sum(torch.cat((x1.view(B,-1,1), x2.view(B,-1,1), y1.view(B,-1,1), \
                        y2.view(B,-1,1), z1.view(B,-1,1), z2.view(B,-1,1)), dim=-1), dim=-1)
        index = (a==6).float().view(B,-1,1).repeat(1,1,3).permute(0,2,1)
        center_index = torch.ones((B,3,1)).to(self.device)
        closing_index, closing_count = _F.ball_query(index, center_index, 0.5, self.gripper_num)
        closing_count = closing_count.view(-1)
        closing_index = closing_index.view(B,-1)
        # print(closing_count, closing_index[0])#closing_count!=0])

        # closing_count   = torch.zeros((B)).to(self.device)
        # closing_index   = torch.zeros((B,self.gripper_num)).long().to(self.device)
        # for i in range(B):
        #     index = torch.nonzero(a[i] == 6).view(-1)
        #     cur_closing_count = len(index)
        #     if cur_closing_count > self.gripper_num:
        #         index = index[np.random.choice(len(index),self.gripper_num,replace=False)]
        #     elif cur_closing_count > 0:
        #         index = index[np.random.choice(len(index),self.gripper_num,replace=True)]
            
        #     if cur_closing_count > 0:
        #         closing_count[i] = cur_closing_count
        #         closing_index[i] = index
        #         # closing_count[i]   = len(index) 
        
        group_mask = (unique_count > 0)
        close_mask = (closing_count > 0)
        mask = (group_mask & close_mask & grasp_mask)
        # formal_mask = grasp_mask.clone()
        # formal_mask[grasp_mask] = mask
        
        # closing_xyz_index: [B, 3, self.gripper_num], closing_feature_index: [B, C_feature, self.gripper_num]
        closing_xyz_index     = closing_index.unsqueeze(1).expand((B,pcs_t.shape[-1],self.gripper_num))
        # closing_feature_index = torch.zeros((len(grasp_mask), group_feature.shape[-1],self.gripper_num)).to(self.device).long()
        closing_feature_index = closing_index.unsqueeze(1).expand((B,group_feature.shape[-1],self.gripper_num))

        # gripper_pc: [B, 6(3), self.gripper_num],  gripper_feature: [B, C_feature, self.gripper_num]
        if self.use_rgb:
            gripper_pc  = torch.cat(( pcs_t.transpose(2,1).gather(2, closing_xyz_index), \
                                group_points[...,3:].transpose(2,1).gather(2, closing_xyz_index) ), dim = 1)
        else:
            gripper_pc  = pcs_t.transpose(2,1).gather(2, closing_xyz_index)
        return gripper_pc, mask
        # gripper_feature = group_feature.transpose(2,1).gather(2, closing_feature_index)
        # return gripper_feature, mask#formal_mask

    def forward(self, grasp, pc, all_feature, stage1_feature, gripper_params): 
        '''
        grasp      : [B, N*anchor_number, self.k_reg]
        pc         : [B, A, 6 (3)]
        all_feature: [B, 256, A]
        '''
        pc, gcenter = pc.transpose(2,1), grasp[:,:,:3].transpose(2,1)
        pc_xyz      = pc[:,:3,:]

        # index: [B, N*anchor_number, num_neigh],  unique_count: [B, N*anchor_number]
        index, unique_count = _F.ball_query(pc_xyz, gcenter, self.radius, self.num_neighbours)
        # group_pc: [B, 6 (3), N*anchor_number, num_neigh], group_feature: [B, 256, N*anchor_number, num_neigh]
        group_pc      = _F.group_points(pc, index)
        group_feature = _F.group_points(all_feature, index)
        # ## gripper_pc: [B, 6 (3), self.gripper_num],  gripper_feature: [B, C_feature, self.gripper_num]
        gripper_feature, mask = self.get_gripper_transform(group_pc.data, group_feature, unique_count, grasp.data, gripper_params)
        # print(gripper_feature.shape)
        # gripper_feature = group_feature.view(group_feature.shape[0], group_feature.shape[1], -1)
        # mask = (unique_count>0).view(-1) & (torch.sum(grasp.view(-1, self.k_reg), dim=1) != -1*(self.k_reg))
        
        # gripper_feature: [B, 256, self.gripper_num]
        stage1_feature = stage1_feature.unsqueeze(2).repeat(1,1,self.gripper_num)
        x = self.mix_mlp(gripper_feature)
        x = torch.cat((stage1_feature, x), dim=1)
        # x_cls = self.mix_mlp_cls(gripper_feature)
        # x: [B, self.k_reg, self.gripper_num]
        x_reg = self.conv_reg(x)
        x_cls = self.conv_cls(x)
        # x: [B, self.k_reg, 1] -> [B, self.k_reg]
        x_reg = self.dc_reg(x_reg).view(-1, self.k_reg)
        x_cls = self.dc_cls(x_cls).view(-1, 2)
        # x_reg = self.mp(x_reg).view(-1, self.k_reg)
        # x_cls = self.mp(x_cls).view(-1, 2)


        # torch.autograd.set_detect_anomaly(True)
        # print(formal_mask.sum(), len(x_reg))
        # x_reg_new = torch.zeros((len(formal_mask), self.k_reg)).to(self.device)
        # x_cls_new = torch.zeros((len(formal_mask), 2)).to(self.device)
        # x_reg_new[formal_mask] = x_reg
        # x_cls_new[formal_mask] = x_cls
        # formal_mask[formal_mask.clone()] = mask
        # x_cls: [B, 2], x_reg: [B, self.reg], mask: [B]
        
        # # one point
        # x = self.mix_mlp(all_feature)
        # x_cls = self.mix_mlp_cls(all_feature)
        # # # x: [B, self.k_reg, self.gripper_num]
        # x = self.conv_reg(x)
        # x_cls = self.conv_cls(x_cls)
        # # # x: [B, self.k_reg, 1] -> [B, self.k_reg]
        # x_reg = x.view(all_feature.shape[0], (self.k_reg-2), -1).transpose(2,1).contiguous().view(-1, (self.k_reg-2))
        # x_cls = x_cls.view(all_feature.shape[0], 2, -1).transpose(2,1).contiguous().view(-1, 2)
        # # # x_cls: [B, 2], x_reg: [B, self.reg], mask: [B]
        # # x_cls, x_reg = x[:,:2], x[:,2:]
        # # # mask = torch.ones((len(x_cls))).bool().to(self.device)
        # mask = torch.sum(grasp.view(-1, self.k_reg-2), dim=1) != -1*(self.k_reg-2)
        return x_cls, x_reg, mask
        # return x_cls_new, x_reg_new, formal_mask

class PointNet2Less(nn.Module):
    """PointNet++ part segmentation with single-scale grouping

    PointNetSA: PointNet Set Abstraction Layer
    PointNetFP: PointNet Feature Propagation Layer

    Args:
        score_classes (int): the number of grasp score classes
        num_centroids (tuple of int): the numbers of centroids to sample in each set abstraction module
        radius (tuple of float): a tuple of radius to query neighbours in each set abstraction module
        num_neighbours (tuple of int): the numbers of neighbours to query for each centroid
        sa_channels (tuple of tuple of int): the numbers of channels within each set abstraction module
        fp_channels (tuple of tuple of int): the numbers of channels for feature propagation (FP) module
        num_fp_neighbours (tuple of int): the numbers of nearest neighbor used in FP
        seg_channels (tuple of int): the numbers of channels in segmentation mlp
        dropout_prob (float): the probability to dropout input features

    References:
        https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_part_seg.py

    """
    _SA_MODULE = PointNetSAModule
    _FP_MODULE = PointnetFPModule

    def __init__(self, input_chann=3, sample_layer=0, k_reg=9, k_anchor=4, k_obj=2, 
                        add_channel_flag=False, dropout_prob=0.5, prior_prob=0.01):
        super(PointNet2Less, self).__init__()
        self.k_reg = k_reg
        self.k_anchor = k_anchor
        self.sample_layer = sample_layer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        num_centroids     = (5120, 1024, 256)
        radius            = (0.02, 0.08, 0.32)
        num_neighbours    = (64, 64, 64)
        sa_channels       = ((128, 128, 256), (256, 256, 512), (512, 512, 1024))
        fp_channels       = ((1024, 1024), (512, 512), (256, 256, 256))
        num_fp_neighbours = (3, 3, 3)
        seg_channels      = (512, 256, 256, 128)

        num_sa_layers = len(num_centroids)
        num_fp_layers = len(fp_channels)

        # Set Abstraction Layers
        feature_channels = input_chann - 3 
        self.sa_modules = nn.ModuleList()
        for ind in range(num_sa_layers):
            sa_module = self._SA_MODULE(in_channels=feature_channels,
                                        mlp_channels=sa_channels[ind],
                                        num_centroids=num_centroids[ind],
                                        radius=radius[ind],
                                        num_neighbours=num_neighbours[ind],
                                        use_xyz=True)
            self.sa_modules.append(sa_module)
            feature_channels = sa_channels[ind][-1]

        inter_channels = [input_chann - 3]
        inter_channels.extend([x[-1] for x in sa_channels])
        # Feature Propagation Layers
        self.fp_modules = nn.ModuleList()
        feature_channels = inter_channels[-1]
        for ind in range(num_fp_layers):
            in_channels = feature_channels + inter_channels[-2 - ind] \
                       if inter_channels[-2 - ind] != 3 else feature_channels + inter_channels[-2 - ind]*2
            fp_module = self._FP_MODULE(in_channels=in_channels,
                                        mlp_channels=fp_channels[ind],
                                        num_neighbors=num_fp_neighbours[ind])
            self.fp_modules.append(fp_module)
            feature_channels = fp_channels[ind][-1]

        # MLP
        feature_channels = fp_channels[-1-self.sample_layer][-1]
        if not add_channel_flag:
            self.mlp = SharedMLP(feature_channels, seg_channels, ndim=1, dropout_prob=dropout_prob)
            self.select_cls_mlp = SharedMLP(feature_channels, seg_channels, ndim=1, dropout_prob=dropout_prob)
            self.select_reg_mlp = SharedMLP(feature_channels, seg_channels, ndim=1, dropout_prob=dropout_prob)
        else:
            self.mlp = SharedMLP(feature_channels*3, seg_channels, ndim=1, dropout_prob=dropout_prob)
            self.select_cls_mlp = SharedMLP(feature_channels*3, seg_channels, ndim=1, dropout_prob=dropout_prob)
            self.select_reg_mlp = SharedMLP(feature_channels*3, seg_channels, ndim=1, dropout_prob=dropout_prob)
            
        self.conv_conf = nn.Conv1d(seg_channels[-1], 1, 1)
        self.bn_conf = nn.BatchNorm1d(1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-2)

        self.conv_cls = nn.Conv1d(seg_channels[-1], self.k_anchor, 1)
        self.bn_cls = nn.BatchNorm1d(self.k_anchor)
        self.conv_reg = nn.Conv1d(seg_channels[-1], self.k_reg*self.k_anchor, 1)
        self.bn_reg = nn.BatchNorm1d(self.k_reg*self.k_anchor)

        for modules in [self.sa_modules, self.fp_modules, self.mlp, self.select_cls_mlp, 
                        self.select_reg_mlp, self.conv_conf, self.conv_cls, self.conv_reg]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv1d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)  
        
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.conv_cls.bias, bias_value)      

    def select_using_conf(self, x_conf, select_feature, select_xyz, conf_times=0.5):
        sample_num = int(x_conf.shape[-1] * conf_times)
        B, feature_channel, xyz_channel = len(select_feature), select_feature.shape[1], select_xyz.shape[1]
        thre_index = torch.sort(x_conf, dim=-1, descending=True)[1][:,:sample_num]
        
        thre_index_feature, thre_index_xyz = thre_index.unsqueeze(1).expand(B, feature_channel, sample_num), \
                                thre_index.unsqueeze(1).expand(B, xyz_channel, sample_num)
        select_feature = select_feature.gather(2, thre_index_feature)
        select_xyz     = select_xyz.gather(2, thre_index_xyz)
        #select_index   = select_index.gather(1, thre_index)
        
        return select_feature, select_xyz, thre_index

    def forward(self, points, conf_times=0.5, add_channel1=None, add_channel2=None):
        B,C,N = points.size()

        xyz = points[:,:3,:]
        feature = points[:,3:6,:]
        
        # save intermediate results
        inter_xyz = [xyz]
        inter_index = [torch.arange(N).view(1,-1).repeat(B,1).to(self.device)]
        inter_feature = [feature]
        # Set Abstraction Layers
        for sa_module in self.sa_modules:
            xyz, feature, index = sa_module(xyz, feature)
            inter_xyz.append(xyz)
            inter_feature.append(feature)
            inter_index.append(inter_index[-1].gather(1, index))

        # Feature Propagation Layers
        sparse_xyz = xyz
        sparse_feature = feature
        # sparse_features, sparse_xyzs = [], []
        for fp_ind, fp_module in enumerate(self.fp_modules):
            dense_xyz = inter_xyz[-2 - fp_ind]
            dense_feature = inter_feature[-2 - fp_ind]
            fp_feature = fp_module(dense_xyz, sparse_xyz, dense_feature, sparse_feature)
            sparse_xyz = dense_xyz
            sparse_feature = fp_feature

            # sparse_features.insert(0, sparse_feature)
            # sparse_xyzs.insert(0, sparse_xyz)
            if 2-fp_ind == self.sample_layer:
                select_feature = sparse_feature
                select_xyz     = sparse_xyz
                select_index   = inter_index[-2 - fp_ind]
                break

        # select_feature = sparse_features[self.sample_layer]
        # select_xyz     = sparse_xyzs[self.sample_layer]

        # if add_channel1 is not None and add_channel2 is not None:
        #     add_channel1 = add_channel1.view(B,1,N).repeat(1,sparse_feature.shape[1],1)
        #     add_channel2 = add_channel2.view(B,1,N).repeat(1,sparse_feature.shape[1],1)
        #     sparse_feature = torch.cat((sparse_feature, add_channel1.float(), add_channel2.float()), dim=1)
        # MLP

        # # (batch_size, channels, point_number)
        # x = self.mlp(select_feature)
        # # (batch_size, channels, point_number*conf_times)
        # x_conf = self.bn_conf(self.conv_conf(x))
        # x_conf = self.sigmoid(x_conf)
        # x_conf = x_conf.transpose(2,1).contiguous().view(B, -1)

        x = self.mlp(sparse_feature)
        # (batch_size, channels, point_number*conf_times)
        x_conf = self.bn_conf(self.conv_conf(x))
        x_conf = self.sigmoid(x_conf)
        x_conf = x_conf.transpose(2,1).contiguous().view(B, -1)

        select_feature, select_xyz, thre_index = self.select_using_conf(x_conf.gather(1, select_index), select_feature, select_xyz, conf_times)
        # (batch_size, channels, point_number*conf_times)
        select_x_cls = self.select_cls_mlp(select_feature)
        # (batch_size, reg_channels, point_number*conf_times)
        # x_cls = self.bn_cls(self.conv_cls(select_x))
        x_cls = self.conv_cls(select_x_cls)
        # (2: grasp classification)
        # x_cls = self.softmax(x_cls.view(B, self.k_anchor, 2, -1))
        # x_cls = x_cls.view(B, self.k_anchor*2, -1)

        select_x_reg = self.select_reg_mlp(select_feature)
        # x_reg = self.bn_reg(self.conv_reg(select_x))
        x_reg = self.conv_reg(select_x_reg)
        # (n: grasp params(x,y,z,rx,ry,rz,theta), grasp scores)
        x_reg = x_reg.view(B, self.k_anchor, self.k_reg, -1)
        x_reg[:,:,7:,:] = self.sigmoid(x_reg[:,:,7:,:])
        x_reg = x_reg.view(B, self.k_anchor * self.k_reg, -1)
 
        return x_conf, x_cls, x_reg, select_xyz, select_index, select_index.gather(1,thre_index)

class PointNet2Direct(nn.Module):
    """PointNet++ part segmentation with single-scale grouping

    PointNetSA: PointNet Set Abstraction Layer
    PointNetFP: PointNet Feature Propagation Layer

    Args:
        score_classes (int): the number of grasp score classes
        num_centroids (tuple of int): the numbers of centroids to sample in each set abstraction module
        radius (tuple of float): a tuple of radius to query neighbours in each set abstraction module
        num_neighbours (tuple of int): the numbers of neighbours to query for each centroid
        sa_channels (tuple of tuple of int): the numbers of channels within each set abstraction module
        fp_channels (tuple of tuple of int): the numbers of channels for feature propagation (FP) module
        num_fp_neighbours (tuple of int): the numbers of nearest neighbor used in FP
        seg_channels (tuple of int): the numbers of channels in segmentation mlp
        dropout_prob (float): the probability to dropout input features

    References:
        https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_part_seg.py

    """
    _SA_MODULE = PointNetSAModule
    _FP_MODULE = PointnetFPModule
    
    def __init__(self, sample_layer=0, k_reg=9, k_obj=2, use_rgb=True, 
                    use_multi=False, dropout_prob=0.5):
        super(PointNet2Direct, self).__init__()
        self.k_reg = k_reg
        self.sample_layer = sample_layer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_rgb   = use_rgb
        self.use_multi = use_multi

        num_centroids     = (5120, 1024, 256)
        radius            = (0.02, 0.08, 0.32)
        num_neighbours    = (64, 64, 64)
        sa_channels       = ((128, 128, 256), (256, 256, 512), (512, 512, 1024))
        fp_channels       = ((1024, 1024), (512, 512), (256, 256, 256))
        num_fp_neighbours = (3, 3, 3)
        seg_channels      = (512, 256, 256, 128)

        num_sa_layers = len(num_centroids)
        num_fp_layers = len(fp_channels)

        # Set Abstraction Layers
        input_chann = 3
        if use_rgb:
            input_chann += 3
        if use_multi:
            input_chann += 3
        feature_channels = input_chann - 3 

        self.sa_modules = nn.ModuleList()
        for ind in range(num_sa_layers):
            sa_module = self._SA_MODULE(in_channels=feature_channels,
                                        mlp_channels=sa_channels[ind],
                                        num_centroids=num_centroids[ind],
                                        radius=radius[ind],
                                        num_neighbours=num_neighbours[ind],
                                        use_xyz=True)
            self.sa_modules.append(sa_module)
            feature_channels = sa_channels[ind][-1]

        inter_channels = [input_chann - 3]
        inter_channels.extend([x[-1] for x in sa_channels])
        # Feature Propagation Layers
        self.fp_modules = nn.ModuleList()
        feature_channels = inter_channels[-1]
        for ind in range(num_fp_layers):
            in_channels = feature_channels + inter_channels[-2 - ind] 
            if inter_channels[-2 - ind] == 3 or inter_channels[-2 - ind] == 6: # use_rgb=True, use_multi=True
                in_channels += 3
            # in_channels = feature_channels + inter_channels[-2 - ind] \
            #            if inter_channels[-2 - ind] != 3 else feature_channels + inter_channels[-2 - ind]*2
            fp_module = self._FP_MODULE(in_channels=in_channels,
                                        mlp_channels=fp_channels[ind],
                                        num_neighbors=num_fp_neighbours[ind])
            self.fp_modules.append(fp_module)
            feature_channels = fp_channels[ind][-1]

        # MLP
        feature_channels = 256#fp_channels[-1-self.sample_layer][-1]
        self.mlp = SharedMLP(feature_channels, seg_channels, ndim=1, dropout_prob=dropout_prob)
        # self.select_cls_mlp = SharedMLP(feature_channels, seg_channels, ndim=1, dropout_prob=dropout_prob)
        # self.select_reg_mlp = SharedMLP(feature_channels, seg_channels, ndim=1, dropout_prob=dropout_prob)
        self.select_cls_mlp = SharedMLP(256, seg_channels, ndim=1, dropout_prob=dropout_prob)
        self.select_reg_mlp = SharedMLP(256, seg_channels, ndim=1, dropout_prob=dropout_prob)
            
        self.conv_conf = nn.Conv1d(seg_channels[-1], 1, 1)
        self.bn_conf = nn.BatchNorm1d(1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-2)

        self.conv_cls = nn.Conv1d(seg_channels[-1], 2, 1)
        self.bn_cls = nn.BatchNorm1d(2)
        self.conv_reg = nn.Conv1d(seg_channels[-1], self.k_reg, 1)
        self.bn_reg = nn.BatchNorm1d(self.k_reg)

        # for modules in [self.sa_modules, self.fp_modules, self.mlp, self.select_reg_mlp, 
        #                     self.select_cls_mlp, self.conv_conf, self.conv_cls, self.conv_reg]:
        #     for layer in modules.modules():
        #         if isinstance(layer, nn.Conv1d):
        #             torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
        #             if layer.bias is not None:
        #                 torch.nn.init.constant_(layer.bias, 0)  
        
        # bias_value = -(math.log((1 - prior_prob) / prior_prob))
        # torch.nn.init.constant_(self.conv_cls.bias, bias_value)      

    def select_using_conf(self, x_conf, select_feature, select_xyz, conf_times=0.5):
        sample_num = int(x_conf.shape[-1] * conf_times)
        B, feature_channel, xyz_channel = len(select_feature), select_feature.shape[1], select_xyz.shape[1]
        thre_index = torch.sort(x_conf, dim=-1, descending=True)[1][:,:sample_num]
        
        thre_index_feature, thre_index_xyz = thre_index.unsqueeze(1).expand(B, feature_channel, sample_num), \
                                thre_index.unsqueeze(1).expand(B, xyz_channel, sample_num)
        select_feature = select_feature.gather(2, thre_index_feature)
        select_xyz     = select_xyz.gather(2, thre_index_xyz)
        #select_index   = select_index.gather(1, thre_index)
        
        return select_feature, select_xyz, thre_index

    def select_using_conf_layer1(self, x_conf, select_feature, select_xyz, conf_times=0.5, select_index=None):
        '''
          Input:
            x_conf          [B, N]
            select_feature  [B, C, N1]
            select_xyz      [B, 3, N]
        '''
        sample_num = int(x_conf.shape[-1] * conf_times)
        B, feature_channel, xyz_channel = len(select_feature), select_feature.shape[1], select_xyz.shape[1]
        # thre_index: [B, len_select]
        thre_index = torch.sort(x_conf, dim=-1, descending=True)[1][:,:sample_num]
        select_thre_index   = select_index.gather(1, thre_index)
        # print(thre_index.shape, select_thre_index.shape)
        
        # thre_index_feature: [B, C, len_select], thre_index_xyz: [B, 3, len_select]
        thre_index_feature, thre_index_xyz = select_thre_index.unsqueeze(1).expand(B, feature_channel, sample_num), \
                                thre_index.unsqueeze(1).expand(B, xyz_channel, sample_num)
        select_feature = select_feature.gather(2, thre_index_feature)
        select_xyz     = select_xyz.gather(2, thre_index_xyz)
        return select_feature, select_xyz, thre_index

    def forward(self, points, conf_times=0.5, add_widths=None):
        B,C,N = points.size()

        xyz = points[:,:3,:]
        feature = points[:,3:6,:] if self.use_rgb else None
        add_feature = add_widths.to(xyz).view(B,1,1).repeat(1,3,N) if self.use_multi else None
        
        if feature is not None:
            if add_feature is not None:
                feature = torch.cat((feature, add_feature), dim=1)
        else:
            if add_feature is not None:
                feature = add_feature

        # save intermediate results
        inter_xyz = [xyz]
        inter_index = [torch.arange(N).view(1,-1).repeat(B,1).to(self.device)]
        inter_feature = [feature]
        # Set Abstraction Layers
        for sa_module in self.sa_modules:
            xyz, feature, index = sa_module(xyz, feature)
            inter_xyz.append(xyz)
            inter_feature.append(feature)
            inter_index.append(inter_index[-1].gather(1, index))

        # Feature Propagation Layers
        sparse_xyz = xyz
        sparse_feature = feature
        for fp_ind, fp_module in enumerate(self.fp_modules):
            dense_xyz = inter_xyz[-2 - fp_ind]
            dense_feature = inter_feature[-2 - fp_ind]
            fp_feature = fp_module(dense_xyz, sparse_xyz, dense_feature, sparse_feature)
            sparse_xyz = dense_xyz
            sparse_feature = fp_feature

            if 2-fp_ind == self.sample_layer:
                select_feature = sparse_feature
                select_xyz     = sparse_xyz
                select_index   = inter_index[-2 - fp_ind]
        
        # # MLP
        # # (batch_size, channels, point_number)
        # x = self.mlp(select_feature)
        # # (batch_size, channels, point_number*conf_times)
        # x_conf = self.bn_conf(self.conv_conf(x))
        # x_conf = self.sigmoid(x_conf)
        # x_conf = x_conf.transpose(2,1).contiguous().view(B, -1)


        x = self.mlp(sparse_feature)
        # (batch_size, channels, point_number*conf_times)
        x_conf = self.bn_conf(self.conv_conf(x))
        x_conf = self.sigmoid(x_conf)
        x_conf = x_conf.transpose(2,1).contiguous().view(B, -1)

        select_feature, select_xyz, thre_index = self.select_using_conf_layer1(x_conf.gather(1, select_index), sparse_feature, select_xyz, conf_times, select_index)
        #select_feature, select_xyz, thre_index = self.select_using_conf(x_conf, select_feature, select_xyz, conf_times)
        # (batch_size, channels, point_number*conf_times)
        # thre_index = select_index
        # print(thre_index)
        select_x_cls = self.select_cls_mlp(select_feature)
        # (batch_size, reg_channels, point_number*conf_times)
        x_cls = self.conv_cls(select_x_cls)
        # (2: grasp classification)

        select_x_reg = self.select_cls_mlp(select_feature)
        x_reg = self.conv_reg(select_x_reg)
        # (n: grasp params(x,y,z,rx,ry,rz,theta), grasp scores)
        x_reg[:,7:,:] = self.sigmoid(x_reg[:,7:,:])
 
        return x_conf, x_cls, x_reg, select_xyz, select_index, select_index.gather(1,thre_index)
        # return x_cls, x_reg, select_xyz#, select_index, select_index.gather(1,thre_index)

class PointNet2TwoStage(nn.Module):
    def __init__(self, num_points, input_chann, k_cls, k_reg, k_reg_theta, add_channel_flag=False):
        super(PointNet2TwoStage, self).__init__()
        self.num_points = num_points
        self.k_reg = k_reg
        self.k_cls = k_cls
        self.k_reg_no_anchor = self.k_reg // self.k_cls
        self.k_reg_theta = k_reg_theta

        if not add_channel_flag:
            self.conv = nn.Conv1d(256, 1024, 1, bias=True)
        else:
            self.conv = nn.Conv1d(256*3, 1024, 1)

        self.bn = nn.BatchNorm1d(1024)

        #self.conv_cls1 = nn.Conv1d(128, 1024, 1)#128128+1024
        self.conv_cls2 = nn.Conv1d(1024, 256, 1)#128+1024
        self.conv_cls3 = nn.Conv1d(256, 128, 1)#128+1024
        self.linear_cls = torch.nn.Linear(128, self.k_cls)
        self.conv_cls4 = nn.Conv1d(128, self.k_cls, 1)
        #self.bn_cls1 = nn.BatchNorm1d(1024)
        self.bn_cls2 = nn.BatchNorm1d(256)
        self.bn_cls3 = nn.BatchNorm1d(128)
        self.bn_cls4 = nn.BatchNorm1d(self.k_cls)

        #self.conv_reg1 = nn.Conv1d(128, 1024, 1)#128+1024
        self.conv_reg2 = nn.Conv1d(1024, 256, 1)#+1024
        self.conv_reg3 = nn.Conv1d(256, 128, 1)#+1024
        self.conv_reg4 = nn.Conv1d(128, self.k_reg, 1)
        #self.bn_reg1 = nn.BatchNorm1d(1024)
        self.bn_reg2 = nn.BatchNorm1d(256)
        self.bn_reg3 = nn.BatchNorm1d(128)
        self.bn_reg4 = nn.BatchNorm1d(self.k_reg)

        #self.conv_reg_theta = nn.Conv1d(int(self.k_reg/7*3), self.k_reg_theta, 1)
        #self.bn_reg_theta = nn.BatchNorm1d(self.k_reg_theta)

        self.mp1 = nn.MaxPool1d(num_points)
        self.ap = torch.nn.AdaptiveAvgPool1d(1)
        self.sigmod = nn.Sigmoid()

    def forward(self, xyz, feature):
        #x = F.avg_pool1d(xyz.float(),self.num_points)
        mp_x = self.mp1(xyz) #[len(true_mask), 128, 1]
        ####mp_x = xyz.view(-1, 128, 1)
        if feature is not None:
            #x[:,:128,:] = x[:,:128,:] + feature.view(feature.shape[0], feature.shape[1],1)
            mp_x = torch.cat((mp_x, feature.view(feature.shape[0], feature.shape[1],1)), dim=1)
            #mp_x = feature.view(feature.shape[0], feature.shape[1],1)
        x = mp_x
        
        x = F.relu(self.bn(self.conv(x)))

        x_cls = F.relu(self.bn_cls2(self.conv_cls2(x)))
        x_cls = F.relu(self.bn_cls3(self.conv_cls3(x_cls)))
        x_cls = self.bn_cls4(self.conv_cls4(x_cls))

        B,C,_ = x_cls.size()
        x_cls = x_cls.view(B,C)

        x_reg = F.relu(self.bn_reg2(self.conv_reg2(x)))
        x_reg = F.relu(self.bn_reg3(self.conv_reg3(x_reg)))
        x_reg = self.bn_reg4(self.conv_reg4(x_reg))

        x_reg = x_reg.view(B,-1,self.k_reg_no_anchor)
        x_reg[:,:,7:] = self.sigmod(x_reg[:,:,7:])
        '''
        x_reg = x_reg.view(B,-1,7)
        x_reg_theta = self.bn_reg_theta(self.conv_reg_theta(x_reg[:,:,3:6].contiguous().view(B,-1,1)))
        x_reg_new = torch.cat([x_reg, x_reg_theta], dim=2)
        x_reg_new = x_reg_new[:,:,[0,1,2,3,4,5,7,6]]
        return x_cls, x_reg_new, mp_x
        '''
        return x_cls, x_reg, mp_x

class PointNet2Refine1(nn.Module):
    def __init__(self, num_points = 2500, input_chann = 3, k_cls = 2, k_reg = 8):
        super(PointNet2Refine1, self).__init__()
        self.num_points = num_points
        self.k_reg = k_reg
        self.k_cls = k_cls

        self.conv_formal = nn.Conv1d(384, 1024, 1)
        self.bn_formal = nn.BatchNorm1d(1024)

        # self.conv_formal_cls2 = nn.Conv1d(1024, 128, 1)
        # self.conv_formal_cls3 = nn.Conv1d(128, self.k_cls, 1)
        # self.bn_formal_cls2 = nn.BatchNorm1d(128)
        # self.bn_formal_cls3 = nn.BatchNorm1d(self.k_cls)
        self.conv_formal_cls2 = nn.Conv1d(1024, 128, 1)
        self.conv_formal_cls3 = nn.Conv1d(128, self.k_cls, 1)
        self.bn_formal_cls2 = nn.BatchNorm1d(128)
        self.bn_formal_cls3 = nn.BatchNorm1d(self.k_cls)

        self.conv_formal_reg2 = nn.Conv1d(1024, 128, 1)
        self.conv_formal_reg3 = nn.Conv1d(128, self.k_reg, 1)
        self.bn_formal_reg2 = nn.BatchNorm1d(128)
        self.bn_formal_reg3 = nn.BatchNorm1d(self.k_reg)

        self.mp1 = nn.MaxPool1d(num_points)
        self.ap = torch.nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, gripper_feature, group_feature):
        '''
        gripper_feature : [B, region_num, 128]
        group_feature: [B, 128]
        '''
        gripper_feature = self.mp1(gripper_feature) #[B, 128, 1]
        # x: [B, 128, 1]
        x = gripper_feature
        if group_feature is not None:
            # x: [B, 256, 1]
            x = torch.cat((x, group_feature.view(group_feature.shape[0], \
                                group_feature.shape[1],1)), dim=1)
        x = F.relu(self.bn_formal(self.conv_formal(x)))

        # x_cls = F.relu(self.bn_formal_cls2(self.conv_formal_cls2(x)))
        # x_cls = self.bn_formal_cls3(self.conv_formal_cls3(x_cls))
        # x_cls = x_cls.view(x_cls.shape[0], x_cls.shape[1])

        x_cls = F.relu(self.bn_formal_cls2(self.conv_formal_cls2(x)))
        x_cls = self.bn_formal_cls3(self.conv_formal_cls3(x_cls))
        x_cls = x_cls.view(x_cls.shape[0], x_cls.shape[1])

        x_reg = F.relu(self.bn_formal_reg2(self.conv_formal_reg2(x)))
        x_reg = self.bn_formal_reg3(self.conv_formal_reg3(x_reg))
        x_reg = x_reg.view(x_reg.shape[0], x_reg.shape[1])
        #x_reg[:,-1] = self.sigmoid(x_reg[:,-1])
        
        return x_cls, x_reg
        # return x_reg

if __name__ == '__main__':
    pass
