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

class PointNet2Cls(nn.Module):
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
        super(PointNet2Cls, self).__init__()
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

        self.region_radius = 0.02
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
        thre_index = torch.sort(x_conf, dim=-1, descending=True)[1][:,:sample_num]
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
            all_thre_index = torch.nonzero(x_conf[i] >= 0.5).view(-1)
            if len(all_thre_index) <= 0:
                thre_index     = _F.farthest_point_sample(select_xyz[i][:3].view(1,3,-1), sample_num)
            else:
                thre_index     = _F.farthest_point_sample(select_xyz[i][:3,all_thre_index].view(1,3,-1), sample_num)
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










if __name__ == '__main__':
    pass
