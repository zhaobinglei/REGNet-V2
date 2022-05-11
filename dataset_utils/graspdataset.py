import os
import glob
import pickle
import torch
import math
import torch.utils.data
import torch.nn as nn
import numpy as np
import random
import open3d #0.7.0
from multi_model.utils.pn2_utils import function as _F

def get_grasp_gt(center_pc, data_paths, score_thre=None,regrad=False):
    '''
        Input:
        center_pc  : [B, 3, select_points_number]
        data_paths : list
        Output:
        grasp_trans: [B, center_num, 8] (center[3], axis_y[3], grasp_angle[1], score[1])
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, pNum, _ = center_pc.shape
    grasp_label           = torch.full((B, pNum, 3, 4), -1.0).to(device)
    gscore, gcenter_score = torch.full((B, pNum), -1.0).to(device), torch.full((B, pNum), -1.0).to(device)
    
    data_paths = data_paths.reshape(-1)
    for i in range(len(data_paths)):
        data = np.load(data_paths[i], allow_pickle=True)
        if 'frame' in data.keys():
            grasp              = torch.Tensor(data['frame']).to(device)
            grasp_score        =  torch.Tensor(data['antipodal_score']).to(device)
            grasp_center_score =  torch.Tensor(data['antipodal_score']).to(device)

        else:
            grasp              = torch.Tensor(data['select_frame']).to(device)
            grasp_score        = torch.Tensor(data['select_antipodal_score']).to(device) if type(data['select_antipodal_score']) \
                                                    is np.ndarray else data['select_antipodal_score'].to(device)
            grasp_center_score = torch.Tensor(data['select_center_score']).to(device) if type(data['select_center_score']) \
                                                    is np.ndarray else data['select_center_score'].to(device) 
            grasp_class_label  = torch.Tensor(data['select_frame_label']).to(device) if type(data['select_frame_label']) \
                                                    is np.ndarray else data['select_frame_label'].to(device)
            # grasp_score           = torch.Tensor(data['select_score']) / 3 if type(data['select_score']) is np.ndarray else data['select_score'] / 3
            # grasp_vertical_score  = torch.Tensor(data['select_vertical_score']) if type(data['select_vertical_score']) is np.ndarray else data['select_vertical_score']
        
        if score_thre:
            select_inedx = (grasp_score > score_thre)
            grasp, grasp_score = grasp[select_inedx], grasp_score[select_inedx]
            grasp_center_score = grasp_center_score[select_inedx]
        # print("antipodal_score mean: ", grasp_score.mean(), "center_score mean: ", grasp_center_score.mean())
        if regrad:
            grasp[:,2,3] += 0.25
        grasp_center, grasp_x = grasp[:,:3,3], grasp[:,:3,0]
        # in_grasp_center     = (grasp_center + grasp_x * depth).float()
        distance              = compute_distance(center_pc[i], grasp_center)
        dist_min_values, dist_min_index = torch.min(distance, dim=1)
        no_grasp_mask                   = (dist_min_values > 0.07) # sqrt(0.005)

        grasp_label[i], gscore[i] = grasp[dist_min_index,:3,:4], grasp_score[dist_min_index]
        grasp_label[i][no_grasp_mask], gscore[i][no_grasp_mask]  = -1, -1
        
        gcenter_score[i] = grasp_center_score[dist_min_index]
        gcenter_score[i][no_grasp_mask] = -1
    
    grasp_trans = _transform_grasp(grasp_label, gscore, gcenter_score)
    # test whether grasps are transformed right
    # grasp_label_ori = _inv_transform_grasp(grasp_trans)
    return grasp_trans

def get_grasp_allobj(pc, predict_score, params, data_paths, use_theta=True):
    '''
      randomly sample grasp center in positive points set (all obj), 
      and get grasps centered at these centers.
      Input:
        pc             :[B,N,6]  input points 
        predict_score  :[B,N]
        params         :list [center_num(int), score_thre(float), group_num(int), r_time_group(float), group_num_more(int), \
                              r_time_group_more(float), width(float), height(float), depth(float)]
        data_paths     :list
      Output:
        center_pc           :[B, center_num, 6]
        center_pc_index     :[B, center_num] index of selected center in sampled points
        pc_group_index      :[B, center_num, group_num]
        pc_group_more_index :[B, center_num, group_num_more]
        pc_group            :[B, center_num, group_num, 6]
        pc_group_more       :[B, center_num, group_num_more, 6]
        grasp_labels        :[B, center_num, 8] the labels of grasps (center[3], axis_y[3], grasp_angle[1], score[1])
    '''
    [center_num, score_thre, group_num, r_time_group, group_num_more, r_time_group_more, \
                                                            width, height, depth] = params
    
    center_pc, center_pc_index = _select_score_center(pc, predict_score, center_num, score_thre)
    pc_group_index, pc_group = _get_group_pc(pc, center_pc, center_pc_index, group_num, width, height, depth, r_time_group)
    pc_group_more_index, pc_group_more = _get_group_pc(pc, center_pc, center_pc_index, group_num_more, width, height, depth, r_time_group_more)

    grasp_labels = None
    if len(data_paths) > 0:
        grasp_labels = _get_center_grasp(center_pc_index, center_pc, data_paths, depth, use_theta)
    return center_pc, center_pc_index, pc_group_index, pc_group, pc_group_more_index, pc_group_more, grasp_labels

def _transform_grasp(grasp_ori, antipodal_score_ori, center_score_ori):
    '''
      Input:
        grasp_ori: [B, center_num, 3, 4] 
                   [[x1, y1, z1, c1],
                    [x2, y2, z2, c2],
                    [x3, y3, z3, c3]]
        antipodal_score_ori: [B, center_num]
        center_score_ori   : [B, center_num]
      Output:
        grasp_trans:[B, center_num, 9] (center[3], axis_y[3], grasp_angle[1], antipodal_score[1], center_score[1])
    '''
    B, CN = antipodal_score_ori.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    grasp_trans[:,:,8]   = center_score_ori
    return grasp_trans

def _inv_transform_grasp(grasp_trans):
    '''
      Input:
        grasp_trans:[B, center_num, 9] (center[3], axis_y[3], grasp_angle[1], antipodal_score[1], center_score[1])
      Output:
        matrix: [B, center_num, 3, 4] 
                   [[x1, y1, z1, c1],
                    [x2, y2, z2, c2],
                    [x3, y3, z3, c3]]
    '''
    B, CN, channels = grasp_trans.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    no_grasp_mask = (grasp_trans.view(B*CN, channels)[:,-1] == -1)

    center, axis_y, angle = grasp_trans.view(-1,channels)[:,:3], grasp_trans.view(-1,channels)[:,3:6], grasp_trans.view(-1,8)[:,6]
    cos_t, sin_t = torch.cos(angle), torch.sin(angle)

    one, zero = torch.ones((B*CN, 1), dtype=torch.float32).to(device), torch.zeros((B*CN, 1), dtype=torch.float32).to(device)
    R1 = torch.cat( (cos_t.view(-1,1), zero, -sin_t.view(-1,1), zero, one, zero, \
                    sin_t.view(-1,1), zero, cos_t.view(-1,1)), dim=1).view(-1,3,3).to(device)

    # R1 = torch.zeros((B*CN, 3, 3))
    # for i in range(B*CN):
    #     r = torch.tensor([[cos_t[i], 0, -sin_t[i]],[0, 1, 0],[sin_t[i], 0, cos_t[i]]]).view(1,3,3)
    #     R1[i,:,:] = r

    norm_y = torch.norm(axis_y, dim=1)
    axis_y = torch.div(axis_y, norm_y.view(-1,1))
    axis_y[torch.nonzero(torch.eq(norm_y, 0))] = torch.tensor(([0,1,0]), dtype=torch.float).to(device)
    
    axis_x = torch.cat((axis_y[:, 1].view(-1,1), -axis_y[:, 0].view(-1,1), zero), 1)
    norm_x = torch.norm(axis_x, dim=1)
    axis_x = torch.div(axis_x, norm_x.view(-1,1))
    axis_x[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float).to(device)
    
    axis_z = torch.cross(axis_x, axis_y, dim=1)
    norm_z = torch.norm(axis_z, dim=1)
    axis_z = torch.div(axis_z, norm_z.view(-1,1))
    axis_z[torch.nonzero(torch.eq(norm_z, 0))] = torch.tensor(([0,0,1]), dtype=torch.float).to(device)
    
    matrix   = torch.cat((axis_x.view(-1,3,1), axis_y.view(-1,3,1), axis_z.view(-1,3,1)), dim=2)
    matrix   = torch.bmm(matrix, R1)
    approach = matrix[:,:,0]
    norm_x   = torch.norm(approach, dim=1)
    approach = torch.div(approach, norm_x.view(-1,1))
    approach[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float).to(device)
    
    minor_normal = torch.cross(approach, axis_y, dim=1)
    matrix       = torch.cat((approach.view(-1,3,1), axis_y.view(-1,3,1), \
                                minor_normal.view(-1,3,1), center.view(-1,3,1)), dim=2)
    matrix[no_grasp_mask] = -1
    matrix = matrix.view(len(grasp_trans), -1, 3, 4)
    return matrix

def compute_distance(points1, points2):
    # compute the distance between points1 and points2
    # points1:  [len(points1), 3]
    # points2:  [len(points2), 3]
    # distance: [len(points1), len(points2)]
    len1, len2 = points1.shape[0], points2.shape[0]
    points1 = points1.repeat(len2, 1)
    points2 = points2.view(len2,1,3).repeat(1, len1, 1).view(-1,3)
    distance = torch.norm(points1-points2, dim=1) 
    distance = distance.view(len2, len1).transpose(1,0)
    #print(distance)
    return distance

def compute_distance2(points1, points2):
    # compute the square of distance between points1 and points2
    #distance [len(points1), len(points2)]
    distance2 = -2 * points1[:, :3].mm(points2.transpose(1,0))
    distance2 += torch.sum(points2.mul(points2), 1).view(1,-1).repeat(points1.size()[0],1)
    distance2 += torch.sum(points1[:, :3].mul(points1[:, :3]), 1).view(-1,1).repeat(1, points2.size()[0])
    return distance2

def _get_local_points_batch(all_points, center, width, height, depth, r_time):
    '''
      Get the points around **one** scored center
        Input:
          all_points: [N, C]
          center:     [center_num, C]
        Return:
          group_points_index: [center_num, N]
    '''
    all_points_repeat = all_points[:, :3].repeat(len(center),1).view(-1,3)
    center_repeat = center[:, :3].repeat(1,len(all_points)).view(-1,3)
    dist = all_points_repeat - center_repeat
    max_radius = max(width, height, depth)*r_time
    # distance:  [center_num, N]
    distance = torch.sqrt(torch.mul(dist[:,0], dist[:,0]) + torch.mul(dist[:,1], dist[:,1]) + torch.mul(dist[:,2], dist[:,2])).view(len(center),-1)
    group_points_index = (distance <= max_radius)
    return group_points_index

def _get_local_points(all_points, center, width, height, depth, r_time):
    '''
      Get the points around **one** scored center
        Input:
          all_points: [N, C]
          center:     [C]
        Return:
          group_points_index: [uncertain number]
    '''
    max_radius = max(width, height, depth)*r_time
    group_points_index = (torch.abs(all_points[:,0]-center[0]) < max_radius) & (torch.abs(all_points[:,1]-center[1]) < max_radius) & (torch.abs(all_points[:,2]-center[2]) < max_radius) #& (torch.sum(all_points[:,:3] != center[:3], dim=1)/3).byte()
    group_points_index = torch.nonzero(group_points_index).view(-1)
    return group_points_index

def _get_group_pc(pc, center_pc, center_pc_index, group_num, width, height, depth, r_time):
    '''
     Input:
        pc              :[B,N,6]
        center_pc       :[B,center_num,6]
        center_pc_index :[B,center_num]
        group_num       :int
     Return:
        pc_group_index  :[B,center_num,group_num] index of grouped points of selected center in sampled points
        pc_group        :[B,center_num,group_num,6]
    '''
    B,A,C = pc.shape
    center_num = center_pc.shape[1]
    pc_group = torch.full((B, center_num, group_num, C), -1.0)
    pc_group_index = torch.full((B, center_num, group_num), -1)
    if pc.is_cuda:
        pc_group, pc_group_index = pc_group.cuda(), pc_group_index.cuda()
        
    # Get the points around one scored center    
    for i in range(B):
        group_points_index = _get_local_points_batch(pc[i], center_pc[i], width, height, depth, r_time)
        for j in range(center_num):
            group_points_index_one = torch.nonzero(group_points_index[j]).view(-1)
            if len(group_points_index_one) >= group_num:
                group_points_index_one = group_points_index_one[np.random.choice(len(group_points_index_one), group_num, replace=False)]
            elif len(group_points_index_one) > 0:
                group_points_index_one = group_points_index_one[np.random.choice(len(group_points_index_one), group_num, replace=True)]
                
            if len(group_points_index_one) > 0:
                pc_group_index[i,j] = group_points_index_one
                pc_group[i,j] = pc[i,group_points_index_one]

    ##if pc.is_cuda:
    ##    pc = pc.cpu()
    ##for i in range(B):
    ##    cur_pc_o3d = open3d.geometry.PointCloud()
    ##    cur_pc_o3d.points = open3d.utility.Vector3dVector(pc[i,:,:3]) 
    ##    cur_pc_tree = open3d.geometry.KDTreeFlann(cur_pc_o3d)
    ##    for j in range(center_num):
    ##        [k, idx, _] = cur_pc_tree.search_radius_vector_3d(cur_pc_o3d.points[center_pc_index[i,j]], max(width, height, depth)*r_time)

    return pc_group_index, pc_group

def _select_score_center(pc, pre_score, center_num, score_thre):
    '''
     Get the points where their scores are positive as regression centers of grasps
     Input:
        pc              :[B,N,6]
        pre_score       :[B,N], belongs to [0,1]
        score_thre      :float, score threshold, belongs to (0,1)
        center_num      :int
     Return:
        center_pc       :[B, center_num, 6]
        center_pc_index :[B, center_num] index of selected center in sampled points
    '''
    B,A,C = pc.shape
    pre_score = pre_score.cpu()
    if B == 1:
        positive_pc_mask = (pre_score.view(-1) > score_thre)
        positive_pc_mask = (pre_score.view(-1) > score_thre)
        positive_pc_mask = positive_pc_mask.cpu().numpy()
        map_index = torch.Tensor(np.nonzero(positive_pc_mask)[0]).view(-1).long()

        center_pc = torch.full((center_num, C), -1.0)
        center_pc_index = torch.full((center_num,), -1)

        pc = pc.view(-1,C)
        cur_pc = pc[map_index,:]
        if len(cur_pc) > center_num:
            center_pc_index = _F.farthest_point_sample(cur_pc[:,:3].view(1,-1,3).transpose(2,1), center_num).view(-1)

            center_pc_index = map_index[center_pc_index.long()]
            center_pc = pc[center_pc_index.long()]
            
        elif len(cur_pc) > 0:
            center_pc_index[:len(cur_pc)] = torch.arange(0, len(cur_pc))
            center_pc_index[len(cur_pc):] = torch.Tensor(np.random.choice(cur_pc.shape[0], center_num-len(cur_pc), replace=True))
            center_pc_index = map_index[center_pc_index.long()]
            center_pc = pc[center_pc_index.long()]
            
        else:
            center_pc_index = torch.Tensor(np.random.choice(pc.shape[0], center_num, replace=False))
            center_pc = pc[center_pc_index.long()]
    
        center_pc = center_pc.view(1,-1,C)
        center_pc_index = center_pc_index.view(1,-1)
        if pc.is_cuda:
            center_pc = center_pc.cuda()
            center_pc_index = center_pc_index.cuda()
        return center_pc, center_pc_index

    # ---------------------- for train -------------------
    positive_pc_mask = (pre_score > score_thre)

    center_pc = torch.full((B, center_num, C), -1.0)
    center_pc_index = torch.full((B, center_num), -1)
    for i in range(B):
        cur_pc = pc[i,positive_pc_mask[i],:]
        if len(cur_pc) > center_num:
            #center_pc_index[i] = torch.Tensor(np.random.choice(cur_pc.shape[0], center_num, replace=False))
            #center_pc_index[i] = _farthest_point_sample(cur_pc, center_num)
            center_pc_index[i] = _F.farthest_point_sample(cur_pc[:,:3].view(1,-1,3).transpose(2,1), center_num).view(-1)

            map_index = torch.nonzero(positive_pc_mask[i]).view(-1)
            center_pc_index[i] = map_index[center_pc_index[i].long()]
            center_pc[i] = pc[i, center_pc_index[i].long()]
            
        elif len(cur_pc) > 0:
            center_pc_index[i,:len(cur_pc)] = torch.arange(0, len(cur_pc))
            center_pc_index[i,len(cur_pc):] = torch.Tensor(np.random.choice(cur_pc.shape[0], center_num-len(cur_pc), replace=True))
            #center_pc[i] = cur_pc[center_pc_index[i].long()]

            map_index = torch.nonzero(positive_pc_mask[i]).view(-1)
            center_pc_index[i] = map_index[center_pc_index[i].long()]
            center_pc[i] = pc[i, center_pc_index[i].long()]
            
        else:
            center_pc_index[i] = torch.Tensor(np.random.choice(pc.shape[1], center_num, replace=False))
            center_pc[i] = pc[i, center_pc_index[i].long()]
    
    if pc.is_cuda:
       center_pc = center_pc.cuda()
       center_pc_index = center_pc_index.cuda()
    return center_pc, center_pc_index

def _farthest_point_sample(xyz, npoint):
    """
      Input:
        xyz: pointcloud data, [N, C]
        npoint: number of samples
      Return:
        centroids: sampled pointcloud index, [npoint]
    """
    cuda = xyz.is_cuda
    N, C = xyz.shape

    #import time
    #s1 = time.time()
    centroids = torch.zeros((npoint), dtype=torch.long)
    distance = torch.ones((N)) * 1e10
    farthest = torch.tensor([np.random.randint(0, N)], dtype=torch.long)
    if cuda:
        centroids, distance, farthest = centroids.cuda(), distance.cuda(), farthest.cuda()
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :3].view(1, 3)
        dist = torch.sum((xyz[:, :3] - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        torch.argmin(distance)
        farthest = int(torch.argmax(distance))
    #e1 = time.time()
    #print(e1-s1)
    '''
    s2 = time.time()
    if cuda:
       xyz = xyz.cpu()
    xyz = np.array(xyz)
    centroids = np.zeros(npoint, dtype=int)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :3].reshape(1, 3)
        dist = np.sum((xyz[:, :3] - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = int(np.argmax(distance))
    centroids = torch.from_numpy(centroids)
    if cuda:
        centroids = centroids.cuda()
    e2 = time.time()
    print(e2-s2)
    '''
    return centroids
