import os, sys
import numpy as np
import time
import pickle
import transforms3d, open3d
import logging

import torch, random
from tensorboardX import SummaryWriter

import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR

sys.path.append('contrast')
from gpd.pointcloud import PointCloud
from gpd.configs import config
from gpd.network.pointnet import PointNetCls
# from network.pointnet2 import PointNet2Cls
from gpd.network.cnn import GPDClassifier


sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../..'))
from dataset_utils.eval_score.eval import eval_test, eval_validate, eval_validate_wo_view, eval_test_batch
from dataset_utils.graspdataset import compute_distance
import multi_model.utils.pn2_utils.function as _F

def get_limit_points(local_cloud, x_limit, y_limit, z_limit, close_region_points_num):
    B, N, _ = local_cloud.shape
    x1 = local_cloud[:,:,0] > 0
    x2 = local_cloud[:,:,0] < x_limit
    y1 = local_cloud[:,:,1] > -y_limit
    y2 = local_cloud[:,:,1] < y_limit
    z1 = local_cloud[:,:,2] > -z_limit
    z2 = local_cloud[:,:,2] < z_limit
    a = torch.sum(torch.cat((x1.view(B,-1,1), x2.view(B,-1,1), y1.view(B,-1,1), \
                    y2.view(B,-1,1), z1.view(B,-1,1), z2.view(B,-1,1)), dim=-1), dim=-1)
    index = (a==6).float().view(B,-1,1).repeat(1,1,3).permute(0,2,1)
    center_index = torch.ones((B,3,1))
    if local_cloud.is_cuda:
        center_index = center_index.cuda()
    closing_index, closing_count = _F.ball_query(index, center_index, 0.5, close_region_points_num)
    closing_count = closing_count.view(-1)
    closing_index = closing_index.view(B,-1)
    return closing_index, closing_count

def get_close_region_points(points, grasps, gripper_params):
    '''
        Input:
            point: [bs,N,6]
            grasps: [bs,M,4,4]
                [[x1, y1, z1, c1],
                 [x2, y2, z2, c2],
                 [x3, y3, z3, c3],
                 [0,  0,  0,  1 ]]
            gripper_params: [width, height, depth, close_region_points_num]
        Output:
            gripper_pc: [bs*M,close_region_points_num,6]
    '''
    bs, N, channel  = points.shape
    _ , M, gc1, gc2 = grasps.shape
    B = bs * M
    width, height, depth, close_region_points_num = gripper_params

    points = points.view(bs,1,-1,channel).repeat(1,M,1,1).view(B,N,channel)
    grasps = grasps.view(B, gc1, gc2)
    # matrix, center = grasps[:,:3,:3], grasps[:,:3,3]
    # ## pcs_t: [B,G,3]
    # pcs_t = torch.bmm(matrix.float(), (points[:,:,:3].float() - \
    #                     center.view(-1,1,3).repeat(1, points.shape[1], 1).float()).permute(0,2,1)).permute(0,2,1)
    if points.is_cuda:
        pc_homo = torch.cat([points[:,:,:3].transpose(2,1), \
                            torch.ones(points[:,:,:3].shape[0], 1, points[:,:,:3].shape[1]).cuda()], dim=1).float()
    else:
        pc_homo = torch.cat([points[:,:,:3].transpose(2,1), \
                            torch.ones(points[:,:,:3].shape[0], 1, points[:,:,:3].shape[1])], dim=1).float()
    pcs_t = torch.bmm(torch.inverse(grasps.float()), pc_homo).transpose(2,1)[:,:,:3]     

    x_limit = depth    
    z_limit = height/2 
    y_limit = width/2

    # closing_index, keep_num = get_limit_points(pcs_t, x_limit, y_limit, z_limit, close_region_points_num)
    # keep = torch.ones((B)).bool()
    # keep[keep_num<=10] = 0

    # for i in range(len(closing_index)):
    #     closing_index[i] = closing_index[i][np.random.choice(int(keep_num[i]), close_region_points_num, replace=True)]
    
    # if channel > 3:
    #     gripper_pc = torch.cat( (pcs_t[:,:,0].gather(1,closing_index).view(-1,close_region_points_num,1),\
    #                     pcs_t[:,:,1].gather(1,closing_index).view(-1,close_region_points_num,1),\
    #                     pcs_t[:,:,2].gather(1,closing_index).view(-1,close_region_points_num,1),\
    #                     points[:,:,3].gather(1,closing_index).view(-1,close_region_points_num,1),\
    #                     points[:,:,4].gather(1,closing_index).view(-1,close_region_points_num,1),\
    #                     points[:,:,5].gather(1,closing_index).view(-1,close_region_points_num,1)) , dim=-1)
    # else:
    #     gripper_pc = torch.cat( (pcs_t[:,:,0].gather(1,closing_index).view(-1,close_region_points_num,1),\
    #                     pcs_t[:,:,1].gather(1,closing_index).view(-1,close_region_points_num,1),\
    #                     pcs_t[:,:,2].gather(1,closing_index).view(-1,close_region_points_num,1)) , dim=-1)
    
    x1 = pcs_t[:,:,0] > 0
    x2 = pcs_t[:,:,0] < x_limit
    y1 = pcs_t[:,:,1] > -y_limit
    y2 = pcs_t[:,:,1] < y_limit
    z1 = pcs_t[:,:,2] > -z_limit
    z2 = pcs_t[:,:,2] < z_limit
    a = torch.sum(torch.cat((x1.view(B,-1,1), x2.view(B,-1,1), y1.view(B,-1,1), \
                    y2.view(B,-1,1), z1.view(B,-1,1), z2.view(B,-1,1)), dim=-1), dim=-1)
    
    gripper_pc = torch.full((B,close_region_points_num,channel), -1.0)
    keep = torch.zeros((B)).bool()
    keep_num =  torch.zeros((B)).int()
    if a.is_cuda:
        gripper_pc, keep = gripper_pc.cuda(), keep.cuda()
    for i in range(B):
        index = torch.nonzero(a[i] == 6).view(-1)
        cur_closing_count = len(index)
        if cur_closing_count > close_region_points_num:
            index = index[np.random.choice(len(index),close_region_points_num,replace=False)]
            keep[i] = 1
        elif cur_closing_count > 10:
            index = index[np.random.choice(len(index),close_region_points_num,replace=True)]
            keep[i] = 1
        keep_num[i] = cur_closing_count

        if cur_closing_count > 10:
            if channel > 3:
                gripper_pc[i] = torch.cat((pcs_t[i,index], points[i,index,3:]), dim=-1)
            else:
                gripper_pc[i] = pcs_t[i,index]
    return gripper_pc[keep], keep, keep_num

def cal_projection(point_cloud_voxel, m_width_of_pic, margin, surface_normal, order, width, voxel_point_num):
    occupy_pic = np.zeros([m_width_of_pic, m_width_of_pic, 1], dtype=np.float32)
    norm_pic = np.zeros([m_width_of_pic, m_width_of_pic, 3], dtype=np.float32)
    norm_pic_num = np.zeros([m_width_of_pic, m_width_of_pic, 1], dtype=np.float32)

    max_x = point_cloud_voxel[:, order[0]].max()
    min_x = point_cloud_voxel[:, order[0]].min()
    max_y = point_cloud_voxel[:, order[1]].max()
    min_y = point_cloud_voxel[:, order[1]].min()
    min_z = point_cloud_voxel[:, order[2]].min()

    tmp = max((max_x - min_x), (max_y - min_y))
    if tmp == 0:
        print("WARNING : the num of input points seems only have one, no possilbe to do learning on"
                "such data, please throw it away.  -- Hongzhuo")
        return occupy_pic, norm_pic
    # Here, we use the gripper width to cal the res:
    res = width / (m_width_of_pic-margin)

    voxel_points_square_norm = []
    x_coord_r = ((point_cloud_voxel[:, order[0]]) / res - m_width_of_pic / 2)
    y_coord_r = ((point_cloud_voxel[:, order[1]]) / res + m_width_of_pic / 2)
    z_coord_r = ((point_cloud_voxel[:, order[2]]) / res + m_width_of_pic / 2)
    x_coord_r = np.floor(x_coord_r).astype(int)
    y_coord_r = np.floor(y_coord_r).astype(int)
    z_coord_r = np.floor(z_coord_r).astype(int)
    voxel_index = np.array([x_coord_r, y_coord_r, z_coord_r]).T  # all point in grid
    coordinate_buffer = np.unique(voxel_index, axis=0)  # get a list of points without duplication
    K = len(coordinate_buffer)
    # [K, 1] store number of points in each voxel grid
    number_buffer = np.zeros(shape=K, dtype=np.int64)
    feature_buffer = np.zeros(shape=(K, voxel_point_num, 6), dtype=np.float32)
    index_buffer = {}
    for i in range(K):
        index_buffer[tuple(coordinate_buffer[i])] = i  # got index of coordinate

    for voxel, point, normal in zip(voxel_index, point_cloud_voxel, surface_normal):
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        if number < voxel_point_num:
            feature_buffer[index, number, :3] = point
            feature_buffer[index, number, 3:6] = normal
            number_buffer[index] += 1

    voxel_points_square_norm = np.sum(feature_buffer[..., -3:], axis=1)/number_buffer[:, np.newaxis]
    voxel_points_square = coordinate_buffer

    if len(voxel_points_square) == 0:
        return occupy_pic, norm_pic
    x_coord_square = voxel_points_square[:, 0]
    y_coord_square = voxel_points_square[:, 1]
    norm_pic[x_coord_square, y_coord_square, :] = voxel_points_square_norm
    occupy_pic[x_coord_square, y_coord_square] = number_buffer[:, np.newaxis]
    occupy_max = occupy_pic.max()
    assert(occupy_max > 0)
    occupy_pic = occupy_pic / occupy_max
    return occupy_pic, norm_pic

def get_gpd_projected_points(points, grasps, gripper_params):
    ''' for gpd baseline, only support input_chann == [3, 12]
        Input:
            point: [bs,N,6]
            grasps: [bs,M,4,4]
                [[x1, y1, z1, c1],
                 [x2, y2, z2, c2],
                 [x3, y3, z3, c3],
                 [0,  0,  0,  1 ]]
            gripper_params: [width, height, depth, close_region_points_num, project_chann]
        Output:
            gripper_pc: [bs*M,close_region_points_num,6]
    '''
    normal_K = 10
    normal_R = 0.01
    voxel_point_num  = 50
    projection_margin = 1
    minimum_point_amount = 150
    project_size = 60

    bs, N, channel  = points.shape
    _ , M, gc1, gc2 = grasps.shape
    B = bs * M
    width, height, depth, close_region_points_num, project_chann = gripper_params  
    x_limit = depth    
    z_limit = height/2 
    y_limit = width/2
    if project_chann not in [3, 12]:
        raise NotImplementedError

    points = points.view(bs,1,-1,channel).repeat(1,M,1,1).view(B,N,channel)
    grasps = grasps.view(B, gc1, gc2)
    if points.is_cuda:
        pc_homo = torch.cat([points[:,:,:3].transpose(2,1), \
                            torch.ones(points[:,:,:3].shape[0], 1, points[:,:,:3].shape[1]).cuda()], dim=1).float()
    else:
        pc_homo = torch.cat([points[:,:,:3].transpose(2,1), \
                            torch.ones(points[:,:,:3].shape[0], 1, points[:,:,:3].shape[1])], dim=1).float()
    pcs_t = torch.bmm(torch.inverse(grasps.float()), pc_homo).transpose(2,1)[:,:,:3]  


    proj_output = torch.zeros((B,project_size,project_size,project_chann))
    keep = torch.ones((B)).bool()
    for idx in range(B):
        x1 = pcs_t[idx,:,0] > 0
        x2 = pcs_t[idx,:,0] < x_limit
        y1 = pcs_t[idx,:,1] > -y_limit
        y2 = pcs_t[idx,:,1] < y_limit
        z1 = pcs_t[idx,:,2] > -z_limit
        z2 = pcs_t[idx,:,2] < z_limit
        a = torch.sum(torch.cat((x1.view(-1,1), x2.view(-1,1), y1.view(-1,1), \
                        y2.view(-1,1), z1.view(-1,1), z2.view(-1,1)), dim=-1), dim=-1)
        close_idx = torch.nonzero(a==6).view(-1).cpu().numpy()
        if len(close_idx) <= 10:
            keep[idx] = 0
            continue

        pc_points = pcs_t[idx].cpu().numpy().astype(np.float32)
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(pc_points) 
        pc.estimate_normals(
            search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=normal_R, max_nn=normal_K))
        pc_normal = np.array(pc.normals)
        grasp_pc = pc_points[close_idx]
        grasp_pc_norm = pc_normal[close_idx]

        m_width_of_pic = project_size
        margin = projection_margin
        order = np.array([0, 1, 2])
        occupy_pic1, norm_pic1 = cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm, order, width, voxel_point_num)
        if project_chann == 3:
            output = norm_pic1
        elif project_chann == 12:
            order = np.array([1, 2, 0])
            occupy_pic2, norm_pic2 = cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm, order, width, voxel_point_num)
            order = np.array([0, 2, 1])
            occupy_pic3, norm_pic3 = cal_projection(grasp_pc, m_width_of_pic, margin, grasp_pc_norm, order, width, voxel_point_num)
            output = np.dstack([occupy_pic1, norm_pic1, occupy_pic2, norm_pic2, occupy_pic3, norm_pic3])
        else:
            raise NotImplementedError
        proj_output[idx] = torch.Tensor(output)
    if pcs_t.is_cuda:
        proj_output, keep = proj_output.cuda(), keep.cuda()
    return proj_output[keep], keep


class GraspSampler(PointCloud):
    def __init__(self, points, center_num, topK, view_num, table_height,
                 depth: float, width: float, gpu: int, visualization=False, regrad=False):
        '''
          points: [N, 3]
        '''
        if gpu != -1:
            torch.cuda.set_device(gpu)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(points)
        PointCloud.__init__(self, cloud, visualization)
        self.table_height = table_height
        self.center_num = center_num
        self.topK = topK
        self.regrad = regrad
        if view_num is None:
            center_camera = np.array([0, 0, 1.658])
        else:
            center_camera = config.CAMERA_POSE[view_num][0:3] if not regrad else config.CAMERA_POSE_REGRAD[view_num][0:3]

        self.depth, self.width = depth, width

        if type(points) == torch.Tensor:
            self.cloud_array = points.float().to(self.device)
        else:
            self.cloud_array = torch.FloatTensor(points).to(self.device)
        self.cloud_array_numpy = np.array(self.cloud_array.cpu())
        self.cloud_array_homo = torch.cat(
            [self.cloud_array.transpose(0, 1), torch.ones(1, self.cloud_array.shape[0], device=self.device)],
            dim=0).float().to(self.device)
        
        self.kd_tree = open3d.geometry.KDTreeFlann(self.cloud)
        self.estimate_normals(center_camera) 
        self.normal_array = torch.tensor(self.normals.T).float().to(self.device)
        
        ## torch.Tensor: self.frame [center_num,3,3]
        self.frame = np.zeros((self.center_num, 3, 3))

        self.global_to_local = torch.eye(4).unsqueeze(0).expand(self.frame.shape[0], 4, 4).to(self.device).contiguous()
        # self.global_to_local[:, 0:3, 0:3] = self.frame.transpose(1, 2)
        # self.global_to_local[:, 0:3, 3:4] = -torch.bmm(self.frame.transpose(1, 2), self.center.unsqueeze(2))
        self.left_normal = torch.tensor([[0, 1, 0]], device=self.device).float()
        self.right_normal = torch.tensor([[0, -1, 0]], device=self.device).float()

        self.valid_grasp = 0
        self.baseline_frame = []
        self.close_pc_num = []

        self._sample_frames()
    
    def _sample_frames(self):
        self.frame_indices = torch.tensor(np.random.choice(len(self.cloud_array), \
                                self.center_num, replace=False), device=self.device).long()
        
        for frame_index in range(len(self.frame_indices)):
            index = self.frame_indices[frame_index]
            self._estimate_frame(index, frame_index)
        self.frame = torch.tensor(self.frame).to(self.device).float()
        self.global_to_local[:, 0:3, 0:3] = self.frame.transpose(1, 2)
        self.global_to_local[:, 0:3, 3:4] = -torch.bmm(self.frame.transpose(1, 2),
                                                       self.cloud_array[self.frame_indices, :].unsqueeze(2))
        
    def _estimate_frame(self, index: int, frame_index: int):
        """
        Estimate the Darboux frame of single point
        In self.frame, each column of one point frame is a vec3, with the order of x, y, z axis
        Note there is a minus sign of the whole frame, which means that x is the negative direction of normal
        :param index: The index of point in all single view cloud
        :param frame_index: The index of point in frame list, which is a subset of all point cloud
        """
        [k, idx, _] = self.kd_tree.search_radius_vector_3d(self.cloud_array_numpy[index, :], config.CURVATURE_RADIUS)
        normal = self.normals[index:index + 1, :]
        if k < 5:
            return None

        M = np.eye(3) - normal.T @ normal
        xyz_centroid = np.mean(M @ self.normals[idx, :].T, axis=1, keepdims=True)
        normal_diff = self.normals[idx, :].T - xyz_centroid
        cov = normal_diff @ normal_diff.T
        eig_value, eig_vec = np.linalg.eigh(cov)

        minor_curvature = eig_vec[:, 0] - eig_vec[:, 0] @ normal.T * np.squeeze(normal)
        minor_curvature /= np.linalg.norm(minor_curvature)

        principal_curvature = np.cross(minor_curvature, np.squeeze(normal))

        self.frame[frame_index, :, :] = np.stack([-self.normals[index, :], -principal_curvature, minor_curvature],
                                                 axis=1)
        if normal[0,0] == 0 and normal[0,1] == 0:        
            theta = 2*np.pi*np.random.random() - np.pi
            matrix = np.array([[np.cos(theta), np.sin(theta), 0],
                               [-np.sin(theta), np.cos(theta), 0],
                               [0, 0, 1]])
            self.frame[frame_index, :, :] = np.dot(matrix, self.frame[frame_index, :, :])

    def generate_grasp_nocoll_view(self):
        for frame_index in range(self.frame.shape[0]):
            self.finger_hand_view(frame_index, self.frame_indices[frame_index])
            if self.valid_grasp >= self.topK:
                break
        #print(self.baseline_frame)
        self.baseline_frame = torch.inverse(torch.cat(self.baseline_frame, dim=0))
        return self.baseline_frame    

    def _table_collision_check(self, point, frame):
        """
        Check whether the gripper collide with the table top with offset
        :param point: torch.tensor(3)
        :param frame: torch.tensor(3, 3)
        :return: a torch boolean tensor with shape (len(config.INDEX_TO_ARRAY))
        """

        T_local_to_global = torch.eye(4, device=self.device).float()
        T_local_to_global[0:3, 0:3] = frame
        T_local_to_global[0:3, 3] = point
        T_local_search_to_global_all = torch.bmm(
                T_local_to_global.squeeze(0).expand(config.LOCAL_SEARCH_TO_LOCAL.shape[0], 4, 4).contiguous(),
                config.LOCAL_SEARCH_TO_LOCAL.to(self.device))
        boundary_global = torch.bmm(T_local_search_to_global_all, config.TORCH_GRIPPER_BOUND.squeeze(0).expand(
                T_local_search_to_global_all.shape[0], -1, -1).contiguous().to(self.device))
        table_collision_bool_all = boundary_global[:, 2, :] < self.table_height - 0.05#+ config.TABLE_COLLISION_OFFSET
        #print( boundary_global[:, 2, :])
        return table_collision_bool_all.any(dim=1, keepdim=False)


    def finger_hand_view(self, frame_index, index):
        """
        :param frame_index: The index of point in frame list, which is a subset of all point cloud
        """
        frame = self.frame[frame_index, :, :]
        point = self.cloud_array[index, :]
        if not self.regrad:
            if point[2] + frame[2, 0] * self.depth < self.table_height - 0.005: # config.FINGER_LENGTH  self.depth 
                # print(point[2] + frame[2, 0] * self.depth)
                return

        table_collision_bool = self._table_collision_check(point, frame)

        T_global_to_local = self.global_to_local[frame_index, :, :]
        local_cloud = torch.matmul(T_global_to_local, self.cloud_array_homo)
        local_cloud_normal = torch.matmul(T_global_to_local[0:3, 0:3], self.normal_array)

        i = 0
        for dl_num, dl in enumerate(config.LENGTH_SEARCH):
            close_plane_bool = (local_cloud[0, :] > dl-config.BOTTOM_LENGTH) & (local_cloud[0, :] < dl+self.depth) # config.FINGER_LENGTH
            if torch.sum(close_plane_bool) < config.NUM_POINTS_THRESHOLD:
                i += config.GRASP_PER_LENGTH
                #print("1..")
                continue
            
            close_plane_points = local_cloud[:, close_plane_bool]  # only filter along x axis
            T_local_to_local_search_all = config.LOCAL_TO_LOCAL_SEARCH.to(self.device)[
                                          dl_num * config.GRASP_PER_LENGTH:(dl_num + 1) * config.GRASP_PER_LENGTH, :, :]
            local_search_close_plane_points_all = torch.matmul(T_local_to_local_search_all.contiguous().view(-1, 4),
                                                               close_plane_points).contiguous().view(
                                                                    config.GRASP_PER_LENGTH, 4, -1)[:, 0:3, :]
        
            for _ in range(config.GRASP_PER_LENGTH):
                if not self.regrad:
                    if table_collision_bool[i]:
                        #print("2..")
                        i += 1
                        continue

                local_search_close_plane_points = local_search_close_plane_points_all[i % config.GRASP_PER_LENGTH, :, :]
                hand_half_bottom_width = self.width / 2 + config.FINGER_WIDTH
                hand_half_bottom_space = self.width / 2 
                z_collision_bool = (local_search_close_plane_points[2, :] < config.HALF_HAND_THICKNESS) & \
                                    (local_search_close_plane_points[2, :] > -config.HALF_HAND_THICKNESS)
                back_collision_bool = (local_search_close_plane_points[1, :] < hand_half_bottom_width) & \
                                        (local_search_close_plane_points[1, :] > -hand_half_bottom_width) & \
                                        (local_search_close_plane_points[0, :] < -config.BACK_COLLISION_MARGIN) & \
                                        z_collision_bool

                if torch.sum(back_collision_bool) > config.BACK_COLLISION_THRESHOLD:
                    i += 1
                    continue

                y_finger_region_bool_left = (local_search_close_plane_points[1, :] < hand_half_bottom_width) & \
                                            (local_search_close_plane_points[1, :] > hand_half_bottom_space)
                y_finger_region_bool_right = (local_search_close_plane_points[1, :] > -hand_half_bottom_width) & \
                                            (local_search_close_plane_points[1, :] < -hand_half_bottom_space)

                y_finger_region_bool = y_finger_region_bool_left | y_finger_region_bool_right
                collision_region_bool = (z_collision_bool & y_finger_region_bool)
                if torch.sum(collision_region_bool) > config.FINGER_COLLISION_THRESHOLD:
                    i += 1
                    continue
                else:
                    close_region_bool = z_collision_bool & \
                                        (local_search_close_plane_points[1, :] < hand_half_bottom_space) & \
                                        (local_search_close_plane_points[1, :] > -hand_half_bottom_space)

                    if torch.sum(close_region_bool.int()) < config.CLOSE_REGION_MIN_POINTS:
                        i += 1
                        continue

                    best_frame = torch.matmul(
                        T_local_to_local_search_all[i % config.GRASP_PER_LENGTH],
                        self.global_to_local[frame_index])
                    
                    matrix_norm = torch.norm(best_frame[:3,:3])
                    if matrix_norm > 0:
                        self.baseline_frame.append(best_frame.view(1,4,4))
                        self.close_pc_num.append(torch.sum(close_region_bool.int()))
                        self.valid_grasp += 1
                        i+=1
                        return 
                    if self.valid_grasp >= self.topK:
                        return 
            

class ModelInit:
    def __init__(self, mode, model_path, params, gripper_num, input_chann, class_num):
        self.mode = mode
        self.model_path = model_path
        self.gripper_num = gripper_num
        self.input_chann = input_chann
        self.class_num   = class_num
        self.gpu_num, self.gpu_id, self.gpu_ids, self.lr = params
        self.checkpoint = None if model_path=='' else torch.load(model_path, map_location='cuda:{}'.format(self.gpu_id))

    def _map_model(self, model):
        device = torch.device("cuda:"+str(self.gpu_id))
        model = model.to(device)
        
        if self.gpu_num > 1:
            device_id = [int(i) for i in self.gpu_ids.split(',')]
            model = nn.DataParallel(model, device_ids=device_id)
        print("Construct network successfully!")
        return model

    def construct_net(self):
        # self.mode = ['train', validate', 'test']
        #-------------- load network----------------
        model = PointNetCls(self.gripper_num, self.input_chann, self.class_num)
        resume_num = 0
        if self.checkpoint:
            new_model_dict = {}
            model_dict, resume_num = self.checkpoint['net'], self.checkpoint['epoch'] + 1
            if 'test' in self.mode:
                resume_num -= 1

            for key in model_dict.keys():
                new_model_dict[key.replace("module.", "")] = model_dict[key]
            model.load_state_dict(new_model_dict)

        model = self._map_model(model)
        return model, resume_num

    def construct_scheduler(self, model, resume_num): 
        optimizer = optim.Adam([{'params':model.parameters(), 'initial_lr':self.lr}], lr=self.lr)
        # if self.checkpoint:
        #     optimizer.load_state_dict(self.checkpoint['optimizer'])
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5, last_epoch=resume_num-1)
        print("optimizer", optimizer)
        return optimizer, scheduler

    def construct_model(self):
        model, resume_num    = self.construct_net()
        optimizer, scheduler = self.construct_scheduler(model, resume_num)
        return model, resume_num, optimizer, scheduler


class GPDModelInit:
    def __init__(self, mode, model_path, params, gripper_num, input_chann):
        self.mode = mode
        self.model_path = model_path
        self.gripper_num = gripper_num
        self.input_chann = input_chann
        self.gpu_num, self.gpu_id, self.gpu_ids, self.lr = params
        self.checkpoint = None if model_path=='' else torch.load(model_path, map_location='cuda:{}'.format(self.gpu_id))

    def _map_model(self, model):
        device = torch.device("cuda:"+str(self.gpu_id))
        model = model.to(device)
        
        if self.gpu_num > 1:
            device_id = [int(i) for i in self.gpu_ids.split(',')]
            model = nn.DataParallel(model, device_ids=device_id)
        print("Construct network successfully!")
        return model

    def construct_net(self):
        # self.mode = ['train', validate', 'test']
        #-------------- load network----------------
        model = GPDClassifier(self.input_chann)
        resume_num = 0
        if self.checkpoint:
            new_model_dict = {}
            model_dict, resume_num = self.checkpoint['net'], self.checkpoint['epoch'] + 1
            if 'test' in self.mode:
                resume_num -= 1

            for key in model_dict.keys():
                new_model_dict[key.replace("module.", "")] = model_dict[key]
            model.load_state_dict(new_model_dict)

        model = self._map_model(model)
        return model, resume_num

    def construct_scheduler(self, model, resume_num): 
        optimizer = optim.Adam([{'params':model.parameters(), 'initial_lr':self.lr}], lr=self.lr)
        # if self.checkpoint:
        #     optimizer.load_state_dict(self.checkpoint['optimizer'])
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5, last_epoch=resume_num-1)
        print("optimizer", optimizer)
        return optimizer, scheduler

    def construct_model(self):
        model, resume_num    = self.construct_net()
        optimizer, scheduler = self.construct_scheduler(model, resume_num)
        return model, resume_num, optimizer, scheduler


class Logger:
    def __init__(self, logger):
        self.logger = logger

    def add_batch_tuple(self, index_tuple, index, mode,):
        self.logger.add_scalar('batch_'+mode+'_acc',       (index_tuple[0].mean()), index)
        self.logger.add_scalar('batch_'+mode+'_precision', (index_tuple[1].mean()), index)
        self.logger.add_scalar('batch_'+mode+'_recall',    (index_tuple[2].mean()), index)

    def add_batch_loss(self, data, index, mode="train"):
        if len(data) == 2:
            loss, index_tuple = data
            self.logger.add_scalar('batch_'+mode+'_loss_total', loss, index)  
            self.add_batch_tuple(index_tuple, index, mode)

        elif len(data) == 3:
            loss, index_tuple, loss_tuple = data
            self.logger.add_scalar('batch_'+mode+'_loss_total', loss, index)  
            self.logger.add_scalar('batch_'+mode+'_loss_cls',   loss_tuple[0], index)  
            self.logger.add_scalar('batch_'+mode+'_loss_trans', loss_tuple[1], index)  
            self.add_batch_tuple(index_tuple, index, mode)

    def add_batch_eval(self, batch_vgr, batch_score, batch_vgr_before, batch_coverage, \
                batch_coverage_all, batch_vgr_before_all, batch_vgr_all, batch_score_all, mode, index):
        if batch_vgr !=0 and batch_score !=0 and batch_vgr_before !=0 :
            self.logger.add_scalar('batch_'+mode+'_vgr',           batch_vgr,          index)
            self.logger.add_scalar('batch_'+mode+'_score',         batch_score,        index)
            self.logger.add_scalar('batch_'+mode+'_vgr_before',    batch_vgr_before,   index)
            self.logger.add_scalar('batch_'+mode+'_converage',     batch_coverage,     index)
            self.logger.add_scalar('batch_'+mode+'_vgr_all',       batch_vgr_all,      index)
            self.logger.add_scalar('batch_'+mode+'_vgr_before_all', batch_vgr_before_all, index)
            self.logger.add_scalar('batch_'+mode+'_score_all',     batch_score_all,    index)
            self.logger.add_scalar('batch_'+mode+'_converage_all', batch_coverage_all, index)

    def add_epoch_eval(self, vgr, score, t_score_coll, vgr_before, coverage, coverage_all, \
                    vgr_all, score_all, vgr_before_all, mode, epoch, stage, score_thre, str_width=None):
        if str_width is None:
            self.logger.add_scalar('epoch_'+mode+'_'+stage+'_vgr_before'+score_thre,   vgr_before,   epoch) 
            self.logger.add_scalar('epoch_'+mode+'_'+stage+'_vgr'+score_thre,          vgr,          epoch) 
            self.logger.add_scalar('epoch_'+mode+'_'+stage+'_score'+score_thre,        score,        epoch) 
            self.logger.add_scalar('epoch_'+mode+'_'+stage+'_coverage'+score_thre,     coverage,     epoch) 
            self.logger.add_scalar('epoch_'+mode+'_'+stage+'_coverage_all'+score_thre, coverage_all, epoch) 
            self.logger.add_scalar('epoch_'+mode+'_'+stage+'_vgr_all_'+score_thre,          vgr_all,      epoch) 
            self.logger.add_scalar('epoch_'+mode+'_'+stage+'_score_all_'+score_thre,        score_all,    epoch) 
        else: # mode in 'test' or 'pretest'
            self.logger.add_scalar(str_width + 'epoch_'+mode+'_'+stage+'_vgr_before'+score_thre,   vgr_before,   epoch) 
            self.logger.add_scalar(str_width + 'epoch_'+mode+'_'+stage+'_vgr'+score_thre,          vgr,          epoch) 
            self.logger.add_scalar(str_width + 'epoch_'+mode+'_'+stage+'_score'+score_thre,        score,        epoch) 
            self.logger.add_scalar(str_width + 'epoch_'+mode+'_'+stage+'_coverage'+score_thre,     coverage,     epoch) 
            self.logger.add_scalar(str_width + 'epoch_'+mode+'_'+stage+'_coverage_all'+score_thre, coverage_all, epoch) 
            self.logger.add_scalar(str_width + 'epoch_'+mode+'_'+stage+'_vgr_all_'+score_thre,     vgr_all,          epoch) 
            self.logger.add_scalar(str_width + 'epoch_'+mode+'_'+stage+'_score_all_'+score_thre,   score_all,        epoch) 

        logging.info("vgr\t\t"+str(vgr))
        logging.info("score_wo_collision\t\t"+str(score))
        logging.info("score\t\t"+str(t_score_coll))
        logging.info("coverage\t"+str(coverage))
        logging.info("coverage_all\t"+str(coverage_all))
        logging.info("vgr_scene\t"+str(vgr_all))
        logging.info("score_scene\t"+str(score_all))

class Eval:
    def __init__(self, log_machine, center_num=1000, topK=100, dis_thre=0.01, epoch=-1, eval_time=1, regrad=False):
        self.K = topK
        self.dis_thre = dis_thre
        self.epoch = epoch
        self.batchs = 0
        self.log_machine = log_machine
        self.center_num = center_num
        self.total_score_collision=0
        self.eval_time = eval_time
        self.regrad=regrad
        
        self.total_vgr             = [0] * eval_time
        self.total_score           = [0] * eval_time
        self.total_score_collision = [0] * eval_time
        self.total_grasp_formal    = [0] * eval_time
        self.total_grasp_woco_view = [0] * eval_time

        self.batch_vgr             = [0] * eval_time
        self.batch_score           = [0] * eval_time
        self.batch_score_collision = [0] * eval_time
        self.batch_coverage        = [0] * eval_time
        self.batch_coverage_all    = [0] * eval_time

        self.cur_batch_vgr             = [0] * eval_time
        self.cur_batch_score           = [0] * eval_time
        self.cur_batch_score_collision = [0] * eval_time
        self.cur_batch_coverage        = [0] * eval_time
        self.cur_batch_coverage_all    = [0] * eval_time

    def _eval_coverage(self, ground, predicted_grasp):
        if 'select_frame' in ground.keys():
            ground_grasp = torch.FloatTensor(ground['select_frame']) 
        else:
            ground_grasp = torch.FloatTensor(ground['frame']) 
        predicted_grasp  = predicted_grasp if type(predicted_grasp) == torch.Tensor else torch.FloatTensor(predicted_grasp)
        ground_center    = ground_grasp[:,:3, 3]
        predic_center    = predicted_grasp[:,:3]
        if predicted_grasp.is_cuda:
            ground_center = ground_center.cuda()
            
        coverage_rate = 0
        if len(predicted_grasp) > 0:
            #### distance: [len(ground), len(predicted)]
            distance = compute_distance(ground_center, predic_center)
            distance_mask = torch.max((distance < self.dis_thre), dim=1)[0]
            coverage_rate = float(torch.sum(distance_mask)) / float(len(distance_mask))
        return coverage_rate

    def _get_data_view_num(self, data_path):
        if 'noise' not in data_path.split('_')[-1]:
            view_num = int(data_path.split('_')[-1].split('.')[0])  # eg. .../4080_view_1.p
        else:
            view_num = int(data_path.split('_')[-2])                # eg. .../4080_view_1_noise.p
        return view_num

    def _get_data_width(self, data_path, set_width):
        width = set_width
        if '0.' in data_path.split('/')[-3]:
            if set_width not in [0.06, 0.08, 0.10, 0.12]:
                width = float(data_path.split('/')[-3])  # eg. .../4080_view_1.p
        return width
    
    def eval_one_grasp(self, data_path, model, params, train_params, cur_eval_time):
        depths, width, table_height, gpu = params
        cur_data = np.load(data_path, allow_pickle=True)
        float_str_map = {0.06:'0.06', 0.08:'0.08', 0.10:'0.10', 0.12:'0.12'}
        if width in [0.06, 0.08, 0.10, 0.12]:
            if '0.' in data_path.split('/')[-3]:
                re_width = data_path.split('/')[-3]
                data_path = data_path.replace(re_width, float_str_map[width])
        # if width is None:
        #     width = self._get_data_width(data_path, width)
        view_num = self._get_data_view_num(data_path)
        if self.regrad:
            view_num-=1
        
        view = cur_data['view_cloud']
        grasp = self.sample_grasps(view, view_num, model, params, train_params)
        grasp = grasp[grasp[:,-2] > 0]

        vgr, score, score_coll, grasp_nocoll_view_num, grasp_nocoll_view, grasp_nocoll_scene = \
                eval_validate_wo_view(cur_data, grasp[:,:8], view_num, table_height, depths, width, gpu, self.regrad)
        
        print("predicted grasp length: {} -> top 100: no coll with view: {}, no coll with scene: {}".\
                                                            format(len(grasp), grasp_nocoll_view_num, vgr))
        formal_num     = len(grasp)
        # formal_num_all = len(cur_grasp_all)
        grasp_nocoll_view_num = max(1, grasp_nocoll_view_num) # if grasp_nocoll_view_num == 0 
        formal_num            = max(1, formal_num)            # if formal_num == 0 
        # grasp_nocoll_view_num_all = max(1, grasp_nocoll_view_num_all) # if grasp_nocoll_view_num == 0 
        # formal_num_all            = max(1, formal_num_all)            # if formal_num == 0 

        if grasp_nocoll_view_num <= 0:
            vgr_all, score_all, score_coll_all = 0, 0, 0
        else:
            vgr_all, score_all, score_coll_all = vgr/grasp_nocoll_view_num, \
                        score/grasp_nocoll_view_num, score_coll/grasp_nocoll_view_num

        coverage_rate_all = 0
        coverage_rate = self._eval_coverage(cur_data, grasp_nocoll_view)  

        self.total_vgr[cur_eval_time]             += vgr
        self.total_score[cur_eval_time]           += score
        self.total_score_collision[cur_eval_time] += score_coll
        self.total_grasp_formal[cur_eval_time]    += formal_num
        self.total_grasp_woco_view[cur_eval_time] += grasp_nocoll_view_num
        
        self.batch_vgr[cur_eval_time]             += vgr_all
        self.batch_score[cur_eval_time]           += score_all
        self.batch_score_collision[cur_eval_time] += score_coll_all
        self.batch_coverage[cur_eval_time]        += coverage_rate
        self.batch_coverage_all[cur_eval_time]    += coverage_rate_all

        self.cur_batch_vgr[cur_eval_time]             += vgr_all
        self.cur_batch_score[cur_eval_time]           += score_all
        self.cur_batch_score_collision[cur_eval_time] += score_coll_all
        self.cur_batch_coverage[cur_eval_time]        += coverage_rate
        self.cur_batch_coverage_all[cur_eval_time]    += coverage_rate_all

    def update_epoch(self):
        self.epoch += 1

    def sample_grasps(self, pc, view_num, model, params, train_params):
        depths, width, table_height, gpu = params
        GSampler = GraspSampler(pc, self.center_num, self.K, view_num, table_height, depths, width, gpu, regrad=self.regrad)
        grasps = GSampler.generate_grasp_nocoll_view()
        device   = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')

        if model is None:
            score = torch.ones((len(grasps))).to(device)
            grasps = self._transform_grasp(grasps, score, gpu)
        else:
            model.eval()
            pc       = torch.tensor(pc).to(device).view(1, -1, 3)
            if len(train_params) == 4:
                #if self.regrad:
                #    pc[:,:,2] += 0.25
                #    grasps[:,2,3] += 0.25
                # close_pc: [len(grasp), close_region_points_num, 3]
                close_pc, keep_idx, keep_num = get_close_region_points(pc, grasps.view(1,-1,4,4), train_params)
                with torch.no_grad():
                    pred, _ = model(close_pc.permute(0,2,1)[:,:3])
                    pred_cls = pred.data.max(1, keepdim=True)[1].view(-1)
                    #print(pred)
                    score = torch.zeros((len(keep_idx))).to(device)
                    score[keep_idx] = pred_cls.float()
                    grasps = self._transform_grasp(grasps, score, gpu)
            else:
                #if self.regrad:
                #    pc[:,:,2] += 0.25
                #    grasps[:,2,3] += 0.25
                proj_pic, keep_idx = get_gpd_projected_points(pc, grasps.view(1,-1,4,4), train_params)
                with torch.no_grad():
                    pred = model(proj_pic.permute(0,3,1,2))
                    pred_cls = pred.data.max(1, keepdim=True)[1].view(-1)
                    score = torch.zeros((len(keep_idx))).to(device)
                    score[keep_idx] = pred_cls.float()
                    grasps = self._transform_grasp(grasps, score, gpu)
        return grasps

    def _transform_grasp(self, grasp_ori, score_ori, gpu_id):
        '''
        Input:
            grasp_ori: [B, 4, 4] 
                    [[x1, y1, z1, c1],
                        [x2, y2, z2, c2],
                        [x3, y3, z3, c3]]
            score_ori: [B]
        Output:
            grasp_trans:[B, 9] (center[3], axis_y[3], grasp_angle[1], antipodal_score[1], center_score[1])
        '''
        B, _, _ = grasp_ori.shape
        device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
        grasp_trans = torch.full((B, 9), -1.0).to(device)

        axis_x = grasp_ori[:,:3,0].view(B, 3)
        axis_y = grasp_ori[:,:3,1].view(B, 3)
        axis_z = grasp_ori[:,:3,2].view(B, 3)
        grasp_angle = torch.atan2(axis_x[:,2], axis_z[:,2])  ## torch.atan(torch.div(axis_x[:,2], axis_z[:,2])) is not OK!!!

        grasp_angle[axis_y[:,0] < 0]         = np.pi-grasp_angle[axis_y[:,0] < 0]
        axis_y[axis_y[:,0] < 0]              = -axis_y[axis_y[:,0] < 0]
        grasp_angle[grasp_angle >= 2*np.pi]  = grasp_angle[grasp_angle >= 2*np.pi] - 2*np.pi
        grasp_angle[grasp_angle <= -2*np.pi] = grasp_angle[grasp_angle <= -2*np.pi] + 2*np.pi
        grasp_angle[grasp_angle > np.pi]     = grasp_angle[grasp_angle > np.pi] - 2*np.pi
        grasp_angle[grasp_angle <= -np.pi]   = grasp_angle[grasp_angle <= -np.pi] + 2*np.pi

        grasp_trans[:,:3]  = grasp_ori[:,:3,3].view(B, 3)
        grasp_trans[:,3:6] = axis_y.view(B, 3)
        grasp_trans[:,6]   = grasp_angle.view(B)
        grasp_trans[:,7]   = score_ori
        grasp_trans[:,8]   = 0.
        return grasp_trans

    def eval_grasps_with_gt(self, data_paths, model, params, train_params, cur_eval_time):
        self.batchs += len(data_paths)
        for i in range(len(data_paths)):
            data_path = data_paths[i]
            self.eval_one_grasp(data_path, model, params, train_params, cur_eval_time)
        
        print("#batch vgr \t\t",        self.cur_batch_vgr[cur_eval_time]/len(data_paths))
        print("#batch score \t\t",      self.cur_batch_score[cur_eval_time]/len(data_paths))
        print("#batch score coll \t\t", self.cur_batch_score_collision[cur_eval_time]/len(data_paths))
        print("#batch coverage\t\t",    self.cur_batch_coverage[cur_eval_time]/len(data_paths))
        print("#batch coverage all\t\t",self.cur_batch_coverage_all[cur_eval_time]/len(data_paths))

        self.cur_batch_vgr[cur_eval_time]             = 0
        self.cur_batch_score[cur_eval_time]           = 0
        self.cur_batch_score_collision[cur_eval_time] = 0
        self.cur_batch_coverage[cur_eval_time]        = 0
        self.cur_batch_coverage_all[cur_eval_time]    = 0

    def new_epoch(self):
        self.total_vgr             = [0] * self.eval_time
        self.total_score           = [0] * self.eval_time
        self.total_score_collision = [0] * self.eval_time
        self.total_grasp_formal    = [0] * self.eval_time
        self.total_grasp_woco_view = [0] * self.eval_time

        self.batch_vgr             = [0] * self.eval_time
        self.batch_score           = [0] * self.eval_time
        self.batch_score_collision = [0] * self.eval_time
        self.batch_coverage        = [0] * self.eval_time
        self.batch_coverage_all    = [0] * self.eval_time

    def eval_batch(self, data_path, model, params, train_params, index, mode, cur_eval_time=0):
        # eval parameters of width is generated from its data path
        print("=======================evaluate grasps=======================")
        self.eval_grasps_with_gt(data_path, model, params, train_params, cur_eval_time)

    def eval_epoch(self, batch_nums, mode, str_width=None):
        i = 0
        tvgr        = self.total_vgr[i] / self.total_grasp_woco_view[i]
        tscore      = self.total_score[i] / self.total_grasp_woco_view[i]
        tscore_coll = self.total_score_collision[i] / self.total_grasp_woco_view[i]

        tvgr_scene               = self.batch_vgr[i]     / self.batchs
        tscore_scene             = self.batch_score[i] / self.batchs
        tscore_coll_scene        = self.batch_score_collision[i]     / self.batchs
        tcoverage                = self.batch_coverage[i]     / self.batchs
        tcoverage_all            = self.batch_coverage_all[i] / self.batchs
            
        log_str = "".join( ("vgr: {}\t\t score: {}\t\t score_collision: {}\n".format(tvgr, tscore, tscore_coll), 
                            "SCENE vgr: {}\t\t score: {}\t\t score_collision: {}\n".format(tvgr_scene, tscore_scene, tscore_coll_scene),
                            "coverage: {}\t\t coverage_all: {}\n".format(tcoverage, tcoverage_all) ))
        
        print(log_str)
        if str_width is not None:
            str_width = str(float(str_width))
        logging.info("{} Epoch {}, Eval Width {}".format(mode, self.epoch, str_width))
        logging.info(log_str)

        self.batchs = 0
        self.new_epoch()


class EvalNoTruth:
    def __init__(self, center_num=1000, topK=100):
        self.K = topK
        self.batchs = 0
        self.center_num = center_num

    def sample_grasps(self, pc, view_num, model, params, train_params, trans_flag=True):
        depths, width, table_height, gpu = params
        with torch.no_grad():
            start = time.time()
            GSampler = GraspSampler(pc, self.center_num, self.K, view_num, table_height, depths, width, gpu)
            grasps = GSampler.generate_grasp_nocoll_view()
            torch.cuda.synchronize()
            processing_time = time.time() - start
            device   = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')

            if model is None:
                score = torch.ones((len(grasps))).to(device)
                forward_passing_time = 0
                if trans_flag:
                    grasps = self._transform_grasp(grasps, score, gpu)
            else:
                model.eval()
                pc       = torch.tensor(pc).to(device).view(1, -1, 3)
                if len(train_params) == 4:
                    # close_pc: [len(grasp), close_region_points_num, 3]
                    start = time.time()
                    close_pc, keep_idx, keep_num = get_close_region_points(pc, grasps.view(1,-1,4,4), train_params)
                    torch.cuda.synchronize()
                    processing_time += time.time() - start
                    
                    start = time.time()
                    pred, _ = model(close_pc.permute(0,2,1)[:,:3])
                    torch.cuda.synchronize()
                    forward_passing_time = time.time() - start
                else:
                    start = time.time()
                    proj_pic, keep_idx = get_gpd_projected_points(pc, grasps.view(1,-1,4,4), train_params)
                    torch.cuda.synchronize()
                    processing_time += time.time() - start
                    
                    start = time.time()
                    pred = model(proj_pic.permute(0,3,1,2))
                    torch.cuda.synchronize()
                    forward_passing_time = time.time() - start

                pred_cls = pred.data.max(1, keepdim=True)[1].view(-1)
                score = torch.zeros((len(keep_idx))).to(device)
                score[keep_idx] = pred_cls.float()
                if trans_flag:
                    grasps = self._transform_grasp(grasps, score, gpu)
                else:
                    grasps = grasps[score > 0]
                    
        return grasps, forward_passing_time, processing_time

    def _transform_grasp(self, grasp_ori, score_ori, gpu_id):
        '''
        Input:
            grasp_ori: [B, 4, 4] 
                    [[x1, y1, z1, c1],
                        [x2, y2, z2, c2],
                        [x3, y3, z3, c3]]
            score_ori: [B]
        Output:
            grasp_trans:[B, 9] (center[3], axis_y[3], grasp_angle[1], antipodal_score[1], center_score[1])
        '''
        B, _, _ = grasp_ori.shape
        device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
        grasp_trans = torch.full((B, 9), -1.0).to(device)

        axis_x = grasp_ori[:,:3,0].view(B, 3)
        axis_y = grasp_ori[:,:3,1].view(B, 3)
        axis_z = grasp_ori[:,:3,2].view(B, 3)
        grasp_angle = torch.atan2(axis_x[:,2], axis_z[:,2])  ## torch.atan(torch.div(axis_x[:,2], axis_z[:,2])) is not OK!!!

        grasp_angle[axis_y[:,0] < 0]         = np.pi-grasp_angle[axis_y[:,0] < 0]
        axis_y[axis_y[:,0] < 0]              = -axis_y[axis_y[:,0] < 0]
        grasp_angle[grasp_angle >= 2*np.pi]  = grasp_angle[grasp_angle >= 2*np.pi] - 2*np.pi
        grasp_angle[grasp_angle <= -2*np.pi] = grasp_angle[grasp_angle <= -2*np.pi] + 2*np.pi
        grasp_angle[grasp_angle > np.pi]     = grasp_angle[grasp_angle > np.pi] - 2*np.pi
        grasp_angle[grasp_angle <= -np.pi]   = grasp_angle[grasp_angle <= -np.pi] + 2*np.pi

        grasp_trans[:,:3]  = grasp_ori[:,:3,3].view(B, 3)
        grasp_trans[:,3:6] = axis_y.view(B, 3)
        grasp_trans[:,6]   = grasp_angle.view(B)
        grasp_trans[:,7]   = score_ori
        grasp_trans[:,8]   = 0.
        return grasp_trans

    def eval_notruth(self, model, pc, params, train_params):
        print(params)
        view_num = None
        Tptime, Tfptime = 0, 0
        for i in range(len(pc)):
            cur_pc = pc[i,:,:3].cpu().numpy()
            grasp, forward_passing_time, processing_time = self.sample_grasps(cur_pc, view_num, \
                                                                model, params, train_params)
            print("Keep {} grasps".format(len(grasp)))
            Tptime += processing_time
            Tfptime += forward_passing_time
        # print("forward_passing_time: {}, processing_time: {}".format(Tfptime/len(pc), Tptime/len(pc)  ))
        return Tfptime/len(pc), Tptime/len(pc)        

    def eval_notruth_returngrasp(self, model, pc, params, train_params):
        view_num = None
        for i in range(len(pc)):
            cur_pc = pc[i,:,:3].cpu().numpy()
            grasp, _, _ = self.sample_grasps(cur_pc, view_num, model, params, train_params, trans_flag=False)
            print("Keep {} grasps".format(len(grasp)))
        return grasp

class Eval_S4G:
    def __init__(self, topK=100, dis_thre=0.01, epoch=-1, eval_time=1, regrad=False):
        self.K = topK
        self.dis_thre = dis_thre
        self.epoch = epoch
        self.batchs = 0
        self.eval_time = eval_time
        self.regrad=regrad

        self.total_vgr             = [0] * eval_time
        self.total_score           = [0] * eval_time
        self.total_score_collision = [0] * eval_time
        self.total_grasp_formal    = [0] * eval_time
        self.total_grasp_woco_view = [0] * eval_time

        self.batch_vgr             = [0] * eval_time
        self.batch_score           = [0] * eval_time
        self.batch_score_collision = [0] * eval_time
        self.batch_coverage        = [0] * eval_time
        self.batch_coverage_all    = [0] * eval_time

        self.cur_batch_vgr             = [0] * eval_time
        self.cur_batch_score           = [0] * eval_time
        self.cur_batch_score_collision = [0] * eval_time
        self.cur_batch_coverage        = [0] * eval_time
        self.cur_batch_coverage_all    = [0] * eval_time

    def _eval_coverage(self, ground, predicted_grasp):
        if 'select_frame' in ground.keys():
            ground_grasp = torch.FloatTensor(ground['select_frame']) 
        else:
            ground_grasp = torch.FloatTensor(ground['frame']) 
        predicted_grasp  = predicted_grasp if type(predicted_grasp) == torch.Tensor else torch.FloatTensor(predicted_grasp)
        ground_center    = ground_grasp[:,:3, 3]
        predic_center    = predicted_grasp[:,:3]
        if predicted_grasp.is_cuda:
            ground_center = ground_center.cuda()
            
        coverage_rate = 0
        if len(predicted_grasp) > 0:
            #### distance: [len(ground), len(predicted)]
            distance = compute_distance(ground_center, predic_center)
            distance_mask = torch.max((distance < self.dis_thre), dim=1)[0]
            coverage_rate = float(torch.sum(distance_mask)) / float(len(distance_mask))
        return coverage_rate

    def _get_data_view_num(self, data_path):
        if 'noise' not in data_path.split('_')[-1]:
            view_num = int(data_path.split('_')[-1].split('.')[0])  # eg. .../4080_view_1.p
        else:
            view_num = int(data_path.split('_')[-2])                # eg. .../4080_view_1_noise.p
        return view_num

    def _get_data_width(self, data_path, set_width):
        width = set_width
        if '0.' in data_path.split('/')[-3]:
            if set_width not in [0.06, 0.08, 0.10, 0.12]:
                width = float(data_path.split('/')[-3])  # eg. .../4080_view_1.p
        return width
    
    def select_topK(self, grasp, cur_k=None):
        _, sort_idx = torch.sort(grasp[:,7], descending=True)
        if cur_k is not None:
            cur_grasp = grasp[sort_idx[:cur_k]]
        else:
            cur_grasp = grasp[sort_idx[:self.K]]
        return cur_grasp

    def select_scorethre(self, grasp, score_thre=0.5):
        cur_grasp = grasp[grasp[:,7]>=score_thre]
        return cur_grasp

    def select_scorethre_topK(self, grasp, score_thre=0.5):
        cur_grasp = grasp[grasp[:,7]>=score_thre]
        if len(cur_grasp) <= self.K:
            return cur_grasp
        
        point_xy = cur_grasp[:,:2]
        zeros = torch.zeros(len(cur_grasp), 1)
        if point_xy.is_cuda:
            zeros = zeros.cuda()
        point_xy_zero = torch.cat((point_xy, zeros), dim=1)
        center_pc_index = _F.farthest_point_sample(point_xy_zero.view(1,-1,3).transpose(2,1), self.K).view(-1)
        return cur_grasp[center_pc_index]

    def eval_one_grasp(self, data_path, grasp, params, cur_eval_time, score_thre=None):
        depths, width, table_height, gpu = params
        cur_data = np.load(data_path, allow_pickle=True)
        float_str_map = {0.06:'0.06', 0.08:'0.08', 0.10:'0.10', 0.12:'0.12'}
        if width in [0.06, 0.08, 0.10, 0.12]:
            if '0.' in data_path.split('/')[-3]:
                re_width = data_path.split('/')[-3]
                data_path = data_path.replace(re_width, float_str_map[width])
        # if width is None:
        #     width = self._get_data_width(data_path, width)
        view_num = self._get_data_view_num(data_path)
        if self.regrad:
            view_num-=1

        if len(grasp) > 1000:
            grasp = self.select_topK(grasp, 1000)
        keep_idx, keep_num = eval_test_batch(cur_data['view_cloud'].reshape(1,-1,3), grasp[:,:8].view(1,-1,8), table_height, depths, width, gpu)
        # cur_grasp_all = self.select_scorethre(grasp, 0.1)
        cur_grasp_all = grasp[keep_idx[0]]
        # cur_grasp = self.select_topK(grasp)
        cur_grasp = self.select_topK(cur_grasp_all)
        
        vgr, score, score_coll, grasp_nocoll_view_num, grasp_nocoll_view, grasp_nocoll_scene = \
                eval_validate_wo_view(cur_data, cur_grasp[:,:8], view_num, table_height, depths, width, gpu, self.regrad)
        
        print("predicted grasp length: {} -> top 100: no coll with view: {}, no coll with scene: {}".\
                                                            format(len(grasp), grasp_nocoll_view_num, vgr))
        formal_num     = len(cur_grasp)
        # formal_num_all = len(cur_grasp_all)
        grasp_nocoll_view_num = max(1, grasp_nocoll_view_num) # if grasp_nocoll_view_num == 0 
        formal_num            = max(1, formal_num)            # if formal_num == 0 
        # grasp_nocoll_view_num_all = max(1, grasp_nocoll_view_num_all) # if grasp_nocoll_view_num == 0 
        # formal_num_all            = max(1, formal_num_all)            # if formal_num == 0 

        if grasp_nocoll_view_num <= 0:
            vgr_all, score_all, score_coll_all = 0, 0, 0
        else:
            vgr_all, score_all, score_coll_all = vgr/grasp_nocoll_view_num, \
                        score/grasp_nocoll_view_num, score_coll/grasp_nocoll_view_num

        coverage_rate_all = self._eval_coverage(cur_data, cur_grasp_all)  
        coverage_rate = self._eval_coverage(cur_data, grasp_nocoll_view)  

        self.total_vgr[cur_eval_time]             += vgr
        self.total_score[cur_eval_time]           += score
        self.total_score_collision[cur_eval_time] += score_coll
        self.total_grasp_formal[cur_eval_time]    += formal_num
        self.total_grasp_woco_view[cur_eval_time] += grasp_nocoll_view_num
        
        self.batch_vgr[cur_eval_time]             += vgr_all
        self.batch_score[cur_eval_time]           += score_all
        self.batch_score_collision[cur_eval_time] += score_coll_all
        self.batch_coverage[cur_eval_time]        += coverage_rate
        self.batch_coverage_all[cur_eval_time]    += coverage_rate_all

        self.cur_batch_vgr[cur_eval_time]             += vgr_all
        self.cur_batch_score[cur_eval_time]           += score_all
        self.cur_batch_score_collision[cur_eval_time] += score_coll_all
        self.cur_batch_coverage[cur_eval_time]        += coverage_rate
        self.cur_batch_coverage_all[cur_eval_time]    += coverage_rate_all


    def update_epoch(self):
        self.epoch += 1

    def eval_grasps_with_gt(self, data_paths, grasp, params, cur_eval_time):
        self.batchs += len(data_paths)
        for i in range(len(data_paths)):
            batch_grasp = grasp[i]
            score_mask = torch.nonzero(batch_grasp[:,7] >= 0).view(-1)
            if len(score_mask) <= 0:
                print("**there has no predicted grasp:", len(score_mask))
                continue
            batch_grasp = batch_grasp[score_mask]

            data_path = data_paths[i]
            self.eval_one_grasp(data_path, batch_grasp, params, cur_eval_time)
            
        print("#batch vgr \t\t",        self.cur_batch_vgr[cur_eval_time]/len(data_paths))
        print("#batch score \t\t",      self.cur_batch_score[cur_eval_time]/len(data_paths))
        print("#batch score coll \t\t", self.cur_batch_score_collision[cur_eval_time]/len(data_paths))
        print("#batch coverage\t\t",    self.cur_batch_coverage[cur_eval_time]/len(data_paths))
        print("#batch coverage all\t\t",self.cur_batch_coverage_all[cur_eval_time]/len(data_paths))

        self.cur_batch_vgr[cur_eval_time]             = 0
        self.cur_batch_score[cur_eval_time]           = 0
        self.cur_batch_score_collision[cur_eval_time] = 0
        self.cur_batch_coverage[cur_eval_time]        = 0
        self.cur_batch_coverage_all[cur_eval_time]    = 0

    def new_epoch(self):
        self.total_vgr             = [0] * self.eval_time
        self.total_score           = [0] * self.eval_time
        self.total_score_collision = [0] * self.eval_time
        self.total_grasp_formal    = [0] * self.eval_time
        self.total_grasp_woco_view = [0] * self.eval_time

        self.batch_vgr             = [0] * self.eval_time
        self.batch_score           = [0] * self.eval_time
        self.batch_score_collision = [0] * self.eval_time
        self.batch_coverage        = [0] * self.eval_time
        self.batch_coverage_all    = [0] * self.eval_time

    def eval_batch(self, data_path, grasp, params, cur_eval_time=0):
        # eval parameters of width is generated from its data path
        print("=======================evaluate grasps=======================")
        self.eval_grasps_with_gt(data_path, grasp, params, cur_eval_time)


    def eval_epoch(self, batch_nums, mode, str_width=None):
        i = 0
        tvgr        = self.total_vgr[i] / self.total_grasp_woco_view[i]
        tscore      = self.total_score[i] / self.total_grasp_woco_view[i]
        tscore_coll = self.total_score_collision[i] / self.total_grasp_woco_view[i]

        tvgr_scene               = self.batch_vgr[i]     / self.batchs
        tscore_scene             = self.batch_score[i] / self.batchs
        tscore_coll_scene        = self.batch_score_collision[i]     / self.batchs
        tcoverage                = self.batch_coverage[i]     / self.batchs
        tcoverage_all            = self.batch_coverage_all[i] / self.batchs
            
        log_str = "".join( ("vgr: {}\t\t score: {}\t\t score_collision: {}\n".format(tvgr, tscore, tscore_coll), 
                            "SCENE vgr: {}\t\t score: {}\t\t score_collision: {}\n".format(tvgr_scene, tscore_scene, tscore_coll_scene),
                            "coverage: {}\t\t coverage_all: {}\n".format(tcoverage, tcoverage_all) ))
        
        print(log_str)
        if str_width is not None:
            str_width = str(float(str_width))
        logging.info("{} Epoch {}, Eval Width {}".format(mode, self.epoch, str_width))
        logging.info(log_str)

        # self.logging(tvgr, tscore, t_score_coll, tvgr_before, tcoverage, tcoverage_all, \
        #             tvgr_all, tscore_all, tvgr_before_all, mode, self.epoch, str_width)
        self.batchs = 0
        self.new_epoch()

    def logging(self, vgr, score, t_score_coll, vgr_before, coverage, coverage_all, \
                    vgr_all, score_all, vgr_before_all, mode, epoch, str_width=None):
        logging.info("{} Epoch {}, Eval Width {}".format(mode, epoch, str_width))
        logging.info("vgr\t\t"+str(vgr))
        logging.info("score_wo_collision\t\t"+str(score))
        logging.info("score\t\t"+str(t_score_coll))
        logging.info("coverage\t"+str(coverage))
        logging.info("coverage_all\t"+str(coverage_all))
        logging.info("vgr_scene\t"+str(vgr_all))
        logging.info("score_scene\t"+str(score_all))

    def eval_notruth(self, pc, color, grasp, params, score_thre=None, grasp_save_path=None):
        print(params)
        depths, width, table_height, gpu = params
        view_num = None
        
        grasp = grasp[0]
        score_mask = torch.nonzero(grasp[:,7] >= 0).view(-1)
        if len(score_mask) <= 0:
            print("**there has no predicted grasp:", len(score_mask))
            return None
        grasp = grasp[score_mask]
        cur_grasp_all = grasp.clone()
        cur_grasp = self.select_topK(grasp)

        grasp_nocoll = eval_test(pc, cur_grasp[:,:8], view_num, table_height, depths, width, gpu)
        all_grasp_nocoll = eval_test(pc, cur_grasp_all[:,:8], view_num, table_height, depths, width, gpu)
        print("predicted grasp length:", len(score_mask), ' no coll with view:\t', len(all_grasp_nocoll))
        print("top 100: no coll with view:\t\t\t", len(grasp_nocoll))
        output_dict = {
            'points'    : pc,
            'colors'    : color,
            'grasp'     : all_grasp_nocoll.cpu().numpy(),
            'grasp100'  : grasp_nocoll.cpu().numpy(),
        }

        if score_thre:
            for thre in score_thre:
                cur_grasp = self.select_scorethre_topK(grasp, thre)
                grasp_score = eval_test(pc, cur_grasp[:,:8], view_num, table_height, depths, width, gpu)
                output_dict.update({'grasp'+str(thre): grasp_score.cpu().numpy()})
                print('scorethre', str(thre), ": no coll with view:\t\t", len(grasp_score))
        print(grasp_save_path)
        if grasp_save_path:
            with open(grasp_save_path, 'wb') as file:
                pickle.dump(output_dict, file)

