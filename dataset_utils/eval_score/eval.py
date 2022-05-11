from .eval_utils.evaluation_data_generator import inv_transform_predicted_grasp, trans_points2grasps
from .eval_utils.evaluation_data_generator import EvalDataValidate, EvalDataTest
import torch
import numpy as np

def eval_test(points, predicted_grasp, view_num, table_height, depth, width, gpu: int):
    '''
    points         : [N, 3]
    predicted_grasp: [B, 8]
    '''
    # print(depth, width)
    view_cloud = EvalDataTest(points, predicted_grasp, view_num, table_height, depth, width, gpu)
    grasp_nocoll_view = view_cloud.run_collision_view()
    return grasp_nocoll_view

def eval_test_batch(points, predicted_grasp, table_height, depth, width, gpu: int):
    '''
    points         : [B, N1, 3]
    predicted_grasp: [B, K, 8]
    '''
    device = torch.device('cpu')
    if gpu != -1 and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        device = torch.device('cuda')
        
    if type(points) == torch.Tensor:
        cloud_array  = points.float().to(device)
    else:
        cloud_array  = torch.FloatTensor(points).to(device)
    cloud_array_homo = torch.cat(
            [cloud_array[...,:3].transpose(1,2), torch.ones(cloud_array.shape[0], 1, \
            cloud_array.shape[1], device=device)], dim=1).float().to(device)
    
    if type(predicted_grasp) == torch.Tensor:
        grasp = predicted_grasp.float().to(device)
    else:
        grasp = torch.FloatTensor(predicted_grasp).to(device)
    B, K, C = grasp.shape
    
    ### torch.Tensor: frame [B*K,3,3], center [B*K,3], score [B*K,1]
    frame_r, frame_center, frame_score = inv_transform_predicted_grasp(grasp.view(-1, C))
    global_to_local = torch.eye(4).unsqueeze(0).expand(frame_r.shape[0], 4, 4).to(device).contiguous()
    global_to_local[:, 0:3, 0:3] = frame_r
    global_to_local[:, 0:3, 3] = frame_center
    global_to_local = torch.inverse(global_to_local)
    
    frame_r      = frame_r.view(B, K, 3, 3)
    frame_center = frame_center.view(B, K, 3)
    frame_score  = frame_score.view(B, K, 1)
    global_to_local = global_to_local.view(B, K, 4, 4)
    params = [table_height, depth, width, gpu]
    keep_idx = trans_points2grasps(cloud_array_homo, global_to_local, frame_r, frame_center, params).view(B, -1)
    keep_num = torch.zeros((B)).to(device).int()
    for i in range(B):
        keep_num[i] = keep_idx[i].sum()
    return keep_idx, keep_num


def eval_validate(formal_dict, predicted_grasp, view_num: int, table_height, depth, width, gpu: int):
    '''
    formal_dict:  {}
    predicted_grasp: [B, 8]
    '''
    view_cloud = EvalDataValidate(formal_dict, predicted_grasp, view_num, table_height, depth, width, gpu)
    vgr, score, score_coll, grasp_nocoll_view_num, grasp_nocoll_view, grasp_nocoll_scene = view_cloud.run_collision()
    return vgr, score, score_coll, grasp_nocoll_view_num, grasp_nocoll_view, grasp_nocoll_scene
    
def eval_validate_wo_view(formal_dict, predicted_grasp, view_num: int, table_height, depth, width, gpu: int, regrad=False):
    '''
    formal_dict:  {}
    predicted_grasp: [B, 8]
    '''
    view_cloud = EvalDataValidate(formal_dict, predicted_grasp, view_num, table_height, depth, width, gpu, regrad=regrad)
    vgr, score, score_coll, grasp_nocoll_view_num, grasp_nocoll_view, grasp_nocoll_scene = view_cloud.run_collision(use_view=False)
    return vgr, score, score_coll, grasp_nocoll_view_num, grasp_nocoll_view, grasp_nocoll_scene
if __name__ == "__main__":
    pass
