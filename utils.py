import os
import numpy as np
import time
import pickle
import transforms3d
import logging

import torch, random
from tensorboardX import SummaryWriter

import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from dataset_utils.scoredataset import ScoreDataset, ScoreDatasetMultiWidth
from dataset_utils.scoredataset_regrad import ScoreDatasetREGRAD
from dataset_utils.graspdataset import compute_distance
from multi_model.proposal_network import GraspProposalNetwork
from multi_model.proposal_network_less import GraspLessProposalNetwork
from multi_model.proposal_network_direct import GraspDirectNetwork
from multi_model.gripper_refine_network import GripperRefineNetwork

from dataset_utils.eval_score.eval import eval_test, eval_validate, eval_test_batch, eval_validate_wo_view
from multi_model.utils.pn2_utils import function as _F

def mkdir_output(base_path, tag, mode="train", log_flag=False):
    path = os.path.join(base_path, tag)
    if not os.path.exists(path):
        os.makedirs(path)

    if log_flag:
        logger = SummaryWriter(path)
        return logger
      
def get_dataset(all_points_num, base_path, tag="train", seed=1, width=None, regrad=False):
    if not regrad:
        dataset = ScoreDataset(
                all_points_num = all_points_num,
                path = base_path,
                tag = tag,
                data_seed = seed,
                data_width = width)
    else:
        dataset = ScoreDatasetREGRAD(
                all_points_num = all_points_num,
                path = base_path,
                tag = tag,
                data_seed = seed,
                data_width = width)

    print(len(dataset))
    return dataset

def get_dataset_multi(all_points_num, base_path, tag="train", seed=1):
    dataset = ScoreDatasetMultiWidth(
                all_points_num = all_points_num,
                path = base_path,
                tag = tag,
                data_seed = seed)
    print(len(dataset))
    return dataset

def get_dataloader(dataset, batchsize, shuffle=True, num_workers=8, pin_memory=True):
    def my_worker_init_fn(pid):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        # np.random.seed(torch.initial_seed() % (2**31-1))
    def my_collate(batch):
        batch = list(filter(lambda x:x[0] is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batchsize,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    shuffle=shuffle,
                    worker_init_fn=my_worker_init_fn,
                    collate_fn=my_collate,
                )
    return dataloader

class ModelInit:
    def __init__(self, mode, model_path, method, params, rgb_flag=True, multi_flag=False, \
                sample_layer=1, conf_times=0.0025, use_region=True, use_fps=False, regrad=False):
        self.mode = mode
        self.model_path = model_path
        self.method = method
        self.multi_flag = multi_flag
        self.rgb_flag = rgb_flag
        self.sample_layer = sample_layer
        self.conf_times = conf_times
        self.use_region = use_region
        self.use_fps = use_fps
        self.regrad=regrad
        self.grasp_channel, self.radius, self.radius_refine, _, self.gripper_num, \
                        self.gpu_num, self.gpu_id, self.gpu_ids, self.lr = params
        self.checkpoint = None if model_path=='' else torch.load(model_path, map_location='cuda:{}'.format(self.gpu_id))
        # self.checkpoint = None if model_path=='' else torch.load(model_path, map_location={'cuda:0': 'cuda:'+str(self.gpu_id)})
        
    def _construct_region(self):
        if self.method == 'class_01':
            model = GraspProposalNetwork(self.radius, self.grasp_channel, self.rgb_flag, self.multi_flag, \
                                        sample_layer=self.sample_layer, conf_times=self.conf_times, \
                                        use_region=self.use_region, use_fps=self.use_fps, regrad=self.regrad) 
        elif self.method == 'class_anchornum':
            model = GraspLessProposalNetwork(self.radius, self.grasp_channel, self.rgb_flag, \
                                        self.multi_flag, sample_layer=self.sample_layer, conf_times=self.conf_times)
        elif self.method == 'noanchor':
            model = GraspDirectNetwork(self.radius, self.grasp_channel, self.rgb_flag, self.multi_flag, \
                                        sample_layer=self.sample_layer, conf_times=self.conf_times)
        
        resume_num = 0
        if self.checkpoint:
            model_dict, resume_num = self.checkpoint['net'], self.checkpoint['epoch'] + 1
            if 'test' in self.mode:
                resume_num -= 1
            new_model_dict = {}
            for key in model_dict.keys():
                new_model_dict[key.replace("module.", "")] = model_dict[key]
            model.load_state_dict(new_model_dict)
        return model, resume_num

    def _construct_refine(self):
        #-------------- load refine network----------------
        model = GripperRefineNetwork(self.gripper_num, self.radius_refine, self.radius, self.grasp_channel, \
                    self.method, self.rgb_flag, self.multi_flag, self.sample_layer, self.conf_times, self.use_region, self.use_fps, self.regrad)
        resume_num = 0
        if self.checkpoint:
            new_model_dict = {}
            model_dict, resume_num = self.checkpoint['net'], self.checkpoint['epoch'] + 1
            if 'test' in self.mode:
                resume_num -= 1

            if 'region' in self.model_path.split('/')[-1]:
                for key in model_dict.keys():
                    new_model_dict[key.replace("module.", "region_model.")] = model_dict[key]
                cur_dict = model.state_dict()
                cur_dict.update(new_model_dict)
                model.load_state_dict(cur_dict)
            else:
                for key in model_dict.keys():
                    new_model_dict[key.replace("module.", "")] = model_dict[key]
                model.load_state_dict(new_model_dict)
        return model, resume_num

    def _map_model(self, model):
        device = torch.device("cuda:"+str(self.gpu_id))
        model = model.to(device)
        
        if self.gpu_num > 1:
            device_id = [int(i) for i in self.gpu_ids.split(',')]
            model = nn.DataParallel(model, device_ids=device_id)
        print("Construct network successfully!")
        return model

    def construct_net(self):
        # self.mode = ['train', 'pretrain', 'validate', 'prevalidate', 'test', 'pretest']
        if 'pre' in self.mode:
            model, resume_num = self._construct_region()
        else:
            model, resume_num = self._construct_refine()
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

    def add_batch_tuple(self, dtuple, index, mode, stage):
        self.logger.add_scalar('batch_'+mode+'_loss1'+'_'+stage, (dtuple[0].mean()), index)
        self.logger.add_scalar('batch_'+mode+'_loss2'+'_'+stage, (dtuple[1].mean()), index)
        self.logger.add_scalar('batch_'+mode+'_loss3'+'_'+stage, (dtuple[2].mean()), index)
        self.logger.add_scalar('batch_'+mode+'_loss4'+'_'+stage, (dtuple[3].mean()), index)
        
        self.logger.add_scalar('batch_'+mode+'_pre_loss_center'+'_'+stage, (dtuple[4].mean()), index)
        self.logger.add_scalar('batch_'+mode+'_pre_loss_ori'+'_'+stage,    (dtuple[5].mean()), index)
        self.logger.add_scalar('batch_'+mode+'_pre_loss_theta'+'_'+stage,  (dtuple[6].mean()), index)  
        self.logger.add_scalar('batch_'+mode+'_pre_loss_score'+'_'+stage,  (dtuple[7].mean()), index)  
    
        self.logger.add_scalar('batch_'+mode+'_acc'+'_'+stage,    (dtuple[8].mean()), index) 
        self.logger.add_scalar('batch_'+mode+'_recall'+'_'+stage, (dtuple[9].mean()), index) 

        self.logger.add_scalar('batch_'+mode+'_loss_cls'+'_'+stage,   (dtuple[10].mean()), index)
        self.logger.add_scalar('batch_'+mode+'_loss_grasp'+'_'+stage, (dtuple[11].mean()), index) 
        if len(dtuple) > 12 and stage=='stage1':
            self.logger.add_scalar('batch_'+mode+'_loss_conf'+'_'+stage,  (dtuple[12].mean()), index) 

    def add_batch_loss(self, data, index, mode="train"):
        if len(data) == 2:
            loss, loss_tuple = data
            self.logger.add_scalar('batch_'+mode+'_loss_stage1', loss, index)  
            self.add_batch_tuple(loss_tuple, index, mode, 'stage1')

        elif len(data) == 3:
            loss, loss_tuple1, loss_tuple2 = data
            self.logger.add_scalar('batch_'+mode+'_loss_stage1and2', loss, index)  
            self.add_batch_tuple(loss_tuple1, index, mode, 'stage1')
            self.add_batch_tuple(loss_tuple2, index, mode, 'stage2')

    def add_batch_eval(self, batch_vgr, batch_score, batch_vgr_before, batch_coverage, \
                batch_coverage_all, batch_vgr_before_all, batch_vgr_all, batch_score_all, mode, index, stage):
        if batch_vgr !=0 and batch_score !=0 and batch_vgr_before !=0 :
            self.logger.add_scalar('batch_'+mode+'_vgr_'+stage,           batch_vgr,          index)
            self.logger.add_scalar('batch_'+mode+'_score_'+stage,         batch_score,        index)
            self.logger.add_scalar('batch_'+mode+'_vgr_before_'+stage,    batch_vgr_before,   index)
            self.logger.add_scalar('batch_'+mode+'_converage_'+stage,     batch_coverage,     index)
            self.logger.add_scalar('batch_'+mode+'_vgr_all_'+stage,       batch_vgr_all,      index)
            self.logger.add_scalar('batch_'+mode+'_vgr_before_all_'+stage, batch_vgr_before_all, index)
            self.logger.add_scalar('batch_'+mode+'_score_all_'+stage,     batch_score_all,    index)
            self.logger.add_scalar('batch_'+mode+'_converage_all_'+stage, batch_coverage_all, index)

    def add_epoch_eval(self, vgr, score, score_coll, vgr_before, coverage, coverage_all, \
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

        # logging.info("vgr\t\t"+str(vgr))
        # logging.info("score_no_collision\t"+str(score))
        # logging.info("score\t\t"+str(score_coll))
        # logging.info("coverage\t"+str(coverage))
        # logging.info("coverage_all\t"+str(coverage_all))
        # logging.info("vgr_scene\t"+str(vgr_all))
        # logging.info("score_scene\t"+str(score_all))
        

class Eval:
    def __init__(self, log_machine, topK=100, dis_thre=0.01, epoch=-1, eval_time=1, regrad=False):
        self.K = topK
        self.dis_thre = dis_thre
        self.epoch = epoch
        self.batchs = 0
        self.log_machine = log_machine
        self.eval_time = eval_time
        self.regrad = regrad

        self.total_vgr             = [0] * eval_time
        self.total_score           = [0] * eval_time
        self.total_score_collision = [0] * eval_time
        self.total_grasp_formal    = [0] * eval_time
        self.total_grasp_woco_view = [0] * eval_time

        self.batch_vgr             = [0] * eval_time
        self.batch_score           = [0] * eval_time
        self.batch_score_collision = [0] * eval_time
        self.before_batch_vgr             = [0] * eval_time
        self.before_batch_score           = [0] * eval_time
        self.before_batch_score_collision = [0] * eval_time
        self.batch_coverage        = [0] * eval_time
        self.batch_coverage_all    = [0] * eval_time

        self.cur_batch_vgr             = [0] * eval_time
        self.cur_batch_score           = [0] * eval_time
        self.cur_batch_score_collision = [0] * eval_time
        self.cur_before_batch_vgr             = [0] * eval_time
        self.cur_before_batch_score           = [0] * eval_time
        self.cur_before_batch_score_collision = [0] * eval_time
        self.cur_batch_coverage        = [0] * eval_time
        self.cur_batch_coverage_all    = [0] * eval_time

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
    
    def eval_one_grasp(self, data_path, grasp, params, cur_eval_time=0):
        depths, width, table_height, gpu = params
        print(data_path)
        cur_data = np.load(data_path, allow_pickle=True)
        float_str_map = {0.06:'0.06', 0.08:'0.08', 0.10:'0.10', 0.12:'0.12'}
        if width in [0.06, 0.08, 0.10, 0.12]:
            if '0.' in data_path.split('/')[-3]:
                re_width = data_path.split('/')[-3]
                data_path = data_path.replace(re_width, float_str_map[width])
        if width is None:
            width = self._get_data_width(data_path, width)
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
        #print(cur_grasp[:5])
        
        # vgr, score, score_coll, grasp_nocoll_view_num, grasp_nocoll_view, grasp_nocoll_scene = \
        #         eval_validate(cur_data, cur_grasp[:,:8], view_num, table_height, depths, width, gpu)
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
            before_vgr_all, before_score_all, before_score_coll_all = 0, 0, 0
        else:
            vgr_all, score_all, score_coll_all = vgr/grasp_nocoll_view_num, \
                        score/grasp_nocoll_view_num, score_coll/grasp_nocoll_view_num
            before_vgr_all, before_score_all, before_score_coll_all = vgr/formal_num, \
                                                score/formal_num, score_coll/formal_num


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
        self.before_batch_vgr[cur_eval_time]             += before_vgr_all
        self.before_batch_score[cur_eval_time]           += before_score_all
        self.before_batch_score_collision[cur_eval_time] += before_score_coll_all
        self.batch_coverage[cur_eval_time]        += coverage_rate
        self.batch_coverage_all[cur_eval_time]    += coverage_rate_all

        self.cur_batch_vgr[cur_eval_time]             += vgr_all
        self.cur_batch_score[cur_eval_time]           += score_all
        self.cur_batch_score_collision[cur_eval_time] += score_coll_all
        self.cur_before_batch_vgr[cur_eval_time]             += before_vgr_all
        self.cur_before_batch_score[cur_eval_time]           += before_score_all
        self.cur_before_batch_score_collision[cur_eval_time] += before_score_coll_all
        self.cur_batch_coverage[cur_eval_time]        += coverage_rate
        self.cur_batch_coverage_all[cur_eval_time]    += coverage_rate_all

        # return vgr, score, grasp_nocoll_view_num, formal_num, coverage_rate, \
        #         coverage_rate_all, vgr_all, score_all, grasp_nocoll_view_num_all, formal_num_all

    def update_epoch(self):
        self.epoch += 1

    def eval_grasps_with_gt1(self, data_paths, grasp, record_data, params, cur_eval_time):
        total_vgr, total_score, grasp_num, grasp_num_before, total_coverage_rate, \
                total_coverage_rate_all, total_vgr_all, total_score_all, grasp_num_all, grasp_num_before_all = record_data
        batch_vgr, batch_score, batch_vgr_before, batch_grasp_num, batch_grasp_before_num, batch_coverage, \
                batch_coverage_all, batch_vgr_all, batch_score_all, batch_vgr_before_all, batch_grasp_num_all, batch_grasp_before_num_all = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        self.batchs += len(data_paths)
        for i in range(len(data_paths)):
            batch_grasp = grasp[i]
            score_mask = torch.nonzero(batch_grasp[:,7] >= 0).view(-1)
            if len(score_mask) <= 0:
                print("**there has no predicted grasp:", len(score_mask))
                continue
            batch_grasp = batch_grasp[score_mask]

            data_path = data_paths[i]
            vgr, score, grasp_nocoll_view_num, formal_num, coverage_rate, \
                coverage_rate_all, vgr_all, score_all, grasp_nocoll_view_num_all, formal_num_all = \
                                self.eval_one_grasp(data_path, batch_grasp, params, cur_eval_time)
            # if self.epoch%5 == 0:
            #     print("predicted grasp length:", len(score_mask), ' -> no coll with view:', \
            #                                     grasp_nocoll_view_num_all, ' no coll with scene:', vgr_all)
            print("predicted grasp length:", len(score_mask), ' -> top 100: no coll with view:', \
                                                grasp_nocoll_view_num, ' no coll with scene:', vgr)
            grasp_nocoll_view_num = max(1, grasp_nocoll_view_num) # if grasp_nocoll_view_num == 0 
            formal_num            = max(1, formal_num)            # if formal_num == 0 
            grasp_nocoll_view_num_all = max(1, grasp_nocoll_view_num_all) # if grasp_nocoll_view_num == 0 
            formal_num_all            = max(1, formal_num_all)            # if formal_num == 0 

            total_vgr[0]           += vgr
            total_score[0]         += score
            grasp_num[0]           += grasp_nocoll_view_num
            grasp_num_before[0]    += formal_num
            total_coverage_rate[0] += coverage_rate
            
            total_vgr_all[0]       += vgr_all
            total_score_all[0]     += score_all
            grasp_num_all[0]       += grasp_nocoll_view_num_all
            grasp_num_before_all[0]    += formal_num_all
            total_coverage_rate_all[0] += coverage_rate_all
            
            batch_grasp_before_num     += formal_num
            batch_grasp_before_num_all += formal_num_all
            batch_vgr              += vgr
            batch_score            += score
            batch_grasp_num        += grasp_nocoll_view_num
            batch_coverage         += coverage_rate
            batch_coverage_all     += coverage_rate_all
            batch_vgr_all          += vgr_all
            batch_score_all        += score_all
            batch_grasp_num_all    += grasp_nocoll_view_num_all

        batch_grasp_before_num     = max(1, batch_grasp_before_num)         # if batch_grasp_before_num == 0 
        batch_grasp_before_num_all = max(1, batch_grasp_before_num_all)     # if batch_grasp_before_num_all == 0
        batch_grasp_num            = max(1, batch_grasp_num)                # if batch_grasp_num == 0
        batch_grasp_num_all        = max(1, batch_grasp_num_all)            # if batch_grasp_num_all == 0
        batch_vgr_before    = batch_vgr / batch_grasp_before_num
        batch_vgr          /= batch_grasp_num
        batch_score        /= batch_grasp_num
        batch_coverage     /= len(data_paths)
        batch_coverage_all /= len(data_paths)
        batch_vgr_before_all= batch_vgr_all / batch_grasp_before_num_all
        batch_vgr_all       /= len(data_paths)
        batch_score_all     /= len(data_paths)
        print("#before batch vgr \t", batch_vgr_before)
        print("#batch vgr \t\t",        batch_vgr)
        print("#batch score \t\t",      batch_score)
        print("#batch coverage\t\t",  batch_coverage)

        print("#batch vgr all\t\t",        batch_vgr_all)
        print("#batch score all\t\t",      batch_score_all)
        # if self.epoch%5 == 0:
        #     print("\n")
        #     print("#before batch vgr all\t", batch_vgr_before_all)
        #     print("#batch vgr all\t\t",     batch_vgr_all)
        #     print("#batch score all\t",     batch_score_all)
        #     print("#batch coverage_all\t",  batch_coverage_all)

        record_data = (total_vgr, total_score, grasp_num, grasp_num_before, total_coverage_rate, \
                        total_coverage_rate_all, total_vgr_all, total_score_all, grasp_num_all, grasp_num_before_all)
        return batch_vgr, batch_score, batch_vgr_before, batch_coverage, batch_coverage_all, \
                batch_vgr_before_all, batch_vgr_all, batch_score_all, record_data

    def eval_batch1(self, data_path, grasp, record_data, params, index, mode, stage, cur_eval_time):
        # eval parameters of width is generated from its data path
        print("=======================evaluate grasp from {}=======================".format(stage))
        batch_vgr, batch_score, batch_vgr_before, batch_coverage, batch_coverage_all, \
                batch_vgr_before_all, batch_vgr_all, batch_score_all, record_data = \
                    self.eval_grasps_with_gt1(data_path, grasp, record_data, params, cur_eval_time)
        
        self.log_machine.add_batch_eval(batch_vgr, batch_score, batch_vgr_before, batch_coverage, \
                batch_coverage_all, batch_vgr_before_all, batch_vgr_all, batch_score_all, mode, index, stage)
        print("=========================================================================")
        return record_data

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
            # vgr, score, grasp_nocoll_view_num, formal_num, coverage_rate, \
            #     coverage_rate_all, vgr_all, score_all, grasp_nocoll_view_num_all, formal_num_all = \
            self.eval_one_grasp(data_path, batch_grasp, params, cur_eval_time)
        
        print("#before batch vgr \t",        self.cur_before_batch_vgr[cur_eval_time]/len(data_paths))
        print("#before batch score \t",      self.cur_before_batch_score[cur_eval_time]/len(data_paths))
        print("#before batch score coll \t", self.cur_before_batch_score_collision[cur_eval_time]/len(data_paths))
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
        self.cur_before_batch_vgr[cur_eval_time]             = 0
        self.cur_before_batch_score[cur_eval_time]           = 0
        self.cur_before_batch_score_collision[cur_eval_time] = 0

    def new_epoch(self):
        self.total_vgr             = [0] * self.eval_time
        self.total_score           = [0] * self.eval_time
        self.total_score_collision = [0] * self.eval_time
        self.total_grasp_formal    = [0] * self.eval_time
        self.total_grasp_woco_view = [0] * self.eval_time

        self.batch_vgr             = [0] * self.eval_time
        self.batch_score           = [0] * self.eval_time
        self.batch_score_collision = [0] * self.eval_time
        self.before_batch_vgr             = [0] * self.eval_time
        self.before_batch_score           = [0] * self.eval_time
        self.before_batch_score_collision = [0] * self.eval_time
        self.batch_coverage        = [0] * self.eval_time
        self.batch_coverage_all    = [0] * self.eval_time

    def eval_batch(self, data_path, grasp, params, index, mode, stage, cur_eval_time):
        # eval parameters of width is generated from its data path
        print("=======================evaluate grasp from {}=======================".format(stage))
        self.eval_grasps_with_gt(data_path, grasp, params, cur_eval_time)
        
        self.log_machine.add_batch_eval(self.cur_batch_vgr[cur_eval_time], self.cur_batch_score[cur_eval_time], \
                self.cur_before_batch_vgr[cur_eval_time], self.cur_batch_coverage[cur_eval_time], \
                self.cur_batch_coverage_all[cur_eval_time], self.cur_before_batch_vgr[cur_eval_time], \
                self.cur_batch_vgr[cur_eval_time], self.cur_batch_score[cur_eval_time], mode, index, stage)

    def eval_epoch1(self, data, batch_nums, mode, stages, str_width=None):
        self.batchs /= len(data)
        for i in range(len(data)):
            data_i  = data[i]
            stage_i = stages[i]
            nocoll_scene_num, score, nocoll_view_num, formal_num, coverage_rate, \
                coverage_rate_all, nocoll_scene_num_all, score_all, nocoll_view_num_all, formal_num_all = data_i
            
            for j in range(len(nocoll_scene_num)):
                tvgr_before   = nocoll_scene_num[j]  / formal_num[j]
                tvgr          = nocoll_scene_num[j]  / nocoll_view_num[j]
                tscore        = score[j]             / nocoll_view_num[j]
                t_score_coll  = self.total_score_collision[i] / nocoll_view_num[j]
                tcoverage     = coverage_rate[j]     / self.batchs

                tvgr_before_all = nocoll_scene_num_all[j]  / self.batchs#formal_num_all[j]
                tvgr_all        = nocoll_scene_num_all[j]  / self.batchs#nocoll_view_num_all[j]
                tscore_all      = score_all[j]             / self.batchs#nocoll_view_num_all[j]
                tcoverage_all   = coverage_rate_all[j]     / self.batchs

                print("{} before_vgr: \t\t{}".format(        stage_i, tvgr_before) )
                print("{} total_vgr: \t\t{}".format(         stage_i, tvgr) )
                print("{} total_score_no_collision: \t{}".format(       stage_i, tscore) )
                print("{} total_score: \t\t{}".format(       stage_i, t_score_coll) )
                print("{} total_coverage: \t\t{}\n".format(    stage_i, tcoverage) )
                print("{} total_vgr all: \t\t{}".format(         stage_i, tvgr_all) )
                print("{} total_score all: \t\t{}".format(       stage_i, tscore_all) )

                # if self.epoch%5 == 0:
                #     print("{} before_vgr_all: \t\t{}".format(    stage_i, tvgr_before_all) )
                #     print("{} total_vgr_all: \t\t{}".format(     stage_i, tvgr_all) )
                #     print("{} total_score_all: \t{}".format(     stage_i, tscore_all) )
                #     print("{} total_coverage_all: \t{}".format(  stage_i, tcoverage_all) )

                if str_width is not None:
                    str_width = str(float(str_width))
                self.log_machine.add_epoch_eval(tvgr, tscore, t_score_coll, tvgr_before, tcoverage, tcoverage_all, \
                                            tvgr_all, tscore_all, tvgr_before_all, mode, self.epoch, \
                                            stage_i, 'None', str_width)
                # if self.epoch%5 != 0:
                #     break

        self.batchs = 0

    def eval_epoch(self, batch_nums, mode, stages, str_width=None):
        self.batchs /= len(stages)
        for i in range(len(stages)):
            stage_i = stages[i]
            
            tbefore_vgr        = self.total_vgr[i] / self.total_grasp_formal[i]
            tbefore_score      = self.total_score[i] / self.total_grasp_formal[i]
            tbefore_score_coll = self.total_score_collision[i] / self.total_grasp_formal[i]
            tvgr        = self.total_vgr[i] / self.total_grasp_woco_view[i]
            tscore      = self.total_score[i] / self.total_grasp_woco_view[i]
            tscore_coll = self.total_score_collision[i] / self.total_grasp_woco_view[i]

            tbefore_vgr_scene        = self.before_batch_vgr[i]     / self.batchs
            tbefore_score_scene      = self.before_batch_score[i]     / self.batchs
            tbefore_score_coll_scene = self.before_batch_score_collision[i]     / self.batchs
            tvgr_scene               = self.batch_vgr[i]     / self.batchs
            tscore_scene             = self.batch_score[i] / self.batchs
            tscore_coll_scene        = self.batch_score_collision[i]     / self.batchs
            tcoverage                = self.batch_coverage[i]     / self.batchs
            tcoverage_all            = self.batch_coverage_all[i] / self.batchs
                
            # log_str = stage_i.join( ("before_vgr: {}\t\t before_score: {}\t\t before_score_collision: {}\n".format(tbefore_vgr, tbefore_score, tbefore_score_coll), 
            #                        "vgr: {}\t\t score: {}\t\t score_collision: {}\n".format(tvgr, tscore, tscore_coll), 
            #                        "SCENE before_vgr: {}\t\t before_score: {}\t\t before_score_collision: {}\n".format(tbefore_vgr_scene, tbefore_score_scene, tbefore_score_coll_scene),
            #                        "SCENE vgr: {}\t\t score: {}\t\t score_collision: {}\n".format(tvgr_scene, tscore_scene, tscore_coll_scene),
            #                        "coverage: {}\t\t coverage_all: {}\n".format(tcoverage, tcoverage_all) ))
            log_str = stage_i.join( ("vgr: {}\t\t score: {}\t\t score_collision: {}\n".format(tvgr, tscore, tscore_coll), 
                                   "SCENE vgr: {}\t\t score: {}\t\t score_collision: {}\n".format(tvgr_scene, tscore_scene, tscore_coll_scene),
                                   "coverage: {}\t\t coverage_all: {}\n".format(tcoverage, tcoverage_all) ))
            print(log_str)
            logging.info(log_str)
            if str_width is not None:
                str_width = str(float(str_width))
            self.log_machine.add_epoch_eval(tvgr, tscore, tscore_coll, tbefore_vgr, tcoverage, tcoverage_all, \
                            tvgr_scene, tscore_scene, tbefore_vgr_scene, mode, self.epoch, stage_i, 'None', str_width)
                

        self.batchs = 0
        self.new_epoch()

    def eval_notruth(self, pc, color, grasp, params, score_thre=None, grasp_save_path=None):
        print(params)
        depths, width, table_height, gpu = params
        view_num = None
        
        grasp = grasp[0]
        score_mask = torch.nonzero(grasp[:,7] >= 0.1).view(-1)
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
 
class EvalNoTruth:
    def __init__(self, topK=100):
        self.K = topK

    def select_topK(self, grasp):
        _, sort_idx = torch.sort(grasp[:,7], descending=True)
        cur_grasp = grasp[sort_idx[:self.K]]
        return cur_grasp

    def eval_notruth(self, pc, grasp, params):
        '''
          @ This function is for quickly checking collision for batch_pc 
          Input:
            pc:     torch.Tensor [B, N, 3/6]
            grasp:  torch.Tensor [B, K, 3, 4]
            params: tuple
        '''
        # print(params)
        depths, width, table_height, gpu = params
        B, K, C = grasp.shape

        select_grasp = torch.full((B ,self.K, C), -1.0)
        select_grasp_num = torch.zeros((B))
        for idx in range(B):
            cur_grasp = grasp[idx]
            score_mask = torch.nonzero(cur_grasp[:,7] >= 0.1).view(-1)
            len_select = len(score_mask)
            if len_select < self.K:
                select_grasp_num[idx] = len_select 
                select_grasp[idx, :len_select] = cur_grasp[score_mask]
            else:
                select_grasp_num[idx] = self.K 
                select_grasp[idx]     = self.select_topK(cur_grasp[score_mask])
        
        # keep_idx: [B, K] bool
        # keep_num: [B] int
        keep_idx, keep_num = eval_test_batch(pc, select_grasp[...,:8], table_height, depths, width, gpu)
        print("After collision check with view, there are {} grasps".format(keep_num))
        return keep_idx

####___________________________test function_____________________________
def eval_notruth(pc, color, grasp_stage2, grasp_stage3, grasp_stage3_score, grasp_stage3_stage2, output_score, params, grasp_save_path=None):
    depths, width, table_height, gpu, _ = params
    view_num = None
    if len(grasp_stage2) >= 1:
        grasp_stage2 = eval_test(pc, grasp_stage2[:,:8], view_num, table_height, depths, width, gpu)
    if len(grasp_stage3_stage2) >= 1:
        grasp_stage3_stage2 = eval_test(pc, grasp_stage3_stage2[:,:8], view_num, table_height, depths, width, gpu)
    if len(grasp_stage3) >= 1:
        grasp_stage3 = eval_test(pc, grasp_stage3[:,:8], view_num, table_height, depths, width, gpu)
    if len(grasp_stage3_score) >= 1:
        grasp_stage3_score = eval_test(pc, grasp_stage3_score[:,:8], view_num, table_height, depths, width, gpu)

    if gpu != -1:
        output_score        = output_score.view(-1,1).cpu()
        grasp_stage2        = grasp_stage2.cpu()
        grasp_stage3_stage2 = grasp_stage3_stage2.cpu()
        grasp_stage3        = grasp_stage3.cpu()
        grasp_stage3_score  = grasp_stage3_score.cpu()
    print("stage2 grasp num:", len(grasp_stage2))
    print("stage3 grasp num:", len(grasp_stage2))
    print("stage3 grasp num (with scorethre):", len(grasp_stage3_score))
    output_dict = {
        'points'             : pc,
        'colors'             : color,
        'scores'             : output_score.numpy(),
        'grasp_stage2'       : grasp_stage2.numpy(),
        'grasp_stage3_stage2': grasp_stage3_stage2.numpy(),
        'grasp_stage3'       : grasp_stage3.numpy(),
        'grasp_stage3_score' : grasp_stage3_score.numpy(),
    }
    print(grasp_save_path)
    if grasp_save_path:
        with open(grasp_save_path, 'wb') as file:
            pickle.dump(output_dict, file)

def noise_color(pc_color):
    obj_color_time = 1-np.random.rand(3) / 5
    print("noise color time", obj_color_time)
    for i in range(3,6):
        pc_color[:,i] *= obj_color_time[i-3]
    return pc_color

def local_to_global_transformation_quat(point):
    T_local_to_global = np.eye(4)
    #quat = transforms3d.quaternions.axangle2quat([1,0,0], np.pi*1.13)
    quat = transforms3d.euler.euler2quat(-0.87*np.pi, 0, 0)
    frame = transforms3d.quaternions.quat2mat(quat)
    T_local_to_global[0:3, 0:3] = frame
    T_local_to_global[0:3, 3] = point
    return T_local_to_global  

def transform_grasp(grasp_ori):
    '''
      Input:
        grasp_ori: [B, 13] 
      Output:
        grasp_trans:[B, 8] (center[3], axis_y[3], grasp_angle[1], score[1])
    '''
    B, _ = grasp_ori.shape
    grasp_trans = torch.full((B, 8), -1)

    # axis_x = torch.cat([grasp_ori[:,0:1], grasp_ori[:,4:5], grasp_ori[:,8:9]], dim=1)
    # axis_y = torch.cat([grasp_ori[:,1:2], grasp_ori[:,5:6], grasp_ori[:,9:10]], dim=1)
    # axis_z = torch.cat([grasp_ori[:,2:3], grasp_ori[:,6:7], grasp_ori[:,10:11]], dim=1)
    # center = torch.cat([grasp_ori[:,3:4], grasp_ori[:,7:8], grasp_ori[:,11:12]], dim=1)
    axis_x, axis_y, axis_z, center = grasp_ori[:,0:3], grasp_ori[:,3:6], grasp_ori[:,6:9], grasp_ori[:,9:12]
    grasp_angle = torch.atan2(axis_x[:,2], axis_z[:,2])  ## torch.atan(torch.div(axis_x[:,2], axis_z[:,2])) is not OK!!!

    '''
    grasp_angle[axis_y[:,0] < 0] = np.pi-grasp_angle[axis_y[:,0] < 0]
    axis_y[axis_y[:,0] < 0] = -axis_y[axis_y[:,0] < 0]

    grasp_angle[grasp_angle >= 2*np.pi] = grasp_angle[grasp_angle >= 2*np.pi] - 2*np.pi
    grasp_angle[grasp_angle <= -2*np.pi] = grasp_angle[grasp_angle <= -2*np.pi] + 2*np.pi
    grasp_angle[grasp_angle > np.pi] = grasp_angle[grasp_angle > np.pi] - 2*np.pi
    grasp_angle[grasp_angle <= -np.pi] = grasp_angle[grasp_angle <= -np.pi] + 2*np.pi
    '''

    grasp_trans[:,:3]  = center
    grasp_trans[:,3:6] = axis_y
    grasp_trans[:,6]   = grasp_angle
    grasp_trans[:,7]   = grasp_ori[:,12]
    return grasp_trans
