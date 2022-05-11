import argparse
import os
import pickle

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import open3d
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import time

from dataset_utils.graspdataset import get_grasp_allobj
from dataset_utils.eval_score.eval import eval_test, eval_validate
import utils
import glob
import logging

parser = argparse.ArgumentParser(description='GripperRegionNetwork')
parser.add_argument('--tag', type=str, default='default')
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=101)
parser.add_argument('--mode', choices=['train', 'pretrain', 'validate',
                                        'prevalidate', 'test', 'pretest'], required=True)
parser.add_argument('--method', choices=['class_01', 'class_anchornum', 'noanchor'], required=True)
parser.add_argument('--batch-size', type=int, default=15)#16)#16
parser.add_argument('--num-refine', type=int, default=0, help='number of interative refinement iterations')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu-num', type=int, default=2)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpus', type=str, default='0,2,3')
parser.add_argument('--lr' , type=float, default=0.005) #0.001
parser.add_argument('--layer' , type=int, default=1) 
parser.add_argument('--conf_times', type=float, default=0.15)
parser.add_argument('--use_region', action='store_true')
parser.add_argument('--use_fps', action='store_true')
parser.add_argument('--use_rgb', action='store_true')

# parser.add_argument('--load-path', type=str, default='/data1/cxg6/Multigrasp/assets/models/premulti_5120_0.15/region_19.model')
parser.add_argument('--load-path', type=str, default='')

# parser.add_argument('--use-rgb',   type=bool, default=True)
parser.add_argument('--data-path', type=str, default='/data2/zbl/dataset_small', help='data path')

parser.add_argument('--model-path', type=str, default='/data1/cxg6/Multigrasp/assets/models/', help='to saved model path')
parser.add_argument('--log-path', type=str, default='/data1/cxg6/Multigrasp/assets/log/', help='to saved log path')
parser.add_argument('--folder-name', type=str, default='/data1/cxg6/Multigrasp/test_file/real_data')

parser.add_argument('--file-name', type=str, default='')
parser.add_argument('--log-interval', type=int, default=1)
parser.add_argument('--save-interval', type=int, default=1)

parser.add_argument('--eval-diffwidth', action='store_true')
parser.add_argument('--eval-width', type=str, default='0.10')

args = parser.parse_args()
args.cuda = args.cuda if torch.cuda.is_available else False
if args.use_fps:
    args.conf_times = 0.0025
    args.sample_layer = 0

np.random.seed(int(time.time()))
if args.cuda:
    torch.cuda.manual_seed(1)
torch.cuda.set_device(args.gpu)

logger = utils.mkdir_output(args.log_path, args.tag, args.mode, log_flag=True)
utils.mkdir_output(args.model_path, args.tag)

log_dir = os.path.join(args.log_path, args.tag)
logging.basicConfig(
    filename=os.path.join(log_dir, 'train.log'),
    level=logging.INFO
)
logging.info(args)

all_points_num = 25600
obj_class_num = 43

width, height, depth = 0.080, 0.010, 0.060
radius = 0.06
max_radius = max(width, height, depth)
table_height = 0.75
score_thre = 0.5
gripper_num = 64
use_theta = True
grasp_channel = 9
        
gripper_params = [width, height, depth]
model_params   = [grasp_channel, radius, max_radius, obj_class_num, gripper_num, args.gpu_num, args.gpu, args.gpus, args.lr]

# eval_params
topK=100
dis_thre=0.02
model_machine = utils.ModelInit(args.mode, args.load_path, args.method, model_params, \
                rgb_flag=args.use_rgb, multi_flag=True, sample_layer=args.layer, \
                conf_times=args.conf_times, use_region=args.use_region, use_fps=args.use_fps)
model, resume_epoch, optimizer, scheduler = model_machine.construct_model()

train_score_dataset = utils.get_dataset_multi(all_points_num, args.data_path, "train", 1)   #Train
val_score_dataset = utils.get_dataset_multi(all_points_num, args.data_path, "validate", 1)  #Validation
# test_dataset_path = os.path.join(args.data_path, (str(args.eval_width) if args.eval_width != 0.1 else str(args.eval_width)+'0') )
# test_score_dataset = utils.get_dataset(all_points_num, test_dataset_path, "test", 1, args.eval_width)     #Test

train_score_loader = utils.get_dataloader(train_score_dataset, args.batch_size, shuffle=True)
val_score_loader   = utils.get_dataloader(val_score_dataset, args.batch_size, shuffle=True)
test_score_loader  = None
# test_score_loader  = utils.get_dataloader(test_score_dataset, 1, shuffle=False)


class RegionModule():
    def __init__(self, start_epoch=0, end_epoch=1, train_data_loader=None, 
                            val_data_loader=None, test_data_loader=None):
        self.start_epoch = start_epoch
        self.end_epoch   = end_epoch
        self.train_data_loader = train_data_loader 
        self.val_data_loader   = val_data_loader  
        self.test_data_loader  = test_data_loader
        self.eval_params    = [depth, None, table_height, args.gpu]
        if "test" in args.mode:
            self.eval_params[1]= float(args.eval_width)
        self.gripper_params = gripper_params
        self.log_machine    = utils.Logger(logger)
        self.eval_machine   = utils.Eval(self.log_machine, topK, dis_thre, start_epoch-1)
        
        self.saved_base_path = os.path.join(args.model_path, args.tag)

    def change_params(self, widths):
        self.gripper_params[0] = widths

    def train_val(self, epoch, mode='train', use_eval=True):
        if mode == 'train':
            self.eval_machine.update_epoch()
            model.train()
            torch.set_grad_enabled(True)
            dataloader = self.train_data_loader   

        else:
            model.eval()
            torch.set_grad_enabled(False)
            if mode == 'validate':
                dataloader = self.val_data_loader
            elif mode == 'test':
                dataloader = self.test_data_loader
        batch_size = args.batch_size

        for batch_idx, (pc, pc_score, pc_label, data_path, data_width) in enumerate(dataloader):
            if mode == 'train':
                optimizer.zero_grad()
            data_path = np.array(data_path)
            cur_idx = torch.arange(len(pc))
            if args.gpu != -1:
                pc, pc_score, pc_label, data_width = pc.cuda(), pc_score.cuda(), pc_label.cuda(), data_width.cuda()
                cur_idx = cur_idx.cuda()
    
            self.change_params(data_width)
            # pre_grasp: [B, N*anchor_number, 9]
            _, _, pre_grasp, loss, loss_tuple, gt = model(pc, pc_score, pc_label, data_path, cur_idx, data_width)
             
            loss_total = loss.mean() 
            if mode == 'train':
                loss_total.backward()
                optimizer.step()

            data = (loss_total.data, loss_tuple)
            self.log_machine.add_batch_loss(data, batch_idx + epoch * len(dataloader), mode)

            if use_eval:
                self.eval_machine.eval_batch(data_path, pre_grasp, self.eval_params, \
                        batch_idx + epoch*len(dataloader), mode, 'stage1', cur_eval_time=0)

            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{}'.format(mode,
                        epoch, batch_idx * batch_size, len(dataloader.dataset),
                        100. * batch_idx * batch_size / len(dataloader.dataset), loss_total.data, args.tag))
        
        if use_eval:
            stages  = ['stage1']
            self.eval_machine.eval_epoch(len(dataloader.dataset), mode, stages)
        if mode == 'train':
            scheduler.step()

    def pretrain_region(self, epoch):
        self.train_val(epoch, mode='train', use_eval=False)

    def validate_region(self, epoch):
        self.train_val(epoch, mode='validate', use_eval=True)

    def test_region(self, epoch):
        self.train_val(epoch, mode='test', use_eval=True)

    def train(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            print("---------------pretrain_region epoch", epoch, "------------------")
            path_region  = os.path.join(self.saved_base_path, 'region_{}.model'.format(epoch))
            logging.info("epoch"+str(epoch))
            self.pretrain_region(epoch)
            region_state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            torch.save(region_state, path_region)
            logging.info('validate')
            self.validate_region(epoch)
            # logging.info('test')
            # self.test_region(epoch)

    def validate(self, epoch):
        print("---------------validate_region epoch", epoch, "------------------")
        self.validate_region(epoch)

    def test(self, epoch):
        print("---------------DATALOADER: test_region epoch", epoch, "------------------")
        self.test_region(epoch)

class RefineModule():
    def __init__(self, start_epoch=0, end_epoch=1, train_data_loader=None, 
                            val_data_loader=None, test_data_loader=None):
        self.start_epoch = start_epoch
        self.end_epoch   = end_epoch
        self.train_data_loader = train_data_loader 
        self.val_data_loader   = val_data_loader  
        self.test_data_loader  = test_data_loader
        self.eval_params    = [depth, None, table_height, args.gpu]
        if "test" in args.mode:
            self.eval_params[1]= args.eval_width
        self.gripper_params = gripper_params
        self.log_machine    = utils.Logger(logger)
        self.eval_machine   = utils.Eval(self.log_machine, topK, dis_thre, start_epoch-1, eval_time=2)
        
        self.saved_base_path = os.path.join(args.model_path, args.tag)

    def change_params(self, widths):
        self.gripper_params[0] = widths

    def train_val(self, epoch, mode='train', use_eval=True):
        if mode == 'train':
            self.eval_machine.update_epoch()
            model.train()
            torch.set_grad_enabled(True)
            dataloader = self.train_data_loader   

        else:
            model.eval()
            torch.set_grad_enabled(False)
            if mode == 'validate':
                dataloader = self.val_data_loader
            elif mode == 'test':
                dataloader = self.test_data_loader
            # batch_size = 1
        batch_size = args.batch_size

        for batch_idx, (pc, pc_score, pc_label, data_path, data_width) in enumerate(dataloader):
            if mode == 'train':
                optimizer.zero_grad()
            data_path = np.array(data_path)
            cur_idx = torch.arange(len(pc))
            if args.gpu != -1:
                pc, pc_score, pc_label, data_width = pc.cuda(), pc_score.cuda(), pc_label.cuda(), data_width.cuda()
                cur_idx = cur_idx.cuda()
    
            self.change_params(data_width)
            # pre2_grasp: [B, N*anchor_number, 9], pre1_grasp: [B, N*anchor_number, 9]
            pre2_grasp, pre1_grasp, loss, loss_tuple2, loss_tuple1 = model(pc, self.gripper_params, pc_score, pc_label, data_path, cur_idx, data_width)
             
            loss_total = loss.mean() 
            if mode == 'train':
                loss_total.backward()
                optimizer.step()

            data = (loss_total.data, loss_tuple1, loss_tuple2)
            self.log_machine.add_batch_loss(data, batch_idx + epoch * len(dataloader), mode)

            if use_eval:
                self.eval_machine.eval_batch(data_path, pre1_grasp, self.eval_params, \
                        batch_idx + epoch*len(dataloader), mode, stage='stage1', cur_eval_time=0)
                self.eval_machine.eval_batch(data_path, pre2_grasp, self.eval_params, \
                        batch_idx + epoch*len(dataloader), mode, stage='stage2', cur_eval_time=1)

            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{}'.format(mode,
                        epoch, batch_idx * batch_size, len(dataloader.dataset),
                        100. * batch_idx * batch_size / len(dataloader.dataset), loss_total.data, args.tag))
        
        if use_eval:
            stages  = ['stage1', 'stage2']
            self.eval_machine.eval_epoch(len(dataloader.dataset), mode, stages)
        if mode == 'train':
            scheduler.step()

    def train_refine(self, epoch):
        self.train_val(epoch, mode='train', use_eval=False)

    def validate_refine(self, epoch):
        self.train_val(epoch, mode='validate', use_eval=True)

    def test_refine(self, epoch):
        self.train_val(epoch, mode='test', use_eval=True)

    def train(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            print("---------------train_refine epoch", epoch, "------------------")
            path_refine  = os.path.join(self.saved_base_path, 'refine_{}.model'.format(epoch))
            logging.info("epoch"+str(epoch))
            self.train_refine(epoch)
            refine_state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            torch.save(refine_state, path_refine)
            logging.info('validate')
            self.validate_refine(epoch)
            # logging.info('test')
            # self.test_refine(epoch)

    def validate(self, epoch):
        print("---------------validate_refine epoch", epoch, "------------------")
        self.validate_refine(epoch)

    def test(self, epoch):
        print("---------------DATALOADER: test_refine epoch", epoch, "------------------")
        self.test_refine(epoch)


def main():
    if args.mode == 'pretrain':
        regionModule = RegionModule(resume_epoch, args.epoch, train_score_loader, val_score_loader, test_score_loader)
        regionModule.train()
            
    elif args.mode == 'train':
        refineModule = RefineModule(resume_epoch, args.epoch, train_score_loader, val_score_loader, test_score_loader)
        refineModule.train()

    elif args.mode == 'prevalidate':
        regionModule = RegionModule(resume_epoch, val_data_loader=val_score_loader)
        regionModule.validate(resume_epoch)

    elif args.mode == 'validate':
        refineModule = RefineModule(resume_epoch, val_data_loader=val_score_loader)
        refineModule.validate(resume_epoch)

    elif args.mode == 'pretest':
        regionModule = RegionModule(resume_epoch, test_data_loader=test_score_loader)
        regionModule.test(resume_epoch)

    elif args.mode == 'test':
        refineModule = RefineModule(resume_epoch, test_data_loader=test_score_loader)
        refineModule.test(resume_epoch)

if __name__ == "__main__":
    main()
