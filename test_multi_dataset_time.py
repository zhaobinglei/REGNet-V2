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
parser.add_argument('--mode', choices=['test', 'pretest'], required=True)
parser.add_argument('--method', choices=['class_01', 'class_anchornum', 'noanchor'], required=True)
parser.add_argument('--batch-size', type=int, default=6)
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

parser.add_argument('--load-path', type=str, default='')

# parser.add_argument('--use-rgb',   type=bool, default=False)
parser.add_argument('--data-path', type=str, default='/data2/zbl/dataset', help='data path')

parser.add_argument('--model-path', type=str, default='/data1/cxg6/Multigrasp/contrast/assets/models/', help='to saved model path')
parser.add_argument('--log-path', type=str, default='/data1/cxg6/Multigrasp/contrast/assets/log/', help='to saved log path')
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

log_dir = os.path.join(args.log_path, args.tag)
logging.basicConfig(
    filename=os.path.join(log_dir, 'test.log'),
    level=logging.INFO
)
logging.info(args)

all_points_num = 25600
obj_class_num = 43

width, height, depth = 0.080, 0.010, 0.060
radius = 0.06
max_radius = 0.10#max(width, height, depth)
table_height = 0.75
score_thre = 0.5
gripper_num = 64
use_theta = True
grasp_channel = 9
        
gripper_params = [width, height, depth]
model_params   = [grasp_channel, radius, max_radius, obj_class_num, gripper_num, args.gpu_num, args.gpu, args.gpus, args.lr]

# eval_params, vmax
topK=1000
dis_thre=0.02
score_thres=[]#0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] 
model_machine = utils.ModelInit(args.mode, args.load_path, args.method, model_params, \
                rgb_flag=args.use_rgb, multi_flag=True, sample_layer=args.layer, \
                conf_times=args.conf_times, use_region=args.use_region, use_fps=args.use_fps)
model, resume_epoch, optimizer, scheduler = model_machine.construct_model()

Tforward_passing_time = 0 
Tprocessing_time = 0
Ttimes = 0
processing_time = 0

if args.eval_diffwidth:
    # test_score_dataset = utils.get_dataset_multi(all_points_num, args.data_path, "test", 1)  
    # test_score_loader = utils.get_dataloader(test_score_dataset, args.batch_size, shuffle=False)
    widths = ['0.06', '0.08', '0.10', '0.12']
    test_dataset_path = [os.path.join(args.data_path, i) for i in widths]
    test_score_dataset = [utils.get_dataset(all_points_num, test_dataset_path[i], "test", 1, float(widths[i])) \
                            for i in range(len(widths))]
    test_score_loader = [utils.get_dataloader(test_score_dataset[i], args.batch_size, shuffle=False) \
                            for i in range(len(widths))]

else:
    widths = [args.eval_width]
    test_dataset_path = os.path.join(args.data_path, args.eval_width)
    test_score_dataset = utils.get_dataset(all_points_num, test_dataset_path, "test", 1, float(args.eval_width))  
    test_score_loader  = [utils.get_dataloader(test_score_dataset, args.batch_size, shuffle=False)]


class RegionModule():
    def __init__(self, start_epoch=0, end_epoch=1, train_data_loader=None, 
                            val_data_loader=None, test_data_loader=None, cur_width=None):
        self.start_epoch = start_epoch
        self.end_epoch   = end_epoch
        self.train_data_loader = train_data_loader 
        self.val_data_loader   = val_data_loader  
        self.test_data_loader  = test_data_loader
        self.eval_params    = [depth, None, table_height, args.gpu]
        if "test" in args.mode:
            self.eval_params[1]= float(args.eval_width) # eval_width will be changed in utils.py using the width in file name
            if cur_width is not None:
                 self.eval_params[1]= float(cur_width)
        self.gripper_params = gripper_params
        self.log_machine    = utils.Logger(logger)
        self.eval_machine   = utils.EvalNoTruth(topK)
        
    def change_params(self, widths):
        self.gripper_params[0] = widths

    def train_val(self, epoch, mode='train', use_eval=True):
        model.eval()
        torch.set_grad_enabled(False)
        dataloader = self.test_data_loader
        batch_size = args.batch_size

        for batch_idx, (pc, pc_score, pc_label, data_path, data_width) in enumerate(dataloader):
            data_path = np.array(data_path)
            cur_idx = torch.arange(len(pc))
            if args.gpu != -1:
                pc, pc_score, pc_label, data_width = pc.cuda(), pc_score.cuda(), pc_label.cuda(), data_width.cuda()
                cur_idx = cur_idx.cuda()
            print("datawidth:",data_width)
            self.change_params(data_width)
            # pre_grasp: [B, N*anchor_number, 9]

            start = time.time()
            _, _, pre_grasp, _, _, _ = model(pc, data_width=data_width)
            torch.cuda.synchronize()
            forward_passing_time = time.time() - start

            if use_eval:
                start = time.time()
                self.eval_machine.eval_notruth(pc, pre_grasp, self.eval_params)
                torch.cuda.synchronize()
                processing_time = time.time() - start
            global Ttimes, Tprocessing_time, Tforward_passing_time
            Ttimes += 1

            print('{} Epoch: {} [{}/{} ({:.0f}%)]\t{}'.format(mode,
                        epoch, batch_idx * batch_size, len(dataloader.dataset),
                        100. * batch_idx * batch_size / len(dataloader.dataset), args.tag))
        
            print('forward_passing_time: {}, processing_time: {}ms'.format(forward_passing_time*1000, processing_time*1000))
            Tprocessing_time += processing_time
            Tforward_passing_time += forward_passing_time

    def test(self, epoch):
        print("---------------DATALOADER: test_region epoch", epoch, "------------------")
        self.train_val(epoch, mode='test', use_eval=True)

class RefineModule():
    def __init__(self, start_epoch=0, end_epoch=1, train_data_loader=None, 
                            val_data_loader=None, test_data_loader=None, cur_width=None):
        self.start_epoch = start_epoch
        self.end_epoch   = end_epoch
        self.train_data_loader = train_data_loader 
        self.val_data_loader   = val_data_loader  
        self.test_data_loader  = test_data_loader
        self.eval_params    = [depth, None, table_height, args.gpu]
        if "test" in args.mode:
            self.eval_params[1]= float(args.eval_width)
            if cur_width is not None:
                 self.eval_params[1]= float(cur_width)
        self.gripper_params = gripper_params
        self.log_machine    = utils.Logger(logger)
        self.eval_machine   = utils.EvalNoTruth(topK)

    def change_params(self, widths):
        self.gripper_params[0] = widths

    def train_val(self, epoch, mode='train', use_eval=True):
        model.eval()
        torch.set_grad_enabled(False)
        dataloader = self.test_data_loader
        batch_size = args.batch_size

        for batch_idx, (pc, pc_score, pc_label, data_path, data_width) in enumerate(dataloader):
            data_path = np.array(data_path)
            cur_idx = torch.arange(len(pc))
            if args.gpu != -1:
                pc, pc_score, pc_label, data_width = pc.cuda(), pc_score.cuda(), pc_label.cuda(), data_width.cuda()
                cur_idx = cur_idx.cuda()
    
            self.change_params(data_width)
            # pre2_grasp: [B, N*anchor_number, 9], pre1_grasp: [B, N*anchor_number, 9]
            start = time.time()
            pre2_grasp, pre1_grasp, _, _, _ = model(pc, self.gripper_params, data_width=data_width)
            torch.cuda.synchronize()
            forward_passing_time = time.time() - start

            if use_eval:
                # self.eval_machine.eval_notruth(pc, pre1_grasp, self.eval_params)
                start = time.time()
                self.eval_machine.eval_notruth(pc, pre2_grasp, self.eval_params)
                torch.cuda.synchronize()
                processing_time = time.time() - start

            global Ttimes, Tprocessing_time, Tforward_passing_time
            Ttimes += 1
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\t{}'.format(mode,
                        epoch, batch_idx * batch_size, len(dataloader.dataset),
                        100. * batch_idx * batch_size / len(dataloader.dataset), args.tag))
        
            print('forward_passing_time: {}, processing_time: {}ms'.format(forward_passing_time*1000, processing_time*1000))
            Tprocessing_time += processing_time
            Tforward_passing_time += forward_passing_time

    def test(self, epoch):
        print("---------------DATALOADER: test_refine epoch", epoch, "------------------")
        self.train_val(epoch, mode='test', use_eval=True)


def main():
    if args.mode == 'pretest':
        for i in range(len(test_score_loader)):
            cur_loader = test_score_loader[i]
            cur_width = widths[i]
            logging.info("cur_width: " + cur_width)

            regionModule = RegionModule(resume_epoch, test_data_loader=cur_loader, cur_width=cur_width)
            regionModule.test(resume_epoch)

    elif args.mode == 'test':
        for i in range(len(test_score_loader)):
            cur_loader = test_score_loader[i]
            cur_width = widths[i]
            logging.info("cur_width: " + cur_width)

            refineModule = RefineModule(resume_epoch, test_data_loader=cur_loader, cur_width=cur_width)
            refineModule.test(resume_epoch)
    print('forward_passing_time: {}, processing_time: {}ms'.format(Tforward_passing_time/Ttimes*1000, Tprocessing_time/Ttimes*1000))
    logging.info("Mean Time")
    logging.info("forward_passing_time: {}".format(Tforward_passing_time/Ttimes*1000))
    logging.info("processing_time: {}".format(Tprocessing_time/Ttimes*1000))

if __name__ == "__main__":
    main()
