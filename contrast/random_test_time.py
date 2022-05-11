import argparse
import os, sys
import pickle

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

import torch, random
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import open3d
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import time

from gpd.dataset import PointNetGPDDataset
from dataset_utils.eval_score.eval import eval_test, eval_validate
import utils
import model_utils
import glob
import logging

parser = argparse.ArgumentParser(description='GripperRegionNetwork')
parser.add_argument('--tag', type=str, default='default')
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=41)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu-num', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpus', type=str, default='0,2,3')
parser.add_argument('--lr' , type=float, default=0.005) 

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

np.random.seed(1)#int(time.time()))
random.seed(1)
torch.cuda.set_device(args.gpu)
if args.cuda:
    torch.cuda.manual_seed(1)
if args.gpu_num > 1:
    torch.cuda.manual_seed_all(1)

logger = utils.mkdir_output(args.log_path, args.tag, 'test', log_flag=True)

log_dir = os.path.join(args.log_path, args.tag)
logging.basicConfig(
    filename=os.path.join(log_dir, 'train.log'),
    level=logging.INFO
)
logging.info(args)

all_points_num = 51200
grasp_points_num = 1000
thresh_score  = 0.6 

width = float(args.data_path.split('/')[-1]) if '0.' in args.data_path else 0.080
height, depth = 0.010, 0.06
gripper_params = [width, height, depth]

table_height = 0.75
score_thre = 0.5
# eval_params
center_num = 1000
topK=100
dis_thre=0.02

Tforward_passing_time = 0 
Tprocessing_time = 0
Ttimes = 0
class RandomModule():
    def __init__(self, train_data_loader=None, 
                    val_data_loader=None, test_data_loader=None, cur_width=None):
        self.train_data_loader = train_data_loader 
        self.val_data_loader   = val_data_loader  
        self.test_data_loader  = test_data_loader
        self.eval_params    = [depth, args.eval_width, table_height, args.gpu]
        if cur_width is not None:
            self.eval_params[1]= float(cur_width)
        self.cur_width = cur_width

        self.gripper_params = gripper_params
        # self.log_machine    = model_utils.Logger(logger)
        # self.eval_machine   = model_utils.Eval(self.log_machine, center_num, topK, dis_thre, 0)
        self.eval_machine   = model_utils.EvalNoTruth(center_num, topK)
        
        self.saved_base_path = os.path.join(args.model_path, args.tag)

    def train_val(self, mode='train', use_eval=True):
        if mode == 'train':
            self.eval_machine.update_epoch()
            torch.set_grad_enabled(True)
            dataloader = self.train_data_loader   
            batch_size = args.batch_size

        else:
            torch.set_grad_enabled(False)
            if mode == 'validate':
                dataloader = self.val_data_loader
            elif mode == 'test':
                dataloader = self.test_data_loader
            batch_size = args.batch_size

        for batch_idx, (pc, grasps, labels, data_path) in enumerate(dataloader):
            data_path = np.array(data_path)

            print('[{}/{} ({:.0f}%)]\t{}'\
                    .format(batch_idx * batch_size, len(dataloader.dataset), 100. * batch_idx * \
                    batch_size / len(dataloader.dataset), args.tag))
            if use_eval:
                forward_passing_time, processing_time = self.eval_machine.eval_notruth(\
                                            None, pc, self.eval_params, self.gripper_params)

                global Ttimes, Tprocessing_time, Tforward_passing_time
                Ttimes += 1
                print('forward_passing_time: {}, processing_time: {}ms'.format(forward_passing_time*1000, processing_time*1000))
                Tprocessing_time += processing_time
                Tforward_passing_time += forward_passing_time

    def test(self):
        print("-------------------------test width {}--------------------------".format(self.cur_width))
        logging.info("Test width {}".format(self.cur_width))
        self.train_val(mode='test', use_eval=True)

def main():            
    float_str_map = {0.06:'0.06', 0.08:'0.08', 0.10:'0.10', 0.12:'0.12'}
    if args.eval_diffwidth:
        eval_widths = [0.06, 0.08, 0.10, 0.12]
        for eval_width in eval_widths:
            if eval_width in float_str_map:
                str_width = float_str_map[eval_width]
            else:
                str_width = '0.08'
            test_dataset  = PointNetGPDDataset(
                            all_points_num = all_points_num,
                            path = os.path.join(args.data_path, str_width),
                            tag = 'test',
                            data_seed = 1,
                            thresh_score = thresh_score,
                            frame_num = 16
                            )
            test_loader  = utils.get_dataloader(test_dataset, args.batch_size, shuffle=True)
 
            model = RandomModule(test_data_loader=test_loader, cur_width=eval_width)
            model.test()

    else:
        if args.eval_width in float_str_map:
            str_width = float_str_map[args.eval_width]
        else:
            str_width = '0.08'
        test_dataset  = PointNetGPDDataset(
                        all_points_num = all_points_num,
                        path = os.path.join(args.data_path, str_width),
                        tag = 'test',
                        data_seed = 1,
                        thresh_score = thresh_score,
                        frame_num = 16
                        )
        test_loader  = utils.get_dataloader(test_dataset, args.batch_size, shuffle=True)
        model = RandomModule(test_data_loader=test_loader, cur_width=args.eval_width)
        model.test()

    print('final forward_passing_time: {}, processing_time: {}ms'.format(Tforward_passing_time/Ttimes*1000, Tprocessing_time/Ttimes*1000))
    logging.info("Mean Time")
    logging.info("forward_passing_time: {}".format(Tforward_passing_time/Ttimes*1000))
    logging.info("processing_time: {}".format(Tprocessing_time/Ttimes*1000))

if __name__ == "__main__":
    main()

