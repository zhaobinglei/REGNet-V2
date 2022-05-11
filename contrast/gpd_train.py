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
parser.add_argument('--mode', choices=['train', 'validate', 'test'], required=True)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu-num', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpus', type=str, default='0,2,3')
parser.add_argument('--lr' , type=float, default=0.001) 

# parser.add_argument('--load-path', type=str, default='/data1/cxg6/Multigrasp/assets/models/pretrain_5120_0.15_mp_0.02_layer1_256/region_10.model')
parser.add_argument('--load-path', type=str, default='')
parser.add_argument('--data-path', type=str, default='/data2/zbl/dataset/0.08', help='data path')

parser.add_argument('--model-path', type=str, default='/data1/cxg6/Multigrasp/contrast/assets/models/', help='to saved model path')
parser.add_argument('--log-path', type=str, default='/data1/cxg6/Multigrasp/contrast/assets/log/', help='to saved log path')
parser.add_argument('--folder-name', type=str, default='/data1/cxg6/Multigrasp/test_file/real_data')

parser.add_argument('--file-name', type=str, default='')
parser.add_argument('--log-interval', type=int, default=1)
parser.add_argument('--save-interval', type=int, default=1)

parser.add_argument('--project_chann', type=int, default=3, choices=[3,12])
parser.add_argument('--eval-diffwidth', action='store_true')
parser.add_argument('--eval-width', type=str, default='0.10')

# torch.multiprocessing.set_start_method('spawn')

args = parser.parse_args()
args.cuda = args.cuda if torch.cuda.is_available else False

np.random.seed(int(time.time()))
random.seed(int(time.time()))
torch.cuda.set_device(args.gpu)
if args.cuda:
    torch.cuda.manual_seed(1)
if args.gpu_num > 1:
    torch.cuda.manual_seed_all(1)

logger = utils.mkdir_output(args.log_path, args.tag, args.mode, log_flag=True)
utils.mkdir_output(args.model_path, args.tag)

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
train_dataset = PointNetGPDDataset(
                all_points_num = all_points_num,
                path = args.data_path,
                tag = 'train',
                data_seed = 1,
                thresh_score = thresh_score,
                frame_num = 16
                )
val_dataset   = PointNetGPDDataset(
                all_points_num = all_points_num,
                path = args.data_path,
                tag = 'validate',
                data_seed = 1,
                thresh_score = thresh_score,
                frame_num = 16
                )
test_dataset  = PointNetGPDDataset(
                all_points_num = all_points_num,
                path = args.data_path,
                tag = 'test',
                data_seed = 1,
                thresh_score = thresh_score,
                frame_num = 16
                )
train_loader = utils.get_dataloader(train_dataset, args.batch_size, shuffle=True)
val_loader   = utils.get_dataloader(val_dataset, args.batch_size, shuffle=True)
test_loader  = utils.get_dataloader(test_dataset, args.batch_size, shuffle=True)
 
model_params   = [args.gpu_num, args.gpu, args.gpus, args.lr]
model_machine = model_utils.GPDModelInit(args.mode, args.load_path, model_params, grasp_points_num, args.project_chann)
model, resume_epoch, optimizer, scheduler = model_machine.construct_model()

table_height = 0.75
score_thre = 0.5
# eval_params
center_num = 1000
topK=100
dis_thre=0.02

class GPDModule():
    def __init__(self, start_epoch=0, end_epoch=1, train_data_loader=None, 
                            val_data_loader=None, test_data_loader=None, cur_width=None):
        self.start_epoch = start_epoch
        self.end_epoch   = end_epoch
        self.train_data_loader = train_data_loader 
        self.val_data_loader   = val_data_loader  
        self.test_data_loader  = test_data_loader
        self.eval_params    = [depth, width, table_height, args.gpu]
        if "test" in args.mode:
            self.eval_params[1]= float(args.eval_width)
        if cur_width is not None:
            self.eval_params[1]= float(cur_width)
        self.cur_width = cur_width

        self.gripper_params = gripper_params + [grasp_points_num, args.project_chann]
        self.log_machine    = model_utils.Logger(logger)
        self.eval_machine   = model_utils.Eval(self.log_machine, center_num, topK, dis_thre, start_epoch-1)
        
        self.saved_base_path = os.path.join(args.model_path, args.tag)

    def train_val(self, epoch, mode='train', use_eval=True):
        if mode == 'train':
            self.eval_machine.update_epoch()
            model.train()
            torch.set_grad_enabled(True)
            dataloader = self.train_data_loader   
            batch_size = args.batch_size

        else:
            model.eval()
            torch.set_grad_enabled(False)
            if mode == 'validate':
                dataloader = self.val_data_loader
            elif mode == 'test':
                dataloader = self.test_data_loader
            batch_size = args.batch_size

        acc_mean, precision_mean, recall_mean = 0, 0, 0
        num = 0
        for batch_idx, (pc, grasps, labels, data_path) in enumerate(dataloader):
            if mode == 'train':
                optimizer.zero_grad()
            data_path = np.array(data_path)
            if args.gpu > -1:
                pc, grasps, labels = pc.cuda(), grasps.cuda(), labels.cuda()

            # gripper_pc: [bs, close_region_points_num, 6]
            proj_pic, keep_idx = model_utils.get_gpd_projected_points(pc, grasps, self.gripper_params)
            pred = model(proj_pic.permute(0,3,1,2))
            pred_cls = pred.data.max(1, keepdim=True)[1].view(-1)
            labels   = labels.view(-1)[keep_idx]

            loss_cls = F.nll_loss(pred, labels)#.mean() 
            loss     = loss_cls
            if mode == 'train':
                loss.backward()
                optimizer.step()

            tp = ((pred_cls == 1) & (labels == 1) ).sum()
            tn = ((pred_cls == 0) & (labels == 0) ).sum()
            fp = ((pred_cls == 1) & (labels == 0) ).sum()
            fn = ((pred_cls == 0) & (labels == 1) ).sum()
            acc = (tp+tn) / (fp+fn+tp+tn)
            precision = tp / (tp+fp)
            recall    = tp / (tp+fn)
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}% Precision: {:.2f}% Recall: {:.2f}%\t{}'\
                    .format(mode, epoch, batch_idx * batch_size, len(dataloader.dataset), 100. * batch_idx * \
                    batch_size / len(dataloader.dataset), loss.data, acc*100, precision*100, recall*100, args.tag))
            
            data = (loss.data, (acc, precision, recall))
            self.log_machine.add_batch_loss(data, batch_idx + epoch * len(dataloader), mode)
            acc_mean += acc
            precision_mean += precision
            recall_mean += recall
            num += 1
            if use_eval:
                self.eval_machine.eval_batch(data_path, model, self.eval_params, \
                        self.gripper_params, batch_idx + epoch*len(dataloader), mode)

        if use_eval:
            self.eval_machine.eval_epoch(len(dataloader.dataset), mode, self.cur_width)
        
        acc_mean /= num
        precision_mean /= num
        recall_mean /= num
        logging.info('Acc: {:.2f}% Precision: {:.2f}% Recall: {:.2f}%'.format(acc_mean, precision_mean, recall_mean))
        if mode == 'train':
            scheduler.step()

    def pretrain(self, epoch):
        self.train_val(epoch, mode='train', use_eval=False)

    def validate(self, epoch):
        print("---------------validate epoch", epoch, "------------------")
        self.train_val(epoch, mode='validate', use_eval=False)

    def test(self, epoch):
        print("---------------DATALOADER: test epoch", epoch, "------------------")
        self.train_val(epoch, mode='test', use_eval=False)

    def train(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            print("---------------train epoch", epoch, "------------------")
            path  = os.path.join(self.saved_base_path, '{}.model'.format(epoch))
            logging.info("epoch"+str(epoch))
            self.pretrain(epoch)
            state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            torch.save(state, path)
            logging.info('validate')
            self.validate(epoch)
            logging.info('test')
            self.test(epoch)

def main():            
    if args.mode == 'train':
        refineModule = GPDModule(resume_epoch, args.epoch, train_loader, val_loader, test_loader, cur_width=0.08)
        refineModule.train()

    elif args.mode == 'validate':
        refineModule = GPDModule(resume_epoch, val_data_loader=val_loader, cur_width=0.08)
        refineModule.validate(resume_epoch)

    elif args.mode == 'test':
        refineModule = GPDModule(resume_epoch, test_data_loader=test_loader, cur_width=0.08)
        refineModule.test(resume_epoch)

if __name__ == "__main__":
    main()

