#!/usr/bin/env python
import argparse
import os.path as osp
import logging
import time
import sys, os
import open3d

sys.path.insert(0, os.path.abspath(osp.dirname(__file__) + '/..'))
import numpy as np

import torch
import torch.nn as nn

# from tdgpd.config import load_cfg_from_file
from tdgpd.yacs_config import load_cfg_from_file
from tdgpd.utils.logger import setup_logger
from tdgpd.utils.torch_utils import set_random_seed
from tdgpd.models.build_model import build_model
from tdgpd.solver import build_optimizer, build_scheduler
from tdgpd.dataset import build_data_loader
from tdgpd.utils.tensorboard_logger import TensorboardLogger
from tdgpd.utils.metric_logger import MetricLogger
from tdgpd.utils.file_logger import file_logger_noselect as file_logger_cls
import utils, model_utils
#torch.cuda.set_device(7)

topK=100
dis_thre=0.02
score_thres=[]
depth = 0.06
table_height = 0.75

Tforward_passing_time = 0 
Tprocessing_time = 0
Ttimes = 0
processing_time = 0

def parse_args():
    parser = argparse.ArgumentParser(description="S4G Training")
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="contrast/configs/curvature_model.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--tag', type=str, default='default')
    parser.add_argument('--eval-diffwidth', action='store_true')
    parser.add_argument('--eval-width', type=str, default='0.10')

    args = parser.parse_args()
    return args

def test_model(model,
                data_loader,
                curr_epoch,
                batch_size=1,
                output_dir="",
                log_tag='',
                eval_logger = None,
                eval_width=0.08, 
                eval_params=None,
                   ):
    logger = logging.getLogger("tdgpd.test")
    total_iteration = data_loader.dataset.__len__()
    model.eval()
    torch.set_grad_enabled(False)
    
    for iteration, data_batch in enumerate(data_loader):
        data_path  = data_batch['data_path']
        data_batch = {k: v.cuda() for k, v in data_batch.items() if isinstance(v, torch.Tensor)}
        print(data_batch['scene_points'].shape)
        start = time.time()
        # start_gpu = torch.cuda.Event(enable_timing=True)
        # end_gpu = torch.cuda.Event(enable_timing=True)
        # start_gpu.record()
        preds = model(data_batch)
        torch.cuda.synchronize()
        # end_gpu.record()
        # times = start_gpu.elapsed_time(end_gpu)
        # print(times)
        forward_passing_time = time.time() - start

        pc = data_batch['scene_points'].transpose(1,2)

        start = time.time()
        grasps = file_logger_cls(data_batch, preds, 
                        curr_epoch * total_iteration + (iteration + 1) * batch_size, 
                        output_dir, prefix="test", with_label=False, gpu_id=eval_params[-1])
        
        eval_logger.eval_notruth(pc, grasps, eval_params)
        torch.cuda.synchronize()
        processing_time = time.time() - start
        global Ttimes, Tprocessing_time, Tforward_passing_time
        Ttimes += 1       

        print('Test Epoch: {} [{}/{} ({:.0f}%)]\t{}'.format(
                curr_epoch, iteration * batch_size, len(data_loader.dataset),
                    100. * iteration * batch_size / len(data_loader.dataset), log_tag))
        print('forward_passing_time: {}, processing_time: {}ms'.format(forward_passing_time*1000, processing_time*1000))
        Tprocessing_time += processing_time
        Tforward_passing_time += forward_passing_time

def _map_model(cfg, model):
    gpu = cfg.SCHEDULER.GPU
    gpu_num = cfg.SCHEDULER.GPU_NUM
    gpus = cfg.SCHEDULER.GPUS

    device = torch.device("cuda:"+str(cfg.SCHEDULER.GPU))
    model = model.to(device)
    
    if gpu_num > 1:
        device_id = list(gpus)
        model = nn.DataParallel(model, device_ids=device_id)
    return model

def _load_model(cfg, model):
    load_path = cfg.MODEL.MODEL_PATH 
    resume_num = 0
    if load_path != "":
        gpu = cfg.SCHEDULER.GPU
        checkpoint = torch.load(load_path, map_location='cuda:{}'.format(gpu))
        
        new_model_dict = {}
        model_dict, resume_num = checkpoint['net'], checkpoint['epoch'] + 1

        for key in model_dict.keys():
            new_model_dict[key.replace("module.", "")] = model_dict[key]
        model.load_state_dict(new_model_dict)
    return model, resume_num


def test(cfg, output_dir="", eval_width=0.08, eval_params=None):
    logger = logging.getLogger("tdgpd.test")

    # build model
    #set_random_seed(cfg.RNG_SEED)
    model, _, _ = build_model(cfg)
    model, resume_num = _load_model(cfg, model)
    model     = _map_model(cfg, model)
    eval_mechine = utils.EvalNoTruth(topK)

    test_data_loader = build_data_loader(cfg, mode="test all") 
    start_epoch = resume_num 
    
    print("---------------------------test width {}-----------------------------".format(eval_width))
    logger.info("Test width {}".format(eval_width))
    test_model(model,
            data_loader=test_data_loader,
            curr_epoch=start_epoch,
            batch_size=cfg.TEST.BATCH_SIZE,
            output_dir=output_dir,
            log_tag=cfg.OUTPUT_TAG,
            eval_logger=eval_mechine,
            eval_width=eval_width, 
            eval_params=eval_params,
            )

def main():
    args = parse_args()
    
    cfg = load_cfg_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    output_tag = cfg.OUTPUT_TAG
    log_dir = os.path.abspath(os.path.join(output_dir, 'log'))
    
    utils.mkdir_output(log_dir, output_tag)

    logging.basicConfig(
        filename=os.path.join(log_dir, output_tag, 'train.log'),
        level=logging.INFO
    )
    logging.info(args)

    if args.eval_diffwidth:
        eval_widths = [0.06, 0.08, 0.10, 0.12]
        for eval_width in eval_widths:
            print("Test Width {}".format(eval_width))
            eval_params = [depth, eval_width, table_height, cfg.SCHEDULER.GPU]
            test(cfg, output_dir, eval_width, eval_params)
            
    else:
        eval_width = float(args.eval_width)
        eval_params    = [depth, eval_width, table_height, cfg.SCHEDULER.GPU]
        test(cfg, output_dir, eval_width, eval_params)
    print('forward_passing_time: {}, processing_time: {}ms'.format(Tforward_passing_time/Ttimes*1000, Tprocessing_time/Ttimes*1000))
    logging.info("Mean Time")
    logging.info("forward_passing_time: {}".format(Tforward_passing_time/Ttimes*1000))
    logging.info("processing_time: {}".format(Tprocessing_time/Ttimes*1000))

if __name__ == "__main__":
    main()

