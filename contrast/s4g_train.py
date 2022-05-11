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
from tdgpd.utils.checkpoint import CheckPointer
from tdgpd.dataset import build_data_loader
from tdgpd.utils.tensorboard_logger import TensorboardLogger
from tdgpd.utils.metric_logger import MetricLogger
from tdgpd.utils.file_logger import file_logger_noselect as file_logger_cls
# from tdgpd.utils.file_logger_cls import file_logger as file_logger_cls
import utils, model_utils

topK=100
dis_thre=0.02
eval_width = 0.08
depth = 0.06
table_height = 0.75
eval_params    = [depth, eval_width, table_height, 0]
# eval_params    = [depth, None, table_height, args.gpu]
# if "test" in args.mode and args.eval_use_diffwidth:
#     self.eval_params[1]= args.eval_width

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
    parser.add_argument('--gpu-num', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpus', type=str, default='0,2,3')
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--tag', type=str, default='default')
    parser.add_argument('--eval-diffwidth', action='store_true')
    parser.add_argument('--eval-width', type=str, default='0.10')

    args = parser.parse_args()
    return args


def train_model(model,
                loss_fn,
                metric_fn,
                data_loader,
                optimizer,
                curr_epoch,
                tensorboard_logger,
                batch_size=1,
                log_period=1,
                file_log_period=100,
                output_dir="",
                epoch=0,
                log_tag='',
                eval_logger = None,
                ):
    logger = logging.getLogger("tdgpd.train")
    meters = MetricLogger(delimiter="  ")
    model.train()
    end = time.time()
    total_iteration = data_loader.dataset.__len__()
    cls_logits = []
    cls_labels = []
    mov_logits = []
    mov_labels = []
    for iteration, data_batch in enumerate(data_loader):
        data_batch = {k: v.cuda() for k, v in data_batch.items() if isinstance(v, torch.Tensor)}
        data_time = time.time() - end
        
        preds = model(data_batch)
        for k, v in preds.items():
            if "logits" in k:
                if "movable" in k:
                    mov_logits.append(v.detach().cpu().numpy())
                else:
                    cls_logits.append(v.detach().cpu().numpy())

        for k, v in data_batch.items():
            if "labels" in k:
                if "movable" in k:
                    mov_labels.append(v.cpu().numpy())
                if "scene_score_logits" in preds.keys() and "scene_score_labels" in k:
                    cls_labels.append(v.cpu().numpy())
                if "local_search_logits" in preds.keys() and "scored_grasp_labels" in k:
                    cls_labels.append(v.cpu().numpy())
                if "grasp_logits" in preds.keys() and "grasp_score_labels" in k:
                    cls_labels.append(v.cpu().numpy())
        optimizer.zero_grad()

        loss_dict = loss_fn(preds, data_batch)
        metric_dict = metric_fn(preds, data_batch)
        losses = sum(loss_dict.values())
        meters.update(loss=losses, **loss_dict, **metric_dict)

        losses.mean().backward()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{}'.format(
                epoch, iteration * batch_size, len(data_loader.dataset),
                    100. * iteration * batch_size / len(data_loader.dataset), losses.mean().data, log_tag))
        
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        if iteration % log_period == 0:
            # logger.info(
            #     meters.delimiter.join(
            #         [
            #             "EPOCH: {epoch:2d}",
            #             "iter: {iter:4d}",
            #             "{meters}",
            #             "lr: {lr:.2e}",
            #             "max mem: {memory:.0f}",
            #         ]
            #     ).format(
            #         epoch=curr_epoch,
            #         iter=iteration,
            #         meters=str(meters),
            #         lr=optimizer.param_groups[0]["lr"],
            #         memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
            #     )
            # )
            tensorboard_logger.add_scalars(loss_dict, curr_epoch * total_iteration + (iteration + 1) * batch_size,
                                           prefix="train")
            tensorboard_logger.add_scalars(metric_dict, curr_epoch * total_iteration + (iteration + 1) * batch_size,
                                           prefix="train")

        # if iteration % file_log_period == 0:
        #     file_logger_cls(data_batch, preds, curr_epoch * total_iteration + (iteration + 1) * batch_size, output_dir,
        #                     prefix="train", with_label=False)
    cls_logits = np.concatenate(cls_logits, axis=0)
    preds = np.argmax(cls_logits, axis=1)
    cls_labels = np.concatenate(cls_labels)

    ####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # if len(cls_logits.shape) == 2:
    #     score_classes = cls_logits.shape[-1]
    #     for i in range(score_classes):
    #         pred = preds == i
    #         gt = cls_labels == i
    #         true_pos = np.logical_and(pred, gt)
    #         if np.sum(pred) == 0:
    #             precision = recall = 0
    #         else:
    #             precision = np.sum(true_pos).astype(float) / max(np.sum(pred).astype(float), 1e-4)
    #             recall = np.sum(true_pos).astype(float) / max(np.sum(gt).astype(float), 1e-4)
    #         logger.info("Class {}: number: {}, precision: {:.2f}%, recall: {:.2f}%".format(
    #             i, np.sum(gt), precision * 100, recall * 100
    #         ))
    # else:
    #     score_classes = cls_logits.shape[1]
    #     for i in range(score_classes):
    #         pred = preds == i
    #         gt = cls_labels == i
    #         true_pos = np.logical_and(pred, gt)
    #         if np.sum(pred) == 0:
    #             precision = recall = 0
    #         else:
    #             precision = np.sum(true_pos).astype(float) / max(np.sum(pred).astype(float), 1e-4)
    #             recall = np.sum(true_pos).astype(float) / max(np.sum(gt).astype(float), 1e-4)
    #         logger.info("Class {}: number: {}, precision: {:.2f}%, recall: {:.2f}%".format(
    #             i, np.sum(gt), precision * 100, recall * 100
    #         ))

    # if len(mov_logits) > 0:
    #     mov_logits = np.concatenate(mov_logits, axis=0)
    #     # preds = np.argmax(mov_logits, axis=1)
    #     preds = (mov_logits > 0.5).astype(np.int)
    #     mov_labels = np.concatenate(mov_labels, axis=0).astype(np.int)
    #     for i in range(2):
    #         pred = preds == i
    #         gt = mov_labels == i
    #         true_pos = np.logical_and(pred, gt)
    #         if np.sum(pred) == 0:
    #             precision = recall = 0
    #         else:
    #             precision = np.sum(true_pos).astype(float) / max(np.sum(pred).astype(float), 1e-4)
    #             recall = np.sum(true_pos).astype(float) / max(np.sum(gt).astype(float), 1e-4)
    #         logger.info("Movable Class {}: number: {}, precision: {:.2f}%, recall: {:.2f}%".format(
    #             i, np.sum(gt), precision * 100, recall * 100
    #         ))
    ####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # if len(mov_logits.shape) == 2:
    #     score_classes = mov_logits.shape[-1]
    #     for i in range(score_classes):
    #         pred = preds == i
    #         gt = mov_labels == i
    #         true_pos = np.logical_and(pred, gt)
    #         if np.sum(pred) == 0:
    #             precision = recall = 0
    #         else:
    #             precision = np.sum(true_pos).astype(float) / max(np.sum(pred).astype(float), 1e-4)
    #             recall = np.sum(true_pos).astype(float) / max(np.sum(gt).astype(float), 1e-4)
    #         logger.info("Movable Class {}: number: {}, precision: {:.2f}%, recall: {:.2f}%".format(
    #             i, np.sum(gt), precision * 100, recall * 100
    #         ))
    # else:
    #     score_classes = mov_logits.shape[1]
    #     for i in range(score_classes):
    #         pred = preds == i
    #         gt = mov_labels == i
    #         true_pos = np.logical_and(pred, gt)
    #         if np.sum(pred) == 0:
    #             precision = recall = 0
    #         else:
    #             precision = np.sum(true_pos).astype(float) / max(np.sum(pred).astype(float), 1e-4)
    #             recall = np.sum(true_pos).astype(float) / max(np.sum(gt).astype(float), 1e-4)
    #         logger.info("Movable Class {}: number: {}, precision: {:.2f}%, recall: {:.2f}%".format(
    #             i, np.sum(gt), precision * 100, recall * 100
    #         ))

    return meters


def validate_model(model,
                   loss_fn,
                   metric_fn,
                   data_loader,
                   curr_epoch,
                   tensorboard_logger,
                   batch_size=1,
                   log_period=1,
                   file_log_period=100,
                   output_dir="",
                   epoch=0,
                   log_tag='',
                   eval_logger = None,
                   ):
    logger = logging.getLogger("tdgpd.validate")
    meters = MetricLogger(delimiter="  ")
    model.train()
    end = time.time()
    total_iteration = data_loader.dataset.__len__()
    cls_logits = []
    cls_labels = []
    mov_logits = []
    mov_labels = []

    with torch.no_grad():
        for iteration, data_batch in enumerate(data_loader):
            data_path  = data_batch['data_path']
            data_batch = {k: v.cuda() for k, v in data_batch.items() if isinstance(v, torch.Tensor)}
            data_time = time.time() - end

            preds = model(data_batch)
            for k, v in preds.items():
                if "logits" in k:
                    if "movable" in k:
                        mov_logits.append(v.detach().cpu().numpy())
                    else:
                        cls_logits.append(v.detach().cpu().numpy())

            for k, v in data_batch.items():
                if "labels" in k:
                    if "movable" in k:
                        mov_labels.append(v.cpu().numpy())
                    if "scene_score_logits" in preds.keys() and "scene_score_labels" in k:
                        cls_labels.append(v.cpu().numpy())
                    if "local_search_logits" in preds.keys() and "scored_grasp_labels" in k:
                        cls_labels.append(v.cpu().numpy())
                    if "grasp_logits" in preds.keys() and "grasp_score_labels" in k:
                        cls_labels.append(v.cpu().numpy())

            loss_dict = loss_fn(preds, data_batch)
            metric_dict = metric_fn(preds, data_batch)
            losses = sum(loss_dict.values())
            meters.update(loss=losses, **loss_dict, **metric_dict)
            print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{}'.format(
                    epoch, iteration * batch_size, len(data_loader.dataset),
                        100. * iteration * batch_size / len(data_loader.dataset), losses.mean().data, log_tag))
        

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            if iteration % log_period == 0:
                # logger.info(
                #     meters.delimiter.join(
                #         [
                #             "EPOCH: {epoch:2d}",
                #             "iter: {iter:4d}",
                #             "{meters}",
                #         ]
                #     ).format(
                #         epoch=curr_epoch,
                #         iter=iteration,
                #         meters=str(meters),
                #     )
                # )
                tensorboard_logger.add_scalars(meters.meters,
                                               curr_epoch * total_iteration + (iteration + 1) * batch_size,
                                               prefix="valid")

            # if iteration % file_log_period == 0:
            #     file_logger_cls(data_batch, preds, curr_epoch * total_iteration + (iteration + 1) * batch_size,
            #                     output_dir,
            #                     prefix="valid", with_label=False)
            grasps = file_logger_cls(data_batch, preds, 
                                    curr_epoch * total_iteration + (iteration + 1) * batch_size, 
                                    output_dir, prefix="validation", with_label=False)
            
            eval_logger.eval_batch(data_path, grasps, eval_params)
            # for i in range(len(data_path)):
            #     pc = np.array(data_batch['scene_points'][i].T.cpu())
            #     eval_logger.eval_notruth(pc, None, grasps, eval_params, score_thre=None, grasp_save_path='{}.p'.format(i))
    
    eval_logger.eval_epoch(len(data_loader.dataset), "validate", eval_width)

    return meters

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


def train(cfg, output_dir=""):
    logger = logging.getLogger("tdgpd.trainer")

    # build model
    set_random_seed(cfg.RNG_SEED)
    model, loss_fn, metric_fn = build_model(cfg)
    model, resume_num = _load_model(cfg, model)
    model     = _map_model(cfg, model)
    loss_fn   = _map_model(cfg, loss_fn)
    metric_fn = _map_model(cfg, metric_fn)

    # build optimizer
    optimizer = build_optimizer(cfg, model)

    # build lr scheduler
    scheduler = build_scheduler(cfg, optimizer, resume_num)
    eval_mechine = model_utils.Eval_S4G(topK, dis_thre, resume_num)

    # # build checkpointer
    # checkpointer = CheckPointer(model,
    #                             optimizer=optimizer,
    #                             scheduler=scheduler,
    #                             save_dir=output_dir,
    #                             logger=logger)

    # # checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, resume=cfg.AUTO_RESUME)
    # ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # build data loader
    train_data_loader = build_data_loader(cfg, mode="train")
    val_period = cfg.TRAIN.VAL_PERIOD
    val_data_loader  = build_data_loader(cfg, mode="val") if val_period > 0 else None
    test_data_loader = build_data_loader(cfg, mode="test") 
    
    # build tensorboard logger (optionally by comment)
    tensorboard_logger = TensorboardLogger(os.path.join(output_dir, 'log', cfg.OUTPUT_TAG))

    # train
    max_epoch = cfg.SCHEDULER.MAX_EPOCH
    start_epoch = resume_num # checkpoint_data.get("epoch", 0)
    # best_metric_name = "best_{}".format(cfg.TRAIN.VAL_METRIC)
    # best_metric = checkpoint_data.get(best_metric_name, None)
    best_metric = None
    logger.info("Start training from epoch {}".format(start_epoch))
    for epoch in range(start_epoch, max_epoch):
        cur_epoch = epoch
        scheduler.step()
        start_time = time.time()
        train_meters = train_model(model,
                                  loss_fn,
                                  metric_fn,
                                  data_loader=train_data_loader,
                                  optimizer=optimizer,
                                  curr_epoch=epoch,
                                  tensorboard_logger=tensorboard_logger,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  log_period=cfg.TRAIN.LOG_PERIOD,
                                  file_log_period=cfg.TRAIN.FILE_LOG_PERIOD,
                                  output_dir=output_dir,
                                  epoch=cur_epoch,
                                  log_tag=cfg.OUTPUT_TAG,
                                  eval_logger=None,
                                  )
        epoch_time = time.time() - start_time
        logger.info("Epoch[{}]-Train {}  total_time: {:.2f}s".format(
                    cur_epoch, train_meters.summary_str, epoch_time))

        # checkpoint
        checkpoint_state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':cur_epoch}
        torch.save(checkpoint_state, os.path.abspath(os.path.join(output_dir, 'models', \
                                           cfg.OUTPUT_TAG, '{}.model'.format(cur_epoch))))

        # if cur_epoch % ckpt_period == 0 or cur_epoch == max_epoch:
        #     checkpoint_data["epoch"] = cur_epoch
        #     checkpoint_data[best_metric_name] = best_metric
        #     checkpointer.save("model_{:03d}".format(cur_epoch), **checkpoint_data)

        # validate
        if val_period < 1:
            continue
        if cur_epoch % val_period == 0 or cur_epoch == max_epoch:
            print("-----------------------------validate---------------------------------")
            val_meters = validate_model(model,
                                        loss_fn,
                                        metric_fn,
                                        data_loader=val_data_loader,
                                        curr_epoch=epoch,
                                        tensorboard_logger=tensorboard_logger,
                                        batch_size=cfg.TEST.BATCH_SIZE,
                                        log_period=cfg.TEST.LOG_PERIOD,
                                        file_log_period=cfg.TEST.FILE_LOG_PERIOD,
                                        output_dir=output_dir,
                                        epoch=cur_epoch,
                                        log_tag=cfg.OUTPUT_TAG,
                                        eval_logger=eval_mechine,
                                        )
            logger.info("Epoch[{}]-Val {}".format(cur_epoch, val_meters.summary_str))

            # best validation
            cur_metric = val_meters.meters[cfg.TRAIN.VAL_METRIC].global_avg
            if best_metric is None or cur_metric > best_metric:
               best_metric = cur_metric
               # checkpoint_data["epoch"] = cur_epoch
               # checkpoint_data[best_metric_name] = best_metric
               # checkpointer.save("model_best", **checkpoint_data)
               # checkpointer.save("model_best_"+str(cur_epoch), **checkpoint_data)

               torch.save(checkpoint_state, os.path.abspath(os.path.join(output_dir, 'models', \
                                                   cfg.OUTPUT_TAG, 'model_best.model'.format(cur_epoch))))

            test_meters = validate_model(model,
                                        loss_fn,
                                        metric_fn,
                                        data_loader=test_data_loader,
                                        curr_epoch=epoch,
                                        tensorboard_logger=tensorboard_logger,
                                        batch_size=cfg.TEST.BATCH_SIZE,
                                        log_period=cfg.TEST.LOG_PERIOD,
                                        file_log_period=cfg.TEST.FILE_LOG_PERIOD,
                                        output_dir=output_dir,
                                        epoch=cur_epoch,
                                        log_tag=cfg.OUTPUT_TAG,
                                        eval_logger=eval_mechine,
                                        )
            logger.info("Epoch[{}]-Val {}".format(cur_epoch, test_meters.summary_str))

    logger.info("Best val-{} = {}".format(cfg.TRAIN.VAL_METRIC, best_metric))
    return model

def main():
    args = parse_args()
    
    cfg = load_cfg_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    output_tag = cfg.OUTPUT_TAG
    log_dir = os.path.abspath(os.path.join(output_dir, 'log'))
    model_dir = os.path.abspath(os.path.join(output_dir, 'models'))
    
    utils.mkdir_output(log_dir, output_tag)
    utils.mkdir_output(model_dir, output_tag)

    logging.basicConfig(
        filename=os.path.join(log_dir, output_tag, 'train.log'),
        level=logging.INFO
    )
    logging.info(args)

    train(cfg, output_dir)


if __name__ == "__main__":
    main()

