# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from util.misc import NestedTensor
import numpy as np
import time
import torchvision.transforms as standard_transforms
import cv2
from tqdm import tqdm
from post_ import deduplicate_points,filter_by_thres,optimize_points
import torch.nn.functional as F
from dis_compute import chamfer_distance,hausdorff_distance

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def vis(samples, targets, pred, vis_dir, des=None):
    '''
    samples -> tensor: [batch, 3, H, W]
    targets -> list of dict: [{'points':[], 'image_id': str}]
    pred -> list: [num_preds, 2]
    '''
    gts = [t['point'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])
    # draw one by one
    for idx in range(samples.shape[0]):
        sample = restore_transform(samples[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_gt = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        sample_pred = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        max_len = np.max(sample_gt.shape)

        size = 2
        # draw gt
        for t in gts[idx]:
            sample_gt = cv2.circle(sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
        # draw predictions
        for p in pred[idx]:
            sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        name = targets[idx]['image_id']
        # save the visualized images
        if des is not None:
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_gt.jpg'.format(int(name), 
                                                des, len(gts[idx]), len(pred[idx]))), sample_gt)
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_pred.jpg'.format(int(name), 
                                                des, len(gts[idx]), len(pred[idx]))), sample_pred)
        else:
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_gt.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_gt)
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_pred.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_pred)

# the training routine
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # iterate all training samples
    for samples, targets in tqdm(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # forward
        outputs = model(samples)
        # calc the losses
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce all losses
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        # backward
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # update logger
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def calculate_iou_batch(tensor1, tensor2, batch=True, mean=True):
    """
    计算两个张量之间的 IoU（交并比），支持形状为 BxHxW 的批量张量。
    
    参数:
        tensor1 (torch.Tensor): 第一个张量，形状为 BxHxW 或 HxW。
        tensor2 (torch.Tensor): 第二个张量，形状为 BxHxW 或 HxW。
        batch (bool): 如果为 True，则返回批量平均 IoU；否则返回每个样本的 IoU。
    
    返回:
        如果 batch=True，返回标量（批量平均 IoU）。
        如果 batch=False，返回一个包含每个样本 IoU 的张量。
    """
    # 确保输入张量的形状一致
    if tensor1.shape != tensor2.shape:
        raise ValueError("输入张量的形状必须一致")
    
    # 如果输入是二维张量（HxW），增加一个批次维度
    if len(tensor1.shape) == 2:
        tensor1 = tensor1.unsqueeze(0)  # 变为 1xHxW
        tensor2 = tensor2.unsqueeze(0)  # 变为 1xHxW
    
    # 计算交集和并集
    intersection = (tensor1 & tensor2).sum(dim=(1, 2)).float()  # 形状: [B]
    union = (tensor1 | tensor2).sum(dim=(1, 2)).float()  # 形状: [B]
    
    # 处理零除问题
    iou = torch.zeros_like(intersection)  # 初始化 IoU 为 0
    non_zero_union = union != 0  # 找到并集不为零的样本
    iou[non_zero_union] = intersection[non_zero_union] / union[non_zero_union]
    
    # 如果交集和并集都为零，则 IoU 为 1
    both_zero = (intersection == 0) & (union == 0)
    iou[both_zero] = 1.0
    
    # 根据 batch 参数返回结果
    if batch:
        if mean==False:
            return iou
        return iou.mean().item()  # 返回批量平均 IoU
    else:
        return iou.tolist() 

def calculate_iou(tensor1, tensor2):
    # 计算交集
    intersection = (tensor1 & tensor2).sum().float()
    # 计算并集
    union = (tensor1 | tensor2).sum().float()
    # 避免除以零的情况
    if union == 0:
        return torch.tensor(1.0) if intersection == 0 else torch.tensor(0.0)
    # 计算 IoU
    iou = intersection / union
    return iou.item()



# the inference routine
@torch.no_grad()
def evaluate_crowd_no_overlap(model, data_loader, device, vis_dir=None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE
    maes = []
    mses = []
    diss1 = []
    diss2 = []
    hyper_param = {"distance_threshold":[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0],"radius":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]}
    ious = []
    for samples, targets in tqdm(data_loader):
        samples = samples.to(device)
        _, _, h, w = samples.shape
        
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]


        outputs_points,outputs_scores = filter_by_thres(outputs_points,outputs_scores,0.3)
        outputs_points, outputs_scores = deduplicate_points(outputs_points,outputs_scores,2.5)
        outputs_points, outputs_scores = filter_by_thres(outputs_points, outputs_scores,0.5)
        outputs_points,outputs_scores = optimize_points(samples,outputs_points,outputs_scores)
        
        '''
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits_aux'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points_aux'][0]
        '''
        #outputs_points,outputs_scores = filter_points_in_mask(outputs_mask,outputs_points,outputs_scores)
        dis1 = chamfer_distance(outputs_points,targets[0]['point'].to(outputs_points.device))
        dis2 = hausdorff_distance(outputs_points,targets[0]['point'].to(outputs_points.device))
        
        gt_cnt = targets[0]['point'].shape[0]
        points = outputs_points.detach().cpu().numpy().tolist()
        predict_cnt = int(outputs_scores.shape[0])
        # if specified, save the visualized images
        if vis_dir is not None: 
            vis(samples, targets, [points], vis_dir)
        # accumulate MAE, MSE
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        maes.append(float(mae))
        mses.append(float(mse))
        diss1.append(float(dis1))
        diss2.append(float(dis2))
    # calc MAE, MSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    dis1 = np.mean(diss1)
    dis2 = np.mean(diss2)
    print()
    print("chamfer_distance:", dis1)
    print()
    
    print()
    print("hausdorff_distance:", dis2)
    print()

    

    return mae, mse, dis1, dis2