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
from compute import chamfer_distance, hausdorff_distance, compute_tp
from util.misc import NestedTensor
import numpy as np

import torchvision.transforms as standard_transforms
import cv2
from tqdm import tqdm
from post_ import deduplicate_points, filter_by_thres, optimize_points, visualize_postproc_effects
import torch.nn.functional as F
import random


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
                                                                                  des, len(gts[idx]), len(pred[idx]))),
                        sample_gt)
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_pred.jpg'.format(int(name),
                                                                                    des, len(gts[idx]),
                                                                                    len(pred[idx]))), sample_pred)
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


def scale_coords_back(pred_points, src_size, dst_size):
    """
    将预测点从缩放后坐标系映射回原始坐标系
    :param pred_points: 模型输出的坐标 (B, N, 2) 或 (N, 2)
    :param src_size: 原始图像尺寸 (H_src, W_src)
    :param dst_size: 缩放后图像尺寸 (H_dst, W_dst)
    :return: 原始坐标系坐标 (B, N, 2)
    """
    H_src, W_src = src_size
    H_dst, W_dst = dst_size

    # 计算宽高缩放比例
    scale_w = W_src / W_dst
    scale_h = H_src / H_dst

    # 创建缩放系数张量 (1,1,2) 便于广播
    scale_factors = torch.tensor(
        [[scale_w, scale_h]],
        dtype=torch.float32,
        device=pred_points.device
    )


    # 坐标映射
    orig_points = pred_points * scale_factors

    return orig_points

# the inference routine
@torch.no_grad()
def evaluate_crowd_no_overlap(model, data_loader, device, vis_dir=None, resize=True):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE
    maes = []
    mses = []
    diss1 = []
    diss2 = []
    tp_sum_4 = 0
    gt_sum = 0
    et_sum = 0
    tp_sum_8 = 0
    times_cd = []
    times_hd = []
    times_tp_4 = []
    times_tp_8 = []
    #vis_flag = 0

    for samples, targets in tqdm(data_loader):
        samples = samples.to(device)
        _, _, h, w = samples.shape

        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]


        if resize:
            src = targets[0]['ori_size']
            if h!=src[0] or w!=src[1]:
                outputs_points = scale_coords_back(outputs_points,src,[h,w])

        #if random.random() < 0.3:
            #visualize_postproc_effects(samples, outputs_points, outputs_scores, targets[0]['point'], save_path=f"t_{vis_flag}.png")
            #vis_flag += 1

        outputs_points,outputs_scores = filter_by_thres(outputs_points,outputs_scores,0.3)
        outputs_points, outputs_scores = deduplicate_points(outputs_points,outputs_scores,2)
        outputs_points, outputs_scores = filter_by_thres(outputs_points, outputs_scores, 0.5)
        outputs_points,outputs_scores = optimize_points(samples,outputs_points,outputs_scores)

        '''
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits_aux'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points_aux'][0]
        '''
        # outputs_points,outputs_scores = filter_points_in_mask(outputs_mask,outputs_points,outputs_scores)
        dis1, mean_time_chamfer_dis = chamfer_distance(outputs_points, targets[0]['point'].to(outputs_points.device))
        dis2, mean_time_hausdorff_dis = hausdorff_distance(outputs_points,
                                                           targets[0]['point'].to(outputs_points.device))
        times_cd.append(mean_time_chamfer_dis)
        times_hd.append(mean_time_hausdorff_dis)

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

        tp_4, mean_time_tp_4 = compute_tp(points, targets[0]['point'], 4)
        tp_8, mean_time_tp_8 = compute_tp(points, targets[0]['point'], 8)
        tp_sum_4 += tp_4
        gt_sum += gt_cnt
        et_sum += predict_cnt
        tp_sum_8 += tp_8
        times_tp_4.append(mean_time_tp_4)
        times_tp_8.append(mean_time_tp_8)

    # calc MAE, MSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    dis1 = np.mean(diss1)
    dis2 = np.mean(diss2)

    ap_4 = tp_sum_4 / float(et_sum + 1e-10)
    ar_4 = tp_sum_4 / float(gt_sum + 1e-10)
    f1_4 = 2 * ap_4 * ar_4 / (ap_4 + ar_4 + 1e-10)
    ap_8 = tp_sum_8 / float(et_sum + 1e-10)
    ar_8 = tp_sum_8 / float(gt_sum + 1e-10)
    f1_8 = 2 * ap_8 * ar_8 / (ap_8 + ar_8 + 1e-10)
    local_result = {'ap_4': ap_4, 'ar_4': ar_4, 'f1_4': f1_4, 'ap_8': ap_8, 'ar_8': ar_8, 'f1_8': f1_8}

    times = {'time_cd': np.mean(times_cd), 'time_hd': np.mean(times_hd), 'time_tp_4': np.mean(times_tp_4),
             'time_tp_8': np.mean(times_tp_8)}

    return mae, mse, dis1, dis2, local_result, times
