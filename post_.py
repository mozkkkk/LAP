
import torch

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

import torch
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import scipy.ndimage as ndi

from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

def optimize_points(
    image: torch.Tensor, 
    points: torch.Tensor, 
    confidences: torch.Tensor, 
    local_region_size: int = 15,
    confidence_weight: float = 0.6,
    color_weight: float = 0.2,
    text_weight: float = 0.2,
    threshold: float = 0.5
):
    """
    使用ROIAlign方法提取局部图像特征，优化点选择。

    参数:
    - image: 输入图像张量，形状为 (1, 3, H, W)
    - points: 点坐标张量，形状为 (N, 2)
    - confidences: 原始点置信度，形状为 (N,)
    - local_region_size: 局部区域大小，必须为奇数 (默认5)
    - confidence_weight: 原始置信度权重 (默认0.7)
    - feature_weight: 图像特征权重 (默认0.3)
    - threshold: 点选择阈值 (默认0.6)

    返回:
    - optimized_points: 优化后的点坐标
    - final_confidences: 优化后的点置信度
    """
    # 输入验证
    assert local_region_size % 2 == 1, "局部区域大小必须为奇数"
    assert image.dim() == 4 and points.dim() == 2, "张量维度不匹配"

    # 确保所有张量在同一设备上
    device = points.device
    image = image.to(device)
    confidences = confidences.to(device)
    if confidences.shape[0] == 0:
        return points,confidences
    # 提取图像和点的基本信息
    H, W = image.shape[2], image.shape[3]
    N = points.shape[0]
    pad = local_region_size // 2

    # 准备ROI
    # 为每个点创建对应的ROI
    # ROI格式: [batch_index, x1, y1, x2, y2]
    batch_indices = torch.zeros(N, device=device)
    rois = torch.zeros(N, 5, device=device)
    
    # 设置ROI坐标 - 以点为中心的方形区域
    rois[:, 0] = batch_indices  # batch索引
    rois[:, 1] = points[:, 0] - pad  # x1
    rois[:, 2] = points[:, 1] - pad  # y1
    rois[:, 3] = points[:, 0] + pad  # x2
    rois[:, 4] = points[:, 1] + pad  # y2

    # 使用ROIAlign提取特征
    local_region = ops.roi_align(
        image, 
        rois, 
        output_size=(local_region_size, local_region_size),
        spatial_scale=1.0,  # 因为我们使用原始图像坐标
        sampling_ratio=-1  # 自动选择采样点数
    )

    # 计算特征相似性（使用更高效的距离计算）
    text_mean_feature = local_region.mean(dim=0, keepdim=True)
    text_local_region = local_region.view(local_region.shape[0], -1)  # (240, 147)
    text_mean_feature = text_mean_feature.view(1, -1)    # (1, 147)

    # 计算余弦相似度（在 dim=1 上计算）
    text_similarities = F.cosine_similarity(text_local_region, text_mean_feature, dim=1) 
    text_similarities = (text_similarities + 1) / 2

    color_local_features = local_region.mean(dim=(2, 3))

    # 计算特征相似性
    color_mean_feature = color_local_features.mean(dim=0, keepdim=True)
    distances = torch.norm(color_local_features - color_mean_feature, dim=1)
    color_similarities = 1.0 / (1.0 + distances)

    # 结合原始置信度和特征相似性
    new_confidences = (
        confidence_weight * confidences + 
        color_weight * color_similarities + 
        text_weight * text_similarities
    )

    # 根据阈值筛选点
    selected_mask = new_confidences > threshold
    optimized_points = points[selected_mask]
    final_confidences = new_confidences[selected_mask]

    return optimized_points, final_confidences

def deduplicate_points(points, confidence, distance_threshold=5):
    # 按置信度降序排序
    sorted_conf, sorted_indices = torch.sort(confidence, descending=True)
    sorted_points = points[sorted_indices].detach().cpu().numpy()
    
    # 使用KD树加速查询
    tree = cKDTree(sorted_points)
    processed = np.zeros(len(sorted_points), dtype=bool)
    keep_indices = []
    
    for i in range(len(sorted_points)):
        if not processed[i]:
            indices = tree.query_ball_point(sorted_points[i], distance_threshold, p=2)
            processed[indices] = True
            keep_indices.append(sorted_indices[i].item())  # 原始索引
    
    # 收集结果
    fused_points = points[keep_indices]
    fused_confidence = confidence[keep_indices]
    
    return fused_points,fused_confidence

def filter_by_thres(points, confidence, conf_threshold=0.5):
    mask = confidence >= conf_threshold
    points = points[mask]
    confidence = confidence[mask]
    return points,confidence
