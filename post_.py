from matplotlib.gridspec import GridSpec
from scipy.spatial import cKDTree


import torch.nn.functional as F
import torchvision.ops as ops
from matplotlib.patches import Patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


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
        return points, confidences
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
    text_mean_feature = text_mean_feature.view(1, -1)  # (1, 147)

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

    return fused_points, fused_confidence


def filter_by_thres(points, confidence, conf_threshold=0.5):
    mask = confidence >= conf_threshold
    points = points[mask]
    confidence = confidence[mask]
    return points, confidence

def optimize_points_for_study(
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
        return points, confidences
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
    text_mean_feature = text_mean_feature.view(1, -1)  # (1, 147)

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

    return optimized_points, final_confidences, selected_mask


def visualize_postproc_effects(
        image: torch.Tensor,
        points: torch.Tensor,
        confidences: torch.Tensor,
        gt_points: torch.Tensor = None,
        save_path: str = "./postproc_comparison.png",
        figsize: tuple = (20, 14),  # 缩小画布高度
        title_pad: float = 0.9
):
    """
    精准标注优化版：紧凑布局 + 增强过滤点显示
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import ConnectionPatch
    import numpy as np
    from scipy.spatial import cKDTree

    # 图像预处理
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image.device)
    image_denorm = image * std + mean
    image_np = image_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_np = np.clip(image_np, 0, 1)

    # 创建紧凑画布
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=figsize, dpi=150)
    gs = GridSpec(2, 2, figure=fig,
                  top=0.92, bottom=0.05,
                  hspace=0.08, wspace=0.06)  # 缩小子图间距

    # 增强颜色配置
    colors = {
        'raw': '#d62728',  # 红色
        'dedup': '#2ca02c',  # 绿色
        'opt': '#1f77b4',  # 蓝色
        'gt': '#ff7f0e',  # 橙色
        'discard': '#ff0000',  # 高亮红色
        'arrow': '#ff0000'  # 红色箭头
    }

    # 数据处理流程
    # 原始预测
    raw_mask = confidences >= 0.5
    raw_keep = points[raw_mask].cpu()
    raw_discard = points[~raw_mask].cpu()

    # 去重处理
    pre_point,pre_conf = filter_by_thres(points,confidences,0.3)
    dedup_pts, dedup_conf = deduplicate_points(pre_point, pre_conf, 2)
    dedup_mask = dedup_conf >= 0.5
    dedup_keep = dedup_pts[dedup_mask].cpu()
    dedup_discard = dedup_pts[~dedup_mask].cpu()

    # 优化处理
    opt_pts, opt_conf, mask = optimize_points_for_study(image, dedup_pts[dedup_mask], dedup_conf[dedup_mask])
    opt_keep = opt_pts.cpu()
    opt_discard = dedup_pts[dedup_mask][~mask].cpu()  # 确保获取过滤点

    # 核心绘图函数
    def draw_phase(ax, keep_pts, discard_pts, prev_keep, title, phase_color):
        """增强版绘图逻辑"""
        ax.imshow(image_np, interpolation='lanczos')

        # 绘制保留点
        ax.scatter(keep_pts[:, 0], keep_pts[:, 1],
                   s=50, color=phase_color, marker='o',
                   edgecolor='white', linewidth=1.2, zorder=3)

        # 绘制被过滤点与箭头（仅在非首个子图）
        if prev_keep is not None and len(discard_pts) > 0:
            prev_np = prev_keep.numpy()
            discard_np = discard_pts.numpy()

            # 建立点对应关系
            tree = cKDTree(prev_np)
            _, nn_indices = tree.query(discard_np, k=1)

            # 绘制连接箭头
            for i, pt in enumerate(discard_np):
                # 高亮过滤点
                ax.scatter(pt[0], pt[1], s=180,
                           marker='X', color=colors['discard'],
                           linewidths=2.5, zorder=4,
                           edgecolor='black')  # 添加黑色边框

        ax.set_title(title, fontsize=12, weight='bold', pad=title_pad)
        ax.axis('off')

    # 绘制所有处理阶段
    draw_phase(fig.add_subplot(gs[0]),
               raw_keep, raw_discard, None,
               f"A. just confidence filter(Total: {len(raw_keep)})", colors['raw'])

    draw_phase(fig.add_subplot(gs[1]),
               dedup_keep, dedup_discard, raw_keep,
               f"B. Dedup (Changes in quantity compared to the previous stage: {len(pre_point) - len(dedup_keep)})", colors['dedup'])

    draw_phase(fig.add_subplot(gs[2]),
               opt_keep, opt_discard, dedup_keep,  # 确保传入优化阶段的过滤点
               f"C. Optim (Changes in quantity compared to the previous stage: {len(dedup_pts[dedup_mask]) - len(opt_keep)})", colors['opt'])

    # 最终对比图
    ax3 = fig.add_subplot(gs[3])
    ax3.imshow(image_np)
    ax3.scatter(opt_keep[:, 0], opt_keep[:, 1],
                s=50, color=colors['opt'], marker='o',
                edgecolor='white', linewidth=1.2)
    if gt_points is not None:
        gt_np = gt_points.cpu().numpy()
        ax3.scatter(gt_np[:, 0], gt_np[:, 1],
                    s=180, marker='*', color=colors['gt'],
                    edgecolor='white', linewidth=1.2, zorder=4)
    ax3.set_title(f"D. Final vs GT (pred: {len(opt_keep)} gt:{len(gt_np)})",
                  fontsize=12, weight='bold', pad=title_pad)
    ax3.axis('off')

    # 精简图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Kept Points',
                   markerfacecolor='#1f77b4', markersize=10),
        plt.Line2D([0], [0], marker='X', color=colors['discard'], label='Removed Points',
                   markersize=12, markeredgewidth=2, linestyle='None'),
        plt.Line2D([0], [0], marker='*', color=colors['gt'], label='Ground Truth',
                   markersize=15, linestyle='None')
    ]
    fig.legend(handles=legend_elements,
               loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 0.97),
               frameon=True, fontsize=10,
               handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
    plt.close()