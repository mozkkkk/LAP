import time

import numpy as np
import torch
from matplotlib import pyplot as plt
# For Localizaiton
from scipy import spatial as ss


def chamfer_distance(pred, gt):
    """
    计算两个点集之间的倒角距离。
    
    Args:
        pred (Tensor): 预测点集，形状为(N1, 2)
        gt (Tensor): 真实点集，形状为(N2, 2)
    
    Returns:
        float: 平均倒角距离
    """
    t1 = time.time()
    if pred.size(0) == 0 or gt.size(0) == 0:
        if pred.size(0) == 0 and gt.size(0) == 0:
            t2 = time.time()
            mean_time = t2 - t1
            return 0.0,mean_time  # 两者均为空，距离为0
        else:
            t2 = time.time()
            mean_time = t2 - t1
            return float('999'),mean_time  # 一方为空，返回无穷大
    # 计算所有点对之间的欧氏距离矩阵
    dist_matrix = torch.cdist(pred, gt)  # 形状(N1, N2)
    
    # 预测点到最近真值点的距离
    min_dist_p_to_g = dist_matrix.min(dim=1)[0]
    
    # 真值点到最近预测点的距离
    min_dist_g_to_p = dist_matrix.min(dim=0)[0]
    t2 = time.time()
    mean_time = t2-t1
    # 计算平均距离并返回
    return (min_dist_p_to_g.mean() + min_dist_g_to_p.mean()).item(), mean_time

def hausdorff_distance(pred, gt):
    """
    计算两个点集之间的豪斯多夫距离。
    
    Args:
        pred (Tensor): 预测点集，形状为(N1, 2)
        gt (Tensor): 真实点集，形状为(N2, 2)
    
    Returns:
        float: 豪斯多夫距离
    """
    t1 = time.time()
    if pred.size(0) == 0 or gt.size(0) == 0:
        if pred.size(0) == 0 and gt.size(0) == 0:
            t2 = time.time()
            mean_time = t2 - t1
            return 0.0,mean_time  # 两者均为空，距离为0
        else:
            t2 = time.time()
            mean_time = t2 - t1
            return float('999'),mean_time  # 一方为空，返回无穷大
    dist_matrix = torch.cdist(pred, gt)  # 形状(N1, N2)
    
    # 两个方向的最大最小距离
    max_pred = dist_matrix.min(dim=1)[0].max()
    max_gt = dist_matrix.min(dim=0)[0].max()
    t2 = time.time()
    mean_time = t2-t1
    
    return torch.max(max_pred, max_gt).item(), mean_time

def hungarian(matrixTF):
    # matrix to adjacent matrix
    edges = np.argwhere(matrixTF)
    lnum, rnum = matrixTF.shape
    graph = [[] for _ in range(lnum)]
    for edge in edges:
        graph[edge[0]].append(edge[1])
    # deep first search
    match = [-1 for _ in range(rnum)]
    vis = [-1 for _ in range(rnum)]

    def dfs(u):
        for v in graph[u]:
            if vis[v]: continue
            vis[v] = True
            if match[v] == -1 or dfs(match[v]):
                match[v] = u
                return True
        return False

    # for loop
    ans = 0
    for a in range(lnum):
        for i in range(rnum): vis[i] = False
        if dfs(a): ans += 1
    # assignment matrix
    assign = np.zeros((lnum, rnum), dtype=bool)
    for i, m in enumerate(match):
        if m >= 0:
            assign[m, i] = True
    return ans, assign


def compute_tp(pred, gt, threshold):
    t1 = time.time()
    if len(pred) == 0 or gt.size(0) == 0:
        t2 = time.time()
        mean_time = t2 - t1
        return 0,mean_time
    dist_matrix = ss.distance_matrix(pred, gt, p=2)
    match_matrix = np.zeros(dist_matrix.shape, dtype=bool)
    for i_pred_p in range(len(pred)):
        pred_dist = dist_matrix[i_pred_p, :]
        match_matrix[i_pred_p, :] = pred_dist <= threshold

    tp, assign = hungarian(match_matrix)
    t2 = time.time()
    mean_time = t2-t1

    return tp, mean_time


def compute_f_r_p(pred, gt, threshold):
    """计算F1指标（返回精度、召回率、F1和总时间）"""
    t_start = time.time()
    tp, tp_time = compute_tp(pred, gt, threshold)
    ap = tp / (len(pred) + 1e-10)
    ar = tp / (gt.shape[0] + 1e-10)
    f1 = 2 * ap * ar / (ap + ar + 1e-10)
    total_time = time.time() - t_start
    return ap, ar, f1, total_time


# ------------------ 实验1：离群点鲁棒性 ------------------
def experiment_outlier():
    # 生成数据
    theta = torch.linspace(0, 2 * np.pi, 100)
    gt = torch.stack([theta.cos(), theta.sin()], 1) * 0.5
    pred = torch.cat([
        gt[:95] + torch.randn(95, 2) * 0.05,
        torch.tensor([[3., 0], [2.5, 2.5], [-3, 0], [0, 3], [0, -3]])
    ])
    gt += 100
    pred += 100

    # 计算指标
    THRESHOLD = 0.1
    cd, _ = chamfer_distance(pred, gt)
    hd, _ = hausdorff_distance(pred, gt)
    ap, ar, f1, _ = compute_f_r_p(pred.numpy().tolist(), gt, THRESHOLD)

    # 可视化
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    gt_scatter = plt.scatter(gt[:, 0], gt[:, 1], c='#2ca02c', alpha=0.7, label='Ground Truth', marker='o')
    pred_scatter = plt.scatter(pred[:, 0], pred[:, 1], c='#d62728', alpha=0.7, label='Prediction', marker='s')
    plt.title('Point Distribution with Outliers')
    plt.legend(handles=[gt_scatter, pred_scatter])

    # 标准化指标可视化（CD/HD越小越好，F1越大越好）
    plt.subplot(122)
    x = np.arange(3)
    plt.bar(x, [1 / (cd + 1e-6), 1 / (hd + 1e-6), f1], width=0.5,
            color=['#1f77b4', '#ff7f0e', '#2ca02c'], label=['1/(CD+1)', '1/(HD+1)', 'F1'])
    plt.xticks(x, ['Inverse CD', 'Inverse HD', 'F1 Score'])
    plt.ylabel('Normalized Value')
    plt.title(f'Metric Comparison (Threshold={THRESHOLD})')
    plt.ylim(0, 2)


def experiment_density_sensitivity():
    # 生成具有结构特征的测试数据
    np.random.seed(42)
    torch.manual_seed(42)

    # 真值点：左半平面高密度连续结构 + 右半平面稀疏点
    gt_left = torch.stack([
        torch.linspace(-3, -1, 100),
        torch.sin(torch.linspace(-3, -1, 100) * 3 * 0.5)], dim=1) + torch.randn(100, 2) * 0.05

    gt_right = torch.rand(20, 2) * 2 + torch.tensor([1.0, -1.0])
    gt = torch.cat([gt_left, gt_right])

    # 预测点：左半断裂结构 + 右半噪声点
    pred_left = gt_left[::3] + torch.randn(34, 2) * 0.1  # 下采样1/3
    pred_right = torch.rand(70, 2) * 6 - 3  # [-3,3]范围随机点
    pred = torch.cat([pred_left, pred_right])

    # 计算全局指标
    THRESHOLD = 0.2
    cd, _ = chamfer_distance(pred, gt)
    hd, _ = hausdorff_distance(pred, gt)
    ap, ar, f1, _ = compute_f_r_p(pred.numpy().tolist(), gt, THRESHOLD)

    # 计算左半区域指标
    left_mask_gt = (gt[:, 0] < -0.5)
    left_mask_pred = (pred[:, 0] < -0.5)
    gt_left = gt[left_mask_gt]
    pred_left = pred[left_mask_pred].numpy()

    # 左半区域指标
    cd_left, _ = chamfer_distance(pred[left_mask_pred], gt[left_mask_gt])
    hd_left, _ = hausdorff_distance(pred[left_mask_pred], gt[left_mask_gt])
    ap_left, ar_left, f1_left, _ = compute_f_r_p(pred_left.tolist(), gt_left, THRESHOLD)

    # 右半区域指标（对比用）
    right_mask_gt = (gt[:, 0] > 0.5)
    right_mask_pred = (pred[:, 0] > 0.5)
    ap_right, ar_right, f1_right, _ = compute_f_r_p(pred[right_mask_pred].numpy().tolist(),
                                                    gt[right_mask_gt], THRESHOLD)

    # 可视化
    plt.figure(figsize=(18, 6))

    # 子图1：点分布与结构连接
    ax1 = plt.subplot(131)
    ax1.scatter(gt[:, 0], gt[:, 1], c='#2ca02c', alpha=0.7, label='GT', marker='o', s=30)
    ax1.scatter(pred[:, 0], pred[:, 1], c='#d62728', alpha=0.7, label='Pred', marker='s', s=30)
    ax1.legend()
    ax1.set_title(f'Points\nGlobal F1={f1:.2f} vs Local F1={f1_left:.2f}')

    # 子图2：区域指标对比
    ax2 = plt.subplot(132)
    metrics = ['CD', 'HD', 'F1', 'AP', 'AR']
    global_vals = [cd, hd, f1, ap, ar]
    local_vals = [cd_left, hd_left, f1_left, ap_left, ar_left]

    x = np.arange(len(metrics))
    width = 0.35
    ax2.bar(x - width / 2, global_vals, width, label='Global', color='#1f77b4', alpha=0.6)
    ax2.bar(x + width / 2, local_vals, width, label='Left Region', color='#ff7f0e', alpha=0.6)

    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.set_title(f'Metric Comparison (Threshold={THRESHOLD})')

    # 添加数值标注
    for i in x:
        ax2.text(i - width / 2, global_vals[i] + 0.1, f'{global_vals[i]:.2f}', ha='center')
    ax2.text(i + width / 2, local_vals[i] + 0.1, f'{local_vals[i]:.2f}', ha='center')

    # 子图3：区域F1对比分析
    ax3 = plt.subplot(133)
    regions = ['Global', 'Left Structure', 'Right Noise']
    f1_values = [f1, f1_left, f1_right]
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    bars = ax3.bar(regions, f1_values, color=colors, alpha=0.6)
    for bar in bars:
        height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
             f'{height:.2f}', ha='center')

    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Score Regional Comparison')

    plt.tight_layout()
def experiment_asymmetric():
    # 生成数据
    gt = torch.randn(100, 2) * 0.5
    pred = torch.randn(50, 2) * 0.5 + 1.0

    gt += 100
    pred += 100

    # 计算指标
    THRESHOLD = 0.5
    cd, _ = chamfer_distance(pred, gt)
    hd, _ = hausdorff_distance(pred, gt)
    ap, ar, f1, _ = compute_f_r_p(pred.numpy().tolist(), gt, THRESHOLD)

    # 综合可视化
    plt.figure(figsize=(15, 5))
    plt.subplot(131, title='Point Distribution')
    plt.scatter(gt[:, 0], gt[:, 1], c='#2ca02c', alpha=0.5, label='GT')
    plt.scatter(pred[:, 0], pred[:, 1], c='#d62728', alpha=0.5, label='Pred')
    plt.legend()

    plt.subplot(132)
    plt.bar(['CD', 'HD'], [cd, hd], color=['#1f77b4', '#ff7f0e'])
    plt.ylabel('Distance'), plt.title('Error Metrics')

    plt.subplot(133)
    plt.bar(['F1', 'AP', 'AR'], [f1, ap, ar], color=['#2ca02c', '#9467bd', '#8c564b'])
    plt.ylabel('Score'), plt.title('Loc Metrics')


# ------------------ 实验4：时间效率对比 ------------------
def experiment_time():
    sizes = [650, 700, 750, 800, 850, 900, 1000]
    metrics = ['CD', 'HD', 'F1']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    time_data = {m: [] for m in metrics}

    for n in sizes:
        gt = torch.rand(n, 2)
        pred = torch.rand(n, 2)

        # 转换为毫秒
        _, t = chamfer_distance(pred, gt)
        time_data['CD'].append(t * 1000)

        _, t = hausdorff_distance(pred, gt)
        time_data['HD'].append(t * 1000)

        _, _, _, t = compute_f_r_p(pred.numpy().tolist(), gt, 0.1)
        time_data['F1'].append(t * 1000)

    # 线性坐标可视化
    plt.figure(figsize=(10, 6))
    for m, c in zip(metrics, colors):
        plt.plot(sizes, time_data[m], marker='o', color=c, label=m)

    plt.xlabel('Number of Points')
    plt.ylabel('Time (milliseconds)')
    plt.title('Computation Time Comparison (Linear Scale)')
    plt.yscale('log')  # 保持对数坐标但优化显示
    plt.grid(True, which='both', linestyle='--')
    plt.legend()


if __name__ == '__main__':
    plt.style.use('seaborn')
    experiment_outlier()
    experiment_density_sensitivity()
    experiment_asymmetric()
    experiment_time()
    plt.show()

'''
if __name__ == '__main__':
    pred = torch.ones([3,2]).float()
    gt = torch.ones([1,2]).float()
    cd, t1 = chamfer_distance(pred, gt)
    hd, t2 = hausdorff_distance(pred, gt)
    print(cd)
    print(hd)
'''