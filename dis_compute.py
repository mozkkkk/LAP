import torch

def chamfer_distance(pred, gt):
    """
    计算两个点集之间的倒角距离。
    
    Args:
        pred (Tensor): 预测点集，形状为(N1, 2)
        gt (Tensor): 真实点集，形状为(N2, 2)
    
    Returns:
        float: 平均倒角距离
    """
    if pred.size(0) == 0 or gt.size(0) == 0:
        if pred.size(0) == 0 and gt.size(0) == 0:
            return 0.0  # 两者均为空，距离为0
        else:
            return float('999')  # 一方为空，返回无穷大
    # 计算所有点对之间的欧氏距离矩阵
    dist_matrix = torch.cdist(pred, gt)  # 形状(N1, N2)
    
    # 预测点到最近真值点的距离
    min_dist_p_to_g = dist_matrix.min(dim=1)[0]
    
    # 真值点到最近预测点的距离
    min_dist_g_to_p = dist_matrix.min(dim=0)[0]
    
    # 计算平均距离并返回
    return (min_dist_p_to_g.mean() + min_dist_g_to_p.mean()).item()

def hausdorff_distance(pred, gt):
    """
    计算两个点集之间的豪斯多夫距离。
    
    Args:
        pred (Tensor): 预测点集，形状为(N1, 2)
        gt (Tensor): 真实点集，形状为(N2, 2)
    
    Returns:
        float: 豪斯多夫距离
    """

    if pred.size(0) == 0 or gt.size(0) == 0:
        if pred.size(0) == 0 and gt.size(0) == 0:
            return 0.0  # 两者均为空，距离为0
        else:
            return float('999')  # 一方为空，返回无穷大
    dist_matrix = torch.cdist(pred, gt)  # 形状(N1, N2)
    
    # 两个方向的最大最小距离
    max_pred = dist_matrix.min(dim=1)[0].max()
    max_gt = dist_matrix.min(dim=0)[0].max()
    
    return torch.max(max_pred, max_gt).item()

if __name__ == '__main__':
    pred = torch.ones([3,2]).float()
    gt = torch.ones([1,2]).float()
    print(chamfer_distance(pred,gt))
    print(hausdorff_distance(pred,gt))