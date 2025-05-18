import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
from post_ import filter_by_thres
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn_light', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='test',
                        help='path where to save')
    parser.add_argument('--weight_path', default='runs_adapter/best_mae.pth',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def main(args, debug=False):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model(args,adapter=True)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # set your image path here
    img_path = "../../dataset/test/images/IMG_166.jpg"
    label_path = "../../dataset/test/labels/IMG_166.txt"


    data = []
    with open(label_path, 'r') as file:  # 替换为你的文件路径
        for line in file:
            stripped_line = line.strip()  # 去除首尾空白
            if not stripped_line:
                continue  # 跳过空行
            parts = stripped_line.split()  # 按空白分割，适用于空格、制表符分隔
            if len(parts) != 2:
                continue  # 跳过不符合两列的行
            try:
                # 转换为浮点数，并添加到数据列表
                x, y = map(float, parts)
                data.append([x, y])
            except ValueError:
                continue  # 忽略无法转换的行

    # 获取行数并创建NumPy数组
    n = len(data)
    array = np.array(data)
    # load the images
    img_raw = Image.open(img_path).convert('RGB')
    # round the size
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128

    array[:,1]  = array[:,1] * new_height / height
    array[:,0]  = array[:,0] * new_width / width
    img_raw = img_raw.resize((new_width, new_height), Image.Resampling.BILINEAR)
    # pre-proccessing
    img = transform(img_raw)

    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)
    # run inference
    outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]
    ori_s, ori_p = outputs_scores, outputs_points
    outputs_points, outputs_scores = filter_by_thres(outputs_points, outputs_scores, 0.3)
    outputs_points, outputs_scores = deduplicate_points(outputs_points, outputs_scores, 2)
    outputs_points, outputs_scores = filter_by_thres(outputs_points, outputs_scores, 0.5)
    outputs_points, outputs_scores = optimize_points(samples, outputs_points, outputs_scores)

    # filter the predictions
    points = outputs_points.detach().cpu().numpy().tolist()
    predict_cnt = int(outputs_scores.shape[0])

    # draw the predictions
    size = 2
    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    for p in array:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 255, 0), -1)
    # save the visualized image
    cv2.imwrite(os.path.join(args.output_dir, 'gt{}.jpg'.format(n)), img_to_draw)

    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    for p in points:
        img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
    # save the visualized image
    cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(predict_cnt)), img_to_draw)

    img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
    heat,_ = generate_heatmap(img_to_draw, ori_p, ori_s)
    cv2.imwrite(os.path.join(args.output_dir, 'heatmap.jpg'), heat)


def generate_heatmap(image, points, confidences, sigma=5, alpha=0.5):
    """
    生成热力图并与原图叠加。

    参数:
        image (numpy.ndarray): 原图，形状为(H, W, 3)，BGR格式。
        points (torch.Tensor): Nx2的坐标点张量，每个点为(x, y)。
        confidences (torch.Tensor): N长度的置信度张量。
        sigma (int): 高斯模糊的标准差，控制热力扩散程度。
        alpha (float): 热力图的透明度，0为完全透明，1为不透明。

    返回:
        overlay (numpy.ndarray): 叠加后的图像。
        heatmap_colored (numpy.ndarray): 彩色热力图。
    """
    # 转换张量为NumPy数组
    points_np = points.detach().cpu().numpy()
    confidences_np = confidences.detach().cpu().numpy()
    H, W = image.shape[:2]

    # 创建热力图基板
    heatmap = np.zeros((H, W), dtype=np.float32)

    # 遍历每个点，累加置信度
    for (x, y), conf in zip(points_np, confidences_np):
        x_round = int(round(x))
        y_round = int(round(y))
        if 0 <= x_round < W and 0 <= y_round < H:
            heatmap[y_round, x_round] += conf

    # 应用高斯模糊
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # 归一化到0-255并转换为uint8
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = heatmap.astype(np.uint8)

    # 应用颜色映射（JET）
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 叠加热力图与原图
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay, heatmap_colored


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)