import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time
import os
import argparse

from models.pretrain_backbone import build

parser = argparse.ArgumentParser('Set parameters for training P2PNet', add_help=False)
# * Backbone
parser.add_argument('--backbone', default='vgg16_bn_light', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--num_classes', default=100, type=int)
args = parser.parse_args()

# 参数配置
config = {
    'batch_size': 32,
    'num_workers': 32,
    'epochs': 70,
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'print_freq': 100,
    'data_root': '../imagenet-100/'  # 修改为你的ImageNet路径
}

# 数据预处理
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dir = os.path.join(config['data_root'], 'train')
val_dir = os.path.join(config['data_root'], 'val')

train_dataset = datasets.ImageFolder(train_dir, train_transform)
val_dataset = datasets.ImageFolder(val_dir, val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=config['num_workers'],
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=config['num_workers'],
    pin_memory=True
)


def load_pretrain(model, checkpoint):
    pretrained_dict = checkpoint

    # 获取当前模型的参数字典
    model_dict = model.state_dict()

    # 过滤掉新模块的权重（因为预训练权重中没有新模块的权重）
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 更新当前模型的参数字典
    model_dict.update(pretrained_dict)

    # 加载更新后的参数字典到模型
    model.load_state_dict(model_dict)
    return model


# 初始化模型
model, criterion = build(args, training=True)
optimizer = optim.SGD(
    model.parameters(),
    lr=config['lr'],
    momentum=config['momentum'],
    weight_decay=config['weight_decay']
)

# 修改为自动学习率衰减
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',  # 监控准确率
    factor=0.1,  # 学习率衰减因子
    patience=5,  # 3次验证无提升后衰减
    verbose=True  # 显示调整信息
)

# 使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


def train():
    model.train()
    best_acc = 0.0
    for epoch in range(config['epochs']):
        epoch_start = time.time()

        # 训练阶段
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # 前向传播
            output = model(images)
            loss = criterion(output, target)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 打印训练信息
            if i % config['print_freq'] == 0:
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                      f'Loss: {loss.item():.4f}\t'
                      f'Time: {time.time() - epoch_start:.3f}s')

        # 验证阶段（每个epoch都验证）
        acc = validate(epoch)

        # 更新学习率（基于验证准确率）
        scheduler.step(acc)

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save({'model': model.state_dict()}, 'pretrain/best_model.pth')

        # 定期保存模型
        if epoch % 5 == 0:
            torch.save({'model': model.state_dict()},
                       f'pretrain/imagenet_pretrained_{epoch}.pth')


def validate(epoch):
    model.eval()
    top1_acc = 0
    total = 0
    with torch.no_grad():
        for images, target in val_loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(images)
            _, pred = torch.max(output, 1)

            total += target.size(0)
            top1_acc += (pred == target).sum().item()

    acc = 100 * top1_acc / total
    print(f'Validation Epoch: {epoch}\tTop-1 Accuracy: {acc:.2f}%')
    return acc  # 返回准确率用于学习率调整


if __name__ == '__main__':
    if os.path.exists("pretrain/pretrain.pth"):
        checkpoint = torch.load("pretrain/pretrain.pth", map_location='cpu')
        model.load_state_dict(checkpoint["model"])

    train()

    # 保存最终模型
    torch.save({
        'model': model.state_dict(),
    }, 'pretrain/pretrain.pth')