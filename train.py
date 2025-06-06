import argparse
import datetime
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
from tensorboardX import SummaryWriter
import warnings
from thop import profile

import copy

warnings.filterwarnings('ignore')


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training P2PNet', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--lr_drop', default=[], type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn_light', type=str,
                        help="Name of the convolutional backbone to use")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="L1 point coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)

    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    # dataset parameters
    parser.add_argument('--dataset_file', default='SHHA')
    parser.add_argument('--data_root', default='../../dataset',
                        help='path where the dataset is')

    parser.add_argument('--output_dir', default='./runs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoints_dir', default='./runs',
                        help='path where to save checkpoints, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='./runs',
                        help='path where to save, empty for no saving')

    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='./pretrain/best_model.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')

    return parser


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    # create the logging file
    run_log_name = os.path.join(args.output_dir, 'run_log.txt')
    with open(run_log_name, "w") as log_file:
        log_file.write('Eval Log %s\n' % time.strftime("%c"))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    # backup the arguments
    print(args)
    with open(run_log_name, "a") as log_file:
        log_file.write("{}".format(args))
    device = torch.device('cuda')
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # get the P2PNet model
    model, criterion = build_model(args, training=True)
    # move to GPU
    model.to(device)
    criterion.to(device)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    test_input = torch.randn(1, 3, 128, 128).float().to(device)  # 输入尺寸需匹配模型

    # 计算FLOPs和参数量
    flops, _ = profile(copy.deepcopy(model), inputs=(test_input,))
    gflops = flops / 1e9  # 转换为GFLOPs
    print(f"FLOPs: {flops}")
    print(f"GFLOPs: {gflops:.2f}")
    with open(run_log_name, "a") as log_file:
        log_file.write(f"\n")
        log_file.write(f"number of params: {n_parameters}")
        log_file.write(f"\nFLOPs: {flops}")
        log_file.write(f"    GFLOPs: {gflops:.2f}")
        log_file.write(f"\n")
    # print(f"Parameters: {params / 1e6:.2f}M")
    # use different optimation params for different parts of the model
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    # Adam is used by default
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    # optimizer = torch.optim.AdamW(param_dicts, lr=args.lr)
    # lr_scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = 50,eta_min = 0.000001,last_epoch = -1)
    lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop, gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    # create the dataset
    loading_data = build_dataset(args=args)
    # create the training and valiation set
    if not hasattr(args, 'resize'):
        setattr(args, 'resize', False)
    train_set, val_set = loading_data(args.data_root,ps_size=128,resize=args.resize)
    # create the sampler used during training
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    # the dataloader for training
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
    # resume the weights and training state if exists
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp = load_pretrain(model_without_ddp, checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler2.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        t1 = time.time()
        result = evaluate_crowd_no_overlap(model, data_loader_val, device, resize=args.resize)
        t2 = time.time()

        # print the evaluation results
        print('=======================================test=======================================')
        print("ap:", result[4], "times:", result[5])
        print("cd:", result[2], "hd:", result[3])
        print("mae:", result[0], "mse:", result[1], "time:", t2 - t1)
        return

    print("Start training")
    start_time = time.time()
    # save the performance during the training
    mae = []
    mse = []
    dis1 = []
    dis2 = []
    # the logger writer
    writer = SummaryWriter(args.tensorboard_dir)
    step = 0
    # training starts here
    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()
        stat = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)

        # record the training states after every epoch
        if writer is not None:
            with open(run_log_name, "a") as log_file:
                log_file.write("loss/loss@{}: {}".format(epoch, stat['loss']))
                log_file.write("loss/loss_ce@{}: {}".format(epoch, stat['loss_ce']))

            writer.add_scalar('loss/loss', stat['loss'], epoch)
            writer.add_scalar('loss/loss_ce', stat['loss_ce'], epoch)

        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % \
              (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        with open(run_log_name, "a") as log_file:
            log_file.write('[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        # change lr according to the scheduler
        lr_scheduler2.step()

        if epoch == 800:
            param_dicts = [
                {"params": [p for n, p in model_without_ddp.named_parameters() if
                            "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model_without_ddp.named_parameters() if
                               "backbone" in n and p.requires_grad],
                    "lr": 1e-5,
                },
            ]
            # Adam is used by default
            optimizer = torch.optim.Adam(param_dicts, lr=1e-4)
            lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)
        # save latest weights every epoch
        checkpoint_latest_path = os.path.join(args.checkpoints_dir, 'latest.pth')
        torch.save({
            'model': model_without_ddp.state_dict(),
        }, checkpoint_latest_path)
        # run evaluation
        if epoch % args.eval_freq == 0 and epoch != 0:
            t1 = time.time()
            result = evaluate_crowd_no_overlap(model, data_loader_val, device, resize=args.resize)
            t2 = time.time()

            mae.append(result[0])
            mse.append(result[1])
            dis1.append(result[2])
            dis2.append(result[3])
            # print the evaluation results
            print('=======================================test=======================================')
            print("ap:", result[4], "times:", result[5])
            print("cd:", result[2], "hd:", result[3])
            print("mae:", result[0], "mse:", result[1], "time:", t2 - t1, "best mae:", np.min(mae), )

            with open(run_log_name, "a") as log_file:
                log_file.write("\nap:{}".format(result[4]))
                log_file.write(
                    "\ncd:{}, hd:{}, best cd:{}, best hd:{}".format(result[2], result[3], np.min(dis1), np.min(dis2)))
                log_file.write("\nmae:{}, mse:{}, time:{}, best mae:{}".format(result[0],
                                                                               result[1], t2 - t1, np.min(mae)))

            print('=======================================test=======================================')
            # recored the evaluation results
            if writer is not None:
                with open(run_log_name, "a") as log_file:
                    log_file.write("metric/mae@{}: {}".format(step, result[0]))
                    log_file.write("metric/mse@{}: {}\n".format(step, result[1]))
                writer.add_scalar('metric/mae', result[0], step)
                writer.add_scalar('metric/mse', result[1], step)
                step += 1

            # save the best model since begining
            if abs(np.min(mae) - result[0]) < 0.01:
                checkpoint_best_path = os.path.join(args.checkpoints_dir, 'best_mae.pth')
                torch.save({
                    'model': model_without_ddp.state_dict(),
                }, checkpoint_best_path)

            if abs(np.min(dis1) - result[2]) < 0.01:
                checkpoint_best_path = os.path.join(args.checkpoints_dir, 'best_dis.pth')
                torch.save({
                    'model': model_without_ddp.state_dict(),
                }, checkpoint_best_path)
    # total time for training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
