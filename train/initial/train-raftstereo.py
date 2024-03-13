import os
import sys
import time
import logging
from pathlib import Path
from collections import namedtuple

import yaml
from tensorboardX import SummaryWriter

# from nets import Model
from model.other.raft_stereo.core.raft_stereo import RAFTStereo
# from dataset import CREStereoDataset
import data.data_load.stereo_datasets as datasets

import evaluate.evaluate_stereo_raftstereo as eva

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def parse_yaml(file_path: str) -> namedtuple:
    """Parse yaml configuration file and return the object in `namedtuple`."""
    with open(file_path, "rb") as f:
        cfg: dict = yaml.safe_load(f)
    args = namedtuple("train_args", cfg.keys())(*cfg.values())
    return args


def format_time(elapse):
    elapse = int(elapse)
    hour = elapse // 3600
    minute = elapse % 3600 // 60
    seconds = elapse % 60
    return "{:02d}:{:02d}:{:02d}".format(hour, minute, seconds)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def adjust_learning_rate(optimizer, epoch):

    warm_up = 0.02
    const_range = 0.6
    min_lr_rate = 0.05

    if epoch <= args.n_total_epoch * warm_up:
        lr = (1 - min_lr_rate) * args.base_lr / (
            args.n_total_epoch * warm_up
        ) * epoch + min_lr_rate * args.base_lr
    elif args.n_total_epoch * warm_up < epoch <= args.n_total_epoch * const_range:
        lr = args.base_lr
    else:
        lr = (min_lr_rate - 1) * args.base_lr / (
            (1 - const_range) * args.n_total_epoch
        ) * epoch + (1 - min_lr_rate * const_range) / (1 - const_range) * args.base_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.9, max_flow=700):
    '''
    valid: (2, 384, 512) (B, H, W) -> (B, 1, H, W)
    flow_preds[0]: (B, 2, H, W)
    flow_gt: (B, 2, H, W)
    '''


    n_predictions = len(flow_preds)
    assert n_predictions >= 1


    # exlude invalid pixels and extremely large diplacements
    #  排除无效像素和超大像素
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()

    # exclude extremly large displacements：排除非常大的位移
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    # valid = torch.cat([valid, valid * 0], dim=1)

    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    # pre-process:当为CRES时，进行预处理
    # flow_gt = torch.cat([flow_gt, flow_gt * 0], dim=1)

    flow_loss = 0.0
    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        # 我们调整损耗_gamma，使其在任何迭代次数下都保持一致
        # adjusted_loss_gamma = gamma ** (15 / (n_predictions - 1))

        # i_loss.shape:[4,2,384,512]
        # i_loss[valid.bool()].shape:786432
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        # assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * (valid.unsqueeze(1) * i_loss).mean()


    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss,metrics


def main(args):
    # initial info
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # rank, world_size = dist.get_rank(), dist.get_world_size()
    world_size = torch.cuda.device_count()  # number of GPU(s)

    # directory check
    log_model_dir = os.path.join(args.log_dir, "models")
    ensure_dir(log_model_dir)

    # model / optimizer
    # model = Model(
    #     max_disp=args.max_disp, mixed_precision=args.mixed_precision, test_mode=False
    # )
    # raft-stereo
    model = nn.DataParallel(RAFTStereo(args))

    # 当为raft-stereo时，不需要执行下面这步
    # model = nn.DataParallel(model,device_ids=[i for i in range(world_size)])

    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
    # model = nn.DataParallel(model,device_ids=[0])

    tb_log = SummaryWriter(os.path.join(args.log_dir+'/'+'train_event', time.strftime('%m-%d-%H')))

    # worklog
    logging.basicConfig(level=eval(args.log_level))
    worklog = logging.getLogger("train_logger")
    worklog.propagate = False
    fileHandler = logging.FileHandler(
        os.path.join(args.log_dir, "worklog.txt"), mode="a", encoding="utf8"
    )
    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )
    fileHandler.setFormatter(formatter)
    consoleHandler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="\x1b[32m%(asctime)s\x1b[0m %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )
    consoleHandler.setFormatter(formatter)
    worklog.handlers = [fileHandler, consoleHandler]

    # params stat
    worklog.info(f"Use {world_size} GPU(s)")
    worklog.info("Params: %s" % sum([p.numel() for p in model.parameters()]))

    # load pretrained model if exist
    chk_path = os.path.join(log_model_dir, "latest.pth")
    # if args.loadmodel is not None:
    #     chk_path = args.loadmodel
    if not os.path.exists(chk_path):
        chk_path = None

    if chk_path is not None:
        # if rank == 0:
        worklog.info(f"loading model: {chk_path}")
        state_dict = torch.load(chk_path)
        model.load_state_dict(state_dict['state_dict'])
        optimizer.load_state_dict(state_dict['optim_state_dict'])
        resume_epoch_idx = state_dict["epoch"]
        resume_iters = state_dict["iters"]
        start_epoch_idx = resume_epoch_idx + 1
        start_iters = resume_iters
    else:
        start_epoch_idx = 1
        start_iters = 0

    # datasets
    # dataset = CREStereoDataset(args.training_data_path)
    dataset = datasets.fetch_dataloader(args)
    # if rank == 0:
    worklog.info(f"Dataset size: {len(dataset)}")
    worklog.info(f"batch_size_single: {args.batch_size_single}")
    # num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6)) - 2
    # worklog.info(f"num_workers: {num_workers}")
    dataloader = DataLoader(dataset, args.batch_size_single, shuffle=False,
                            num_workers=0, drop_last=True, persistent_workers=False, pin_memory=True)

    # counter
    cur_iters = start_iters
    total_iters = args.minibatch_per_epoch * args.n_total_epoch
    worklog.info(f"total_iters: {total_iters}")
    t0 = time.perf_counter()
    # 进行TEST的频率
    validation_frequency=1000
    for epoch_idx in range(start_epoch_idx, args.n_total_epoch + 1):

        # adjust learning rate
        epoch_total_train_loss = 0
        adjust_learning_rate(optimizer, epoch_idx)
        model.train()
        # model.module.freeze_bn()

        t1 = time.perf_counter()
        # batch_idx = 0

        # for mini_batch_data in dataloader:
        for batch_idx, (_,*mini_batch_data) in enumerate(dataloader):

            if batch_idx % args.minibatch_per_epoch == 0 and batch_idx != 0:
                break
            # batch_idx += 1
            cur_iters += 1

            # parse data
            left, right, gt_flow, valid_mask = [x.cuda() for x in mini_batch_data]
                # (
                # mini_batch_data["left"],
                # mini_batch_data["right"],
                # mini_batch_data["disparity"].cuda(),
                # mini_batch_data["mask"].cuda(),
            # )
            # batch_size_s = len(targets)  # 不足一个batch_size直接停止训练
            # if batch_size_s < BATCH_SIZE:
            #     break

            t2 = time.perf_counter()
            optimizer.zero_grad()

            # pre-process
            # gt_disp = torch.unsqueeze(gt_disp, dim=1)  # [2, 384, 512] -> [2, 1, 384, 512]
            # gt_flow = torch.cat([gt_disp, gt_disp * 0], dim=1)

            # valid = torch.unsqueeze(valid, dim=1)  # [2, 384, 512] -> [2, 1, 384, 512]
            # valid_mask = torch.cat([valid, valid * 0], dim=1)
            # [2, 2, 384, 512]

            # forward :CRES内部已经设定:iters
            # flow_predictions.shape:[4,2,384,512]
            # flow_predictions = model(left.cuda(), right.cuda())

            # raft-stereo:flow_predictions.shape:[4,1,384,512]
            flow_predictions = model(left.cuda(), right.cuda(), iters=args.train_iters)

            loss,metrics = sequence_loss(
                flow_predictions, gt_flow, valid_mask, gamma=0.9,max_flow=700
            )

            # loss stats
            loss_item = loss.data.item()
            epoch_total_train_loss += loss_item
            loss.backward()
            optimizer.step()
            t3 = time.perf_counter()

            #
            tb_log.add_scalar("train/loss_iters", loss_item, cur_iters)
            tb_log.add_scalar("train/epe",metrics['epe'],cur_iters)
            tb_log.add_scalar("train/1px", metrics['1px'], cur_iters)
            tb_log.add_scalar("train/3px", metrics['3px'], cur_iters)
            tb_log.add_scalar("train/5px", metrics['5px'], cur_iters)

            if cur_iters % 5 == 0:
                tdata = t2 - t1
                time_train_passed = t3 - t0
                time_iter_passed = t3 - t1
                step_passed = cur_iters - start_iters
                eta = (
                    (total_iters - cur_iters)
                    / max(step_passed, 1e-7)
                    * time_train_passed
                )

                meta_info = list()
                meta_info.append("{:.2g} b/s".format(1.0 / time_iter_passed))
                meta_info.append("passed:{}".format(format_time(time_train_passed)))
                meta_info.append("eta:{}".format(format_time(eta)))
                meta_info.append(
                    "data_time:{:.2g}".format(tdata / time_iter_passed)
                )

                meta_info.append(
                    "lr:{:.5g}".format(optimizer.param_groups[0]["lr"])
                )
                meta_info.append(
                    "[{}/{}:{}/{}]".format(
                        epoch_idx,
                        args.n_total_epoch,
                        batch_idx,
                        args.minibatch_per_epoch,
                    )
                )
                meta_info.append('epe:{}'.format(metrics['epe']))
                meta_info.append('1px:{}'.format(metrics['1px']))
                meta_info.append('3px:{}'.format(metrics['3px']))
                meta_info.append('5px:{}'.format(metrics['5px']))
                loss_info = [" ==> {}:{:.4g}".format("loss", loss_item)]
                # exp_name = ['\n' + os.path.basename(os.getcwd())]

                info = [",".join(meta_info)] + loss_info
                worklog.info("".join(info))

                # minibatch loss
                tb_log.add_scalar("train/loss_batch", loss_item, cur_iters)
                tb_log.add_scalar(
                    "train/lr", optimizer.param_groups[0]["lr"], cur_iters
                )
                tb_log.flush()
            else:
                meta2_info = list()
                meta2_info.append("lr:{:.5g}".format(optimizer.param_groups[0]["lr"]))
                meta2_info.append(
                    "[{}/{}:{}/{}]".format(
                        epoch_idx,
                        args.n_total_epoch,
                        batch_idx,
                        args.minibatch_per_epoch,
                    )
                )
                loss_info = [" ==> {}:{:.4g}".format("loss", loss_item)]
                info = [",".join(meta2_info)] + loss_info
                worklog.info("".join(info))

            t1 = time.perf_counter()

            total_steps = (epoch_idx-1) * args.minibatch_per_epoch + batch_idx
            if total_steps % validation_frequency == validation_frequency - 1:
                save_path = Path('checkpoints_raft_s_b4_i50000/%d_%s.pth' % (total_steps + 1, args.name))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)

                results = evaluate_stereo_raftstereo.validate_things(model.module,total_steps + 1, iters=args.valid_iters)

                for key in results:
                    tb_log.add_scalar(key, results[key], total_steps)
                model.train()
                model.module.freeze_bn()

        tb_log.add_scalar(
            "train/loss",
            epoch_total_train_loss / args.minibatch_per_epoch,
            epoch_idx,
        )
        tb_log.flush()

        # save model params
        ckp_data = {
            "epoch": epoch_idx,
            "iters": cur_iters,
            "batch_size":  args.batch_size_single * args.nr_gpus,
            "epoch_size": args.minibatch_per_epoch,
            "train_loss": epoch_total_train_loss / args.minibatch_per_epoch,
            "state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
        }
        torch.save(ckp_data, os.path.join(log_model_dir, "latest.pth"))
        # 打印模型的名称
        # for name in model.state_dict():
        #     print(name)
        if epoch_idx % args.model_save_freq_epoch == 0:
            save_path = os.path.join(log_model_dir, time.strftime('%m-%d-%H')+"epoch-%d.pth" % epoch_idx)
            worklog.info(f"Model params saved: {save_path}")
            torch.save(ckp_data, save_path)


        # 输出权重信息
        if epoch_idx % 1 == 0:
            for name, param in model.named_parameters():
                if 'bn' not in name:
                    tb_log.add_histogram(name, param, epoch_idx)
            # weight = model.state_dict()['module.update_block.mask.2.weight']
            # weight.shape = (-1, 1)
            # txt_path = os.path.join(args.log_dir, 'weight-%d' % epoch_idx + '.txt')
            # print(weight)
            # txt = open(txt_path, 'w')
            # for w in weight:
            #     txt.write('%f,' % w)
            # txt.close()

    worklog.info("Training is done, exit.")


if __name__ == "__main__":
    print("连接上远程ssh")
    # train configuration
    # args = parse_yaml("cfgs/train.yaml")
    # print('batch_size_single:', args.batch_size_single)
    # print("已读取yaml")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='CREStereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--training_data_path', default='E:/yxz/Deep_learning/RAFT/RAFT-Stereo-main/SceneFlow', help='use mixed precision')
    parser.add_argument('--log_dir', default='./train_log_raft_s_b4_i50000', help='use mixed precision')
    parser.add_argument('--log_level', default='logging.INFO', help='use mixed precision')
    parser.add_argument('--seed', type=int, default=0, help="batch size used during training.")
    # parser.add_argument('--loadmodel', default='~', help='use mixed precision')


    # Training parameters
    # parser.add_argument('--mixed_precision', default='false', help="-")
    parser.add_argument('--base_lr', type=float, default=4.0e-4, help="batch size used during training.")
    parser.add_argument('--nr_gpus', type=int, default=1, help="batch size used during training.")
    parser.add_argument('--batch_size_single', type=int, default=4, help="batch size used during training.")
    parser.add_argument('--n_total_epoch', type=int, default=250, help="batch size used during training.")
    parser.add_argument('--minibatch_per_epoch', type=int, default=200, help="batch size used during training.")
    parser.add_argument('--model_save_freq_epoch', type=int, default=10, help="batch size used during training.")
    parser.add_argument('--max_disp', type=int, default=256, help="batch size used during training.")
    parser.add_argument('--image_width', type=int, default=512, help="batch size used during training.")
    parser.add_argument('--image_height', type=int, default=384, help="batch size used during training.")


    # parser.add_argument('--batch_size', type=int, default=2, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512],
                        help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=16,
                        help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=16,
                        help='number of flow-field updates during validation forward pass')

    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg",
                        help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true',
                        help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                        help="hidden state and context dimensions")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'],
                        help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()


    main(args)
