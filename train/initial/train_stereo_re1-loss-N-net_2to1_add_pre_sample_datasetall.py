import os
import sys
import time
import logging
from pathlib import Path
from tqdm import tqdm
from collections import namedtuple

from tensorboardX import SummaryWriter

from model.cres.useful.nets_2to1_add_pre import Model
# from dataset import CREStereoDataset
import data.data_load_datasetall.stereo_datasets as datasets

import evaluate.evaluate_stereo_datasatall_2to1_add_pre as eva

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def format_time(elapse):
    elapse = int(elapse)
    hour = elapse // 3600
    minute = elapse % 3600 // 60
    seconds = elapse % 60
    return "{:02d}:{:02d}:{:02d}".format(hour, minute, seconds)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """
    # 流量预测序列上定义的损失函数

    n_predictions = len(flow_preds)
    assert n_predictions >= 1

    # exlude invalid pixels and extremely large diplacements
    #  排除无效像素和超大像素
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()

    # exclude extremly large displacements：排除非常大的位移
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    # pre-process
    # flow_gt = torch.cat([flow_gt, flow_gt * 0], dim=1)
    flow_loss = 0.0
    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        # 我们调整损耗_gamma，使其在任何迭代次数下都保持一致
        # adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = loss_gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        # assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        # flow_loss += i_weight * i_loss[valid.bool()].mean()
        flow_loss += (i_weight * (valid.unsqueeze(1) * i_loss).mean()) / n_predictions

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

class Logger:
    SUM_FREQ = 10

    def __init__(self, model, scheduler,tb_log):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = tb_log

    def _print_training_status(self, worklog,i_epoch,max_epoch,loss,time_train_passed):
        metrics_data = [self.running_loss[k] / Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.5f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        worklog.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        # i_epoch=self.total_steps//len_dataset+1
        meta_info=list()
        meta_info.append("passed:{}".format(format_time(time_train_passed)))
        meta_info.append(
            "[{}/{}:{}/{}]".format(
                i_epoch,
                max_epoch,
                self.total_steps+1,
                args.num_steps,
            )
        )
        meta_info.append('lr:{:.5g}'.format(self.scheduler.get_last_lr()[0]))
        meta_info.append('1px:{}'.format( metrics_data[0]))
        meta_info.append('3px:{}'.format( metrics_data[1]))
        meta_info.append('5px:{}'.format( metrics_data[2]))
        meta_info.append('epe:{}'.format(metrics_data[3]))
        loss_info = [" ==> {}:{:.4g}".format("loss", loss)]
        info = [",".join(meta_info)] + loss_info
        worklog.info("".join(info))

        if self.writer is None:
            self.writer = SummaryWriter(log_dir='train_logger' + '/' + 'train_event')

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics,worklog,i_epoch,max_epoch,loss,time_train_passed):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            self._print_training_status(worklog,i_epoch,max_epoch,loss,time_train_passed)
            self.running_loss = {}

    def write_dict(self, results,tb_log):
        if self.writer is None:
            self.writer = tb_log

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def main(args):
    # initial info
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # directory check
    log_model_dir = os.path.join(args.log_dir, "models")
    ensure_dir(log_model_dir)

    log_checkpoints_dir = os.path.join(args.log_dir, "checkpoints")
    ensure_dir(log_checkpoints_dir)

    world_size = torch.cuda.device_count()  # number of GPU(s)
    # model / optimizer
    model = Model(
        max_disp=args.max_disp, mixed_precision=args.mixed_precision, test_mode=False
    )
    model = nn.DataParallel(model,device_ids=[i for i in range(world_size)])
    model.cuda()

    tb_log = SummaryWriter(os.path.join(args.log_dir+'/'+'train_event', time.strftime('%m-%d-%H')))

    # worklog
    logging.basicConfig(level=eval(args.log_level))
    worklog = logging.getLogger("train_logger")
    worklog.propagate = False
    file_path=os.path.join(args.log_dir, "worklog.txt")
    fileHandler = logging.FileHandler(
       file_path, mode="a", encoding="utf8"
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

    # datasets
    dataset = datasets.fetch_dataloader(args)
    # if rank == 0:
    worklog.info(f"Dataset size: {len(dataset)}")
    worklog.info(f"batch_size_single: {args.batch_size_single}")
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 6)) - 2
    worklog.info(f"num_workers: {num_workers}")
    dataloader = DataLoader(dataset, args.batch_size_single, shuffle=True,
                            num_workers=num_workers, drop_last=True, persistent_workers=False, pin_memory=True)

    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    logger = Logger(model, scheduler,tb_log)

    # load pretrained model if exist
    chk_path = os.path.join(log_model_dir, "latest.pth")
    if not os.path.exists(chk_path):
        chk_path = None

    if chk_path is not None:
        # if rank == 0:
        worklog.info(f"loading model: {chk_path}")
        state_dict = torch.load(chk_path)
        model.load_state_dict(state_dict, strict=True)
        logging.info(f"Done loading latest checkpoint")

    model.cuda()
    model.train()
    model.module.freeze_bn()  # We keep BatchNorm frozen

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = 0
    worklog.info(f"total_iters: {args.num_steps}")
    # 进行test的频率

    validation_frequency = 2000
    max_epoch = args.num_steps // len(dataloader) + 1
    worklog.info(f"max_epoch: {max_epoch}")
    t0 = time.perf_counter()
    while should_keep_training:

        for i_batch, (_, *data_blob) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            left, right, gt_flow, valid_mask = [x.cuda() for x in data_blob]
            # b = flow.shape
            # c = valid.shape

            assert model.training
            flow_predictions = model(left, right)
            assert model.training

            # a=flow_predictions[0].shaoe

            loss, metrics = sequence_loss(flow_predictions, gt_flow, valid_mask)
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            t3 = time.perf_counter()
            time_train_passed = t3 - t0
            i_epoch = (global_batch_num-1) // len(dataloader) + 1
            meta_info = list()
            meta_info.append("passed:{}".format(format_time(time_train_passed)))
            meta_info.append(
                "[{}/{}:{}/{}]".format(
                    i_epoch,
                    max_epoch,
                    global_batch_num,
                    args.num_steps,
                )
            )
            meta_info.append('lr:{:.5g}'.format(scheduler.get_last_lr()[0]))
            loss_info = [" ==> {}:{:.5g}".format("loss", loss.item())]
            info = [",".join(meta_info)] + loss_info
            worklog.info("".join(info))



            logger.push(metrics,worklog, i_epoch,max_epoch,loss.item(),time_train_passed)

            if total_steps % validation_frequency == validation_frequency - 1:
                save_path = Path(log_checkpoints_dir + '/%d_%s.pth' % (total_steps + 1, args.name))
                worklog.info(f"Model params saved: {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)

                save_txt = args.log_dir + '/' + args.save_txt
                results = eva.validate_things(model.module, save_txt, total_steps + 1,
                                                                  iters=args.valid_iters)

                logger.write_dict(results,tb_log)

                model.train()
                model.module.freeze_bn()

            total_steps += 1

            if total_steps >= args.num_steps:
                should_keep_training = False
                break

        if len(dataloader) >= 10000:
            save_path = Path(log_checkpoints_dir + '/%d_epoch_%s.pth.gz' % (total_steps + 1, args.name))
            worklog.info(f"Saving file {save_path}")
            torch.save(model.state_dict(), save_path)

    print("FINISHED TRAINING")
    logger.close()
    PATH = log_checkpoints_dir + '/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    worklog.info("Training is done, exit.")
    return PATH


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

    parser.add_argument('--training_data_path', default='./SceneFlow_all', help='use mixed precision')
    parser.add_argument('--log_dir', default='./results/train_stereo_log_cres_loss_N_net_2to1_add_pre_sample_datasetall_b4_i20000', help='use mixed precision')
    parser.add_argument('--save_txt', default='FlyingThings3D (TEST).txt', help='use mixed precision')
    parser.add_argument('--log_level', default='logging.INFO', help='use mixed precision')
    parser.add_argument('--seed', type=int, default=0, help="batch size used during training.")
    # parser.add_argument('--loadmodel', default='~', help='use mixed precision')



    # Training parameters
    # parser.add_argument('--mixed_precision', default='false', help="-")
    parser.add_argument('--base_lr', type=float, default=4.0e-4, help="batch size used during training.")
    parser.add_argument('--nr_gpus', type=int, default=1, help="batch size used during training.")
    parser.add_argument('--batch_size_single', type=int, default=4, help="batch size used during training.")
    parser.add_argument('--n_total_epoch', type=int, default=100, help="batch size used during training.")
    parser.add_argument('--minibatch_per_epoch', type=int, default=200, help="batch size used during training.")
    parser.add_argument('--model_save_freq_epoch', type=int, default=10, help="batch size used during training.")
    parser.add_argument('--max_disp', type=int, default=256, help="batch size used during training.")
    parser.add_argument('--image_width', type=int, default=512, help="batch size used during training.")
    parser.add_argument('--image_height', type=int, default=384, help="batch size used during training.")



    # parser.add_argument('--batch_size', type=int, default=2, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0004, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=20000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512],
                        help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=16,
                        help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32,
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
    # parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    # parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
    #                     help="hidden state and context dimensions")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'],
                        help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()


    main(args)
