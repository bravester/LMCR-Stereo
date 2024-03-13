from option import args,parser
import torch
import logging
from tensorboardX import SummaryWriter

import os

from nets import Model
import data_load.stereo_datasets as datasets

import evaluate_stereo


def save_path():
    tb_log = SummaryWriter(os.path.join(args.log_dir + '/' + 'train_event', time.strftime('%m-%d-%H')))

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

    return  tb_log,worklog

# 模型保存路径
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

def train(local_rank,train_data_loader,model,optimizer):

    # directory check
    log_model_dir = os.path.join(args.log_dir, "models")
    ensure_dir(log_model_dir)

    log_checkpoints_dir = os.path.join(args.log_dir, "checkpoints")
    ensure_dir(log_checkpoints_dir)

    # start_epoch=0
    # start_step=0

    chk_path = os.path.join(log_model_dir, "latest.pth")
    if not os.path.exists(chk_path):
        chk_path = None

    if chk_path is not None:
        # if rank == 0:
        worklog.info(f"loading model: {chk_path}")
        state_dict = torch.load(chk_path,map_location=torch.device("cuda:0"))
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        resume_epoch_idx = state_dict["epoch"]
        resume_iters = state_dict["iters"]
        start_epoch_idx = resume_epoch_idx
        start_iters = resume_iters
    else:
        start_epoch_idx = 0
        start_iters = 0

        # model:模型拷贝
        model=torch.nn.parallel.DistributedDataParallel(model.cuda(),device_ids=[local_rank])



    for epoch_idx in range(start_epoch_idx, args.n_total_epoch):

        epoch_total_train_loss = 0
        adjust_learning_rate(optimizer, epoch_idx)








if __name__ == "__main__":

    tb_log, worklog=save_path()

    # 增加local_rank
    parser.add_argument("--local_rank",help="local device id on current node")
    if torch.cuda.is_available():
        logging.warning("cuda is available")
        if torch.cuda.device_count() > 1:
            worklog.warning(f"find{torch.cuda.device_count()}GPU")
        else:
            worklog.warning("too few GPU!")
            # return
        # os.environ["CUDA_VISIBILE_DEVICES"]=0
    else:
        worklog.warning("cuda is not available")


    # 分布式训练
    n_gpus=2
    torch.distributed.init_process("nccl",world_size=n_gpus,rank=args.local_rank)
    torch.cuda.set_device(args.local_rank)

    model = Model(
        max_disp=args.max_disp, mixed_precision=args.mixed_precision, test_mode=False
    )
    worklog.info("Params: %s" % sum([p.numel() for p in model.parameters()]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

    dataset = datasets.fetch_dataloader(args)
    # if rank == 0:
    worklog.info(f"Dataset size: {len(dataset)}")
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size_single, shuffle=False,
                            num_workers=0, drop_last=True, persistent_workers=False, pin_memory=True)
