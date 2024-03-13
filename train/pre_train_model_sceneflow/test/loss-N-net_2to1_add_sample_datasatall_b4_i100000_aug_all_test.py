# import argparse
import os
import sys

import time
import logging
import numpy as np
import torch
from tqdm import tqdm
# from model.pre_train_model_sceneflow.nets_feature_pre_zeroresize_dw32_16_8_loop_end.crestereo import Model
from model.pre_train_model_sceneflow.nets_2to1.crestereo import autocast
from model.pre_train_model_sceneflow.nets_2to1 import Model
import data.data_load_datasetall_aug_all.stereo_datasets as datasets
from data.data_load_datasetall_aug_all.data_utils.utils import InputPadder


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_time(elapse):
    elapse = int(elapse)
    hour = elapse // 3600
    minute = elapse % 3600 // 60
    seconds = elapse % 60
    return "{:02d}:{:02d}:{:02d}".format(hour, minute, seconds)

@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    return {'eth3d-epe': epe, 'eth3d-d1': d1}


@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end-start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id+1)%10 == 0:
            logging.info(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(f"Validation KITTI: EPE {epe}, D1 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'kitti-epe': epe, 'kitti-d1': d1}


@torch.no_grad()
def validate_things(model,worklog,save_txt,tall,total_steps,iters=10,mixed_prec=False):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets_testall(dstype='frames_finalpass', things_test=True)
    worklog.info(f"Dataset size:{len(val_dataset)}")
    worklog.info(f"valid_iters：{iters}")

    out_list, epe_list = [], []
    out_list3,out_list5= [], []
    validate_time = []
    for val_id in tqdm(range(len(val_dataset))):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            t0 = time.perf_counter()
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            t1 = time.perf_counter()
        validate_time_pass = t1-t0
        validate_time_passall=format_time(t1-tall)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)
        # flow_gt=torch.cat([flow_gt, flow_gt * 0], dim=0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
        epe = epe.flatten()
        nan =epe[val].mean().item()
        # assert not torch.isnan(flow_pr).any() and not torch.isinf(flow_pr).any()
        # if torch.isnan(epe).any() or torch.isinf(epe).any():
        if np.sum(nan!=nan) != 0:
            worklog.info(f"This is nan image,id:{imageL_file}")
        else:
            # epe = epe.flatten()
            # val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

            out = (epe > 1.0)
            out3 = (epe > 3.0)
            out5 = (epe > 5.0)
            epe_list.append(epe[val].mean().item())
            out_list.append(out[val].cpu().numpy())
            out_list3.append(out3[val].cpu().numpy())
            out_list5.append(out5[val].cpu().numpy())
            validate_time.append(validate_time_pass)
            # worklog.info(
            #     f"Validation FlyingThings epe: {epe[val].mean().item()},d1:{out[val].cpu().numpy()},"
            #     f"d3:{out3[val].cpu().numpy()},d5:{out5[val].cpu().numpy()},validate_time_pass:{validate_time_pass}")
            worklog.info(
                f"Validation FlyingThings epe: {epe[val].mean().item()},validate_time_pass:{validate_time_pass}, validate_time_passall：{validate_time_passall}，image size:{image1.shape},flow_gt:{flow_gt.shape}")

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    out_list3 = np.concatenate(out_list3)
    out_list5 = np.concatenate(out_list5)
    d3 = 100 * np.mean(out_list3)
    d5 = 100 * np.mean(out_list5)

    # validate_time = np.concatenate(validate_time)
    validate_time_mean = np.mean(validate_time)
    worklog.info(f"validate_time_mean:{validate_time_mean}")


    with open(save_txt, 'a+') as f:
        f.write('\n'+"此时的迭代次数为："+str(total_steps)+'\n'
                +"epw_list:"+'\n'
                +str(epe_list)+'\n'
                +"epe的均值为："+str(epe)+'\n'
                +"d1的均值为："+str(d1)+'\n'
                +"d3的均值为："+str(d3)+'\n'
                +"d5的均值为："+str(d5)+'\n'
                +"平均运行时间："+str(validate_time_mean)+'\n')
    f.close()

    worklog.info(
        f"Validation FlyingThings epe: {epe},d1:{d1},d3:{d3},d5:{d5},validate_time_mean:{validate_time_mean}")
    # return {'things-epe': epe, 'things-d1': d1,'things-d3': d3,'things-d5': d5}

@torch.no_grad()
def validate_middlebury(model,worklog,save_txt,tall,total_steps,iters=32, split='F', mixed_prec=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    # aug_params = {}
    aug_params = None
    val_dataset = datasets.Middlebury(aug_params, split=split)
    worklog.info(f"Dataset size:{len(val_dataset)}")
    worklog.info(f"valid_iters：{iters}")

    out_list1, epe_list = [], []
    out_list2, out_list4,out_list5 = [], [],[]
    validate_time = []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            t0 = time.perf_counter()
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            t1 = time.perf_counter()
        validate_time_pass = t1 - t0
        validate_time_passall = format_time(t1-tall)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.reshape(-1) >= -0.5) & (flow_gt[0].reshape(-1) > -1000)

        out5 = (epe_flattened > 0.5)
        out1 = (epe_flattened > 1)
        out2 = (epe_flattened > 2)
        out4 = (epe_flattened > 4)
        image_out5 = out5[val].float().mean().item()
        image_out1 = out1[val].float().mean().item()
        image_out2 = out2[val].float().mean().item()
        image_out4 = out4[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        validate_time.append(validate_time_pass)
        worklog.info(f"Middlebury {imageL_file} Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D0.5 {round(image_out5,4)},"
                     f"D1 {round(image_out1,4)},D2 {round(image_out2,4)},D4 {round(image_out4,4)},validate_time_passall:{validate_time_passall}")
        epe_list.append(image_epe)
        out_list5.append(image_out5)
        out_list1.append(image_out1)
        out_list2.append(image_out2)
        out_list4.append(image_out4)

    epe_list = np.array(epe_list)
    out_list5 = np.array(out_list5)
    out_list1 = np.array(out_list1)
    out_list2 = np.array(out_list2)
    out_list4 = np.array(out_list4)

    epe = np.mean(epe_list)
    d5 = 100 * np.mean(out_list5)
    d1 = 100 * np.mean(out_list1)
    d2 = 100 * np.mean(out_list2)
    d4 = 100 * np.mean(out_list4)

    # validate_time = np.concatenate(validate_time)
    validate_time_mean = np.mean(validate_time)
    worklog.info(f"validate_time_mean:{validate_time_mean}")

    with open(save_txt, 'a+') as f:
        f.write('\n'+"此时的迭代次数为："+str(total_steps)+'\n'
                +"epw_list:"+'\n'
                +str(epe_list)+'\n'
                +"epe的均值为："+str(epe)+'\n'
                + "d0.5的均值为：" + str(d5) + '\n'
                +"d1的均值为："+str(d1)+'\n'
                +"d2的均值为："+str(d2)+'\n'
                +"d4的均值为："+str(d4)+'\n'
                +"平均运行时间："+str(validate_time_mean)+'\n')
    f.close()

    worklog.info(f"Validation Middlebury{split}: EPE: {epe}, D0.5: {d5}, D1: {d1},D2: {d2},D4: {d4},validate_time_mean:{validate_time_mean}")
    # return {f'middlebury{split}-epe': epe, f'middlebury{split}-d0.5': d5,f'middlebury{split}-d1': d1,f'middlebury{split}-d2': d2,f'middlebury{split}-d4': d4}





if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--max_disp', type=int, default=256, help="batch size used during training.")
    parser.add_argument('--log_level', default='logging.INFO', help='use mixed precision')
    parser.add_argument('--save_txt', default='FlyingThings3D (TEST)_iter20.txt', help='use mixed precision')

    # model parameter setting
    parser.add_argument('--log_dir', default='./test/cres_loss_N_net_2to1_sample_datasetall_b4_i100000_aug_all_test',
                        help='use mixed precision')
    parser.add_argument('--log_model_dir',
                        default='E:\\yxz\\Deep_learning\\CREStereo_pytorch-re\\wangyu\\CREStereo-Pytorch-main-re31_1\\train\pre_train_model_sceneflow\\results\cres_loss_N_net_2to1_sample_datasetall_aug_all_b4_i100000\\checkpoints',
                        help='model save pth')
    # parser.add_argument('--dataset', help="dataset for evaluation", required=True,
    #                     choices=["eth3d", "kitti", "things"] + [f"middlebury_{s}" for s in 'FHQ'])
    # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # parser.add_argument('--valid_iters', type=int, default=[24,3,2,2,5], help='number of flow-field updates during forward pass')
    parser.add_argument('--valid_iters', type=int, default=20,
                        help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                        help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg",
                        help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true',
                        help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    args = parser.parse_args()

    # model / optimizer
    model = Model(
        max_disp=args.max_disp, mixed_precision=args.mixed_precision, test_mode=False
    )
    model = torch.nn.DataParallel(model, device_ids=[0])


    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    # worklog
    logging.basicConfig(level=eval(args.log_level))
    worklog = logging.getLogger("train_logger")
    worklog.propagate = False
    file_path = os.path.join(args.log_dir, f"{(args.valid_iters)}worklogtest.txt")
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
    world_size = torch.cuda.device_count()  # number of GPU(s)
    worklog.info(f"Use {world_size} GPU(s)")
    worklog.info("Params: %s" % sum([p.numel() for p in model.parameters()]))

    import os
    path = args.log_model_dir
    model_names=os.listdir(path)
    tall=time.perf_counter()
    for model_name in model_names:
        portion = os.path.splitext(model_name)
        # if portion[-1]==".pth":
        if model_name == "CREStereo.pth":
            chk_path = os.path.join(path, model_name)
            worklog.info(f"model path: {chk_path}")
            if not os.path.exists(chk_path):
                chk_path = None
                worklog.info(f"loading model doesn't success")

            if chk_path is not None:
                # if rank == 0:
                worklog.info(f"loading model: {chk_path}")
                state_dict = torch.load(chk_path)
                model.load_state_dict(state_dict, strict=True)
                worklog.info(f"Done loading {model_name} checkpoint")
            model.cuda()
            model.eval()
            save_txt = args.log_dir +'/'+ args.save_txt
            use_mixed_precision = args.corr_implementation.endswith("_cuda")
            validate_things(model,worklog,save_txt, tall, portion[0],iters=args.valid_iters,mixed_prec=use_mixed_precision)