from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from model.pre_train_model_sceneflow.nets_feature_pre_zeroresize_dw32_16_8_loop_end_light.crestereo import CREStereo, autocast
# from model.cres.useful.L37671_211.nets_2to1_add_pre_zeroresize_grade_dw32_16_8_loop_end.crestereo import CREStereo, autocast
import data.data_load_datasetall_aug_all.stereo_datasets as datasets
from data.data_load_datasetall_aug_all.data_utils.utils import InputPadder

import torch.nn.functional as F
import os
import skimage.io
import data.data_load_datasetall_aug_all.data_utils.frame_utils as data

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_time(elapse):
    elapse = int(elapse)
    hour = elapse // 3600
    minute = elapse % 3600 // 60
    seconds = elapse % 60
    return "{:02d}:{:02d}:{:02d}".format(hour, minute, seconds)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

@torch.no_grad()
def validate_eth3d(model, input_image,output_image, worklog, save_txt, tall, total_steps, infer_level=1, iters=32, split='F',
                        mixed_prec=False,muti=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)
    worklog.info(f"Dataset size:{len(val_dataset)}")
    worklog.info(f"valid_iters：{iters}")

    # 图像输出保存路径
    output_image = output_image + "/" + "eth3d" + str(infer_level) + "_" + str(muti) + "_" + str(input_image)
    ensure_dir(output_image)

    out_list05, epe_list = [], []
    out_list1, out_list2, out_list3 = [], [], []
    out_list4, out_list5 = [], []
    validate_time = []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1_initial, image2_initial, flow_gt, valid_gt = val_dataset[val_id]
        image1_initial = image1_initial[None].cuda()
        image2_initial = image2_initial[None].cuda()

        t_size = input_image[1] / image1_initial.shape[3]
        assert t_size != 0
        image1 = F.interpolate(
            image1_initial,
            size=(input_image[0], input_image[1]),
            mode="bilinear",
            align_corners=True,
        )
        image2 = F.interpolate(
            image2_initial,
            size=(input_image[0], input_image[1]),
            mode="bilinear",
            align_corners=True,
        )

        if infer_level == 1:
            with autocast(enabled=mixed_prec):
                t0 = time.perf_counter()
                pred_flow = model(image1, image2, iters=iters, test_mode=True)
                t1 = time.perf_counter()

                flow_pr = 1 / t_size * F.interpolate(
                    pred_flow,
                    size=(image2_initial.shape[2], image2_initial.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                flow_pr = flow_pr[:, :1]
                flow_pr = flow_pr.cpu().squeeze(0)

        if infer_level == 2:
            imgL_dw2 = F.interpolate(
                image1,
                size=(image1.shape[2] // 2, image1.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )
            imgR_dw2 = F.interpolate(
                image2,
                size=(image2.shape[2] // 2, image2.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )
            with autocast(enabled=mixed_prec):
                t0 = time.perf_counter()
                pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=iters, flow_init=None, test_mode=True)
                pred_flow = model(image1, image2, iters=iters, flow_init=pred_flow_dw2, muti=muti, test_mode=True)
                t1 = time.perf_counter()

                flow_pr = 1 / t_size * F.interpolate(
                    pred_flow,
                    size=(image2_initial.shape[2], image2_initial.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                flow_pr = flow_pr[:, :1]
                flow_pr = flow_pr.cpu().squeeze(0)

        if infer_level == 3:
            imgL_dw4 = F.interpolate(
                image1,
                size=(image1.shape[2] // 4, image1.shape[3] // 4),
                mode="bilinear",
                align_corners=True,
            )
            imgR_dw4 = F.interpolate(
                image2,
                size=(image2.shape[2] // 4, image2.shape[3] // 4),
                mode="bilinear",
                align_corners=True,
            )
            with autocast(enabled=mixed_prec):
                t0 = time.perf_counter()
                pred_flow_dw4 = model(imgL_dw4, imgR_dw4, iters=iters, flow_init=None, test_mode=True)
                pred_flow = model(image1, image2, iters=iters, flow_init=pred_flow_dw4, muti=muti, test_mode=True)
                t1 = time.perf_counter()

                # 上采样
                flow_pr = 1 / t_size * F.interpolate(
                    pred_flow,
                    size=(image2_initial.shape[2], image2_initial.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                flow_pr = flow_pr[:, :1]
                flow_pr = flow_pr.cpu().squeeze(0)

        if infer_level == 4:
            imgL_dw2 = F.interpolate(
                image1,
                size=(image1.shape[2] // 2, image1.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )
            imgR_dw2 = F.interpolate(
                image2,
                size=(image2.shape[2] // 2, image2.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )

            imgL_dw4 = F.interpolate(
                image1,
                size=(image1.shape[2] // 4, image1.shape[3] // 4),
                mode="bilinear",
                align_corners=True,
            )
            imgR_dw4 = F.interpolate(
                image2,
                size=(image2.shape[2] // 4, image2.shape[3] // 4),
                mode="bilinear",
                align_corners=True,
            )

            with autocast(enabled=mixed_prec):
                t0 = time.perf_counter()
                pred_flow_dw4 = model(imgL_dw4, imgR_dw4, iters=iters, flow_init=None, test_mode=True)
                pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=iters, flow_init=pred_flow_dw4, muti=muti,
                                      test_mode=True)
                pred_flow = model(image1, image2, iters=iters, flow_init=pred_flow_dw2, muti=muti, test_mode=True)
                t1 = time.perf_counter()

                # 上采样
                flow_pr = 1 / t_size * F.interpolate(
                    pred_flow,
                    size=(image2_initial.shape[2], image2_initial.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                flow_pr = flow_pr[:, :1]
                flow_pr = flow_pr.cpu().squeeze(0)

        flow_p2 = flow_pr.abs()
        disp = flow_p2[0].detach().cpu().numpy()
        file_center = imageL_file.split('\\')[2]
        save_name = output_image + "/" + file_center
        save_name = output_image + "/" + file_center.split('/')[0]
        # save_name = os.path.join(output_image, imageL_file.split('/')[-2])
        # ensure_dir(save_name)
        input_path = save_name + ".pfm"
        # 可视化视差
        import cv2
        disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
        disp_vis = disp_vis.astype("uint8")
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(save_name + ".jpg", disp_vis)
        # skimage.io.imsave(save_name+"2.jpg", (disp * 256.).astype(np.uint16))
        # 保存视差
        data.write_pfm(input_path, disp)

        validate_time_pass = t1 - t0
        validate_time_passall = format_time(t1 - tall)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.flatten() > 0) & (flow_gt.abs().flatten() < 70)
        # val = valid_gt.flatten() >= 0

        out05 = (epe_flattened > 0.5)
        out1 = (epe_flattened > 1)
        out2 = (epe_flattened > 2)
        out3 = (epe_flattened > 3)
        out4 = (epe_flattened > 4)
        out5 = (epe_flattened > 5)
        image_out05 = out05[val].float().mean().item()
        image_out1 = out1[val].float().mean().item()
        image_out2 = out2[val].float().mean().item()
        image_out3 = out3[val].float().mean().item()
        image_out4 = out4[val].float().mean().item()
        image_out5 = out5[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        validate_time.append(validate_time_pass)
        image_name = imageL_file.split('/')[-2] + "/" + imageL_file.split('/')[-1]
        worklog.info(
            f"Eth3d {image_name} Iter {val_id + 1} out of {len(val_dataset)}. image size:{image1.shape},flow_gt size:{flow_gt.shape},EPE {round(image_epe, 4)} D0.5 {round(image_out05, 4)},"
            f"D1 {round(image_out1, 4)},D2 {round(image_out2, 4)},D3 {round(image_out3, 4)},D4 {round(image_out4, 4)},D5 {round(image_out5, 4)},validate_time_pass：{validate_time_pass}，validate_time_passall:{validate_time_passall}")
        epe_list.append(image_epe)
        out_list05.append(image_out05)
        out_list1.append(image_out1)
        out_list2.append(image_out2)
        out_list3.append(image_out3)
        out_list4.append(image_out4)
        out_list5.append(image_out5)

    epe_list = np.array(epe_list)
    out_list05 = np.array(out_list05)
    out_list1 = np.array(out_list1)
    out_list2 = np.array(out_list2)
    out_list3 = np.array(out_list3)
    out_list4 = np.array(out_list4)
    out_list5 = np.array(out_list5)

    epe = np.mean(epe_list)
    d05 = 100 * np.mean(out_list05)
    d1 = 100 * np.mean(out_list1)
    d2 = 100 * np.mean(out_list2)
    d3 = 100 * np.mean(out_list3)
    d4 = 100 * np.mean(out_list4)
    d5 = 100 * np.mean(out_list5)

    # validate_time = np.concatenate(validate_time)
    validate_time_mean = np.mean(validate_time)
    worklog.info(f"validate_time_mean:{validate_time_mean}")
    save_txt = save_txt + '_' + split + '_' + str(infer_level) + "_" + f"{muti}" + '.txt'
    with open(save_txt, 'a+') as f:
        f.write('\n' + "此时的迭代次数为：" + str(total_steps) + '\n'
                + "epw_list:" + '\n'
                + str(epe_list) + '\n'
                + "validate_time:" + '\n'
                + str(validate_time) + '\n'
                + "epe的均值为：" + str(epe) + '\n'
                + "d0.5的均值为：" + str(d05) + '\n'
                + "d1的均值为：" + str(d1) + '\n'
                + "d2的均值为：" + str(d2) + '\n'
                + "d3的均值为：" + str(d3) + '\n'
                + "d4的均值为：" + str(d4) + '\n'
                + "d5的均值为：" + str(d5) + '\n'
                + "平均运行时间：" + str(validate_time_mean) + '\n')
    f.close()

    worklog.info(
        f"Validation Eth3d{split}_{infer_level}_{muti}: EPE: {epe}, D0.5: {d05}, D1: {d1},D2: {d2},D3: {d3},D4: {d4},D5: {d5},validate_time_mean:{validate_time_mean}")
    return {f'Eth3d{split}-epe': epe, f'Eth3d{split}-d0.5': d05, f'Eth3d{split}-d1': d1,
            f'Eth3d{split}-d2': d2, f'Eth3d{split}-d3': d3, f'Eth3d{split}-d4': d4,
            f'Eth3d{split}-d5': d5}

@torch.no_grad()
def validate_eth3d_test(model, input_image,output_image, worklog, save_txt, tall, total_steps, infer_level=1, iters=32, split='F',
                        mixed_prec=False,muti=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    # aug_params = {}
    aug_params = None
    val_dataset = datasets.ETH3D_test(aug_params)
    worklog.info(f"Dataset size:{len(val_dataset)}")
    worklog.info(f"valid_iters：{iters}")

    # 图像输出保存路径
    output_image = output_image + "/" + "eth3d" + str(infer_level) + "_" + str(muti) + "_" + str(input_image)
    ensure_dir(output_image)

    validate_time = []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1_initial, image2_initial, flow_gt, valid_gt = val_dataset[val_id]
        image1_initial = image1_initial[None].cuda()
        image2_initial = image2_initial[None].cuda()

        t_size = input_image[1] / image1_initial.shape[3]
        assert t_size != 0
        image1 = F.interpolate(
            image1_initial,
            size=(input_image[0], input_image[1]),
            mode="bilinear",
            align_corners=True,
        )
        image2 = F.interpolate(
            image2_initial,
            size=(input_image[0], input_image[1]),
            mode="bilinear",
            align_corners=True,
        )
        # padder = InputPadder(image1_initial.shape, divis_by=3*times)
        # image1, image2 = padder.pad(image1_initial, image2_initial)

        if infer_level == 1:
            with autocast(enabled=mixed_prec):
                t0 = time.perf_counter()
                pred_flow = model(image1, image2, iters=iters, test_mode=True)
                t1 = time.perf_counter()

                flow_pr = 1 / t_size * F.interpolate(
                    pred_flow,
                    size=(image2_initial.shape[2], image2_initial.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                flow_pr=flow_pr[:,:1]
                flow_pr = flow_pr.cpu().squeeze(0)

        if infer_level == 2:
            imgL_dw2 = F.interpolate(
                image1,
                size=(image1.shape[2] // 2, image1.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )
            imgR_dw2 = F.interpolate(
                image2,
                size=(image2.shape[2] // 2, image2.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )
            with autocast(enabled=mixed_prec):
                t0 = time.perf_counter()
                pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=iters, flow_init=None, test_mode=True)
                pred_flow = model(image1, image2, iters=iters, flow_init=pred_flow_dw2,muti=muti,test_mode=True)
                t1 = time.perf_counter()

                flow_pr = 1 / t_size * F.interpolate(
                    pred_flow,
                    size=(image2_initial.shape[2], image2_initial.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                # flow_pr = padder.unpad(pred_flow.float())
                flow_pr = flow_pr[:, :1]
                flow_pr = flow_pr.cpu().squeeze(0)

        if infer_level == 3:
            imgL_dw4 = F.interpolate(
                image1,
                size=(image1.shape[2] // 4, image1.shape[3] // 4),
                mode="bilinear",
                align_corners=True,
            )
            imgR_dw4 = F.interpolate(
                image2,
                size=(image2.shape[2] // 4, image2.shape[3] // 4),
                mode="bilinear",
                align_corners=True,
            )
            with autocast(enabled=mixed_prec):
                t0 = time.perf_counter()
                pred_flow_dw4 = model(imgL_dw4, imgR_dw4, iters=iters, flow_init=None, test_mode=True)
                pred_flow = model(image1, image2, iters=iters, flow_init=pred_flow_dw4,muti=muti,test_mode=True)
                t1 = time.perf_counter()

                # 上采样
                flow_pr = 1 / t_size * F.interpolate(
                    pred_flow,
                    size=(image2_initial.shape[2], image2_initial.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                # flow_pr = padder.unpad(pred_flow.float())
                flow_pr = flow_pr[:, :1]
                flow_pr = flow_pr.cpu().squeeze(0)

        if infer_level == 4:
            imgL_dw2 = F.interpolate(
                image1,
                size=(image1.shape[2] // 2, image1.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )
            imgR_dw2 = F.interpolate(
                image2,
                size=(image2.shape[2] // 2, image2.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )

            imgL_dw4 = F.interpolate(
                image1,
                size=(image1.shape[2] // 4, image1.shape[3] // 4),
                mode="bilinear",
                align_corners=True,
            )
            imgR_dw4 = F.interpolate(
                image2,
                size=(image2.shape[2] // 4, image2.shape[3] // 4),
                mode="bilinear",
                align_corners=True,
            )

            with autocast(enabled=mixed_prec):
                t0 = time.perf_counter()
                pred_flow_dw4 = model(imgL_dw4, imgR_dw4, iters=iters, flow_init=None, test_mode=True)
                pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=iters, flow_init=pred_flow_dw4,muti=muti,
                                      test_mode=True)
                pred_flow = model(image1, image2, iters=iters, flow_init=pred_flow_dw2,muti=muti, test_mode=True)
                t1 = time.perf_counter()

                # 上采样
                flow_pr = 1 / t_size * F.interpolate(
                    pred_flow,
                    size=(image2_initial.shape[2], image2_initial.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                # flow_pr = padder.unpad(pred_flow.float())
                flow_pr = flow_pr[:, :1]
                flow_pr = flow_pr.cpu().squeeze(0)

        flow_p2 = flow_pr.abs()
        disp = flow_p2[0].detach().cpu().numpy()
        file_center=imageL_file.split('\\')[2]
        save_name = output_image + "/" + file_center
        # save_name=output_image+"/"+file_center.split('/')[0]
        # save_name = os.path.join(output_image, imageL_file.split('/')[-2])
        # ensure_dir(save_name)
        input_path = save_name + ".pfm"
        # 可视化视差
        import cv2
        disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
        disp_vis = disp_vis.astype("uint8")
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(save_name + ".jpg", disp_vis)
        # skimage.io.imsave(save_name+"2.jpg", (disp * 256.).astype(np.uint16))
        # 保存视差
        data.write_pfm(input_path, disp)

        validate_time_pass = t1 - t0
        validate_time_passall = format_time(t1 - tall)

        validate_time.append(validate_time_pass)
        image_name = imageL_file.split('/')[-2] + "/" + imageL_file.split('/')[-1]
        worklog.info(
            f"Middlebury {image_name} Iter {val_id + 1} out of {len(val_dataset)}. image size:{image1.shape},flow_gt size:{flow_gt.shape},validate_time_pass：{validate_time_pass}，validate_time_passall:{validate_time_passall}")


    # validate_time = np.concatenate(validate_time)
    validate_time_mean = np.mean(validate_time)
    worklog.info(f"validate_time_mean:{validate_time_mean}")
    save_txt = save_txt + '_' + split + '_' + str(infer_level)+"_"+f"{muti}"+'.txt'
    with open(save_txt, 'a+') as f:
        f.write('\n' + "此时的迭代次数为：" + str(total_steps) + '\n'
                + "validate_time:" + '\n'
                + str(validate_time) + '\n'
                + "平均运行时间：" + str(validate_time_mean) + '\n')
    f.close()

    worklog.info(
        f"Validation Middlebury{split}_{infer_level}_{muti}:validate_time_mean:{validate_time_mean}")
    return {f"validate_time_mean:{validate_time_mean}"}


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
def validate_things(model,input_image,output_image,worklog,save_txt,tall,total_steps,infer_level=1,iters=10,sum_sample=400,mixed_prec=False,muti=False):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass',sum_sample=sum_sample,things_test=True)
    worklog.info(f"Dataset size:{len(val_dataset)}")
    worklog.info(f"valid_iters：{iters}")

    # 图像输出保存路径
    output_image = output_image + "/" + "SceneFlow" + str(infer_level) + "_" + str(muti) + "_" + str(input_image)
    ensure_dir(output_image)

    out_list, epe_list = [], []
    out_list3,out_list5=[], []
    validate_time = []
    for val_id in tqdm(range(len(val_dataset))):
        (imageL_file, _, _), image1_initial, image2_initial, flow_gt, valid_gt = val_dataset[val_id]
        image1_initial = image1_initial[None].cuda()
        image2_initial = image2_initial[None].cuda()

        # padder = InputPadder(image1_initial.shape, divis_by=32)
        # image1, image2 = padder.pad(image1_initial, image2_initial)

        t_size = input_image[1] / image1_initial.shape[3]
        assert t_size != 0
        image1 = F.interpolate(
            image1_initial,
            size=(input_image[0], input_image[1]),
            mode="bilinear",
            align_corners=True,
        )
        image2 = F.interpolate(
            image2_initial,
            size=(input_image[0], input_image[1]),
            mode="bilinear",
            align_corners=True,
        )

        if infer_level == 1:
            with autocast(enabled=mixed_prec):
                t0 = time.perf_counter()
                pred_flow = model(image1, image2, iters=iters, test_mode=True)
                t1 = time.perf_counter()

                # flow_pr = padder.unpad(pred_flow.float())
                flow_pr = 1 / t_size * F.interpolate(
                    pred_flow,
                    size=(image2_initial.shape[2], image2_initial.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                flow_pr = flow_pr[:, :1]
                flow_pr = flow_pr.cpu().squeeze(0)

        if infer_level == 2:
            imgL_dw2 = F.interpolate(
                image1,
                size=(image1.shape[2] // 2, image1.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )
            imgR_dw2 = F.interpolate(
                image2,
                size=(image2.shape[2] // 2, image2.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )
            with autocast(enabled=mixed_prec):
                t0 = time.perf_counter()
                pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=iters, flow_init=None, test_mode=True)
                pred_flow = model(image1, image2, iters=iters, flow_init=pred_flow_dw2, muti=muti, test_mode=True)
                t1 = time.perf_counter()

                flow_pr = 1 / t_size * F.interpolate(
                    pred_flow,
                    size=(image2_initial.shape[2], image2_initial.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                # flow_pr = padder.unpad(pred_flow.float())
                flow_pr = flow_pr[:, :1]
                flow_pr = flow_pr.cpu().squeeze(0)

        if infer_level == 3:
            imgL_dw4 = F.interpolate(
                image1,
                size=(image1.shape[2] // 4, image1.shape[3] // 4),
                mode="bilinear",
                align_corners=True,
            )
            imgR_dw4 = F.interpolate(
                image2,
                size=(image2.shape[2] // 4, image2.shape[3] // 4),
                mode="bilinear",
                align_corners=True,
            )
            with autocast(enabled=mixed_prec):
                t0 = time.perf_counter()
                pred_flow_dw4 = model(imgL_dw4, imgR_dw4, iters=iters, flow_init=None, test_mode=True)
                pred_flow = model(image1, image2, iters=iters, flow_init=pred_flow_dw4, muti=muti, test_mode=True)
                t1 = time.perf_counter()

                # 上采样
                flow_pr = 1 / t_size * F.interpolate(
                    pred_flow,
                    size=(image2_initial.shape[2], image2_initial.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                # flow_pr = padder.unpad(pred_flow.float())
                flow_pr = flow_pr[:, :1]
                flow_pr = flow_pr.cpu().squeeze(0)

        if infer_level == 4:
            imgL_dw2 = F.interpolate(
                image1,
                size=(image1.shape[2] // 2, image1.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )
            imgR_dw2 = F.interpolate(
                image2,
                size=(image2.shape[2] // 2, image2.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )

            imgL_dw4 = F.interpolate(
                image1,
                size=(image1.shape[2] // 4, image1.shape[3] // 4),
                mode="bilinear",
                align_corners=True,
            )
            imgR_dw4 = F.interpolate(
                image2,
                size=(image2.shape[2] // 4, image2.shape[3] // 4),
                mode="bilinear",
                align_corners=True,
            )

            with autocast(enabled=mixed_prec):
                t0 = time.perf_counter()
                pred_flow_dw4 = model(imgL_dw4, imgR_dw4, iters=iters, flow_init=None, test_mode=True)
                pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=iters, flow_init=pred_flow_dw4, muti=muti,
                                      test_mode=True)
                pred_flow = model(image1, image2, iters=iters, flow_init=pred_flow_dw2, muti=muti, test_mode=True)
                t1 = time.perf_counter()

                # 上采样
                flow_pr = 1 / t_size * F.interpolate(
                    pred_flow,
                    size=(image2_initial.shape[2], image2_initial.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                # flow_pr = padder.unpad(pred_flow.float())
                flow_pr = flow_pr[:, :1]
                flow_pr = flow_pr.cpu().squeeze(0)

        flow_p2 = flow_pr.abs()
        disp = flow_p2[0].detach().cpu().numpy()
        file_center = imageL_file.split('\\')[4]
        file_center2 = imageL_file.split('\\')[5]
        file_center3 = imageL_file.split('\\')[7]
        save_name = output_image + "/" + file_center+"/"+file_center2+"/"+file_center3
        # save_name = os.path.join(output_image, imageL_file.split('/')[-2])
        save_name2=output_image + "/" + file_center+"/"+file_center2
        ensure_dir(save_name2)
        input_path = save_name + ".pfm"
        # 可视化视差
        import cv2
        disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
        disp_vis = disp_vis.astype("uint8")
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(save_name + ".jpg", disp_vis)
        # # skimage.io.imsave(save_name+"2.jpg", (disp * 256.).astype(np.uint16))
        # # 保存视差
        # data.write_pfm(input_path, disp)

        validate_time_pass = t1 - t0
        validate_time_passall = format_time(t1 - tall)

        # flow_pr=flow_pr[:,:1]
        # flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        # val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)
        # flow_gt=torch.cat([flow_gt, flow_gt * 0], dim=0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        # epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
        flow_gt = flow_gt.abs()
        epe = torch.sum((flow_p2 - flow_gt) ** 2, dim=0).sqrt()
        epe = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)
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
            d1 = out[val].cpu().numpy().mean().item()
            d3 = out3[val].cpu().numpy().mean().item()
            d5 = out5[val].cpu().numpy().mean().item()
            epe_list.append(epe[val].mean().item())
            out_list.append(out[val].cpu().numpy())
            out_list3.append(out3[val].cpu().numpy())
            out_list5.append(out5[val].cpu().numpy())
            validate_time.append(validate_time_pass)
            worklog.info(
                f"Validation FlyingThings{file_center}_{file_center2}_{file_center3} epe: {epe[val].mean().item()},d1:{d1},d3:{d3},d5:{d5},validate_time_pass:{validate_time_pass}, validate_time_passall：{validate_time_passall}，image size:{image1.shape},flow_gt:{flow_gt.shape}")

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    out_list3 = np.concatenate(out_list3)
    out_list5 = np.concatenate(out_list5)
    d3 = 100 * np.mean(out_list3)
    d5 = 100 * np.mean(out_list5)

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
        f"Validation FlyingThings_{input_image}_{infer_level}_{muti} epe: {epe},d1:{d1},d3:{d3},d5:{d5},validate_time_mean:{validate_time_mean}")
    # print("Validation FlyingThings: %f, %f,%f,%f" % (epe, d1,d3,d5))
    return {f'thing_{sum_sample}_epe': epe, f'thing_{sum_sample}_d1': d1,f'thing_{sum_sample}_d3': d3,f'thing_{sum_sample}_d5': d5}


@torch.no_grad()
def validate_middlebury(model, input_image,output_image, worklog, save_txt, tall, total_steps, infer_level=1, iters=32, split='F',
                        mixed_prec=False,muti=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    # aug_params = {}
    aug_params = None
    val_dataset = datasets.Middlebury(aug_params, split=split)
    worklog.info(f"Dataset size:{len(val_dataset)}")
    worklog.info(f"valid_iters：{iters}")

    # 图像输出保存路径
    output_image = output_image + "/" + "middlebury" + str(infer_level) + "_" + str(muti) + "_" + str(input_image)
    ensure_dir(output_image)

    times=32
    if infer_level==1:
        times = 32
    if infer_level == 2:
        times = 64
    if infer_level == 3:
        times = 128
    if infer_level == 4:
        times = 128

    out_list05, epe_list = [], []
    out_list1, out_list2, out_list3 = [], [], []
    out_list4, out_list5 = [], []
    validate_time = []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1_initial, image2_initial, flow_gt, valid_gt = val_dataset[val_id]
        image1_initial = image1_initial[None].cuda()
        image2_initial = image2_initial[None].cuda()

        t_size = input_image[1] / image1_initial.shape[3]
        assert t_size != 0
        image1 = F.interpolate(
            image1_initial,
            size=(input_image[0], input_image[1]),
            mode="bilinear",
            align_corners=True,
        )
        image2 = F.interpolate(
            image2_initial,
            size=(input_image[0], input_image[1]),
            mode="bilinear",
            align_corners=True,
        )

        # padder = InputPadder(image1_initial.shape, divis_by=times)
        # image1, image2 = padder.pad(image1_initial, image2_initial)

        if infer_level == 1:
            with autocast(enabled=mixed_prec):
                t0 = time.perf_counter()
                pred_flow = model(image1, image2, iters=iters, test_mode=True)
                t1 = time.perf_counter()


                flow_pr = 1 / t_size * F.interpolate(
                    pred_flow,
                    size=(image2_initial.shape[2], image2_initial.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                # flow_pr = padder.unpad(pred_flow.float())
                flow_pr=flow_pr[:,:1]
                flow_pr = flow_pr.cpu().squeeze(0)

        if infer_level == 2:
            imgL_dw2 = F.interpolate(
                image1,
                size=(image1.shape[2] // 2, image1.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )
            imgR_dw2 = F.interpolate(
                image2,
                size=(image2.shape[2] // 2, image2.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )
            with autocast(enabled=mixed_prec):
                t0 = time.perf_counter()
                pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=iters, flow_init=None, test_mode=True)
                pred_flow = model(image1, image2, iters=iters, flow_init=pred_flow_dw2,muti=muti,test_mode=True)
                t1 = time.perf_counter()

                flow_pr = 1 / t_size * F.interpolate(
                    pred_flow,
                    size=(image2_initial.shape[2], image2_initial.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                # flow_pr = padder.unpad(pred_flow.float())
                flow_pr = flow_pr[:, :1]
                flow_pr = flow_pr.cpu().squeeze(0)

        if infer_level == 3:
            imgL_dw4 = F.interpolate(
                image1,
                size=(image1.shape[2] // 4, image1.shape[3] // 4),
                mode="bilinear",
                align_corners=True,
            )
            imgR_dw4 = F.interpolate(
                image2,
                size=(image2.shape[2] // 4, image2.shape[3] // 4),
                mode="bilinear",
                align_corners=True,
            )
            with autocast(enabled=mixed_prec):
                t0 = time.perf_counter()
                pred_flow_dw4 = model(imgL_dw4, imgR_dw4, iters=iters, flow_init=None, test_mode=True)
                pred_flow = model(image1, image2, iters=iters, flow_init=pred_flow_dw4,muti=muti,test_mode=True)
                t1 = time.perf_counter()

                # 上采样
                flow_pr = 1 / t_size * F.interpolate(
                    pred_flow,
                    size=(image2_initial.shape[2], image2_initial.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                # flow_pr = padder.unpad(pred_flow.float())
                flow_pr = flow_pr[:, :1]
                flow_pr = flow_pr.cpu().squeeze(0)

        if infer_level == 4:
            imgL_dw2 = F.interpolate(
                image1,
                size=(image1.shape[2] // 2, image1.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )
            imgR_dw2 = F.interpolate(
                image2,
                size=(image2.shape[2] // 2, image2.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )

            imgL_dw4 = F.interpolate(
                image1,
                size=(image1.shape[2] // 4, image1.shape[3] // 4),
                mode="bilinear",
                align_corners=True,
            )
            imgR_dw4 = F.interpolate(
                image2,
                size=(image2.shape[2] // 4, image2.shape[3] // 4),
                mode="bilinear",
                align_corners=True,
            )

            with autocast(enabled=mixed_prec):
                t0 = time.perf_counter()
                pred_flow_dw4 = model(imgL_dw4, imgR_dw4, iters=iters, flow_init=None, test_mode=True)
                pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=iters, flow_init=pred_flow_dw4,muti=muti,
                                      test_mode=True)
                pred_flow = model(image1, image2, iters=iters, flow_init=pred_flow_dw2,muti=muti, test_mode=True)
                t1 = time.perf_counter()

                # 上采样
                flow_pr = 1 / t_size * F.interpolate(
                    pred_flow,
                    size=(image2_initial.shape[2], image2_initial.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                # flow_pr = padder.unpad(pred_flow.float())
                flow_pr = flow_pr[:, :1]
                flow_pr = flow_pr.cpu().squeeze(0)

        flow_p2 = flow_pr.abs()
        disp = flow_p2[0].detach().cpu().numpy()
        file_center=imageL_file.split('\\')[-1]
        save_name=output_image+"/"+file_center.split('/')[0]
        # save_name = os.path.join(output_image, imageL_file.split('/')[-2])
        # ensure_dir(save_name)
        input_path = save_name + ".pfm"
        # 可视化视差
        import cv2
        disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
        disp_vis = disp_vis.astype("uint8")
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(save_name + ".jpg", disp_vis)
        # skimage.io.imsave(save_name+"2.jpg", (disp * 256.).astype(np.uint16))
        # 保存视差
        data.write_pfm(input_path, disp)

        validate_time_pass = t1 - t0
        validate_time_passall = format_time(t1 - tall)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.reshape(-1) >= -0.5) & (flow_gt[0].reshape(-1) > -1000)


        out05 = (epe_flattened > 0.5)
        out1 = (epe_flattened > 1)
        out2 = (epe_flattened > 2)
        out3 = (epe_flattened > 3)
        out4 = (epe_flattened > 4)
        out5 = (epe_flattened > 5)
        image_out05 = out05[val].float().mean().item()
        image_out1 = out1[val].float().mean().item()
        image_out2 = out2[val].float().mean().item()
        image_out3 = out3[val].float().mean().item()
        image_out4 = out4[val].float().mean().item()
        image_out5 = out5[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        validate_time.append(validate_time_pass)
        image_name = imageL_file.split('/')[-2] + "/" + imageL_file.split('/')[-1]
        worklog.info(
            f"Middlebury {image_name} Iter {val_id + 1} out of {len(val_dataset)}. image size:{image1.shape},flow_gt size:{flow_gt.shape},EPE {round(image_epe, 4)} D0.5 {round(image_out05, 4)},"
            f"D1 {round(image_out1, 4)},D2 {round(image_out2, 4)},D3 {round(image_out3, 4)},D4 {round(image_out4, 4)},D5 {round(image_out5, 4)},validate_time_pass：{validate_time_pass}，validate_time_passall:{validate_time_passall}")
        epe_list.append(image_epe)
        out_list05.append(image_out05)
        out_list1.append(image_out1)
        out_list2.append(image_out2)
        out_list3.append(image_out3)
        out_list4.append(image_out4)
        out_list5.append(image_out5)

    epe_list = np.array(epe_list)
    out_list05 = np.array(out_list05)
    out_list1 = np.array(out_list1)
    out_list2 = np.array(out_list2)
    out_list3 = np.array(out_list3)
    out_list4 = np.array(out_list4)
    out_list5 = np.array(out_list5)

    epe = np.mean(epe_list)
    d05 = 100 * np.mean(out_list05)
    d1 = 100 * np.mean(out_list1)
    d2 = 100 * np.mean(out_list2)
    d3 = 100 * np.mean(out_list3)
    d4 = 100 * np.mean(out_list4)
    d5 = 100 * np.mean(out_list5)

    # validate_time = np.concatenate(validate_time)
    validate_time_mean = np.mean(validate_time)
    worklog.info(f"validate_time_mean:{validate_time_mean}")
    save_txt = save_txt + '_' + split + '_' + str(infer_level)+"_"+f"{muti}"+'.txt'
    with open(save_txt, 'a+') as f:
        f.write('\n' + "此时的迭代次数为：" + str(total_steps) + '\n'
                + "epw_list:" + '\n'
                + str(epe_list) + '\n'
                + "epe的均值为：" + str(epe) + '\n'
                + "d0.5的均值为：" + str(d05) + '\n'
                + "d1的均值为：" + str(d1) + '\n'
                + "d2的均值为：" + str(d2) + '\n'
                + "d3的均值为：" + str(d3) + '\n'
                + "d4的均值为：" + str(d4) + '\n'
                + "d5的均值为：" + str(d5) + '\n'
                + "平均运行时间：" + str(validate_time_mean) + '\n')
    f.close()

    worklog.info(
        f"Validation Middlebury{split}_{infer_level}_{muti}: EPE: {epe}, D0.5: {d05}, D1: {d1},D2: {d2},D3: {d3},D4: {d4},D5: {d5},validate_time_mean:{validate_time_mean}")
    return {f'middlebury{split}-epe': epe, f'middlebury{split}-d0.5': d05,f'middlebury{split}-d1': d1,f'middlebury{split}-d2': d2,f'middlebury{split}-d3': d3,f'middlebury{split}-d4': d4,f'middlebury{split}-d5': d5}

@torch.no_grad()
def validate_middlebury_test(model, input_image,output_image, worklog, save_txt, tall, total_steps, infer_level=1, iters=32, split='F',
                        mixed_prec=False,muti=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    # aug_params = {}
    aug_params = None
    val_dataset = datasets.Middlebury_test(aug_params, split=split)
    worklog.info(f"Dataset size:{len(val_dataset)}")
    worklog.info(f"valid_iters：{iters}")

    # 图像输出保存路径
    output_image = output_image + "/" + "middlebury" + str(infer_level) + "_" + str(muti) + "_" + str(input_image)
    ensure_dir(output_image)

    times = 32
    if infer_level == 1:
        times = 32
    if infer_level == 2:
        times = 64
    if infer_level == 3:
        times = 128
    if infer_level == 4:
        times = 128


    validate_time = []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1_initial, image2_initial, flow_gt, valid_gt = val_dataset[val_id]
        image1_initial = image1_initial[None].cuda()
        image2_initial = image2_initial[None].cuda()

        # t_size = input_image[1] / image1_initial.shape[3]
        # assert t_size != 0
        # image1 = F.interpolate(
        #     image1_initial,
        #     size=(input_image[0], input_image[1]),
        #     mode="bilinear",
        #     align_corners=True,
        # )
        # image2 = F.interpolate(
        #     image2_initial,
        #     size=(input_image[0], input_image[1]),
        #     mode="bilinear",
        #     align_corners=True,
        # )
        padder = InputPadder(image1_initial.shape, divis_by=3*times)
        image1, image2 = padder.pad(image1_initial, image2_initial)

        if infer_level == 1:
            with autocast(enabled=mixed_prec):
                t0 = time.perf_counter()
                pred_flow = model(image1, image2, iters=iters, test_mode=True)
                t1 = time.perf_counter()

                # flow_pr = 1 / t_size * F.interpolate(
                #     pred_flow,
                #     size=(image2_initial.shape[2], image2_initial.shape[3]),
                #     mode="bilinear",
                #     align_corners=True,
                # )
                flow_pr = padder.unpad(pred_flow.float())
                flow_pr=flow_pr[:,:1]
                flow_pr = flow_pr.cpu().squeeze(0)

        if infer_level == 2:
            imgL_dw2 = F.interpolate(
                image1,
                size=(image1.shape[2] // 2, image1.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )
            imgR_dw2 = F.interpolate(
                image2,
                size=(image2.shape[2] // 2, image2.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )
            with autocast(enabled=mixed_prec):
                t0 = time.perf_counter()
                pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=iters, flow_init=None, test_mode=True)
                pred_flow = model(image1, image2, iters=iters, flow_init=pred_flow_dw2,muti=muti,test_mode=True)
                t1 = time.perf_counter()

                # flow_pr = 1 / t_size * F.interpolate(
                #     pred_flow,
                #     size=(image2_initial.shape[2], image2_initial.shape[3]),
                #     mode="bilinear",
                #     align_corners=True,
                # )
                flow_pr = padder.unpad(pred_flow.float())
                flow_pr = flow_pr[:, :1]
                flow_pr = flow_pr.cpu().squeeze(0)

        if infer_level == 3:
            imgL_dw4 = F.interpolate(
                image1,
                size=(image1.shape[2] // 4, image1.shape[3] // 4),
                mode="bilinear",
                align_corners=True,
            )
            imgR_dw4 = F.interpolate(
                image2,
                size=(image2.shape[2] // 4, image2.shape[3] // 4),
                mode="bilinear",
                align_corners=True,
            )
            with autocast(enabled=mixed_prec):
                t0 = time.perf_counter()
                pred_flow_dw4 = model(imgL_dw4, imgR_dw4, iters=iters, flow_init=None, test_mode=True)
                pred_flow = model(image1, image2, iters=iters, flow_init=pred_flow_dw4,muti=muti,test_mode=True)
                t1 = time.perf_counter()

                # 上采样
                # flow_pr = 1 / t_size * F.interpolate(
                #     pred_flow,
                #     size=(image2_initial.shape[2], image2_initial.shape[3]),
                #     mode="bilinear",
                #     align_corners=True,
                # )
                flow_pr = padder.unpad(pred_flow.float())
                flow_pr = flow_pr[:, :1]
                flow_pr = flow_pr.cpu().squeeze(0)

        if infer_level == 4:
            imgL_dw2 = F.interpolate(
                image1,
                size=(image1.shape[2] // 2, image1.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )
            imgR_dw2 = F.interpolate(
                image2,
                size=(image2.shape[2] // 2, image2.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )

            imgL_dw4 = F.interpolate(
                image1,
                size=(image1.shape[2] // 4, image1.shape[3] // 4),
                mode="bilinear",
                align_corners=True,
            )
            imgR_dw4 = F.interpolate(
                image2,
                size=(image2.shape[2] // 4, image2.shape[3] // 4),
                mode="bilinear",
                align_corners=True,
            )

            with autocast(enabled=mixed_prec):
                t0 = time.perf_counter()
                pred_flow_dw4 = model(imgL_dw4, imgR_dw4, iters=iters, flow_init=None, test_mode=True)
                pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=iters, flow_init=pred_flow_dw4,muti=muti,
                                      test_mode=True)
                pred_flow = model(image1, image2, iters=iters, flow_init=pred_flow_dw2,muti=muti, test_mode=True)
                t1 = time.perf_counter()

                # 上采样
                # flow_pr = 1 / t_size * F.interpolate(
                #     pred_flow,
                #     size=(image2_initial.shape[2], image2_initial.shape[3]),
                #     mode="bilinear",
                #     align_corners=True,
                # )
                flow_pr = padder.unpad(pred_flow.float())
                flow_pr = flow_pr[:, :1]
                flow_pr = flow_pr.cpu().squeeze(0)

        flow_p2 = flow_pr.abs()
        disp = flow_p2[0].detach().cpu().numpy()
        file_center=imageL_file.split('\\')[-1]
        save_name=output_image+"/"+file_center.split('/')[0]
        # save_name = os.path.join(output_image, imageL_file.split('/')[-2])
        # ensure_dir(save_name)
        input_path = save_name + ".pfm"
        # 可视化视差
        import cv2
        disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
        disp_vis = disp_vis.astype("uint8")
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(save_name + ".jpg", disp_vis)
        # skimage.io.imsave(save_name+"2.jpg", (disp * 256.).astype(np.uint16))
        # 保存视差
        data.write_pfm(input_path, disp)

        validate_time_pass = t1 - t0
        validate_time_passall = format_time(t1 - tall)

        validate_time.append(validate_time_pass)
        image_name = imageL_file.split('/')[-2] + "/" + imageL_file.split('/')[-1]
        worklog.info(
            f"Middlebury {image_name} Iter {val_id + 1} out of {len(val_dataset)}. image size:{image1.shape},flow_gt size:{flow_gt.shape},validate_time_pass：{validate_time_pass}，validate_time_passall:{validate_time_passall}")


    # validate_time = np.concatenate(validate_time)
    validate_time_mean = np.mean(validate_time)
    worklog.info(f"validate_time_mean:{validate_time_mean}")
    save_txt = save_txt + '_' + split + '_' + str(infer_level)+"_"+f"{muti}"+'.txt'
    with open(save_txt, 'a+') as f:
        f.write('\n' + "此时的迭代次数为：" + str(total_steps) + '\n'
                + "平均运行时间：" + str(validate_time_mean) + '\n')
    f.close()

    worklog.info(
        f"Validation Middlebury{split}_{infer_level}_{muti}:validate_time_mean:{validate_time_mean}")
    return {f"validate_time_mean:{validate_time_mean}"}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--dataset', help="dataset for evaluation", required=True, choices=["eth3d", "kitti", "things"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    args = parser.parse_args()

    model = torch.nn.DataParallel(CREStereo(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    # The CUDA implementations of the correlation volume prevent half-precision
    # rounding errors in the correlation lookup. This allows us to use mixed precision
    # in the entire forward pass, not just in the GRUs & feature extractors. 
    use_mixed_precision = args.corr_implementation.endswith("_cuda")

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, split=args.dataset[-1], mixed_prec=use_mixed_precision)

    elif args.dataset == 'things':
        validate_things(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
