
import numpy as np
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm

device = 'cuda'

def format_time(elapse):
    elapse = int(elapse)
    hour = elapse // 3600
    minute = elapse % 3600 // 60
    seconds = elapse % 60
    return "{:02d}:{:02d}:{:02d}".format(hour, minute, seconds)


def inference(autocast,model, datasets,input_image, worklog, save_txt, tall, total_steps, infer_level=1,iters=32,sum_sample=400,split='F',data_choice="middlebury",
                        mixed_prec=False):
    worklog.info(f"Model Forwarding...")
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    # aug_params = {}
    aug_params = None
    outname=data_choice
    if data_choice=="middlebury":
        val_dataset = datasets.Middlebury(aug_params, split=split)
        outname = data_choice+"_"+split
    elif data_choice == "kitti2015":
        val_dataset = datasets.KITTI(aug_params, image_set='training')
        outname = data_choice
    elif data_choice=='eth3d':
        val_dataset=datasets.ETH3D(aug_params)
        outname = data_choice
    elif data_choice=='sceneflow':
        val_dataset=datasets.SceneFlowDatasets(dstype='frames_finalpass',sum_sample=sum_sample,things_test=True)
        outname = data_choice

    worklog.info(f"Dataset size:{len(val_dataset)}")
    worklog.info(f"valid_iters：{iters}")


    torch.backends.cudnn.benchmark = True

    validate_time, epe_list = [], []
    out_list05,out_list1, out_list2, out_list4 = [], [], [],[]
    out_list3,out_list5=[],[]
    for val_id in tqdm(range(len(val_dataset))):
        (imageL_file, _, _), image1_initial, image2_initial, flow_gt, valid_gt = val_dataset[val_id]
        image1_initial = image1_initial[None].cuda()
        image2_initial = image2_initial[None].cuda()

        t_size = input_image[1] / image1_initial.shape[3]
        assert t_size != 0
        image1 = t_size * F.interpolate(
            image1_initial,
            size=(input_image[0], input_image[1]),
            mode="bilinear",
            align_corners=True,
        )
        image2 = t_size * F.interpolate(
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
                pred_flow = model(image1, image2, iters=iters, flow_init=pred_flow_dw2, test_mode=True)
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
                pred_flow = model(image1, image2, iters=iters, flow_init=pred_flow_dw4, test_mode=True)
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
                pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=iters, flow_init=pred_flow_dw4,
                                      test_mode=True)
                pred_flow = model(image1, image2, iters=iters, flow_init=pred_flow_dw2, test_mode=True)
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
        if data_choice=="sceneflow":
            image_name="sceneflow"
        else:
            image_name = imageL_file.split('/')[-2] + "/" + imageL_file.split('/')[-1]
        worklog.info(
            f"{image_name} Iter {val_id + 1} out of {len(val_dataset)}. image size:{image1.shape},flow_gt size:{flow_gt.shape},EPE {round(image_epe, 4)} D0.5 {round(image_out05, 4)},"
            f"D1 {round(image_out1, 4)},D2 {round(image_out2, 4)},D3 {round(image_out3, 4)},D4 {round(image_out4, 4)},D5 {round(image_out5, 4)},"
            f"Runtime: {format(validate_time_pass, '.6f')}s，({format(1 / validate_time_pass, '.6f')}-FPS),validate_time_passall:{validate_time_passall}")
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
    save_txt = save_txt + '_' + split + '_' + str(infer_level)
    with open(save_txt, 'a+') as f:
        f.write('\n' + "此时的迭代次数为：" + str(total_steps) + '\n'
                + "epw_list:" + '\n'
                + str(epe_list) + '\n'
                + "epe avg：" + str(epe) + '\n'
                + "d0.5 avg：" + str(d05) + '\n'
                + "d1 avg：" + str(d1) + '\n'
                + "d2 avg：" + str(d2) + '\n'
                + "d3 avg：" + str(d3) + '\n'
                + "d4 avg：" + str(d4) + '\n'
                + "d5 avg：" + str(d5) + '\n'
                + "avg runtime：" + str(validate_time_mean) + "_S"+'\n'
                + "frames：" + str(1/validate_time_mean) +"_FPS"+ '\n')
    f.close()

    worklog.info(
        f"Validation {outname}_{infer_level}: EPE: {epe}, D0.5: {d05}, D1: {d1},D2: {d2},D3: {d3},D4: {d4},D5: {d5},Runtime:{format(1/validate_time_mean, '.6f')}-FPS ({format(validate_time_mean, '.6f')}s)")
    return {f'{outname}-epe': epe, f'{outname}-d0.5': d05, f'{outname}-d1': d1,f'{outname}-d2': d2,f'{outname}-d3': d3, f'{outname}-d4': d4,f'{outname}-d5': d5}
