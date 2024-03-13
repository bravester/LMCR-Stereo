import torch
import argparse
import torch.nn.functional as F
import numpy as np
import cv2
import os
# from imread_from_url import imread_from_url

from model.pre_train_model_sceneflow.nets_feature_pre_zeroresize_dw32_16_8_loop_end import Model

device = 'cuda'


def load_model(model_path):
    print("Loading model:", os.path.abspath(model_path))
    pretrained_dict = torch.load(model_path)
    model = Model(max_disp=256, mixed_precision=False, test_mode=True)

    model.load_state_dict(pretrained_dict, strict=False)

    model.eval()
    return model


# Ref: https://github.com/megvii-research/CREStereo/blob/master/test.py
def inference(left, right, model, n_iter=20):
    print("Model Forwarding...")
    imgL = left.transpose(2, 0, 1)
    imgR = right.transpose(2, 0, 1)
    imgL = np.ascontiguousarray(imgL[None, :, :, :])
    imgR = np.ascontiguousarray(imgR[None, :, :, :])

    imgL = torch.tensor(imgL.astype("float32")).to(device)
    imgR = torch.tensor(imgR.astype("float32")).to(device)

    imgL_dw2 = F.interpolate(
        imgL,
        size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
        mode="bilinear",
        align_corners=True,
    )
    imgR_dw2 = F.interpolate(
        imgR,
        size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
        mode="bilinear",
        align_corners=True,
    )
    # print(imgR_dw2.shape)
    with torch.inference_mode():
        pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)

        pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
    pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

    return pred_disp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A demo to run CREStereo.")
    parser.add_argument(
        "--model_path",
        default="./train_log/models/latest.pth",
        help="The path of pre-trained MegEngine model.",
    )
    parser.add_argument(
        "--left", default="doc/datasets/SceneFlow7/val/0000000_left.png", help="The path of left image."
    )
    parser.add_argument(
        "--right", default="doc/datasets/SceneFlow7/val/0000000_right.png", help="The path of right image."
    )
    parser.add_argument(
        "--size",
        default="384x512",
        help="The image size for inference. Te default setting is 1024x1536. \
	                        To evaluate on ETH3D Benchmark, use 768x1024 instead.",
    )
    parser.add_argument(
        "--output", default="disparity.png", help="The path of output disparity."
    )
    args = parser.parse_args()

    assert os.path.exists(args.model_path), "The model path do not exist."
    assert os.path.exists(args.left), "The left image path do not exist."
    assert os.path.exists(args.right), "The right image path do not exist."

    # left_img = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/left.png")
    # right_img = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/right.png")

    # model_func = load_model(args.model_path)
    left = cv2.imread(args.left)
    right = cv2.imread(args.right)

    assert left.shape == right.shape, "The input images have inconsistent shapes."

    in_h, in_w = left.shape[:2]

    print("Images resized:", args.size)
    # in_h, in_w = left_img.shape[:2]
    #
    # Resize image in case the GPU memory overflows
    eval_h, eval_w = (in_h, in_w)
    imgL = cv2.resize(left, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    imgR = cv2.resize(right, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    #
    # model_path = "models/crestereo_eth3d.pth"

    model = Model(max_disp=256, mixed_precision=False, test_mode=True)
    # model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(args.model_path), strict=False)
    model.to(device)
    model.eval()

    pred = inference(imgL, imgR, model, n_iter=20)

    t = float(in_w) / float(eval_w)
    disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t
    # a=disp
    # a=a.astype("uint8")
    # cv2.namedWindow("left_disp1", cv2.WINDOW_NORMAL)
    # cv2.imshow("left_disp1", a)

    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    cv2.namedWindow("left_disp", cv2.WINDOW_NORMAL)
    cv2.imshow("left_disp", disp_vis)
    # cv2.applyColorMap
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    combined_img = np.hstack((left, disp_vis))
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow("output", combined_img)
    cv2.imwrite("output.jpg", disp_vis)
    cv2.waitKey(0)
