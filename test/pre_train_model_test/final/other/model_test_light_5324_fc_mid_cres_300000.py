import os
import sys
import time
import logging
from pathlib import Path
import torch
import torch.nn as nn
import argparse
# from tqdm import tqdm
# from imread_from_url import imread_from_url

# from model.pre_train_model_sceneflow.nets_feature_pre_zeroresize_dw32_16_8_loop_end import Model
# from model.pre_train_model_sceneflow.nets_feature_pre_zeroresize_dw32_16_8_loop_end.crestereo import autocast

from model.pre_train_model_sceneflow.nets_feature_pre_zeroresize_dw32_16_8_loop_end import Model


import evaluate.test.final.evaluate_stereo_datasatall_feature_2to1_add_pre_zeroresize_grade_dw32_16_8_loop_end_light as eva
import data.data_load_datasetall_aug_all.stereo_datasets as datasets
import inference.inference_model as eva2

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


device = 'cuda'

def main(args):
	# directory check
	log_outimage = os.path.join(args.log_dir, "out_image")
	ensure_dir(log_outimage)
	# worklog
	logging.basicConfig(level=eval(args.log_level))
	worklog = logging.getLogger("train_logger")
	worklog.propagate = False
	file_path = os.path.join(args.log_dir, "worklog.txt")
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

	world_size = torch.cuda.device_count()  # number of GPU(s)
	# 加载模型
	model = Model(
		max_disp=args.max_disp, mixed_precision=args.mixed_precision, test_mode=True
	)
	model = nn.DataParallel(model, device_ids=[0])
	model.cuda()
	# 加载现有训练模型
	total_steps=300000-1
	log_model_dir = args.log_model_dir
	chk_path = os.path.join(log_model_dir, str(total_steps+1)+"_CREStereo.pth")
	# chk_path = "Do not import the pre training model"
	worklog.info(f"model path: {chk_path}")
	if not os.path.exists(chk_path):
		chk_path = None
		worklog.info(f"loading model doesn't success")

	if chk_path is not None:
		# if rank == 0:
		worklog.info(f"loading model: {chk_path}")
		state_dict = torch.load(chk_path)
		model.load_state_dict(state_dict, strict=True)
		logging.info(f"Done loading latest checkpoint")
	# model.load_state_dict(torch.load(args.model_path), strict=False)
	model.cuda()
	model.eval()
	t0 = time.perf_counter()
	save_txt = args.log_dir + '/' + args.save_txt
	save_txt2 = args.log_dir + '/' + args.save_txt+"_2"
	save_txt3 = args.log_dir + '/' + args.save_txt+"_3"

	# results2 = eva2.inference(autocast=autocast, model=model.module, datasets=datasets, input_image=args.input_size,
	# 						worklog=worklog, save_txt=save_txt, tall=t0,
	# 						total_steps=step, infer_level=1, iters=args.valid_iters, sum_sample=400,
	# 						data_choice="sceneflow")

	# results = eva.validate_things( model=model.module, worklog=worklog, save_txt=save_txt, tall=t0,
	# 							  total_steps=step,iters=args.valid_iters,sum_sample=4370)

	results_F1 = eva.validate_middlebury_test(model=model.module, input_image=args.input_size,output_image=log_outimage, worklog=worklog,
										 save_txt=save_txt, tall=t0, total_steps=total_steps + 1,
										 infer_level=1, iters=args.valid_iters, split='F')

	results_F2 = eva.validate_middlebury_test(model=model.module, input_image=args.input_size,output_image=log_outimage, worklog=worklog,
										 save_txt=save_txt, tall=t0, total_steps=total_steps + 1,
										 infer_level=2, iters=args.valid_iters, split='F')

	results_F3 = eva.validate_middlebury_test(model=model.module, input_image=args.input_size,output_image=log_outimage, worklog=worklog,
										 save_txt=save_txt, tall=t0, total_steps=total_steps + 1,
										 infer_level=3, iters=args.valid_iters, split='F')

	results_F4 = eva.validate_middlebury_test(model=model.module, input_image=args.input_size,output_image=log_outimage, worklog=worklog,
										 save_txt=save_txt, tall=t0, total_steps=total_steps + 1,
										 infer_level=4, iters=args.valid_iters, split='F')

	results_F2_True = eva.validate_middlebury_test(model=model.module, input_image=args.input_size,output_image=log_outimage, worklog=worklog,
											  save_txt=save_txt, tall=t0, total_steps=total_steps + 1,
											  infer_level=2, iters=args.valid_iters, split='F', muti=True)
	results_F3_True = eva.validate_middlebury_test(model=model.module, input_image=args.input_size,output_image=log_outimage,
											  worklog=worklog,
											  save_txt=save_txt, tall=t0, total_steps=total_steps + 1,
											  infer_level=3, iters=args.valid_iters, split='F', muti=True)
	results_F4_True = eva.validate_middlebury_test(model=model.module, input_image=args.input_size,output_image=log_outimage,
											  worklog=worklog,
											  save_txt=save_txt, tall=t0, total_steps=total_steps + 1,
											  infer_level=4, iters=args.valid_iters, split='F', muti=True)



	results_F12 = eva.validate_middlebury_test(model=model.module, input_image=args.input_size2,output_image=log_outimage, worklog=worklog,
										 save_txt=save_txt2, tall=t0, total_steps=total_steps + 1,
										 infer_level=1, iters=args.valid_iters, split='F')

	results_F22 = eva.validate_middlebury_test(model=model.module, input_image=args.input_size2,output_image=log_outimage, worklog=worklog,
										 save_txt=save_txt2, tall=t0, total_steps=total_steps + 1,
										 infer_level=2, iters=args.valid_iters, split='F')

	results_F32 = eva.validate_middlebury_test(model=model.module, input_image=args.input_size2,output_image=log_outimage, worklog=worklog,
										 save_txt=save_txt2, tall=t0, total_steps=total_steps + 1,
										 infer_level=3, iters=args.valid_iters, split='F')

	results_F42 = eva.validate_middlebury_test(model=model.module, input_image=args.input_size2,output_image=log_outimage, worklog=worklog,
										 save_txt=save_txt2, tall=t0, total_steps=total_steps + 1,
										 infer_level=4, iters=args.valid_iters, split='F')

	results_F22_True = eva.validate_middlebury_test(model=model.module, input_image=args.input_size2,output_image=log_outimage, worklog=worklog,
											  save_txt=save_txt2, tall=t0, total_steps=total_steps + 1,
											  infer_level=2, iters=args.valid_iters, split='F', muti=True)
	results_F32_True = eva.validate_middlebury_test(model=model.module, input_image=args.input_size2,output_image=log_outimage,
											  worklog=worklog,
											  save_txt=save_txt2, tall=t0, total_steps=total_steps + 1,
											  infer_level=3, iters=args.valid_iters, split='F', muti=True)
	results_F42_True = eva.validate_middlebury_test(model=model.module, input_image=args.input_size2,output_image=log_outimage,
											  worklog=worklog,
											  save_txt=save_txt2, tall=t0, total_steps=total_steps + 1,
											  infer_level=4, iters=args.valid_iters, split='F', muti=True)

	return results_F1


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description="A demo to run CREStereo.")
	parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

	parser.add_argument('--log_dir', default='./results_float/light_5324_fc_mid_cres_300000_true2',
						help='use mixed precision')
	parser.add_argument('--log_model_dir',
						default='E:\\yxz\\Deep_learning\\CREStereo_pytorch-re\\wangyu\\CREStereo-Pytorch-main-re31_1\\train\\pre_train_model_sceneflow\\final\\pre+mid+cres\\results\\gi32_16_8_light_5324_pre_mid_fc_b4_i300000_true\\checkpoints',
						help='model save pth')
	parser.add_argument('--save_txt', default='FlyingThings3D', help='use mixed precision')
	parser.add_argument('--log_level', default='logging.INFO', help='use mixed precision')

	parser.add_argument('--max_disp', type=int, default=256, help="batch size used during training.")

	# sceneflow :[544,960] ,middlebury ;[1536.2048],kitti2015: [320,512]或者[320,1024]
	parser.add_argument('--input_size', type=int, default=[1536,2048],
						help='input image size')
	parser.add_argument('--input_size2', type=int, default=[768, 1024],
						help='input image size')
	parser.add_argument('--input_size3', type=int, default=[384, 512],
						help='input image size')
	parser.add_argument('--valid_iters', type=int, default=[5, 3, 2, 4],
						help='number of flow-field updates during validation forward pass')
	args = parser.parse_args()
	print("连接上远程ssh")
	main(args)
	print("已训练完成")
	# os.system("shutdown")

