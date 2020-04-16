# coding: utf-8
# from data_provider import *
from C3D_model import *
import torchvision
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import os 
from torch import save, load
import pickle
import time
import numpy as np
import PIL.Image as Image
import skimage.io as io
from skimage.transform import resize
import h5py
from PIL import Image
import cv2

def feature_extractor():
	#trainloader = Train_Data_Loader( VIDEO_DIR, resize_w=128, resize_h=171, crop_w = 112, crop_h = 112, nb_frames=16)
	net = C3D(487)
	print('net', net)
	## Loading pretrained model from sports and finetune the last layer
	net.load_state_dict(torch.load('./pretrained_models/c3d.pickle'))
	if RUN_GPU : 
		net.cuda(0)
		net.eval()
		print('net', net)
	feature_dim = 4096 if EXTRACTED_LAYER != 5 else 8192

	path_list = []
	video_list = []
	for dir_name, _, file_names in os.walk(VIDEO_DIR):
		for file_name in file_names:
			if file_name.endswith(('.mp4', '.mkv', '.avi', '.wmv', '.iso')):
				path_list.append(os.path.join(dir_name, file_name))
				video_list.append(os.path.splitext(file_name)[0])
	    		#print(os.path.join(dirname, filename))
	gpu_id = args.gpu_id

	if not os.path.isdir(OUTPUT_DIR):
		os.mkdir(OUTPUT_DIR)

	# current location
	temp_path = os.path.join(os.getcwd(), 'temp')
	if not os.path.exists(temp_path):
		os.mkdir(temp_path)

	error_fid = open('error.txt', 'w')
	for video_path, file_name in zip(path_list, video_list): 
		print('video_path', video_path)
		print('file_name', file_name)
		frame_path = os.path.join(temp_path, file_name + '_frames')
		if not os.path.exists(frame_path):
			os.mkdir(frame_path)

		feature_file = open(os.path.join(OUTPUT_DIR, file_name + '_' + OUTPUT_NAME), 'w')

		print('Extracting video frames ...')

		index_w = np.random.randint(resize_w - crop_w) ## crop
		index_h = np.random.randint(resize_h - crop_h) ## crop

		features = []

		capture = cv2.VideoCapture(video_path)  #视频名称
		print(capture.isOpened())
		start_frame = 1
		end_frame = 1
		while True: 
			# cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)	# 设置起始帧
		    ret, img = capture.read()  
		    if not ret:	# last batch
		    	if start_frame == 1 and end_frame == 1:
		    		error_fid.write(file_name+'\n')
		    		print('Fail to extract frames for video: %s'%file_name)
		    		break
		    	if start_frame + nb_frames > end_frame:
		    		break
		    	input_blobs = []
		    	# real_size = int((end_frame - start_frame) / nb_frames) + (0 if (end_frame - start_frame) % nb_frames == 0 else 1)
		    	while(start_frame + nb_frames <= end_frame):
		    		clip = []
		    		#print(range(start_frame, min(start_frame + nb_frames, end_frame)))
		    		clip = np.array([resize(io.imread(os.path.join(frame_path, 'pic_{:05d}.jpg'.format(k))), output_shape=(resize_w, resize_h), preserve_range=True) for k in range(start_frame, start_frame + nb_frames)])
		    		start_frame = start_frame + nb_frames
		    		clip = clip[:, index_w: index_w+ crop_w, index_h: index_h+ crop_h, :]
		    		input_blobs.append(clip)
		    	input_blobs = np.array(input_blobs, dtype='float32')
		    	input_blobs = torch.from_numpy(np.float32(input_blobs.transpose(0, 4, 1, 2, 3)))
		    	input_blobs = Variable(input_blobs).cuda() if RUN_GPU else Variable(input_blobs)
		    	_, batch_output = net(input_blobs, EXTRACTED_LAYER)
		    	batch_feature  = (batch_output.data).cpu()
		    	features.append(batch_feature)
		    	break
		    cv2.imwrite(os.path.join(frame_path, 'pic_%05d.jpg'%end_frame), img)  #写出视频图片.jpg格式
		    end_frame = end_frame + 1
		    if (end_frame - start_frame) % (nb_frames * BATCH_SIZE) == 0:
		    	input_blobs = []
		    	while(start_frame < end_frame):
		    		clip = []
		    		clip = np.array([resize(io.imread(os.path.join(frame_path, 'pic_{:05d}.jpg'.format(k))), output_shape=(resize_w, resize_h), preserve_range=True) for k in range(start_frame, min(start_frame + nb_frames, end_frame))])
		    		start_frame = start_frame + nb_frames
		    		clip = clip[:, index_w: index_w+ crop_w, index_h: index_h+ crop_h, :]
		    		input_blobs.append(clip)
		    	input_blobs = np.array(input_blobs, dtype='float32')
		    	input_blobs = torch.from_numpy(np.float32(input_blobs.transpose(0, 4, 1, 2, 3)))
		    	input_blobs = Variable(input_blobs).cuda() if RUN_GPU else Variable(input_blobs)
		    	_, batch_output = net(input_blobs, EXTRACTED_LAYER)
		    	batch_feature  = (batch_output.data).cpu()
		    	features.append(batch_feature)
		    	try: 
		    		os.system('rm -rf ' + frame_path + '/*')
		    	except: 
		    		pass
		# clear temp frame folders
		try: 
			os.system('rm -rf ' + frame_path)
		except: 
			pass
		capture.release()
		features = torch.cat(features, 0)
		features = features.numpy()
		#print('features', features)
		for feature in features:
			feature_file.write(','.join(feature.astype(str).tolist()))
			feature_file.write('\n')
		print('%s has been processed...'%file_name)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	print('******--------- Extract C3D features ------*******')
	parser.add_argument('-o', '--OUTPUT_DIR', dest='OUTPUT_DIR', type=str, default='./output_frm/', help='Output file name')
	parser.add_argument('-l', '--EXTRACTED_LAYER', dest='EXTRACTED_LAYER', type=int, choices=[5, 6, 7], default=6, help='Feature extractor layer')
	parser.add_argument('-i', '--VIDEO_DIR', dest='VIDEO_DIR', type = str, help='Input Video directory')
	parser.add_argument('-gpu', '--gpu', dest='GPU', action = 'store_true', help='Run GPU?')
	parser.add_argument('--OUTPUT_NAME', default='c3d_features.hdf5', help='The output name of the hdf5 features')
	parser.add_argument('-b', '--BATCH_SIZE', default=30, help='the batch size')
	parser.add_argument('-id', '--gpu_id', default=0, type=int)
	#parser.add_argument('-p', '--video_list_file', type=str, help='the video name list')

	args = parser.parse_args()
	params = vars(args) # convert to ordinary dict
	print('parsed parameters:')

	OUTPUT_DIR = params['OUTPUT_DIR']
	EXTRACTED_LAYER = params['EXTRACTED_LAYER']
	VIDEO_DIR = params['VIDEO_DIR']
	RUN_GPU = params['GPU']
	OUTPUT_NAME = params['OUTPUT_NAME']
	BATCH_SIZE = params['BATCH_SIZE']
	crop_w = 112
	resize_w = 128
	crop_h = 112
	resize_h = 171
	nb_frames = 16	
	feature_extractor()
