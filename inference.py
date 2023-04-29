import os
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm

from util.util import tensor2im

torch.backends.cudnn.benchmark = True
import copy
from util.util import *
from PIL import Image

import datetime

import scipy
import cv2
import moviepy.video.io.ImageSequenceClip

from models.networks import SkyLake_G

netG = SkyLake_G().cuda().eval()

state_dict = torch.load('./checkpoints/d2c_06/latest_net_G.pth', map_location='cuda:0')  # 生成猫猫的网络

netG.load_state_dict(state_dict)

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)
])

z_ab = torch.randn(1, 64) * 0.2
z_ab[:, :32] += 1.0

z_ab = z_ab.to('cuda:0')


def func_pic(img_path='./test_samples/111.jpg'):
    real_A = Image.open(img_path)
    real_A = test_transform(real_A).unsqueeze(0).cuda()

    with torch.no_grad():
        # start = datetime.datetime.now()
        o_img = netG(real_A, z_ab)
        # end = datetime.datetime.now()
        # print('totally time is', end, start, end - start)

        o_img = tensor2im(o_img[0])
        save_image(o_img, './test_samples/111_out.jpg')


def func_video(video_path='./test_samples/in_dog2.mp4'):
    # Frame numbers and length of output video
    start_frame = 0
    end_frame = None
    frame_num = 0
    mp4_fps = 25
    faces = None
    smoothing_sec = .7
    eig_dir_idx = 1  # first eig isn't good so we skip it

    frames = []
    reader = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter('output.mp4', fourcc, mp4_fps, (256, 256), True)

    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    while reader.isOpened():  # read the video
        _, image = reader.read()
        if image is None:
            break

        # Image size
        height, width = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        frame = test_transform(image).unsqueeze(0).cuda()  # 变成网络可输入的模式

        with torch.no_grad():
            o_img, e = netG(frame, z_ab)
            o_img = tensor2im(o_img)  # 换位置了，*255了
            frames.append(o_img)

    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames, fps=mp4_fps)
    clip.write_videofile('./test_samples/out_dog2.mp4')


def mp42gif(file_name='./test_samples/out.mp4'):
    import moviepy.editor as mp

    clip = mp.VideoFileClip(file_name)
    clip.write_videofile('./test_samples/out.gif')


def get_frames(file_name='./test_samples/out_sya.mp4'):
    import moviepy.editor as mp
    clip = mp.VideoFileClip(file_name, target_resolution=(256, 256), resize_algorithm='bilinear')

    output = 'frame_clip_out_sya'
    if not os.path.exists(output):
        os.mkdir(output)

    count = 4
    for frame in clip.iter_frames():
        if count>1000:
            break
        count += 1
        if count % 4 == 0:
            im = Image.fromarray(frame)
            im.save(output + '/%07d.jpg' % count)


func_pic()

# func_video()
# get_frames()
# mp42gif()
