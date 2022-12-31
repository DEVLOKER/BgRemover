import os
import sys
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from modnet.src.models.modnet import MODNet

class BackgroundRemover(object):

    """         static variables      """
    CKPT_PATH = "modnet/pretrained/modnet_photographic_portrait_matting.ckpt"
    INPUT_PATH = "files/uploaded"
    OUTPUT_PATH = "files/processed"
    # TMP_PATH = "files/tmp"
    # define hyper-parameters
    REF_SIZE = 512

    def __init__(self):
        # define image to tensor transform
        self.im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        # create MODNet and load the pre-trained ckpt
        self.modnet = MODNet(backbone_pretrained=False)
        self.modnet = nn.DataParallel(self.modnet)

        if torch.cuda.is_available():
            self.modnet = self.modnet.cuda()
            weights = torch.load(BackgroundRemover.CKPT_PATH)
        else:
            weights = torch.load(BackgroundRemover.CKPT_PATH, map_location=torch.device('cpu'))
        self.modnet.load_state_dict(weights)
        self.modnet.eval()



    def process(self, im_name):
        # read image
        image = Image.open(os.path.join(BackgroundRemover.INPUT_PATH, im_name))
        im = Image.open(os.path.join(BackgroundRemover.INPUT_PATH, im_name))


        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = self.im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < BackgroundRemover.REF_SIZE or min(im_h, im_w) > BackgroundRemover.REF_SIZE:
            if im_w >= im_h:
                im_rh = BackgroundRemover.REF_SIZE
                im_rw = int(im_w / im_h * BackgroundRemover.REF_SIZE)
            elif im_w < im_h:
                im_rw = BackgroundRemover.REF_SIZE
                im_rh = int(im_h / im_w * BackgroundRemover.REF_SIZE)
        else:
            im_rh = im_h
            im_rw = im_w
        
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = self.modnet(im.cuda() if torch.cuda.is_available() else im, True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        matte_name = im_name.split('.')[0] + '.png'
        matte = Image.fromarray(((matte * 255).astype('uint8')), mode='L')
        combined = self.combine(image, matte)
        combined.save(os.path.join(BackgroundRemover.OUTPUT_PATH, matte_name))
        return matte_name


    def processAll(self):
        # inference images
        im_names = os.listdir(BackgroundRemover.INPUT_PATH)
        for im_name in im_names:
            print('Process image: {0}'.format(im_name))
            self.process(im_name)


    def combine(self, image, matte):

        # obtain predicted foreground
        image = np.asarray(image)
        if len(image.shape) == 2:
            image = image[:, :, None]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, 0:3]
        matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
        combined = image * matte + np.full(image.shape, 255) * (1 - matte)
        combined = Image.fromarray(np.uint8(combined))
        return combined