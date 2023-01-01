import os
import sys
import numpy as np
from PIL import Image
import cv2

from tqdm import tqdm
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
        self.torch_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        # create MODNet and load the pre-trained ckpt
        self.modnet = MODNet(backbone_pretrained=False)
        self.modnet = nn.DataParallel(self.modnet)
        
        
        GPU = torch.cuda.is_available() # True if torch.cuda.device_count() > 0 else False
        if GPU:
            print('Use GPU...')
            self.modnet = self.modnet.cuda()
            weights = torch.load(BackgroundRemover.CKPT_PATH)
        else:
            print('Use CPU...')
            weights = torch.load(BackgroundRemover.CKPT_PATH, map_location=torch.device('cpu'))
        self.modnet.load_state_dict(weights)
        self.modnet.eval()



    def processImage(self, im_name):
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
        im = self.torch_transforms(im)

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
        combined = self.__combine(image, matte)
        combined.save(os.path.join(BackgroundRemover.OUTPUT_PATH, matte_name))
        return matte_name



    def processVideo(self, vi_name):

        result_type = 'fg' # matte - save the alpha matte; fg - save the foreground
        matte_name = os.path.splitext(vi_name)[0] + '_{0}.webm'.format(result_type) # mp4 avi webm
        result = os.path.join(BackgroundRemover.OUTPUT_PATH, matte_name)
        alpha_matte = True if result_type == 'matte' else False
        fps = 30

        # video capture
        vc = cv2.VideoCapture(os.path.join(BackgroundRemover.INPUT_PATH, vi_name))

        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False

        if not rval:
            print('Failed to read the video: {0}'.format(vi_name))
            exit()

        num_frame = vc.get(cv2.CAP_PROP_FRAME_COUNT)
        h, w = frame.shape[:2]
        if w >= h:
            rh = 512
            rw = int(w / h * 512)
        else:
            rw = 512
            rh = int(h / w * 512)
        rh = rh - rh % 32
        rw = rw - rw % 32

        # video writer
        fourcc = cv2.VideoWriter_fourcc(*'VP80') #  'MP4V' 'MJPG' 'XVID' 'DIVX' 'MPEG' 'VP80' 'H264'
        video_writer = cv2.VideoWriter(result, fourcc, fps, (w, h))

        print('Start matting...')
        with tqdm(range(int(num_frame)))as t:
            for c in t:
                frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_np = cv2.resize(frame_np, (rw, rh), cv2.INTER_AREA)

                frame_PIL = Image.fromarray(frame_np)
                frame_tensor = self.torch_transforms(frame_PIL)
                frame_tensor = frame_tensor[None, :, :, :]
                if torch.cuda.is_available(): # GPU
                    frame_tensor = frame_tensor.cuda()

                with torch.no_grad():
                    _, _, matte_tensor = self.modnet(frame_tensor, True)

                matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
                matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)
                if alpha_matte:
                    view_np = matte_np * np.full(frame_np.shape, 255.0)
                else:
                    view_np = matte_np * frame_np + (1 - matte_np) * np.full(frame_np.shape, 255.0)
                view_np = cv2.cvtColor(view_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
                view_np = cv2.resize(view_np, (w, h))
                video_writer.write(view_np)

                rval, frame = vc.read()
                c += 1

        video_writer.release()
        print('Save the result video to {0}'.format(result))



    def processAll(self):
        # load files
        files = os.listdir(BackgroundRemover.INPUT_PATH)
        for fileName in files:
            extension = fileName.split('.')[1]
            if extension in ["png", "jpg", "jpeg"]:
                print('Process image: {0}'.format(fileName))
                self.processImage(fileName)
            if extension in ["mp4"]:
                print('Process video: {0}'.format(fileName))
                self.processVideo(fileName)


    def __combine(self, image, matte):

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