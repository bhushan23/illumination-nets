import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import torchvision.utils as vutils
import argparse
import sys

from torch.autograd import Variable
from PIL import Image
import pdb

def setCuda(*args):
    barg = []
    for arg in args:
        barg.append(arg.cuda())
    return barg

def setAsVariable(*args):
    barg = []
    for arg in args:
        barg.append(Variable(arg))
    return barg

def getBaseGrid(N=64, normalize = True, getbatch = False, batchSize = 1):
    a = torch.arange(-(N-1), (N), 2)
    if normalize:
        a = a/(N-1)
    x = a.repeat(N,1)
    y = x.t()
    grid = torch.cat((x.unsqueeze(0), y.unsqueeze(0)),0)
    if getbatch:
        grid = grid.unsqueeze(0).repeat(batchSize,1,1,1)
    return grid

def load_encoder(path):
    model = DAENet.Dense_Encoders_Intrinsic(opt)
    model.load_state_dict(torch.load(path))
    model, = setCuda(model)
    model.eval()
    return model

def load_decoder(path):
    model = DAENet.Dense_DecodersIntegralWarper2_Intrinsic(opt)
    model.load_state_dict(torch.load(path))
    model, = setCuda(model)
    model.eval()
    return model

def visualizeAsImages(img_list, output_dir,
                      n_sample=4, id_sample=None, dim=-1,
                      filename='myimage', nrow=2,
                      normalize=False):
    if id_sample is None:
        images = img_list[0:n_sample,:,:,:]
    else:
        images = img_list[id_sample,:,:,:]
    if dim >= 0:
        images = images[:,dim,:,:].unsqueeze(1)
    vutils.save_image(images,
        '%s'% (output_dir+'.png'),
        nrow=nrow, normalize = normalize, padding=2)

def parseSampledDataPoint(dp0_img, nc):
    dp0_img  = dp0_img.float()/255 # convert to float and rerange to [0,1]
    if nc==1:
        dp0_img  = dp0_img.unsqueeze(3)
    dp0_img  = dp0_img.permute(0,3,1,2).contiguous()  # reshape to [batch_size, 3, img_H, img_W]
    return dp0_img
    
def load_image(image_path):
    with open(image_path, 'rb') as f0:
        img0 = Image.open(f0)
        img0 = img0.convert('RGB')
        img0 = img0.resize((64, 64), Image.ANTIALIAS)
        img0 = np.array(img0)
        return img0
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gth_dir', type=str)
    parser.add_argument('--output_file', type=str)
    opt = parser.parse_args()

    l = []
    
    # output_files = ['00', '03', '03', '03', '05', '05', '06', '07', '08', '09', '10', '11', '11', '14', '14', '15', '16', '17', '18', '19']
    output_files = ['00', '03', '05', '06', '07', '08', '09', '10', '11', '14', '15', '16', '17', '18', '19']

    for out_file in output_files:
        t_img = load_image(opt.gth_dir+'/'+ out_file + '.png')   #str(i).zfill(2)+'.png')
        t_img = np.stack([t_img], axis = 0)
        t_img = torch.tensor(t_img).type(torch.FloatTensor)
        t_img = parseSampledDataPoint(t_img, 3)
        l.append(t_img.squeeze())

    l = np.stack(l, axis=0)
    l = torch.tensor(l)
    
    visualizeAsImages(l.data.clone(), opt.output_file, filename="interpolated", n_sample=15, nrow=5, normalize=False)
