import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import torchvision.utils as vutils
import argparse
import sys

src_path = '../Latent_Space_Transfer/4_manual_mask_CNN_63_instance_norm/'  #   '../3_manual_masking_map_approach_for_lighting/' 
sys.path.insert(0, src_path + 'core')
sys.path.insert(0, src_path + 'models')

import DAENet
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
    
def get_illumination_map(map_type):
    if map_type == 1:
        illumination_map = {}
        illumination_map[0] = [[0] * 63 for i in range(63)]
        illumination_map[1] = [[0] * 54 + [1] * 9 for i in range(63)]
        illumination_map[2] = [[0] * 45 + [1] * 18 for i in range(63)]
        illumination_map[3] = [[0] * 36 + [1] * 27 for i in range(63)]
        illumination_map[4] = [[0] * 27 + [1] * 36 for i in range(63)]
        illumination_map[5] = [[0] * 18 + [1] * 45 for i in range(63)]
        illumination_map[6] = [[0] * 9 + [1] * 54 for i in range(63)]
        illumination_map[7] = [[0] * 0 + [1] * 63 for i in range(63)]
        illumination_map[8] = [[1] * 54 + [0] * 9 for i in range(63)]
        illumination_map[9] = [[1] * 45 + [0] * 18 for i in range(63)]
        illumination_map[10] = [[1] * 36 + [0] * 27 for i in range(63)]
        illumination_map[11] = [[1] * 27 + [0] * 36 for i in range(63)]
        illumination_map[12] = [[1] * 18 + [0] * 45 for i in range(63)]
        illumination_map[13] = [[1] * 9 + [0] * 54 for i in range(63)]
        illumination_map[14] = [[0] * 27 + [1] * 36 for i in range(9)] + [[0] * 18 + [1] * 45 for i in range(45)] + [[0] * 27 + [1] * 36 for i in range(9)]
        illumination_map[15] = [[0] * 27 + [1] * 36 for i in range(9)] + [[0] * 9 + [1] * 54 for i in range(45)] + [[0] * 27 + [1] * 36 for i in range(9)]
        illumination_map[16] = [[0] * 9 + [1] * 45 + [0] * 9 for i in range(45)] + [[0] * 18 + [1] * 27 + [0] * 18 for i in range(18)]
        illumination_map[17] = [[1] * 36 + [0] * 27 for i in range(9)] + [[1] * 45 + [0] * 18 for i in range(45)] + [[1] * 36 + [0] * 27 for i in range(9)]
        illumination_map[18] = [[1] * 36 + [0] * 27 for i in range(9)] + [[1] * 9 + [0] * 54 for i in range(45)] + [[1] * 27 + [0] * 36 for i in range(9)]
        illumination_map[19] = [[0] * 63 for i in range(63)]
    else:
        illumination_map = {}

    return illumination_map
    

def load_image(image_path):
    with open(image_path, 'rb') as f0:
        img0 = Image.open(f0)
        img0 = img0.convert('RGB')
        img0 = img0.resize((64, 64), Image.ANTIALIAS)
        img0 = np.array(img0)
        return img0
    return None

def dae_step(opt, d, e, img, dest_light):
    baseg = getBaseGrid(N=opt.imgSize, getbatch=True, batchSize=img.size()[0])
    img = img.type(torch.cuda.FloatTensor)
    if opt.cuda:
        dp0_img, baseg = setCuda(img, baseg)
    img, = setAsVariable(img)
    baseg = Variable(baseg, requires_grad=False)
    dp0_z, dp0_zS, dp0_zT, dp0_zW = e(img)
    baseg = baseg.type(torch.cuda.FloatTensor)
    #pdb.set_trace()
    dp0_S, dp0_T, dp0_I, dp0_W, dp0_output, dp0_Wact = d(dest_light, dp0_zS, dp0_zT, dp0_zW, baseg)
    return dp0_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--gth_dir', type=str)
    #parser.add_argument('--output_dir', type=str)
    parser.add_argument('--map_type', type=int)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--gpu_ids', type=int, default=0, help='ids of GPUs to use')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--epoch_iter', type=int, default=600, help='number of epochs on entire dataset')
    parser.add_argument('--location', type=int, default=0, help='where is the code running')
    parser.add_argument('-f', type=str, default='', help='dummy input required for jupyter notebook')
    parser.add_argument('--modelPath', default='', help="path to model (to continue training)")

    opt = parser.parse_args()
    opt.imgSize = 64
    opt.cuda = True
    opt.use_dropout = 0
    opt.ngf = 32
    opt.ndf = 32
    # dimensionality: shading latent code
    opt.sdim = 16
    # dimensionality: albedo latent code
    opt.tdim = 16
    # dimensionality: texture (shading*albedo) latent code
    opt.idim = opt.sdim + opt.tdim
    # dimensionality: warping grid (deformation field) latent code
    opt.wdim = 128
    # dimensionality of general latent code (before disentangling)
    opt.zdim = 128
    opt.use_gpu = True
    opt.gpu_ids = 0
    opt.ngpu = 1
    opt.nc = 3
    opt.useDense = True
    
    
    decoder = load_decoder(opt.checkpoint_path+'/wasp_model_epoch_decoders.pth')
    encoder = load_encoder(opt.checkpoint_path+'/wasp_model_epoch_encoders.pth')
    # illumination_map = get_illumination_map()
    l = []
    loss = []
    lossCrit = nn.L1Loss()
    img = load_image(opt.image_path)
    img = np.stack([img], axis = 0)
    img = torch.tensor(img)
    
    loss_file = open(opt.output_file + '.loss.csv', 'w')
    output_files = ['00', '03', '05', '06', '07', '08', '09', '10', '11', '14', '15', '16', '17', '18', '19']
    for out_file in output_files:
        loss_file.write(out_file + ',')
    loss_file.write('\n')
    for out_file in output_files:
        img1 = parseSampledDataPoint(img, opt.nc)
        t_img = load_image(opt.gth_dir+'/'+ out_file + '.png')   #str(i).zfill(2)+'.png')
        t_img = np.stack([t_img], axis = 0)
        t_img = torch.tensor(t_img).type(torch.FloatTensor)
        t_img = parseSampledDataPoint(t_img, opt.nc)
        # t_img = t_img.permute([0, 3, 1, 2])
        pred = dae_step(opt, decoder, encoder, img1, [int(out_file)]).detach().cpu()
        pred = pred.type(torch.FloatTensor)
        # t_img = t_img.squeeze(dim=0)
        # pred = pred.squeeze(dim=0)
        # print(pred.shape, t_img.shape)
        tmp_loss = lossCrit(pred, t_img).item()
        loss_file.write(str(tmp_loss)+',')
        l.append(pred.squeeze())

    l = np.stack(l, axis=0)
    l = torch.tensor(l)
    
    visualizeAsImages(l.data.clone(), opt.output_file, filename="interpolated", n_sample=20, nrow=5, normalize=False)


    
    

