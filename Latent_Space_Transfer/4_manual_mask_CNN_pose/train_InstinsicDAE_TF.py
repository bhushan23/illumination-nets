from __future__ import print_function
import argparse
import os
import sys
sys.path.insert(0, './core')
sys.path.insert(0, './models')

import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import Function
import math

# our data loader
import DAELightTransferDataLoader as lightDL
import gc

ON_SERVER = True

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default = True, action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gpu_ids', type=int, default=0, help='ids of GPUs to use')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--epoch_iter', type=int,default=600, help='number of epochs on entire dataset')
parser.add_argument('--location', type = int, default=0, help ='where is the code running')
parser.add_argument('-f',type=str,default= '', help='dummy input required for jupyter notebook')
parser.add_argument('--modelPath', default='', help="path to model (to continue training)")

if ON_SERVER:
    out_path = './results_CNN_Latent_128_Pose/'  # '/nfs/bigdisk/bsonawane/LightTransfer_Exp/Latent_Space/2_CNN_64/'    #cropped_multipie/4_manual_masking_map_approach_for_lighting_only_multipie_cropped_63/'
    # data_path = '/nfs/bigdisk/zhshu/data/fare/real/multipie_select_batches/'
    data_path = '/nfs/bigmind/add_ssd/zhshu/data/Multipie/crops14/'
    # data_path = '/nfs/bigdisk/multipie_cropped/'
    # data_path = '/nfs/bigdisk/bsonawane/multipie-data/'
else:
    out_path  = '/home/bhushan/work/thesis/Sem2/source/experiment/illumination-nets/1_lighting_transfer_with_unknown_light_source/output'
    data_path = '/home/bhushan/work/thesis/Sem2/source/experiment/illumination-nets/data/multipie_select_batches/'
parser.add_argument('--dirCheckpoints', default=out_path+'/checkpoints/dae-2', help='folder to model checkpoints')
parser.add_argument('--dirImageoutput', default=out_path+'/images/train', help='folder to output images')
parser.add_argument('--dirTestingoutput', default=out_path+'/images/test', help='folder to testing results/images')
parser.add_argument('--dirDataroot', default=data_path, help='folder to dataroot')
parser.add_argument('--useDense', default = True, help='enables dense net architecture')
parser.add_argument('--use_celeba', default= False, help='If true use celebA else Multipie')

opt = parser.parse_args()


# size of image
opt.imgSize=64
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
opt.useDense=True
print(opt)

try:
    os.makedirs(opt.dirCheckpoints)
except OSError:
    pass
try:
    os.makedirs(opt.dirImageoutput)
except OSError:
    pass
try:
    os.makedirs(opt.dirTestingoutput)
except OSError:
    pass


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# sample iamges
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
        '%s/%s'% (output_dir, filename+'.png'),
        nrow=nrow, normalize = normalize, padding=2)

def parseSampledDataPoint(dp0_img, nc):
    dp0_img  = dp0_img.float()/255 # convert to float and rerange to [0,1]
    if nc==1:
        dp0_img  = dp0_img.unsqueeze(3)
    dp0_img  = dp0_img.permute(0,3,1,2).contiguous()  # reshape to [batch_size, 3, img_H, img_W]
    return dp0_img


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


# ---- The model ---- #
# get the model definition/architecture
# get network
import DAENet

if opt.useDense:
    encoders      = DAENet.Dense_Encoders_Intrinsic(opt)
    decoders      = DAENet.Dense_DecodersIntegralWarper2_Intrinsic(opt)
else:
    encoders      = DAENet.Encoders_Intrinsic(opt)
    decoders      = DAENet.DecodersIntegralWarper2_Intrinsic(opt)

# light_transfer    = DAENet.LightingTransfer(opt)

if opt.cuda:
    encoders.cuda()
    decoders.cuda()
    # light_transfer.cuda()

if not opt.modelPath=='':
    # rewrite here
    print('Reload previous model at: '+ opt.modelPath)
    encoders.load_state_dict(torch.load(opt.modelPath+'_encoders.pth'))
    decoders.load_state_dict(torch.load(opt.modelPath+'_decoders.pth'))
    # light_transfer.load_state_dict(torch.load(opt.modelPath+'_lighttransfer.pth'))
else:
    print('No previous model found, initializing model weight.')
    encoders.apply(weights_init)
    decoders.apply(weights_init)
    # light_transfer.apply(weights_init)

print(opt.gpu_ids)
updator_encoders     = optim.Adam(encoders.parameters(), lr = opt.lr, betas=(opt.beta1, 0.999))
updator_decoders     = optim.Adam(decoders.parameters(), lr = opt.lr, betas=(opt.beta1, 0.999))
# updator_lighttran    = optim.Adam(light_transfer.parameters(), lr = opt.lr, betas=(opt.beta1, 0.999))


# criteria/loss
criterionRecon      = nn.L1Loss()
criterionTVWarp     = DAENet.TotalVaryLoss(opt)
criterionBiasReduce = DAENet.BiasReduceLoss(opt)
criterionSmoothL1   = DAENet.TotalVaryLoss(opt)
criterionSmoothL2   = DAENet.SelfSmoothLoss2(opt)

# Training set
TrainingData = []


# Following dataset was used for previous experiments
"""
TrainingData.append(opt.dirDataroot + 'session01_01_select')
TrainingData.append(opt.dirDataroot + 'session01_02_select')
TrainingData.append(opt.dirDataroot + 'session01_03_select')
TrainingData.append(opt.dirDataroot + 'session01_04_select')
TrainingData.append(opt.dirDataroot + 'session01_05_select')
TrainingData.append(opt.dirDataroot + 'session01_06_select')
TrainingData.append(opt.dirDataroot + 'session01_07_select')

TrainingData.append(opt.dirDataroot + 'session02_01_select')
TrainingData.append(opt.dirDataroot + 'session02_02_select')
TrainingData.append(opt.dirDataroot + 'session02_03_select')
TrainingData.append(opt.dirDataroot + 'session02_04_select')
TrainingData.append(opt.dirDataroot + 'session02_05_select')
TrainingData.append(opt.dirDataroot + 'session02_06_select')
TrainingData.append(opt.dirDataroot + 'session02_07_select')

TrainingData.append(opt.dirDataroot + 'session03_01_select')
TrainingData.append(opt.dirDataroot + 'session03_02_select')
TrainingData.append(opt.dirDataroot + 'session03_03_select')
TrainingData.append(opt.dirDataroot + 'session03_04_select')
TrainingData.append(opt.dirDataroot + 'session03_05_select')

TrainingData.append(opt.dirDataroot + 'session04_01_select')
TrainingData.append(opt.dirDataroot + 'session04_02_select')
TrainingData.append(opt.dirDataroot + 'session04_03_select')
TrainingData.append(opt.dirDataroot + 'session04_04_select')
TrainingData.append(opt.dirDataroot + 'session04_05_select')
TrainingData.append(opt.dirDataroot + 'session04_06_select')
TrainingData.append(opt.dirDataroot + 'session04_07_select')


TrainingMask = []
TrainingMask.append(opt.dirDataroot + 'session01_masks')
TrainingMask.append(opt.dirDataroot + 'session02_masks')
TrainingMask.append(opt.dirDataroot + 'session03_masks')
TrainingMask.append(opt.dirDataroot + 'session04_masks')
#
#
# # Testing set
TestingData = []
TestingData.append(opt.dirDataroot + 'session01_select_test')

TestingMask = []
TestingMask.append(opt.dirDataroot + 'session01_masks')
"""

TrainingData = []
# TrainingData.append(opt.dirDataroot + 'small_train')

TrainingData.append(opt.dirDataroot + 'session01_crops14')
# TrainingData.append(opt.dirDataroot + 'session02_crops14')
# TrainingData.append(opt.dirDataroot + 'session03_crops14')
# TrainingData.append(opt.dirDataroot + 'session04_crops14')

TestingData = []
TestingData.append(opt.dirDataroot + 'session_test_crops14')
# TestingData.append(opt.dirDataroot + 'small_test')

TrainingMask = []
TestingMask = []

# ------------ training ------------ #
doTraining = True
doTesting = True
iter_mark=0

print("Loading Dataset")
if opt.use_celeba:
    train_dataset = lightDL.CelebA_DataLoader(dir_path=opt.dirDataroot, batch_size=32, resize=64, is_training=True)
    test_dataset = lightDL.CelebA_DataLoader(dir_path=opt.dirDataroot, batch_size=32, resize=64, is_training=False)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
else:
    dataset = lightDL.FareMultipieLightingTripletsFrontal(None, root=TrainingData, root_mask=TrainingMask, transform = None, resize = 64)
    # train_amount = train_amount + len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))


    dataset_test = lightDL.FareMultipieLightingTripletsFrontal(None, root=TestingData, root_mask = TestingMask, transform = None, resize=64)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
print("Dataset Loaded")


# print('# size of the current (sub)dataset is %d' %len(dataset))
# train_amount = train_amount + len(dataset)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
#
#
# dataset_test = lightDL.FareMultipieLightingTripletsFrontal(None, root=TestingData, root_mask = TestingMask, transform = None, resize=64)
# dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))


# _, src_img, _, dest_img = next(iter(dataloader))
#
# src_img = parseSampledDataPoint(src_img, opt.nc)
# src_img = src_img.type(torch.cuda.FloatTensor)
#
# dest_img = parseSampledDataPoint(dest_img, opt.nc)
# dest_img = dest_img.type(torch.cuda.FloatTensor)
#
# visualizeAsImages(src_img.data.clone(),
#                 opt.dirImageoutput,
#                 filename='TEST_INIT_srcimg0', n_sample = 49, nrow=7, normalize=False)
# visualizeAsImages(dest_img.data.clone(),
#                 opt.dirImageoutput,
#                 filename='TEST_INIT_destImg', n_sample = 49, nrow=7, normalize=False)

print('Log done')
for epoch in range(opt.epoch_iter):
        train_loss = 0
        train_amount = 0+1e-6
        gc.collect() # collect garbage
        encoders.train()
        decoders.train()
        # for dataroot in TrainingData:
        if not doTraining:
           break

        for batch_idx, data_point in enumerate(dataloader, 0):
            #raw_input("Press Enter to continue...")
            gc.collect() # collect garbage
            ### prepare data ###
            dp0_img, dest_light, dest_img = data_point[1], data_point[2], data_point[3]
            # dest_img = dest_img.type(torch.cuda.FloatTensor)
            # dest_img = dest_img.permute(0, 3, 1, 2)

            # print('dest_light: ', dest_light)
            dp0_img = parseSampledDataPoint(dp0_img, opt.nc)
            dp0_img = dp0_img.type(torch.cuda.FloatTensor)

            dest_img = parseSampledDataPoint(dest_img, opt.nc)
            dest_img = dest_img.type(torch.cuda.FloatTensor)

            baseg = getBaseGrid(N=opt.imgSize, getbatch = True, batchSize = dp0_img.size()[0])
            zeroWarp = torch.cuda.FloatTensor(1, 2, opt.imgSize, opt.imgSize).fill_(0)
            if opt.cuda:
                dp0_img, baseg, zeroWarp = setCuda(dp0_img, baseg, zeroWarp)
            dp0_img, = setAsVariable(dp0_img)
            baseg = Variable(baseg, requires_grad=False)
            zeroWarp = Variable(zeroWarp, requires_grad=False)
            updator_decoders.zero_grad()
            updator_encoders.zero_grad()
            #updator_lighttran.zero_grad()
            decoders.zero_grad()
            encoders.zero_grad()
            # light_transfer.zero_grad()
            ### forward training points: dp0
            dp0_z, dp0_zS, dp0_zT, dp0_zW = encoders(dp0_img)
            baseg = baseg.type(torch.cuda.FloatTensor)
            # new_zS = light_transfer(dest_light, do0_zS)
            dp0_S, dp0_T, dp0_I, dp0_W, dp0_output, dp0_Wact = decoders(dest_light, dp0_zS, dp0_zT, dp0_zW, baseg)
            # reconstruction loss
            loss_recon = criterionRecon(dp0_output, dest_img)
            # smooth warping loss
            loss_tvw = criterionTVWarp(dp0_W, weight=1e-6)
            # bias reduce loss
            loss_br = criterionBiasReduce(dp0_W, zeroWarp, weight=1e-2)
            # intrinsic loss :Shading, L2
            loss_intr_S = criterionSmoothL2(dp0_S, weight = 1e-6)
            # all loss functions
            loss_all = loss_recon + loss_tvw + loss_br + loss_intr_S
            loss_all.backward()

            updator_decoders.step()
            updator_encoders.step()

            loss_encdec = loss_recon.data[0] + loss_br.data[0] + loss_tvw.data[0] + loss_intr_S.data[0]

            train_loss += loss_encdec

            iter_mark+=1
            print('Iteration[%d] loss -- all:  %.4f .. recon:  %.4f .. tvw: %.4f .. br: %.4f .. intr_s: %.4f .. '
                % (iter_mark,  loss_encdec, loss_recon.data[0], loss_tvw.data[0], loss_br.data[0], loss_intr_S.data[0]))
            # visualzing training progress
            gx = (dp0_W.data[:,0,:,:]+baseg.data[:,0,:,:]).unsqueeze(1).clone()
            gy = (dp0_W.data[:,1,:,:]+baseg.data[:,1,:,:]).unsqueeze(1).clone()
            visualizeAsImages(dp0_img.data.clone(),
                opt.dirImageoutput,
                filename='iter_'+str(iter_mark)+'_img0_', n_sample = 49, nrow=7, normalize=False)
            visualizeAsImages(dest_img.data.clone(),
                opt.dirImageoutput,
                filename='iter_'+str(iter_mark)+'_destImg_', n_sample = 49, nrow=7, normalize=False)
            #visualizeAsImages(dp0_I.data.clone(),
            #    opt.dirImageoutput,
            #    filename='iter_'+str(iter_mark)+'_tex0_', n_sample = 49, nrow=7, normalize=False)
            #visualizeAsImages(dp0_S.data.clone(),
            #    opt.dirImageoutput,
            #    filename='iter_'+str(iter_mark)+'_intr_shade0_', n_sample = 49, nrow=7, normalize=False)
            #visualizeAsImages(dp0_T.data.clone(),
            #    opt.dirImageoutput,
            #    filename='iter_'+str(iter_mark)+'_intr_tex0_', n_sample = 49, nrow=7, normalize=False)
            visualizeAsImages(dp0_output.data.clone(),
                opt.dirImageoutput,
                filename='iter_'+str(iter_mark)+'_output0_', n_sample = 49, nrow=7, normalize=False)
            #visualizeAsImages((gx+1)/2,
            #    opt.dirImageoutput,
            #    filename='iter_'+str(iter_mark)+'_warp0x_', n_sample = 49, nrow=7, normalize=False)
            #visualizeAsImages((gy+1)/2,
            #    opt.dirImageoutput,
            #    filename='iter_'+str(iter_mark)+'_warp0y_', n_sample = 49, nrow=7, normalize=False)
            if doTraining:
             # do checkpointing
             torch.save(encoders.state_dict(), '%s/wasp_model_epoch_encoders.pth' % (opt.dirCheckpoints))
             torch.save(decoders.state_dict(), '%s/wasp_model_epoch_decoders.pth' % (opt.dirCheckpoints))

     
        # ------------ testing ------------ #
         
        # on synthetic image set
        print('Testing images ... ')
        #raw_input("Press Enter to continue...")
        testing_loss=0
        gc.collect() # collect garbage
        if doTesting:
           encoders.train()
           decoders.train()
           for batch_idx, data_point in enumerate(dataloader_test, 0):
               #raw_input("Press Enter to continue...")
               gc.collect() # collect garbage
               ### prepare data ###
               dp0_img, dest_light, dest_img = data_point[1], data_point[2], data_point[3]

               dest_img = parseSampledDataPoint(dest_img, opt.nc)
               dest_img = dest_img.type(torch.cuda.FloatTensor)
   
               dp0_img = parseSampledDataPoint(dp0_img, opt.nc)
               baseg = getBaseGrid(N=opt.imgSize, getbatch = True, batchSize = dp0_img.size()[0])
               zeroWarp = torch.cuda.FloatTensor(1, 2, opt.imgSize, opt.imgSize).fill_(0)
               if opt.cuda:
                   dp0_img, baseg, zeroWarp = setCuda(dp0_img, baseg, zeroWarp)
               dp0_img, = setAsVariable(dp0_img)
               baseg = Variable(baseg, requires_grad=False)
               zeroWarp = Variable(zeroWarp, requires_grad=False)
               updator_decoders.zero_grad()
               updator_encoders.zero_grad()
               decoders.zero_grad()
               encoders.zero_grad()
               dp0_img = dp0_img.type(torch.cuda.FloatTensor)
               
               ### forward training points: dp0
               dp0_z, dp0_zS, dp0_zT, dp0_zW = encoders(dp0_img)
               baseg = baseg.type(torch.cuda.FloatTensor)
               dp0_S, dp0_T, dp0_I, dp0_W, dp0_output, dp0_Wact = decoders(dest_light, dp0_zS, dp0_zT, dp0_zW, baseg)
               # reconstruction loss
               loss_recon = criterionRecon(dp0_output, dest_img)
               # smooth warping loss
               loss_tvw = criterionTVWarp(dp0_W, weight=1e-6)
               # bias reduce loss
               loss_br = criterionBiasReduce(dp0_W, zeroWarp, weight=1e-2)
               # intrinsic loss :Shading, L2
               loss_intr_S = criterionSmoothL2(dp0_S, weight = 1e-6)
               # all loss functions
               loss_all = loss_recon + loss_tvw + loss_br + loss_intr_S
   
               loss_encdec = loss_recon.data[0] + loss_br.data[0] + loss_tvw.data[0] + loss_intr_S.data[0] 
   
               testing_loss += loss_encdec
               
               print('Iteration[%d] loss -- all:  %.4f .. recon:  %.4f .. tvw: %.4f .. br: %.4f .. intr_s: %.4f .. ' 
                   % (iter_mark,  loss_encdec, loss_recon.data[0], loss_tvw.data[0], loss_br.data[0], loss_intr_S.data[0]))
               # visualzing training progress
               print('Storing:' )
               gx = (dp0_W.data[:,0,:,:]+baseg.data[:,0,:,:]).unsqueeze(1).clone()
               gy = (dp0_W.data[:,1,:,:]+baseg.data[:,1,:,:]).unsqueeze(1).clone()
               visualizeAsImages(dp0_img.data.clone(), 
                   opt.dirTestingoutput, 
                   filename='img0_'+str(iter_mark), n_sample = 49, nrow=7, normalize=False)           
               #visualizeAsImages(dp0_I.data.clone(), 
               #    opt.dirTestingoutput, 
               #    filename='tex0_'+str(iter_mark), n_sample = 49, nrow=7, normalize=False)
               visualizeAsImages(dest_img.data.clone(), 
                   opt.dirTestingoutput, 
                   filename='expected_output0_'+str(iter_mark), n_sample = 49, nrow=7, normalize=False)           
               #visualizeAsImages(dp0_S.data.clone(), 
               #    opt.dirTestingoutput, 
               #    filename='intr_shade0_'+str(iter_mark), n_sample = 49, nrow=7, normalize=False)
               #visualizeAsImages(dp0_T.data.clone(), 
               #    opt.dirTestingoutput, 
               #    filename='intr_tex0_'+str(iter_mark), n_sample = 49, nrow=7, normalize=False)
               visualizeAsImages(dp0_output.data.clone(), 
                   opt.dirTestingoutput, 
                   filename='output0_'+str(iter_mark), n_sample = 49, nrow=7, normalize=False)   
               #visualizeAsImages((gx+1)/2, 
               #    opt.dirTestingoutput, 
               #    filename='warp0x_'+str(iter_mark), n_sample = 49, nrow=7, normalize=False)          
               #visualizeAsImages((gy+1)/2, 
               #    opt.dirTestingoutput, 
               #    filename='warp0y_'+str(iter_mark), n_sample = 49, nrow=7, normalize=False)   
               break
           # put testing code here #
           gc.collect() # collect garbage





























    ##
