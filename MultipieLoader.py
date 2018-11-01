import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import os.path
import scipy.io
import random
import csv

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

NUMPY_EXTENSIONS = ['.npy', '.NPY']

PNG_EXTENSIONS = ['.png', '.PNG']


def duplicates(lst, item, match = True):
    if match:
        return [i for i, x in enumerate(lst) if x == item]
    else:
        return [i for i, x in enumerate(lst) if not x == item]

def DefaultAxisRotate(R):
    DefaultR = torch.Tensor(4,4).fill_(0)
    DefaultR[0,0] = -1
    DefaultR[1,1] = 1
    DefaultR[2,2] = -1
    DefaultR[3,3] = 1
    return torch.mm(DefaultR,R)

def DefaultAxisRotate2(R):
    DefaultR = torch.Tensor(4,4).fill_(0)
    DefaultR[0,0] = -1
    DefaultR[1,1] = -1
    DefaultR[2,2] = 1
    DefaultR[3,3] = 1
    return torch.mm(DefaultR,R)

def getHomogeneousExtrinsicMatrixFromIdp(idp,  pose_file, DefaultRotate = True):
    exR = pose_file['fare_poseList']['exR'][0,idp]
    exT = pose_file['fare_poseList']['exT'][0,idp]
    EXM = torch.Tensor(4,4).fill_(0)
    EXM[0:3,0:3] = torch.from_numpy(exR)
    EXM[0:3,3] = torch.from_numpy(exT)
    EXM[3,3]=1
    if DefaultRotate:
        EXM = DefaultAxisRotate(EXM)
    return EXM

def intersect(*d):
    sets = iter(map(set, d))
    result = sets.next()
    for s in sets:
        result = result.intersection(s)
    return result

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in PNG_EXTENSIONS)

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def parse_imgfilename_fare(fn):
    ids = fn[str.rfind(fn,'_ids_')+5 : str.rfind(fn,'_ide_')]
    ide = fn[str.rfind(fn,'_ide_')+5 : str.rfind(fn,'_idp_')]
    idp = fn[str.rfind(fn,'_idp_')+5 : str.rfind(fn,'_idt_')]
    idt = fn[str.rfind(fn,'_idt_')+5 : str.rfind(fn,'_idl_')]
    idl = fn[str.rfind(fn,'_idl_')+5 : -4]
    return ids, ide, idp, idt , idl


def parse_imgfilename_fare_multipie(fn):
    ids = fn[0:3]
    ide = fn[7:9]
    idp = fn[10:13]
    idl = fn[14:16]
    return ids, ide, idp, idl

class FareMultipieLightingTripletsFrontal(data.Dataset):
    # a full fare dataset object
    def __init__(self, opt, root, 
        resize = 64,
        transform=None, return_paths=False):
        self.opt = opt
        imgs, ids, ide, idp, idl = self.make_dataset_fare_multipie(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.resize = resize
        self.imgs = imgs
        self.ids = ids
        self.ide = ide
        self.idp = idp
        self.idl = idl
        self.transform = transform
        self.return_paths = return_paths
        self.loader = self.fareloader_lighting_triplet

    def __getitem__(self, index):
        while not (self.idp[index] == '051'):
            index = self.resample()
        imgPath0 = self.imgs[index]
        # different lighting
        coindex9 = self.getCoindex9(index)
        # different person
        coindex1 = self.getCoindex1(index)
        imgPath9 = self.imgs[coindex9]
        imgPath1 = self.imgs[coindex1]

        img0,img9,img1 = self.loader(imgPath0, imgPath9, imgPath1)

        return img0, img9, img1

    def __len__(self):
        return len(self.imgs)

    def make_dataset_fare_multipie(self, dirpath_root):
        img_list = [] # list of path to images
        ids_list = [] # list of ids of the images
        ide_list = [] # list of expression of the images
        idp_list = [] # list of pose/camera of the images
        idl_list = [] # list of lighting of the images
        print(dirpath_root)
        assert os.path.isdir(dirpath_root)
        for root, _, fnames in sorted(os.walk(dirpath_root)):
            for fname in fnames:
                if is_image_file(fname):
                    ids, ide, idp, idl = self.parse_imgfilename_fare_multipie(fname)
                    ids_list.append(ids)
                    ide_list.append(ide)
                    idp_list.append(idp)
                    idl_list.append(idl)
                    path_img = os.path.join(root, fname)
                    img_list.append(path_img)
        return img_list, ids_list, ide_list, idp_list, idl_list

    def get_Sample(self, filepath):
        # print(self.ids)
        # print(self.ide)
        # print(self.idp)
        # print(self.idl)
        with open(filepath, 'wb') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(['ids', 'ide', 'idp', 'idl'])
            for i in range(len(self.ids)):
                csvwriter.writerow([self.ids[i], self.ide[i], self.idp[i], self.idl[i]])


    def parse_imgfilename_fare_multipie(self, fn):
        ids = fn[0:3]
        ide = fn[7:9]
        idp = fn[10:13]
        idl = fn[14:16]
        return ids, ide, idp, idl

    def getitem(self, ids, e, p, l):
        # different ids, same ide, same idl, same idt, same idp
        # print(len(self.ids))
        # print(self.idl)
        for i in range(len(self.ids)):
            if(self.ids[i] == ids and self.ide[i] == e and self.idp[i] == p and self.idl[i] == l):
                return self.imgs[i]

    # def fareloader_lighting_triplet(self, imgPath0, imgPath9, imgPath1):
    #     resize=self.resize
    #     with open(imgPath0, 'rb') as f0:
    #         with Image.open(f0) as img0:
    #             img0 = img0.convert('RGB')
    #             if resize:
    #                 img0 = img0.resize((resize, resize),Image.ANTIALIAS)
    #             img0 = np.array(img0)
    #     with open(imgPath9, 'rb') as f9:
    #         with Image.open(f9) as img9:
    #             img9 = img9.convert('RGB')
    #             if resize:
    #                 img9 = img9.resize((resize, resize),Image.ANTIALIAS)
    #             img9 = np.array(img9)
    #     with open(imgPath1, 'rb') as f1:
    #         with Image.open(f1) as img1:
    #             img1 = img1.convert('RGB')
    #             if resize:
    #                 img1 = img1.resize((resize, resize),Image.ANTIALIAS)
    #             img1 = np.array(img1)
    #     return img0, img9, img1

    def fareloader_lighting_triplet(self, imgPath0, imgPath9, imgPath1):
        resize=self.resize
        with open(imgPath0, 'rb') as f0:
            	img0 = Image.open(f0)
                img0 = img0.convert('RGB')
                if resize:
                    img0 = img0.resize((resize, resize),Image.ANTIALIAS)
                img0 = np.array(img0)
        with open(imgPath9, 'rb') as f9:
            	img9 = Image.open(f9)
                img9 = img9.convert('RGB')
                if resize:
                    img9 = img9.resize((resize, resize),Image.ANTIALIAS)
                img9 = np.array(img9)
        with open(imgPath1, 'rb') as f1:
            	img1 = Image.open(f1)
                img1 = img1.convert('RGB')
                if resize:
                    img1 = img1.resize((resize, resize),Image.ANTIALIAS)
                img1 = np.array(img1)
        return img0, img9, img1


    def getCoindex9(self, index):
        # same ids, same ide, different idl, same idt, same idp
        s = duplicates(self.ids, self.ids[index], match = True)
        e = duplicates(self.ide, self.ide[index], match = True)
        l = duplicates(self.idl, self.idl[index], match = False)
        p = duplicates(self.idp, self.idp[index], match = True)
        ava = intersect(s,e,l,p)
        if len(ava)>0:
            return random.sample(ava, 1)[0]
        else:
            return index    

    def getCoindex1(self, index):
        # different ids, same ide, same idl, same idt, same idp
        s = duplicates(self.ids, self.ids[index], match = False)
        e = duplicates(self.ide, self.ide[index], match = True)
        l = duplicates(self.idl, self.idl[index], match = True)
        p = duplicates(self.idp, self.idp[index], match = True)
        ava = intersect(s,e,l,p)
        if len(ava)>0:
            return random.sample(ava, 1)[0]
        else:
            return index  

    def inClique(self, a, b):
        c1 = ['041','050', '051']
        c2 = ['080', '090', '120']
        c3 = ['080', '130', '140', '051']
        c4 = ['190', '200', '041']
        if ((a in c1) and (b in c1)) or ((a in c2) and (b in c2)) or ((a in c3) and (b in c3)) or ((a in c4) and (b in c4)):
            return True
        else:
            return False

    def resample(self):
        index = np.random.randint(len(self.imgs), size=1)[0]
        return index


