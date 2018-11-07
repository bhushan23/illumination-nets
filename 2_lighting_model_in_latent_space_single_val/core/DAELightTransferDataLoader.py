import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import os.path
import scipy.io
import random
import csv
import matplotlib.pyplot as plt
import itertools
import torchvision.utils as vutils

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

NUMPY_EXTENSIONS = ['.npy', '.NPY']

PNG_EXTENSIONS = ['.png', '.PNG']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in PNG_EXTENSIONS)

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class FareMultipieLightingTripletsFrontal(data.Dataset):
    # a full fare dataset object
    def __init__(self, opt, root, root_mask,
        resize = 64,
        transform=None, return_paths=False):
        self.opt = opt
        sess_img_map = self.make_dataset_same_face_diff_light_multipie(root)
        if len(sess_img_map) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.image_masks = self.make_dataset_mask_same_face_diff_light_multipie(root_mask)
        
        self.root = root
        self.resize = resize
        # self.ids = img_to_id
        # debugging- save images during init
        self.images = sess_img_map
        # print(img_map)
        # self.save_images()
        self.data_set = self.convert_one_to_one()
        self.transform = transform
        self.return_paths = return_paths
        #self.loader = self.lighting_net

    def save_images(self, resize = 64):
        # print(self.images)
        for session_name, session_val in self.images.items():
            print('Session: ', session_name)
            for (image_id, image_e), val in session_val.items():
                # print('In ', key, val)
                print('Image_id: ', image_id, image_e)
                for key1, val1 in val:
                    print('Opening ', key1, val1)
                    img0 = self.get_image(val1, resize = resize)
                    plt.imshow(img0)
                    #plt.imshow(img0)
                    plt.show()
                # break
            break

    def convert_one_to_one(self, resize = 64):
        data = []
        for session_name, session_val in self.images.items():
            print('Session: ', session_name)
            current_session_mask = self.image_masks[session_name]
            for image_id, val in session_val.items():
                # print('In ', image_id)
                new_list = list(itertools.combinations(val, 2))
                # print(new_list)
                current_image_mask = current_session_mask[image_id]
                current_image_mask = torch.tensor(self.get_image(current_image_mask, resize = resize), dtype = torch.uint8)
                current_image_mask /= 255

                for data_point in new_list:
                    # print(data_point[0], data_point[1])
                    # source1 = torch.zeros((19,))
                    # source1[int(data_point[0][0])] = 1
                    source1 = int(data_point[0][0])
                    image1  = torch.tensor(self.get_image(data_point[0][1], resize = resize))
                    image1 *= current_image_mask
                    # image1  = image1.permute(2, 0, 1)
                    # source2 = torch.zeros((19,))
                    # source2[int(data_point[1][0])] = 1
                    source2 = int(data_point[1][0])
                    # source2 = int(data_point[1][0])
                    image2  = torch.tensor(self.get_image(data_point[1][1], resize = resize))
                    image2 *= current_image_mask
                    #image2  = image2.permute(2, 0, 1)
                    #vutils.save_image(image2, 
                    #    '/nfs/bigdisk/bsonawane/dae-5-out/Test_Out.png',
                    #    nrow=49, normalize = False, padding=2)
                    data.append([source1, image1, source2, image2])                    
        print(len(data))
        print(data[0][1].shape, data[0][3].shape)
        return data

    def get_image(self, image, resize = None):
        with open(image, 'rb') as f0:
            img0 = Image.open(f0)
            img0 = img0.convert('RGB')
            if resize:
                img0 = img0.resize((resize, resize),Image.ANTIALIAS)
            img0 = np.array(img0)
            return img0
        return None
    
    def __getitem__(self, index):
        if index < len(self.data_set):
            img1 = self.data_set[index][1]
            img2 = self.data_set[index][3]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            return self.data_set[index][0], img1, self.data_set[index][2], img2
        return None, None, None, None

    def __len__(self):
        return len(self.data_set)

    def make_dataset_mask_same_face_diff_light_multipie(self, dirpath_root_list):
        img_list = [] # list of path to images
        session_maps = {} # list of images per session
        # img_to_id = [] # stores main id for every image
        print(dirpath_root_list)
        # assert os.path.isdir(dirpath_root)
        for dirpath_root in dirpath_root_list:
            assert os.path.isdir(dirpath_root)
            # img_map  = {} # maps ids to list of index in idl_list
            # hack for finding session name
            # Assuming directory structure
            
            st = dirpath_root
            st = st[:st.rfind('_')]
            session_name = st[st.find('session'):]
            print(dirpath_root, session_name)
            for root, _, fnames in sorted(os.walk(dirpath_root)):
                # print('ROOT:', folder)
                for fname in fnames:
                    if len(fname) < 15:
                        continue
                    if is_image_file(fname):
                        ids, ide, idp, idl = self.parse_imgfilename_fare_multipie(fname)
                        # Currently only working with front pose
                        if idp != '051':
                            continue
                        path_img = os.path.join(root, fname)
                        img_list.append(path_img)
                        # print(ids, idl)
                        if session_name in session_maps:
                            if (ids, ide) in session_maps[session_name]:
                                session_maps[session_name][(ids, ide)] = path_img
                            else:
                                session_maps[session_name][(ids, ide)] = path_img
                        else:
                            session_maps[session_name] = {(ids, ide): path_img}
                        
                        # if ids in img_map:
                        #     img_map[ids].append((idl, path_img))
                        # else:
                        #     img_map[ids] = [(idl, path_img)]
                        #img_to_id.append(ids)
                        # img0 = self.get_image(path_img)
                        # plt.imshow(img0)
                        # plt.show()
            # print(img_map)
        return session_maps #img_map

    def make_dataset_same_face_diff_light_multipie(self, dirpath_root_list):
        img_list = [] # list of path to images
        ids_list = [] # list of ids of the images
        ide_list = [] # list of expression of the images
        idp_list = [] # list of pose/camera of the images
        idl_list = [] # list of lighting of the images
        session_maps = {} # list of images per session
        # img_to_id = [] # stores main id for every image
        print(dirpath_root_list)
        # assert os.path.isdir(dirpath_root)
        for dirpath_root in dirpath_root_list:
            assert os.path.isdir(dirpath_root)
            # img_map  = {} # maps ids to list of index in idl_list
            # hack for finding session name
            # Assuming directory structure
            
            st = dirpath_root
            st = st[:st[:st.rfind('_')].rfind('_')]
            session_name = st[st.find('session'):]
            print(dirpath_root, session_name)
            for root, _, fnames in sorted(os.walk(dirpath_root)):
                # print('ROOT:', folder)
                for fname in fnames:
                    if is_image_file(fname):
                        ids, ide, idp, idl = self.parse_imgfilename_fare_multipie(fname)
                        # Currently only working with pose
                        if idp != '051':
                            continue
                        ids_list.append(ids)
                        ide_list.append(ide)
                        idp_list.append(idp)
                        idl_list.append(idl)
                        path_img = os.path.join(root, fname)
                        img_list.append(path_img)
                        # print(ids, idl)
                        if session_name in session_maps:
                            if (ids, ide) in session_maps[session_name]:
                                session_maps[session_name][(ids, ide)].append([idl, path_img])
                            else:
                                session_maps[session_name][(ids, ide)] = [[idl, path_img]]
                        else:
                            session_maps[session_name] = {(ids, ide):[[idl, path_img]]}
                        
                        # if ids in img_map:
                        #     img_map[ids].append((idl, path_img))
                        # else:
                        #     img_map[ids] = [(idl, path_img)]
                        #img_to_id.append(ids)
                        # img0 = self.get_image(path_img)
                        # plt.imshow(img0)
                        # plt.show()
            # print(img_map)
        return session_maps #img_map

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
